"""Convert independent xView into a CV_Assignment_AAL-shaped root for the 5 truck classes.

Writes <out>/{train,val}/{images,labels} with images as {id}.png and YOLO-normalised labels, so the
frozen tiler runs on it unchanged:  python src/tile.py --root <out> --out <xview_tiles> --splits train val

Class mapping is by NAME (robust to xView's sparse type_id numbering): the xView class-labels file
maps type_id -> name; we keep the type_ids whose names equal our 5 classes (data.CLASSES) and remap
them to 0..4 in that exact order. A name that fails to match aborts loudly (check the labels file).

Only independent IDs (experiments/splits/xview_overlapfree_ids.txt from xview_dedup.py) are converted.
A small, deterministic slice is held out as pretrain-val (loss/mAP monitoring only; report last.pt).
By default only images containing >=1 of the 5 classes are kept; --include-empty adds truck-free
xView images as extra negatives.

Usage:
  python src/xview_to_yolo.py --geojson /data/xview/xView_train.geojson \
      --xview-images /data/xview/train_images --class-labels /data/xview/xview_class_labels.txt \
      --overlapfree experiments/splits/xview_overlapfree_ids.txt --out /content/xview_root
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from data import CLASSES

Image.MAX_IMAGE_PIXELS = None


def norm(s):
    return " ".join(s.split()).strip().lower()


def load_type_map(class_labels):
    """xView type_id (int) -> our class index 0..4, for the 5 names in data.CLASSES. Aborts if a name is missing."""
    name_to_tid = {}
    for line in Path(class_labels).read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        tid, name = line.split(":", 1)
        name_to_tid[norm(name)] = int(tid)
    mapping = {}
    for idx, cls in enumerate(CLASSES):
        key = norm(cls)
        if key not in name_to_tid:
            raise SystemExit(f"class '{cls}' not found in {class_labels}; present names: {sorted(name_to_tid)}")
        mapping[name_to_tid[key]] = idx
    print("type_id -> class:", {tid: CLASSES[i] for tid, i in mapping.items()})
    return mapping


def collect_boxes(geojson, type_map):
    """{image_id_stem: [(our_cls, x1, y1, x2, y2), ...]} for the 5 truck classes only."""
    feats = json.load(open(geojson))["features"]
    boxes, skipped = defaultdict(list), 0
    for f in feats:
        pr = f.get("properties", {})
        tid = pr.get("type_id")
        if tid not in type_map:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in str(pr["bounds_imcoords"]).split(","))
        except (KeyError, ValueError):
            skipped += 1
            continue
        if x2 <= x1 or y2 <= y1:  # degenerate box (known xView noise)
            skipped += 1
            continue
        boxes[Path(pr["image_id"]).stem].append((type_map[tid], x1, y1, x2, y2))
    print(f"geojson: {len(feats)} features -> {sum(len(v) for v in boxes.values())} truck boxes "
          f"in {len(boxes)} images ({skipped} boxes skipped: degenerate/unparsable)")
    return boxes


def write_image(img_path, boxes, out_dir, split):
    """Save {id}.png + YOLO label; returns per-class counts (or None if the image is unreadable)."""
    stem = Path(img_path).stem
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            im.save(out_dir / split / "images" / f"{stem}.png")
    except Exception as e:
        print(f"  skip unreadable {img_path}: {e}")
        return None
    lines, counts = [], np.zeros(len(CLASSES), int)
    for c, x1, y1, x2, y2 in boxes:
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(W), x2), min(float(H), y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{c} {((x1 + x2) / 2) / W:.6f} {((y1 + y2) / 2) / H:.6f} "
                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        counts[c] += 1
    (out_dir / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--xview-images", required=True, help="directory of xView train images ({id}.tif)")
    ap.add_argument("--class-labels", required=True, help="xView class-labels file (type_id:name per line)")
    ap.add_argument("--overlapfree", default="experiments/splits/xview_overlapfree_ids.txt")
    ap.add_argument("--out", required=True, help="output root (gets train/ and val/ subdirs)")
    ap.add_argument("--val-holdout", type=int, default=20, help="xView images reserved as pretrain-val (monitoring)")
    ap.add_argument("--include-empty", action="store_true", help="also convert truck-free xView images (negatives)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    type_map = load_type_map(args.class_labels)
    boxes = collect_boxes(args.geojson, type_map)
    overlapfree = set(Path(args.overlapfree).read_text().split())
    imgdir = Path(args.xview_images)
    suffix = {p.stem: p.suffix for p in imgdir.glob("*") if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg")}

    ids = sorted(i for i in overlapfree if i in suffix and (args.include_empty or boxes.get(i)))
    random.Random(args.seed).shuffle(ids)
    val_ids = set(ids[:args.val_holdout])
    print(f"converting {len(ids)} independent images ({len(val_ids)} held out as pretrain-val)")

    out = Path(args.out)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
    totals = {"train": np.zeros(len(CLASSES), int), "val": np.zeros(len(CLASSES), int)}
    n = {"train": 0, "val": 0}
    for i in sorted(ids, key=int):
        split = "val" if i in val_ids else "train"
        counts = write_image(imgdir / f"{i}{suffix[i]}", boxes.get(i, []), out, split)
        if counts is not None:
            totals[split] += counts
            n[split] += 1
    for split in ("train", "val"):
        print(f"{split}: {n[split]} images | per-class instances " +
              ", ".join(f"{CLASSES[c]} {totals[split][c]}" for c in range(len(CLASSES))))
    print(f"done -> {out}\nnext: python src/tile.py --root {out} --out <xview_tiles> --splits train val")


if __name__ == "__main__":
    main()
