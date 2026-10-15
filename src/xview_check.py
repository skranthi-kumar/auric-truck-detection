"""Pre-flight validator for the xView pipeline. Run BEFORE spending GPU hours.

Checks the three xView inputs exist and are well-formed, that all 5 class names resolve to type_ids,
and reports how many instances of each class are available (full and, if the dedup has run, independent).
Also warns if any assignment VAL id is still present (reminder to run xview_dedup.py).

Usage:
  python src/xview_check.py --geojson /data/xview/xView_train.geojson \
      --xview-images /data/xview/train_images --class-labels /data/xview/xview_class_labels.txt
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from data import CLASSES, image_paths
from xview_to_yolo import load_type_map  # name-based type_id mapping (single source of truth)

REPO = Path(__file__).resolve().parents[1]
OVERLAPFREE = REPO / "experiments" / "splits" / "xview_overlapfree_ids.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--xview-images", required=True)
    ap.add_argument("--class-labels", required=True)
    args = ap.parse_args()
    ok = True

    for label, p in [("geojson", args.geojson), ("images dir", args.xview_images), ("class-labels", args.class_labels)]:
        exists = Path(p).exists()
        ok &= exists
        print(f"[{'OK' if exists else 'MISSING'}] {label}: {p}")
    if not ok:
        raise SystemExit("fix the missing path(s) above")

    imgs = [q for q in Path(args.xview_images).glob("*") if q.suffix.lower() in (".tif", ".tiff", ".png", ".jpg")]
    print(f"[OK] {len(imgs)} xView images in dir")

    type_map = load_type_map(args.class_labels)  # aborts loudly if a class name is missing

    overlapfree = set(OVERLAPFREE.read_text().split()) if OVERLAPFREE.exists() else None
    feats = json.load(open(args.geojson))["features"]
    img_set = Counter()                       # images with >=1 truck box
    per_class_all = Counter()                 # instances per class, full xView
    per_class_lf = Counter()                  # instances per class, independent subset
    for f in feats:
        pr = f.get("properties", {})
        if pr.get("type_id") not in type_map:
            continue
        c = type_map[pr["type_id"]]
        stem = Path(pr["image_id"]).stem
        per_class_all[c] += 1
        img_set[stem] += 1
        if overlapfree is not None and stem in overlapfree:
            per_class_lf[c] += 1
    print(f"[OK] geojson parsed: {len(feats)} features, {sum(per_class_all.values())} truck boxes "
          f"in {len(img_set)} images")

    overlap = sorted({p.stem for p in image_paths("val")} & {q.stem for q in imgs}, key=int)
    if overlap:
        print(f"[WARN] {len(overlap)} assignment VAL ids present in xView images: {overlap}\n"
              f"       -> run src/xview_dedup.py first; pretraining must exclude these.")
    else:
        print("[OK] no assignment VAL ids found among xView images (run dedup to confirm with --phash too)")

    print("\nper-class instances available for pretraining:")
    for c in range(len(CLASSES)):
        if overlapfree is not None:
            print(f"  {CLASSES[c]:<16} {per_class_all[c]:>6} total | {per_class_lf[c]:>6} independent")
        else:
            print(f"  {CLASSES[c]:<16} {per_class_all[c]:>6} total   (run xview_dedup.py for the independent count)")
    print("\npre-flight complete." + ("" if overlapfree is not None else "  Next: src/xview_dedup.py --phash"))


if __name__ == "__main__":
    main()
