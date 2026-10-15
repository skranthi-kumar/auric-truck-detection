"""Overlap gate for full-xView pretraining.

The assignment images are xView images re-saved as PNG: their stems (e.g. 1181, 20, 2543) are
xView's native numeric image IDs, and both splits sit inside xView's *labelled train* ID range.
So "train on full xView" would put the 22 validation images (with their labels) into training.

This script produces a INDEPENDENT allowlist of xView train image IDs to pretrain on:
  all xView train IDs  -  { any xView image that matches an assignment VAL image }
"match" = same ID (primary, exact) OR perceptual-hash near-duplicate (secondary, catches a val
image that was re-chipped / re-numbered). Assignment TRAIN IDs are kept (they are allowed data).

Primary gate needs only the xView image directory (or geojson). The perceptual check (--phash)
decodes every image, so it is slower; run it once and keep the report.

Usage:
  python src/xview_dedup.py --xview-images /data/xview/train_images [--geojson /data/xview/xView_train.geojson] [--phash]

Outputs:
  experiments/splits/xview_overlapfree_ids.txt   (one xView image ID per line; input to xview_to_yolo.py)
  experiments/results/xview_overlap_report.csv
and a printed verdict.

Limitation: a perceptual hash compares whole images, so it does NOT catch a val image that is a
*crop* of a larger xView scene, nor geographic overlap between adjacent xView chips. Those residual
risks are stated, not resolved, in the report.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from data import image_paths  # assignment images, sorted by int(stem)

Image.MAX_IMAGE_PIXELS = None
REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "experiments" / "splits"
RESULTS = REPO / "experiments" / "results"


def xview_train_ids(images_dir, geojson):
    """The set of xView *train* image IDs. Prefer the image directory (every image is a candidate
    overlap source); fall back to the IDs that appear in the annotation geojson."""
    if images_dir:
        ids = {p.stem for p in Path(images_dir).glob("*") if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg")}
        if ids:
            return ids
    if geojson:
        feats = json.load(open(geojson))["features"]
        return {str(Path(f["properties"]["image_id"]).stem) for f in feats if f.get("properties", {}).get("image_id")}
    raise SystemExit("need --xview-images (a directory) or --geojson to enumerate xView train IDs")


def phash_map(paths, hash_size=16):
    """{stem: imagehash.ImageHash} for the given image paths (perceptual hash, robust to re-encoding)."""
    import imagehash
    out = {}
    for p in paths:
        try:
            with Image.open(p) as im:
                out[Path(p).stem] = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
        except Exception as e:  # unreadable image: skip, note in report
            print(f"  phash skip {p}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xview-images", default=None, help="directory of xView train images ({id}.tif)")
    ap.add_argument("--geojson", default=None, help="xView_train.geojson (fallback for the ID list)")
    ap.add_argument("--root", default=None, help="assignment root (default CV_Assignment_AAL via data.py)")
    ap.add_argument("--phash", action="store_true", help="also run the perceptual near-duplicate check")
    ap.add_argument("--phash-thr", type=int, default=12, help="max Hamming distance (of 256 bits) to call a match")
    args = ap.parse_args()

    val_ids = {p.stem for p in (image_paths("val", args.root) if args.root else image_paths("val"))}
    train_ids = {p.stem for p in (image_paths("train", args.root) if args.root else image_paths("train"))}
    xv = xview_train_ids(args.xview_images, args.geojson)
    print(f"xView train images: {len(xv)} | assignment: {len(train_ids)} train / {len(val_ids)} val")

    # --- primary gate: exact ID membership ---
    val_overlap_id = sorted(val_ids & xv, key=int)
    train_in = sorted(train_ids & xv, key=int)
    print(f"\nID GATE  val IDs present in xView train: {len(val_overlap_id)}/{len(val_ids)}  {val_overlap_id}")
    print(f"         train IDs present in xView train: {len(train_in)}/{len(train_ids)} (allowed; kept)")

    rows = [dict(assignment_split="val", assignment_id=i, match="exact-id", xview_id=i, distance=0) for i in val_overlap_id]
    exclude = set(val_overlap_id)  # xView IDs to drop from pretraining

    # --- secondary gate: perceptual near-duplicates (different ID but same picture) ---
    if args.phash:
        if not args.xview_images:
            raise SystemExit("--phash needs --xview-images to decode")
        print("\nPHASH  hashing assignment val + xView train (slow, one-off)...")
        val_paths = image_paths("val", args.root) if args.root else image_paths("val")
        val_h = phash_map(val_paths)
        xv_paths = [p for p in Path(args.xview_images).glob("*") if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg")]
        xv_h = phash_map(xv_paths)
        for vid, vh in val_h.items():
            for xid, xh in xv_h.items():
                if xid == vid:
                    continue  # already caught by the ID gate
                d = vh - xh
                if d <= args.phash_thr:
                    print(f"       val {vid} ~ xView {xid}  (Hamming {d})")
                    rows.append(dict(assignment_split="val", assignment_id=vid, match="phash",
                                     xview_id=xid, distance=int(d)))
                    exclude.add(xid)
    else:
        print("\nPHASH  skipped (pass --phash for the near-duplicate check)")

    overlapfree = sorted(xv - exclude, key=int)
    SPLITS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (SPLITS / "xview_overlapfree_ids.txt").write_text("\n".join(overlapfree) + "\n")
    import csv
    with open(RESULTS / "xview_overlap_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["assignment_split", "assignment_id", "match", "xview_id", "distance"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nVERDICT  {len(exclude)} xView images excluded as val overlaps "
          f"({len(val_overlap_id)} by ID" + (f", {len(exclude) - len(val_overlap_id)} by pHash" if args.phash else "") + ")")
    print(f"         independent xView pretrain set: {len(overlapfree)} images "
          f"-> {SPLITS / 'xview_overlapfree_ids.txt'}")
    print(f"         report -> {RESULTS / 'xview_overlap_report.csv'}")
    if not val_overlap_id and not args.phash:
        print("         NOTE: 0 val IDs matched. If xView only exposes labels for a *different* split than\n"
              "         these IDs, there may be no overlap to remove - still run --phash to confirm.")


if __name__ == "__main__":
    main()
