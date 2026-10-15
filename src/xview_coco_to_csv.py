"""Convert the checkpoint dataset's COCO crop annotations (xview_coco_crop_1008, *_annotations.json) to full-image
boxes: file '<image>_y<y0>_x<x0>.jpg' -> offsets; duplicates from overlapping crops are merged (same class,
IoU > 0.7, the largest box kept, so crop-truncated copies are dropped).

Usage: python src/xview_coco_to_csv.py --json val_annotations.json --images ids.txt --out xview_heldout_labels.csv
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from torchvision.ops import box_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", default=None, help="optional txt of image stems to keep")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.json))
    keep = set(Path(a.images).read_text().split()) if a.images else None
    cat = {c["id"]: c["name"] for c in d["categories"]}
    im = {}
    for i in d["images"]:
        m = re.match(r"(\d+)_y(\d+)_x(\d+)\.", i["file_name"])
        im[i["id"]] = (m.group(1), int(m.group(2)), int(m.group(3)))
    rows = []
    for an in d["annotations"]:
        stem, y0, x0 = im[an["image_id"]]
        if keep is not None and stem not in keep:
            continue
        x, y, w, h = an["bbox"]
        rows.append((stem, an["category_id"], cat[an["category_id"]], x + x0, y + y0, x + x0 + w, y + y0 + h, w * h))
    df = pd.DataFrame(rows, columns=["image", "cat", "name", "x1", "y1", "x2", "y2", "area"])
    out = []
    for (stem, c), g in df.groupby(["image", "cat"]):
        g = g.sort_values("area", ascending=False)
        b = torch.tensor(g[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32)
        iou = box_iou(b, b)
        kept = []
        for k in range(len(g)):
            if not kept or iou[k, kept].max() <= 0.7:
                kept.append(k)
        out.append(g.iloc[kept])
    res = pd.concat(out)[["image", "cat", "x1", "y1", "x2", "y2", "name"]]
    res.to_csv(a.out, index=False)
    print(f"{len(df)} crop boxes -> {len(res)} image boxes on {res.image.nunique()} images; categories e.g. {list(cat.items())[10:16]}")


if __name__ == "__main__":
    main()
