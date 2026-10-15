"""What do the top-ranked false positives sit on? (diagnostic only; no tuning, no training)

Uses xView's own 60-class labels for the val images (external; provenance in experiments/LOG.md, 13 Oct).
Only images whose xView labels align with our GT are used: >= 80% of our GT boxes matched by an xView box
at IoU >= 0.5 (same criterion as the T3 train-side check).

For each class, the top-N false positives by confidence (N = 400; these set the precision of the high-recall
part of the PR curve) are bucketed:
  wrong subtype      evaluate.py err == "cls" (a GT truck of another of our classes)
  loc / duplicate    err in {"loc", "both", "dup"}
  then, for err == "bg", by the best-overlapping xView box at IoU >= 0.3:
  our-class truck missing from GT | other xView truck / large vehicle | car | other object | nothing labelled

Usage: python src/fp_forensics.py --preds FINAL2 --xview experiments/results/xview_val_labels.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torchvision.ops import box_iou

from data import CLASSES

R = Path(__file__).resolve().parents[1] / "experiments" / "results"
OURS = set(CLASSES)
OTHER_VEH = {"Truck", "Bus", "Trailer", "Utility Truck", "Cargo Car", "Dump Truck", "Pickup Truck", "Cement Mixer",
             "Crane Truck", "Haul Truck", "Tank car", "Flat Car", "Locomotive", "Passenger Car"}
CARS = {"Small Car", "Passenger Vehicle"}
BUCKETS = ["wrong subtype", "loc / duplicate", "our-class truck missing from GT", "other xView vehicle class",
           "car", "other object", "nothing labelled"]


def T(d):
    return torch.tensor(d[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="FINAL2")
    ap.add_argument("--xview", default=str(R / "xview_val_labels.csv"))
    ap.add_argument("--top", type=int, default=400)
    a = ap.parse_args()
    xv = pd.read_csv(a.xview, dtype={"image": str})
    g = pd.read_csv(R / f"{a.preds}_gt.csv", dtype={"image": str})
    m = pd.read_csv(R / f"{a.preds}_matches.csv", dtype={"image": str})

    aligned = []
    for im, gi in g.groupby("image"):
        xi = xv[xv.image == im]
        if len(xi) and (box_iou(T(gi), T(xi)).max(1).values >= 0.5).float().mean() >= 0.8:
            aligned.append(im)
    print(f"aligned val images: {len(aligned)} / {g.image.nunique()}")

    rows = []
    for c, name in enumerate(CLASSES):
        fp = m[(m.cls == c) & (~m.tp) & m.image.isin(aligned)].nlargest(a.top, "conf")
        for im, f in fp.groupby("image"):
            xi = xv[xv.image == im]
            iou = box_iou(T(f), T(xi)) if len(xi) else torch.zeros(len(f), 0)
            for j, r in enumerate(f.itertuples()):
                if r.err == "cls":
                    b = "wrong subtype"
                elif r.err in ("loc", "both", "dup"):
                    b = "loc / duplicate"
                elif iou.shape[1] == 0 or iou[j].max() < 0.3:
                    b = "nothing labelled"
                else:
                    lab = xi.name.values[int(iou[j].argmax())]
                    b = ("our-class truck missing from GT" if lab in OURS else "other xView vehicle class"
                         if lab in OTHER_VEH else "car" if lab in CARS else "other object")
                rows.append((name, b))
    t = pd.crosstab(pd.Series([r[0] for r in rows], name="class"), pd.Series([r[1] for r in rows], name="bucket"))
    t = t.reindex(index=CLASSES, columns=BUCKETS, fill_value=0)
    share = t.div(t.sum(1), axis=0).round(3)
    print(t.to_string())
    print(share.to_string())
    share.to_csv(R / f"{a.preds}_fp_forensics.csv")
    print("wrote", R / f"{a.preds}_fp_forensics.csv")


if __name__ == "__main__":
    main()
