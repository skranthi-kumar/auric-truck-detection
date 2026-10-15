"""Write a cleaned copy of the TRAIN labels (originals untouched): on train images whose xView labels align with
our GT (>= 80% of GT matched at IoU 0.5), a GT box whose IoU >= 0.5 xView box carries a DIFFERENT one of our 5
classes takes xView's class. Val is symlinked unchanged. Motivation: data audit (LOG.md, 15 Oct): train agreement
with xView 0.52-0.99 by class, val 1.00.

Usage: python src/clean_labels.py --out CV_clean
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

from data import CLASSES, ROOT, image_paths, image_size, read_labels

R = Path(__file__).resolve().parents[1] / "experiments" / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "train" / "labels").mkdir(parents=True, exist_ok=True)
    for s in ("val",):
        if not (out / s).exists():
            os.symlink((ROOT / s).resolve(), out / s)
    if not (out / "train" / "images").exists():
        os.symlink((ROOT / "train" / "images").resolve(), out / "train" / "images")
    xv = pd.concat([pd.read_csv(R / f, dtype={"image": str}) for f in ("xview_train_labels.csv", "xview_heldout_labels.csv")])
    changes = []
    for p in image_paths("train"):
        lab_file = ROOT / "train" / "labels" / f"{p.stem}.txt"
        lines = lab_file.read_text().splitlines()
        W, H = image_size(p)
        cls, b = read_labels(p, (W, H))
        xi = xv[xv.image == p.stem]
        if len(b) and len(xi):
            iou = box_iou(torch.tensor(b, dtype=torch.float32), torch.tensor(xi[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32))
            best, j = iou.max(1)
            if (best >= 0.5).float().mean() >= 0.8:
                names = xi.name.values[j.numpy()]
                for k in range(len(b)):
                    if best[k] >= 0.5 and names[k] in CLASSES and CLASSES.index(names[k]) != cls[k]:
                        new = CLASSES.index(names[k])
                        parts = lines[k].split()
                        assert int(parts[0]) == cls[k]
                        lines[k] = " ".join([str(new)] + parts[1:])
                        changes.append((p.stem, CLASSES[cls[k]], CLASSES[new]))
        (out / "train" / "labels" / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    c = pd.DataFrame(changes, columns=["image", "from", "to"])
    c.to_csv(R / "label_cleaning_changes.csv", index=False)
    print(f"changed {len(c)} train boxes in {c.image.nunique()} images")
    print(pd.crosstab(c["from"], c["to"]).to_string())


if __name__ == "__main__":
    main()
