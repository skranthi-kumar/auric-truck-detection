"""GTO: re-score frozen predictions with xView-labelled trucks of our 5 classes that our GT omits (diagnostic only).

Builds two label roots for the xView-aligned val images (criterion as in src/fp_forensics.py): original GT, and
original GT + omitted in-class xView boxes (max IoU with any GT < 0.3). Images are symlinked; only labels differ.
Then runs evaluate.py on both with the same predictions.

Usage: python src/gt_omission_test.py --preds FINAL2 --xview experiments/results/xview_val_labels.csv --work /tmp/gto
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from torchvision.ops import box_iou

from data import CLASSES, image_paths, image_size, read_labels

REPO = Path(__file__).resolve().parents[1]
R = REPO / "experiments" / "results"


def T(a):
    return torch.tensor(a, dtype=torch.float32).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="FINAL2")
    ap.add_argument("--xview", default=str(R / "xview_val_labels.csv"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    xv = pd.read_csv(a.xview, dtype={"image": str})
    work = Path(a.work)
    added = []
    for name in ("orig", "aug"):
        (work / name / "val" / "images").mkdir(parents=True, exist_ok=True)
        (work / name / "val" / "labels").mkdir(parents=True, exist_ok=True)
    aligned = []
    for p in image_paths("val", a.root):
        W, H = image_size(p)
        cls, b = read_labels(p, (W, H))
        xi = xv[xv.image == p.stem]
        if not len(xi) or not len(b):
            continue
        iou = box_iou(T(b), T(xi[["x1", "y1", "x2", "y2"]].values))
        if (iou.max(1).values >= 0.5).float().mean() < 0.8:
            continue
        aligned.append(p.stem)
        lab = (Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")).read_text()
        cand = xi[xi.name.isin(CLASSES)]
        if len(cand):
            m = box_iou(T(cand[["x1", "y1", "x2", "y2"]].values), T(b)).max(1).values.numpy()
            cand = cand[m < 0.3]
        extra = "".join(f"{CLASSES.index(r.name)} {(r.x1 + r.x2) / 2 / W:.8f} {(r.y1 + r.y2) / 2 / H:.8f} "
                        f"{(r.x2 - r.x1) / W:.8f} {(r.y2 - r.y1) / H:.8f}\n" for r in cand.itertuples())
        added += [(p.stem, r.name) for r in cand.itertuples()]
        for name, txt in (("orig", lab), ("aug", lab.rstrip("\n") + "\n" + extra if extra else lab)):
            d = work / name / "val"
            (d / "labels" / f"{p.stem}.txt").write_text(txt)
            link = d / "images" / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
    print(f"aligned images: {len(aligned)}; omitted in-class trucks added: {len(added)}")
    print(pd.Series([c for _, c in added]).value_counts().to_string())
    preds = pd.read_csv(R / f"{a.preds}_val_preds.csv", dtype={"image": str})
    sub = work / f"{a.preds}_aligned_preds.csv"
    preds[preds.image.isin(aligned)].to_csv(sub, index=False)
    for name in ("orig", "aug"):
        subprocess.run([sys.executable, str(REPO / "src" / "evaluate.py"), "--preds", str(sub),
                        "--name", f"GTO_{a.preds}_{name}", "--root", str(work / name)], check=True)


if __name__ == "__main__":
    main()
