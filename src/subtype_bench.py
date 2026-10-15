"""Held-out subtype benchmark: detector vs crop classifier on the SAME objects of the 74 checkpoint-unseen train images.

Detector: for every GT box on those images, take the RFX0 query (train dump) with IoU >= 0.5 and the highest
score over our 5 classes; its predicted subtype = argmax of the 5 mapped sigmoids. GT boxes with no such query
are counted as "no detector box" and excluded from the detector's accuracy (reported separately).
Classifier: probabilities saved by oracle_cls.py --holdout ... --save_holdout <npz> (same GT boxes, same order
as data.read_labels). Both are scored on the GT boxes the detector covers, so the comparison is like for like.

Usage: python src/subtype_bench.py --dump ~/auric_kaggle/DUMP/rfx0_train_dump.npz \
           --holdout experiments/splits/rfx_heldout_train_ids.txt [--cls_probs runs/cls_ctx2.npz ...]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

from data import CLASSES, image_paths, image_size, read_labels

IDS = [11, 12, 15, 13, 16]


def report(name, y, p):
    acc = np.mean(p == y)
    per = {c: round(float(np.mean(p[y == i] == i)), 3) for i, c in enumerate(CLASSES) if (y == i).any()}
    cb = (y <= 1)
    cvb = np.mean(p[cb] == y[cb])
    print(f"\n{name}: accuracy {acc:.3f}  macro-recall {np.mean(list(per.values())):.3f}  Cargo-vs-Box (on Cargo/Box GT) {cvb:.3f}")
    print("per-class recall", per)
    print(pd.crosstab(pd.Series(y, name="GT").map(dict(enumerate(CLASSES))), pd.Series(p, name="pred").map(dict(enumerate(CLASSES)))).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--cls_probs", nargs="*", default=[])
    ap.add_argument("--root", default=None, help="label root (e.g. CV_clean); default the original dataset")
    a = ap.parse_args()
    held = set(Path(a.holdout).read_text().split())
    d = np.load(a.dump)
    names = d["names"]
    img = np.array([n.split("_")[0] for n in names])[d["tile"]]
    off = np.array([[int(n.split("_")[1]), int(n.split("_")[2].split(".")[0])] for n in names], dtype=np.float32)[d["tile"]]
    box = d["box"] + np.tile(off, 2)
    S = 1 / (1 + np.exp(-d["logit"][:, IDS].astype(np.float32)))
    ys, ps, covered, order = [], [], [], []
    for p in image_paths("train", a.root):
        if p.stem not in held:
            continue
        cls, b = read_labels(p, image_size(p))
        if not len(b):
            continue
        k = np.flatnonzero(img == p.stem)
        iou = box_iou(torch.tensor(b, dtype=torch.float32), torch.tensor(box[k], dtype=torch.float32)).numpy()
        sc = np.where(iou >= 0.5, S[k].max(1)[None, :], -1)
        j = sc.argmax(1)
        has = sc.max(1) >= 0
        ys += list(cls)
        ps += list(np.where(has, S[k][j].argmax(1), -1))
        covered += list(has)
    y, pdet, cov = np.array(ys), np.array(ps), np.array(covered)
    print(f"held-out images: {len(held)}; GT boxes {len(y)}; with a detector box (IoU >= 0.5): {cov.sum()} ({cov.mean():.1%})")
    print("GT per class:", dict(zip(CLASSES, np.bincount(y, minlength=5).tolist())))
    report("DETECTOR (RFX0, top query at IoU >= 0.5)", y[cov], pdet[cov])
    for f in a.cls_probs:
        z = np.load(f)
        assert len(z["y"]) == len(y), f"{f}: GT count differs"  # same boxes and order; classes may differ (cleaned labels)
        pc = z["P"].argmax(1)
        report(f"CLASSIFIER {Path(f).stem} (same covered boxes)", y[cov], pc[cov])
        report(f"CLASSIFIER {Path(f).stem} (all held-out GT)", y, pc)
    np.savez(Path(a.holdout).with_name("subtype_bench_detector.npz"), y=y, pred=pdet, covered=cov)


if __name__ == "__main__":
    main()
