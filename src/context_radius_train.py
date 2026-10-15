"""Overlap check for the context-smoothing radius: rerun the forensic neighbour-majority analysis (dataset_forensics.py
section G, which used VAL GT to compare R = 150 / 400 / whole image) on TRAIN GT only.
If train selects the same R by the same criterion, R = 150 is justified without validation labels.

Usage: python src/context_radius_train.py --root CV_Assignment_AAL
"""
import argparse

import numpy as np

from data import image_paths, image_size, read_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    imgs = []
    for p in image_paths(a.split, a.root):
        cls, b = read_labels(p, image_size(p))
        if len(b) >= 2:
            b = np.asarray(b, float)
            imgs.append((np.asarray(cls), np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]))
    n_all = sum(len(c) for c, _ in imgs)
    print(f"{a.split}: {len(imgs)} images with >= 2 boxes, {n_all} boxes")
    for R in (50, 100, 150, 200, 300, 400, 1e9):
        hit = tot = 0
        for c, xy in imgs:
            D = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
            for i in range(len(c)):
                nb = (D[i] <= R) & (np.arange(len(c)) != i)
                if nb.any():
                    hit += np.bincount(c[nb], minlength=5).argmax() == c[i]
                    tot += 1
        lab = "whole image" if R > 1e8 else f"radius {int(R)} px"
        tag = "  <- original candidate" if R in (150, 400, 1e9) else ""
        print(f"- neighbour-majority ({lab}): accuracy {hit / tot:.3f} on {tot} boxes ({tot / n_all:.0%} have a neighbour){tag}")


if __name__ == "__main__":
    main()
