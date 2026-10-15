"""Merge several prediction CSVs (same images) with Weighted Boxes Fusion (Solovyev et al. 2021), per image
and class: boxes from different models that overlap (IoU >= 0.55) are fused into one box at their
confidence-weighted mean coordinates; fused confidence = mean conf * (n boxes in cluster) / (n models),
so a box that only one model predicts is down-weighted. Output is scored by evaluate.py.

Usage: python src/ensemble.py --preds a.csv b.csv c.csv --out experiments/results/ENS_val_preds.csv
"""
import argparse

import numpy as np
import pandas as pd

from evaluate import iou

THR = 0.55


def wbf(df, n_models, thr=THR, weights=None):
    out = []
    for (im, c), g in df.groupby(["image", "cls"]):
        g = g.sort_values("conf", ascending=False)
        B, S = g[["x1", "y1", "x2", "y2"]].values, g.conf.values
        Wm = None if weights is None else np.asarray(weights, float)[g.model.values]
        # same algorithm as before; fused boxes kept in a preallocated array (was rebuilt from a list per box)
        fused, members, k = np.empty((len(B), 4)), [], 0
        for i in range(len(B)):
            if k:
                ious = iou(B[i:i + 1], fused[:k])[0]
                j = int(ious.argmax())
                if ious[j] >= thr:
                    members[j].append(i)
                    w = S[members[j]]
                    fused[j] = (B[members[j]] * w[:, None]).sum(0) / w.sum()
                    continue
            fused[k] = B[i]; k += 1
            members.append([i])
        for f, m in zip(fused[:k], members):
            if weights is None:
                conf = S[m].mean() * min(len(m), n_models) / n_models
            else:  # weighted variant (FU1): sum of model-weighted confidences over the total model weight
                conf = (S[m] * Wm[m]).sum() / sum(weights)
            out.append((im, c, conf, *f))
    return pd.DataFrame(out, columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=THR, help="WBF cluster IoU (default 0.55)")
    ap.add_argument("--weights", type=float, nargs="+", default=None, help="per-model weights (default: equal, original formula)")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p, dtype={"image": str}).assign(model=k) for k, p in enumerate(a.preds)])
    res = wbf(df, len(a.preds), a.iou, a.weights)
    res.to_csv(a.out, index=False)
    print(f"fused {len(df)} boxes from {len(a.preds)} models into {len(res)} -> {a.out}")


if __name__ == "__main__":
    main()
