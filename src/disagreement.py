"""Check-3 proxy: MODEL-DISAGREEMENT rate on val (out-of-sample), not a label-noise rate.
For each val GT box, take the highest-confidence prediction (any class) with IoU >= 0.5. If its confidence >= THR and its
class differs from GT, the model confidently disagrees. Reported per GT class and per (GT -> predicted) pair, with counts.
Diagnostic only; nothing is tuned on it.

Usage: python src/disagreement.py --preds experiments/results/E2_last_val_preds.csv --root CV_Assignment_AAL
"""
import argparse

import numpy as np
import pandas as pd

from data import CLASSES
from evaluate import iou, load_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5, 0.7])
    a = ap.parse_args()
    p = pd.read_csv(a.preds, dtype={"image": str})
    gt = load_gt("val", a.root)
    out = []
    for im, g in gt.groupby("image"):
        q = p[p.image == im]
        if not len(q):
            out += [(c, -1, 0.0) for c in g.cls]
            continue
        I = iou(g[["x1", "y1", "x2", "y2"]].values, q[["x1", "y1", "x2", "y2"]].values)
        conf = np.where(I >= 0.5, q.conf.values[None, :], -1)
        j = conf.argmax(1)
        out += [(c, int(q.cls.values[jj]) if conf[i, jj] >= 0 else -1, max(conf[i, jj], 0.0))
                for i, (c, jj) in enumerate(zip(g.cls.values, j))]
    d = pd.DataFrame(out, columns=["g", "pred", "conf"])  # not "gt": clashes with DataFrame.gt
    for t in a.thr:
        conf_d = d[d.conf >= t]
        dis = conf_d[conf_d.pred != conf_d.g]
        print(f"\nconf >= {t}: {len(conf_d)} of {len(d)} GT boxes have a confident match; "
              f"model-disagreement rate {len(dis) / max(len(conf_d), 1):.3f} ({len(dis)} boxes)")
        per = conf_d.groupby("g").apply(lambda x: pd.Series({"confident": len(x), "disagree": int((x.pred != x.name).sum())}),
                                         include_groups=False)
        per["rate"] = (per.disagree / per.confident).round(3)
        per.index = [CLASSES[i] for i in per.index]
        print(per.to_string())
        pairs = dis.groupby(["g", "pred"]).size().sort_values(ascending=False)
        print("pairs (GT -> pred):", ", ".join(f"{CLASSES[g]}->{CLASSES[k]} {n}" for (g, k), n in pairs.items()))


if __name__ == "__main__":
    main()
