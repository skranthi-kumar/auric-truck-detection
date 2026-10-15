"""Do high-ranked false positives carry a usable signal? (diagnostic on a scored run; no fitting on val)

For predictions with conf >= --min_conf, per predicted class:
  1. confidence distribution of TPs vs each FP cause (wrong subtype, poor localisation, duplicate, background)
  2. AUC (TP vs FP) of label-free per-box signals:
       conf; box side; aspect ratio; margin = own score - best other-class score on the same object
       (other-class rows with IoU >= 0.7); n_classes = number of our classes scored >= 0.05 on the object;
       local density = number of conf >= 0.25 boxes within 150 px.
     AUC near 0.5 = no signal; within-class AP can only change if a signal re-orders TPs vs FPs beyond conf.
     The useful number is the AUC of each signal *conditional on conf* (computed within conf deciles, averaged).

Usage: python src/fp_signal.py --run SC1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from torchvision.ops import box_iou

from data import CLASSES

R = Path(__file__).resolve().parents[1] / "experiments" / "results"


def cond_auc(y, x, conf, bins=10):
    q = pd.qcut(conf, bins, labels=False, duplicates="drop")
    aucs, w = [], []
    for b in np.unique(q):
        k = q == b
        if y[k].min() != y[k].max():
            aucs.append(roc_auc_score(y[k], x[k]))
            w.append(k.sum())
    return float(np.average(aucs, weights=w)) if aucs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="SC1")
    ap.add_argument("--min_conf", type=float, default=0.05)
    a = ap.parse_args()
    m = pd.read_csv(R / f"{a.run}_matches.csv", dtype={"image": str},
                    usecols=["image", "cls", "conf", "x1", "y1", "x2", "y2", "tp", "err"])
    m = m[m.conf >= a.min_conf].reset_index(drop=True)
    m["side"] = np.sqrt((m.x2 - m.x1) * (m.y2 - m.y1))
    w, h = (m.x2 - m.x1), (m.y2 - m.y1)
    m["aspect"] = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1e-3)
    m["margin"], m["n_classes"], m["density"] = 0.0, 1, 0
    for im, f in m.groupby("image"):
        b = torch.tensor(f[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32)
        same_obj = (box_iou(b, b) >= 0.7).numpy()
        c, s = f.cls.values, f.conf.values
        other = np.where(same_obj & (c[None, :] != c[:, None]), s[None, :], 0).max(1)
        m.loc[f.index, "margin"] = s - other
        m.loc[f.index, "n_classes"] = [len(set(c[same_obj[i] & (s >= 0.05)])) for i in range(len(f))]
        hi = f.conf.values >= 0.25
        ctr = np.c_[(f.x1 + f.x2) / 2, (f.y1 + f.y2) / 2]
        if hi.any():
            t = cKDTree(ctr[hi])
            m.loc[f.index, "density"] = [len(x) for x in t.query_ball_point(ctr, 150)]
    rows, dist = [], []
    for c, name in enumerate(CLASSES):
        mc = m[m.cls == c]
        for k in ["tp", "cls", "loc", "both", "dup", "bg"]:
            v = mc.conf[mc.err == k]
            if len(v):
                dist.append((name, k, len(v), *np.percentile(v, [50, 90]).round(3), int((v >= 0.25).sum())))
        y = mc.tp.values.astype(int)
        if y.min() == y.max():
            continue
        r = {"class": name, "n": len(mc), "TP": int(y.sum())}
        r["AUC conf"] = round(roc_auc_score(y, mc.conf), 3)
        for f_ in ["side", "aspect", "margin", "n_classes", "density"]:
            x = mc[f_].values.astype(float)
            r[f"AUC {f_}"] = round(roc_auc_score(y, x), 3)
            r[f"cond-AUC {f_}"] = round(cond_auc(y, x, mc.conf.values), 3)
        rows.append(r)
    D = pd.DataFrame(dist, columns=["class", "type", "n", "conf p50", "conf p90", "n conf>=0.25"])
    S = pd.DataFrame(rows).set_index("class")
    text = (f"{a.run}, predictions with conf >= {a.min_conf}\n1. confidence of TPs vs FP types\n" + D.to_string(index=False)
            + "\n\n2. TP-vs-FP AUC of label-free signals (cond-AUC = within conf deciles; 0.5 = no extra signal)\n" + S.T.to_string())
    print(text)
    (R / f"{a.run}_fp_signal.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
