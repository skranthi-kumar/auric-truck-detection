"""Learned context model (declared in DECISIONS.md, 10 Oct): re-classify detected objects from their own scores
plus neighbourhood and image-level context. Trained on TRAIN-image detections only; applied once to val.

Objects = class-agnostic NMS 0.5 over a run's prediction rows; P[c] = max conf of overlapping rows (IoU >= 0.5)
of class c. Label = class of the GT box with IoU >= 0.5 (else background). Features: P (5), det = max P,
log side, log aspect, det-weighted neighbour class histogram within R px (5), log neighbour count, mean
neighbour det, det-weighted image class histogram (5), log object count. Model: multinomial logistic
regression on standardised features (HistGradientBoosting reported as sensitivity).
Output rows per (object, class c): conf = det * p(c | not bg)  [declared primary]; variant conf = p(c) [sensitivity].

Usage:
  python src/context_model.py --train_preds experiments/results/E2_last_train_preds.csv \
      --val_preds experiments/results/ENS_E2_val_preds.csv --out experiments/results/CTXM
  then: python src/evaluate.py --preds experiments/results/CTXM_val_preds.csv --name CTXM
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torchvision.ops import nms

from data import CLASSES
from evaluate import iou, load_gt

NC = len(CLASSES)
R = 150.0


def objects(g):
    """Class-agnostic anchors and their per-class score vectors from one image's prediction rows."""
    b = g[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
    s, c = g.conf.values.astype(np.float32), g.cls.values
    keep = nms(torch.from_numpy(b), torch.from_numpy(s), 0.5).numpy()
    anchors = b[keep]
    I = iou(anchors, b) >= 0.5
    P = np.zeros((len(keep), NC))
    for cc in range(NC):
        m = (c == cc)
        if m.any():
            P[:, cc] = np.where(I[:, m], s[m][None, :], 0).max(1)
    return anchors, P


def features(anchors, P):
    det = P.max(1)
    w, h = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
    xy = np.stack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2], 1)
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
    D = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    W = (D <= R).astype(float) * det[None, :]
    np.fill_diagonal(W, 0)
    nb_hist = W @ Pn
    nb_hist = nb_hist / np.maximum(nb_hist.sum(1, keepdims=True), 1e-9)
    nb_cnt = (W > 0).sum(1)
    nb_det = np.where(nb_cnt > 0, W.sum(1) / np.maximum(nb_cnt, 1), 0)
    img_hist = (det[:, None] * Pn).sum(0)
    img_hist = img_hist / max(img_hist.sum(), 1e-9)
    F = np.column_stack([P, det, np.log(np.sqrt(w * h) + 1), np.log(np.maximum(w, h) / np.maximum(np.minimum(w, h), 1)),
                         nb_hist, np.log1p(nb_cnt), nb_det, np.repeat(img_hist[None, :], len(P), 0),
                         np.full(len(P), np.log1p(len(P)))])
    return F


def build(preds_csv, split, root):
    p = pd.read_csv(preds_csv, dtype={"image": str})
    gt = load_gt(split, root)
    X, y, meta = [], [], []
    for im, g in p.groupby("image"):
        anchors, P = objects(g)
        F = features(anchors, P)
        G = gt[gt.image == im]
        lab = np.full(len(anchors), NC)  # background
        if len(G):
            I = iou(anchors, G[["x1", "y1", "x2", "y2"]].values)
            j, best = I.argmax(1), I.max(1)
            lab[best >= 0.5] = G.cls.values[j[best >= 0.5]]
        X.append(F), y.append(lab), meta.append(pd.DataFrame({"image": im, "x1": anchors[:, 0], "y1": anchors[:, 1],
                                                              "x2": anchors[:, 2], "y2": anchors[:, 3], "det": P.max(1)}))
    return np.vstack(X), np.concatenate(y), pd.concat(meta, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_preds", required=True)
    ap.add_argument("--val_preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    Xtr, ytr, mtr = build(a.train_preds, "train", a.root)
    print(f"train objects {len(ytr)}: labels {np.bincount(ytr, minlength=NC + 1).tolist()} (last = background)")
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000, C=1.0)
    # image-grouped 5-fold on TRAIN only: does the model learn anything beyond the detector's own class vector?
    base_acc, acc = [], []
    for tr, te in GroupKFold(5).split(Xtr, ytr, mtr.image):
        m = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr[tr]), ytr[tr])
        acc.append(np.mean(m.predict(sc.transform(Xtr[te])) == ytr[te]))
        own = np.where(Xtr[te, :NC].max(1) >= 0.25, Xtr[te, :NC].argmax(1), NC)  # detector alone, conf 0.25 rule
        base_acc.append(np.mean(own == ytr[te]))
    print(f"train 5-fold (by image) accuracy incl. background: detector-alone {np.mean(base_acc):.3f} -> context model {np.mean(acc):.3f}")
    lr.fit(sc.transform(Xtr), ytr)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(Xtr, ytr)

    Xva, yva, mva = build(a.val_preds, "val", a.root)
    for name, model, Xv in (("lr", lr, sc.transform(Xva)), ("gb", gb, Xva)):
        pr = model.predict_proba(Xv)
        p_fg = pr[:, :NC] / np.maximum(pr[:, :NC].sum(1, keepdims=True), 1e-9)
        for variant, conf in (("", mva.det.values[:, None] * p_fg), ("_pfull", pr[:, :NC])):
            rows = [(r.image, c, conf[i, c], r.x1, r.y1, r.x2, r.y2) for i, r in enumerate(mva.itertuples()) for c in range(NC) if conf[i, c] > 0]
            out = f"{a.out}_{name}{variant}_val_preds.csv"
            pd.DataFrame(rows, columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"]).to_csv(out, index=False)
            print("wrote", out)
    print("val objects", len(yva), "(evaluate each CSV with evaluate.py; primary = CTXM_lr)")


if __name__ == "__main__":
    main()
