"""Spatial context smoothing of detector class scores (inference-only, no training).

Forensics: on val GT, the majority class of a box's neighbours within 150 px predicts its class with 0.89
accuracy, versus 0.57-0.59 for crop classifiers. The detector classifies each box alone. Here each detected
object's class vector P is mixed with the confidence-weighted class distribution Q of its neighbours:
    S = (1 - a) * P + a * det * Qn        (det = max_c P, Qn = Q normalised)
Objects are the class-agnostic NMS (0.5) of the prediction rows; P[c] = max conf of overlapping rows of class c.
--oracle uses GT neighbours instead of predicted ones (upper bound on what context could give).

Usage: python src/context_smooth.py --preds experiments/results/E1_up2_s0_val_preds.csv \
           --out experiments/results/E1_ctx_val_preds.csv [--R 150] [--alpha 0.5] [--oracle]
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms

from data import CLASSES
from evaluate import iou, load_gt

NC = len(CLASSES)


def smooth(P, boxes, R, alpha, gt_xy=None, gt_cls=None):
    det = P.max(1)
    xy = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    if gt_xy is not None:  # oracle: GT neighbours, one-hot classes, unit weight
        src_xy, src_P, src_w = gt_xy, np.eye(NC)[gt_cls], np.ones(len(gt_cls))
    else:
        src_xy, src_P, src_w = xy, P / np.maximum(P.sum(1, keepdims=True), 1e-9), det
    D = np.hypot(*(xy[:, None, :] - src_xy[None, :, :]).transpose(2, 0, 1))
    W = (D <= R).astype(float) * src_w[None, :]
    if gt_xy is None:
        np.fill_diagonal(W, 0)  # exclude self
    Q = W @ src_P
    has = Q.sum(1) > 0
    Qn = np.where(has[:, None], Q / np.maximum(Q.sum(1, keepdims=True), 1e-9), 0)
    S = P.copy()
    S[has] = (1 - alpha) * P[has] + alpha * det[has, None] * Qn[has]
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--R", type=float, default=150)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    p = pd.read_csv(a.preds, dtype={"image": str})
    gt = load_gt("val", a.root) if a.oracle else None
    rows = []
    for im, g in p.groupby("image"):
        b = g[["x1", "y1", "x2", "y2"]].values.astype(np.float32)
        s, c = g.conf.values.astype(np.float32), g.cls.values
        keep = nms(torch.from_numpy(b), torch.from_numpy(s), 0.5).numpy()
        anchors = b[keep]
        I = iou(anchors, b) >= 0.5
        P = np.zeros((len(keep), NC))
        for k in range(len(keep)):
            for cc in range(NC):
                m = I[k] & (c == cc)
                P[k, cc] = s[m].max() if m.any() else 0.0
        if a.oracle:
            G = gt[gt.image == im]
            gxy = np.stack([(G.x1 + G.x2) / 2, (G.y1 + G.y2) / 2], 1)
            S = smooth(P, anchors, a.R, a.alpha, gxy, G.cls.values)
        else:
            S = smooth(P, anchors, a.R, a.alpha)
        for k in range(len(keep)):
            for cc in range(NC):
                if S[k, cc] > 0:
                    rows.append((im, cc, S[k, cc], *anchors[k]))
    pd.DataFrame(rows, columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"]).to_csv(a.out, index=False)
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
