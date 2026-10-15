"""Align xView's labels to our TRAIN images that were rescaled or cropped (diagnostic / label-cleaning input).

For each train image with xView labels: if >= 80% of our GT already match an xView box at IoU >= 0.5, keep the
identity transform. Otherwise estimate x' = s * x + t: candidate scales from ratios of matched-size statistics
(median our-GT side / median xView truck side, and a small grid around it); for each scale the offset t is the mode
of all pairwise centre differences (our GT centre - s * xView vehicle centre), refined by the median of the
differences of the resulting nearest matches. The transform is accepted if >= 80% of our GT then match a
transformed xView box at IoU >= 0.5. No val image is used.

Output: experiments/results/xview_train_aligned_labels.csv (xView boxes in our pixel coordinates, accepted images
only) with columns image, cat, x1, y1, x2, y2, name, s, tx, ty.

Usage: python src/align_xview.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

from data import CLASSES, image_paths, image_size, read_labels

R = Path(__file__).resolve().parents[1] / "experiments" / "results"
VEH = set(CLASSES) | {"Truck", "Trailer", "Utility Truck", "Dump Truck", "Bus", "Pickup Truck", "Cement Mixer", "Crane Truck",
                      "Haul Truck", "Small Car", "Passenger Vehicle"}


def match_share(b, xb):
    if not len(xb):
        return 0.0
    return float((box_iou(torch.tensor(b, dtype=torch.float32), torch.tensor(xb, dtype=torch.float32)).max(1).values >= 0.5).float().mean())


def fit(b, xi):
    gc = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
    v = xi[xi.name.isin(VEH)]
    xb = v[["x1", "y1", "x2", "y2"]].values.astype(np.float64)
    xc = np.c_[(xb[:, 0] + xb[:, 2]) / 2, (xb[:, 1] + xb[:, 3]) / 2]
    tr = v[v.name.isin(CLASSES)]
    if not len(tr):
        return None
    s0 = np.median(np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))) / np.median(np.sqrt((tr.x2 - tr.x1) * (tr.y2 - tr.y1)))
    best = (0.0, None)
    for s in s0 * np.array([0.9, 0.95, 1.0, 1.05, 1.1]):
        d = (gc[:, None, :] - s * xc[None, :, :]).reshape(-1, 2)
        if len(d) > 400000:
            d = d[np.random.default_rng(0).choice(len(d), 400000, replace=False)]
        H, ex, ey = np.histogram2d(d[:, 0], d[:, 1], bins=[np.arange(d[:, 0].min(), d[:, 0].max() + 8, 8), np.arange(d[:, 1].min(), d[:, 1].max() + 8, 8)])
        i, j = np.unravel_index(H.argmax(), H.shape)
        t = np.array([ex[i] + 4, ey[j] + 4])
        for _ in range(2):  # refine the offset from the nearest matches
            pc = s * xc + t
            dist = np.linalg.norm(gc[:, None, :] - pc[None, :, :], axis=2)
            k = dist.argmin(1)
            ok = dist[np.arange(len(gc)), k] < 10
            if ok.sum() >= 3:
                t = t + np.median(gc[ok] - pc[k[ok]], axis=0)
        sh = match_share(b, s * xb + np.tile(t, 2))
        if sh > best[0]:
            best = (sh, (s, t))
    return best


def main():
    xv = pd.concat([pd.read_csv(R / f, dtype={"image": str}) for f in ("xview_train_labels.csv", "xview_heldout_labels.csv")])
    out, rows = [], []
    for p in image_paths("train"):
        xi = xv[xv.image == p.stem]
        if not len(xi):
            continue
        cls, b = read_labels(p, image_size(p))
        if not len(b):
            continue
        ident = match_share(b, xi[["x1", "y1", "x2", "y2"]].values)
        if ident >= 0.8:
            s, t, share, kind = 1.0, np.zeros(2), ident, "identity"
        else:
            r = fit(b, xi)
            if r is None or r[1] is None or r[0] < 0.8:
                rows.append((p.stem, "unaligned", ident, r[0] if r else np.nan, np.nan))
                continue
            share, (s, t), kind = r[0], r[1], "fitted"
        a = xi.copy()
        a[["x1", "x2"]] = a[["x1", "x2"]] * s + t[0]
        a[["y1", "y2"]] = a[["y1", "y2"]] * s + t[1]
        a["s"], a["tx"], a["ty"] = s, t[0], t[1]
        out.append(a)
        rows.append((p.stem, kind, ident, share, s))
    pd.concat(out).to_csv(R / "xview_train_aligned_labels.csv", index=False)
    rep = pd.DataFrame(rows, columns=["image", "result", "identity_share", "aligned_share", "scale"])
    rep.to_csv(R / "xview_train_alignment.csv", index=False)
    print(rep.result.value_counts().to_string())
    print("fitted scales:", rep[rep.result == "fitted"].scale.round(2).value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()
