"""Dataset forensics: every label-level problem we can measure without a GPU, plus any exploitable structure.

Checks: label validity, duplicate / conflicting boxes, same-class overlaps that class-wise NMS cannot resolve,
size and aspect by class (size shortcut?), train-vs-val scale shift, class concentration in few images,
per-image class purity, a neighbourhood-context oracle for classification, border/crowding shares,
a val under-annotation proxy from the best detector's background FPs.

Usage: python src/dataset_forensics.py [--name E1_up2_s0]   -> experiments/results/dataset_forensics.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import CLASSES
from evaluate import iou

RES = Path(__file__).resolve().parents[1] / "experiments" / "results"
SHORT = ["Cargo", "Box", "Flatbed", "Tractor", "Liquid"]
L = []


def say(*a):
    L.append(" ".join(str(x) for x in a))


def pairs(g, thr, same_class):
    """Count boxes that overlap another box in the same image at IoU >= thr (same or different class)."""
    b = g[["x1", "y1", "x2", "y2"]].values
    if len(b) < 2:
        return 0
    I = iou(b, b)
    np.fill_diagonal(I, 0)
    same = g.cls.values[:, None] == g.cls.values[None, :]
    M = I >= thr
    M &= same if same_class else ~same
    return int(M.any(1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="E1_up2_s0", help="run whose val matches give the under-annotation proxy")
    a = ap.parse_args()
    e = pd.read_csv(RES / "eda_instances.csv", dtype={"image": str})
    e["x1"], e["y1"], e["x2"], e["y2"] = e.cx - e.w / 2, e.cy - e.h / 2, e.cx + e.w / 2, e.cy + e.h / 2
    e["side"] = np.sqrt(e.w * e.h)
    e["aspect"] = np.maximum(e.w, e.h) / np.maximum(np.minimum(e.w, e.h), 1e-6)
    im = pd.read_csv(RES / "eda_images.csv", dtype={"image": str})
    say("# Dataset forensics\n")

    say("## A. Label validity")
    for s, g in e.groupby("split"):
        say(f"- {s}: {len(g)} boxes; class ids {sorted(g.cls.unique())}; invalid flag {int(g.invalid.sum())}; "
            f"side < 4 px: {int((g.side < 4).sum())}; side > 150 px: {int((g.side > 150).sum())}; "
            f"aspect > 4: {int((g.aspect > 4).sum())}; outside image: "
            f"{int(((g.x1 < -1) | (g.y1 < -1)).sum())}")
    say(f"- empty images: train {int((im[im.split == 'train'].n == 0).sum())}, val {int((im[im.split == 'val'].n == 0).sum())}\n")

    say("## B. Duplicate and conflicting boxes (per image)")
    for s, g in e.groupby("split"):
        dup = sum(pairs(x, 0.7, True) for _, x in g.groupby("image"))
        conf = sum(pairs(x, 0.5, False) for _, x in g.groupby("image"))
        nms = sum(pairs(x, 0.5, True) for _, x in g.groupby("image"))
        say(f"- {s}: same-class IoU>=0.7 (duplicate labels) {dup} ({dup / len(g):.1%}); "
            f"different-class IoU>=0.5 (one object, two labels) {conf} ({conf / len(g):.1%}); "
            f"same-class IoU>=0.5 (class-wise NMS 0.5 can keep only one) {nms} ({nms / len(g):.1%})")
    say("")

    say("## C. Size and aspect by class (px, native resolution)")
    t = e.groupby(["split", "cls"]).side.describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "90%"]]
    t.index = t.index.set_levels([t.index.levels[0], [SHORT[i] for i in t.index.levels[1]]])
    say(t.round(1).to_string())
    t2 = e[e.split == "train"].groupby("cls").aspect.median()
    say("\n- median aspect (train): " + ", ".join(f"{SHORT[c]} {v:.2f}" for c, v in t2.items()))
    # size-only classifier: how much does size alone predict the class? (nearest-median rule, train->val)
    med = e[e.split == "train"].groupby("cls").side.median().values
    v = e[e.split == "val"]
    pred = np.abs(v.side.values[:, None] - med[None, :]).argmin(1)
    say(f"- size-only classifier (nearest class median side) val accuracy: {np.mean(pred == v.cls.values):.3f}, "
        f"majority-class baseline: {v.cls.value_counts(normalize=True).max():.3f}\n")

    say("## D. Train vs val scale shift (per-image median box side)")
    pm = e.groupby(["split", "image"]).side.median().reset_index()
    for s, g in pm.groupby("split"):
        say(f"- {s}: per-image median side  10% {g.side.quantile(.1):.1f}  50% {g.side.median():.1f}  90% {g.side.quantile(.9):.1f} px")
    big = im[(im.split == "val")].sort_values("W", ascending=False).head(3)
    say("- largest val images: " + ", ".join(f"{r.image} ({r.W}x{r.H}, {r.n} boxes, median side "
        f"{pm[(pm.split == 'val') & (pm.image == r.image)].side.values[0]:.1f})" for r in big.itertuples()) + "\n")

    say("## E. Class concentration (how few images hold a class)")
    for s, g in e.groupby("split"):
        rows = []
        for c in range(5):
            cnt = g[g.cls == c].groupby("image").size().sort_values(ascending=False)
            rows.append((SHORT[c], int(cnt.sum()), len(cnt), f"{cnt.head(3).sum() / max(cnt.sum(), 1):.0%}",
                         ", ".join(f"{i}:{n}" for i, n in cnt.head(3).items())))
        say(f"\n{s}:\n" + pd.DataFrame(rows, columns=["class", "boxes", "images", "top-3 images hold", "top-3"]).to_string(index=False))
    say("")

    say("## F. Per-image class purity (share of an image's boxes in its majority class)")
    for s, g in e.groupby("split"):
        pur = g.groupby("image").cls.agg(lambda x: x.value_counts(normalize=True).max())
        n = g.groupby("image").size()
        pur = pur[n >= 5]
        say(f"- {s} (images with >=5 boxes, n={len(pur)}): purity median {pur.median():.2f}, >=0.9 in {np.mean(pur >= 0.9):.0%} of images")
    say("")

    say("## G. Context oracle: can neighbours predict a box's class? (leave-one-out, val GT)")
    v = e[e.split == "val"].reset_index(drop=True)
    for R in (150, 400, 1e9):
        hit = tot = 0
        for _, g in v.groupby("image"):
            c, xy = g.cls.values, g[["cx", "cy"]].values
            if len(g) < 2:
                continue
            D = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
            for i in range(len(g)):
                nb = (D[i] <= R) & (np.arange(len(g)) != i)
                if nb.any():
                    hit += np.bincount(c[nb], minlength=5).argmax() == c[i]
                    tot += 1
        lab = "whole image" if R > 1e8 else f"radius {int(R)} px"
        say(f"- neighbour-majority ({lab}): accuracy {hit / tot:.3f} on {tot} boxes with a neighbour")
    say("  (reference: crop classifiers on GT boxes 0.57-0.59; majority class 0.52)\n")

    say("## H. Border and crowding")
    for s, g in e.groupby("split"):
        say(f"- {s}: touching border {g.border.mean():.1%}; overlapping another GT (IoU>0) {np.mean(g.max_iou_other > 0):.1%}; IoU>0.3 {np.mean(g.max_iou_other > 0.3):.1%}")
    say("")

    say(f"## I. Val under-annotation proxy ({a.name}: confident background FPs vs GT per image)")
    m = pd.read_csv(RES / f"{a.name}_matches.csv", dtype={"image": str})
    fp = m[(m.err == "bg") & (m.conf >= 0.5)].groupby("image").size()
    gt = im[im.split == "val"].set_index("image").n
    t = pd.DataFrame({"gt": gt, "bgFP>=0.5": fp}).fillna(0).astype(int)
    t["ratio"] = (t["bgFP>=0.5"] / t["gt"].clip(lower=1)).round(2)
    say(t.sort_values("ratio", ascending=False).head(8).to_string())
    say(f"\n- total confident bg FPs {int(t['bgFP>=0.5'].sum())} vs {int(t["gt"].sum())} GT; images with ratio >= 0.5: {int((t.ratio >= 0.5).sum())}\n")

    say("## J. Instances per image")
    for s, g in im.groupby("split"):
        say(f"- {s}: boxes/image  min {g.n.min()}  median {g.n.median():.0f}  90% {g.n.quantile(.9):.0f}  max {g.n.max()}")
    out = RES / "dataset_forensics.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
