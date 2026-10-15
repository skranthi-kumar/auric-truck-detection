"""Error budget for a scored run (diagnostic only; oracle numbers are bounds, they overlap and must not be added).

From <run>_matches.csv / <run>_gt.csv (evaluate.py) plus xView's val labels for cause attribution:
  A. per class: AP50, recall (correct class, any conf), recall / precision at conf >= 0.25, GT never detected
  B. per-class oracles (TIDE-style, re-matched as evaluate.oracle): fix subtype (cls), fix boxes (loc),
     delete duplicates, delete background FPs, recover never-detected GT (removed from the GT count)
  C. matched-box classification accuracy: for each GT, the highest-confidence prediction of ANY class with
     IoU >= 0.5 (a TP or a wrong-subtype row targeting it); accuracy = share where that prediction is correct
  D. false positives by cause, per class: wrong subtype / loc+both / duplicate / background split by xView's label
     at the box (13 xView-aligned images; the other images are "no reference"); counts at conf >= 0.25 and the AP
     regained by deleting each cause (rank-aware, all 22 images)
  E. recall by GT size and by image density; concentration of each class's conf >= 0.25 FPs in its top-2 images

Usage: python src/error_budget.py --run SC1 [--agn SC1_agn]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

from data import CLASSES
from evaluate import ap101, match, per_class_ap

R = Path(__file__).resolve().parents[1] / "experiments" / "results"
GENERIC = {"Truck", "Trailer", "Utility Truck", "Dump Truck", "Cement Mixer", "Crane Truck", "Haul Truck",
           "Pickup Truck", "Cargo Car", "Tank car", "Flat Car"}
OTHER_VEH = {"Bus", "Small Car", "Passenger Vehicle", "Passenger Car", "Locomotive", "Engineering Vehicle",
             "Front loader/Bulldozer", "Excavator", "Ground Grader", "Mobile Crane", "Reach Stacker", "Straddle Carrier"}


def oracle_pc(m, g, kind):
    p = m.copy()
    gi = g.set_index("gt_id")
    if kind in ("cls", "loc"):
        fix = p.err == kind
        t = gi.loc[p.target[fix]]
        if kind == "cls":
            p.loc[fix, "cls"] = t.cls.values
        else:
            p.loc[fix, ["x1", "y1", "x2", "y2"]] = t[["x1", "y1", "x2", "y2"]].values
        p = p.assign(fixed=fix)
        r = match(p[["image", "cls", "conf", "x1", "y1", "x2", "y2", "fixed"]], g)
        r = r[~(r.fixed & ~r.tp)]
    else:
        r = p[p.err != kind]
    return np.array(per_class_ap(r, g))


def aligned_images(g, xv):
    out = []
    for im, gi in g.groupby("image"):
        xi = xv[xv.image == im]
        if len(xi) and (box_iou(torch.tensor(gi[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32),
                                torch.tensor(xi[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32)).max(1).values >= 0.5).float().mean() >= 0.8:
            out.append(im)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="SC1")
    ap.add_argument("--agn", default=None, help="name of the class-agnostic evaluation of the same predictions")
    a = ap.parse_args()
    m = pd.read_csv(R / f"{a.run}_matches.csv", dtype={"image": str})
    g = pd.read_csv(R / f"{a.run}_gt.csv", dtype={"image": str})
    xv = pd.read_csv(R / "xview_val_labels.csv", dtype={"image": str})
    out = []

    # A
    base = np.array(per_class_ap(m, g))
    best = m[m.tp].groupby("gt_id").conf.max()
    g["best_tp"] = g.gt_id.map(best).fillna(0)
    hi = m[m.conf >= 0.25]
    A = pd.DataFrame({"AP50": base, "n_gt": g.groupby("cls").size().reindex(range(5)).values,
                      "recall_any_conf": [((g.cls == c) & (g.best_tp > 0)).sum() / (g.cls == c).sum() for c in range(5)],
                      "recall@0.25": [((g.cls == c) & (g.best_tp >= 0.25)).sum() / (g.cls == c).sum() for c in range(5)],
                      "precision@0.25": [hi[(hi.cls == c)].tp.mean() if (hi.cls == c).any() else np.nan for c in range(5)],
                      "never_detected": [int(((g.cls == c) & ~g.gt_id.isin(m.gt_id[m.tp]) & ~g.gt_id.isin(m.target[m.err.isin(["cls", "loc", "both"])])).sum()) for c in range(5)]},
                     index=CLASSES)
    agn = None
    if a.agn and (R / f"{a.agn}_metrics.json").exists():
        agn = json.load(open(R / f"{a.agn}_metrics.json"))["mAP50"]
    out.append(f"{a.run}: mAP50 {np.nanmean(base):.4f}" + (f"; class-agnostic AP50 {agn:.4f}" if agn else "") +
               "\nA. per-class performance\n" + A.round(3).to_string())

    # B
    B = pd.DataFrame(index=CLASSES)
    for k, lab in (("cls", "fix subtype"), ("loc", "fix boxes"), ("dup", "delete duplicates"), ("bg", "delete background FPs")):
        B[lab] = oracle_pc(m, g, k) - base
    unrec = g[g.best_tp == 0]
    B["recover never-matched GT"] = [ap101(m.conf.values[m.cls == c], m.tp.values[m.cls == c],
                                           (g.cls == c).sum() - (unrec.cls == c).sum()) - base[c] for c in range(5)]
    B.loc["mean"] = B.mean()
    out.append("B. oracle gains per class (bounds; overlapping; do not add)\n" + B.round(3).to_string())

    # C
    cand = pd.concat([m[m.tp].assign(g_id=m.gt_id[m.tp], ok=True), m[m.err == "cls"].assign(g_id=m.target[m.err == "cls"], ok=False)])
    top = cand.sort_values("conf", ascending=False).drop_duplicates("g_id").set_index("g_id")
    g["boxed"] = g.gt_id.isin(top.index)
    g["cls_ok"] = g.gt_id.map(top.ok)
    pred_cls = g.gt_id.map(top.cls)
    C = pd.DataFrame({"GT with a matched box": g.groupby("cls").boxed.mean().values,
                      "subtype accuracy on matched boxes": [g[(g.cls == c) & g.boxed].cls_ok.mean() for c in range(5)]}, index=CLASSES)
    conf = pd.crosstab(g.cls[g.boxed].map(dict(enumerate(CLASSES))), pred_cls[g.boxed].map(dict(enumerate(CLASSES))))
    out.append("C. matched-box classification (top-scoring prediction of any class at IoU >= 0.5)\n" + C.round(3).to_string()
               + "\nGT class (rows) x class of the top matched prediction (cols)\n" + conf.to_string())

    # D
    al = set(aligned_images(g, xv))
    cause = pd.Series(index=m.index, dtype=object)
    cause[m.err == "cls"] = "wrong subtype (GT truck)"
    cause[m.err.isin(["loc", "both"])] = "poor localisation"
    cause[m.err == "dup"] = "duplicate"
    bg = m[(m.err == "bg") & (m.conf >= 0.05)]
    cause[(m.err == "bg") & (m.conf < 0.05)] = "background, conf < 0.05 (not attributed)"
    cause[bg.index[~bg.image.isin(al)]] = "background, no xView reference"
    for im, f in bg[bg.image.isin(al)].groupby("image"):
        xi = xv[xv.image == im]
        iou = box_iou(torch.tensor(f[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32),
                      torch.tensor(xi[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32))
        b, j = iou.max(1)
        names = xi.name.values[j.numpy()]
        lab = np.where(b.numpy() < 0.3, "background: nothing labelled",
                       np.where(np.isin(names, CLASSES), "background: our-class truck the GT omits (annotation)",
                                np.where(np.isin(names, list(GENERIC)), "background: generic / non-target truck",
                                         np.where(np.isin(names, list(OTHER_VEH)), "background: other vehicle", "background: non-vehicle object"))))
        cause[f.index] = lab
    fp = m[~m.tp]
    D_cnt = pd.crosstab(cause[fp.index][fp.conf >= 0.25], fp.cls[fp.conf >= 0.25].map(dict(enumerate(CLASSES))))
    D_cnt["ALL"] = D_cnt.sum(1)
    rows = []
    for k in cause.dropna().unique():
        drop = (cause == k) & ~m.tp
        r = m[~drop]
        aps = np.array([ap101(r.conf.values[r.cls == c], r.tp.values[r.cls == c], (g.cls == c).sum()) for c in range(5)])
        rows.append([k, *(aps - base).round(3), round(np.nanmean(aps) - np.nanmean(base), 4)])
    D_ap = pd.DataFrame(rows, columns=["cause", *CLASSES, "mean"]).set_index("cause").sort_values("mean", ascending=False)
    out.append(f"D. false positives by cause (xView reference on {len(al)} aligned images)\ncounts at conf >= 0.25\n" + D_cnt.to_string()
               + "\nAP regained by deleting each cause (rank-aware; bounds, overlapping)\n" + D_ap.to_string())

    # E
    g["side"] = np.sqrt((g.x2 - g.x1) * (g.y2 - g.y1))
    g["size"] = pd.cut(g.side, [0, 16, 24, 32, 1e9], labels=["<16", "16-24", "24-32", ">=32"])
    dens = g.groupby("image").size()
    g["density"] = g.image.map(pd.qcut(dens, 3, labels=["sparse", "medium", "dense"]))
    g["found"] = g.best_tp > 0
    E1 = g.pivot_table(index="cls", columns="size", values="found", aggfunc="mean", observed=False).rename(index=dict(enumerate(CLASSES)))
    E2 = g.pivot_table(index="cls", columns="density", values="found", aggfunc="mean", observed=False).rename(index=dict(enumerate(CLASSES)))
    hfp = fp[fp.conf >= 0.25]
    conc = {CLASSES[c]: round(hfp[hfp.cls == c].image.value_counts().head(2).sum() / max((hfp.cls == c).sum(), 1), 2) for c in range(5)}
    out.append("E. recall (correct class, any conf) by GT size\n" + E1.round(3).to_string() + "\nby image density\n" + E2.round(3).to_string()
               + f"\nshare of each class's conf >= 0.25 FPs in its top-2 images: {conc}")

    text = "\n\n".join(out)
    print(text)
    (R / f"{a.run}_error_budget.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
