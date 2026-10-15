"""Single source of truth for every reported number.

Input : predictions CSV (image, cls, conf, x1, y1, x2, y2) in full-image pixels, from predict.py.
Output: mAP50 + per-class AP50 (COCO 101-point), bootstrap CI over images, TIDE-style error
        breakdown with oracle dAP, confusion matrix, per-prediction and per-GT tables.

Error rules (TIDE thresholds: fg IoU 0.5, bg IoU 0.1), applied to every non-TP prediction in order:
  dup  : IoU>=0.5 with a same-class GT that a higher-scoring prediction already matched
  cls  : IoU>=0.5 with a GT of another class
  loc  : 0.1<=IoU<0.5 with a same-class GT
  both : 0.1<=IoU<0.5 with a GT of another class only
  bg   : IoU<0.1 with every GT
  missed: a GT with no TP and not the target of any cls/loc/both prediction

Usage: python src/evaluate.py --preds experiments/results/E0_val_preds.csv --name E0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import CLASSES, image_paths, image_size, read_labels

FG, BG, NC = 0.5, 0.1, len(CLASSES)
OUT = Path(__file__).resolve().parents[1] / "experiments" / "results"


def iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def load_gt(split, root=None, images=None):
    rows = []
    for p in (image_paths(split, root) if root else image_paths(split)):
        if images is not None and p.stem not in images:
            continue
        cls, b = read_labels(p, image_size(p))
        rows += [dict(image=p.stem, cls=c, x1=x[0], y1=x[1], x2=x[2], y2=x[3]) for c, x in zip(cls, b)]
    return pd.DataFrame(rows).reset_index(names="gt_id")


def match(preds, gt):
    """Greedy per image/class matching by descending confidence. Adds tp, gt_id, err to preds."""
    preds = preds.sort_values("conf", ascending=False).reset_index(drop=True)
    tp = np.zeros(len(preds), bool)
    gid = np.full(len(preds), -1)
    err = np.array(["tp"] * len(preds), dtype=object)
    target = np.full(len(preds), -1)  # GT a cls/loc/both error points at
    for im, P in preds.groupby("image", sort=False):
        G = gt[gt.image == im]
        if len(G) == 0:
            err[P.index] = "bg"
            continue
        ious = iou(P[["x1", "y1", "x2", "y2"]].values, G[["x1", "y1", "x2", "y2"]].values)
        same = P.cls.values[:, None] == G.cls.values[None]
        used = np.zeros(len(G), bool)
        for r, i in enumerate(P.index):  # P is already sorted by conf
            s = np.where(same[r] & ~used, ious[r], -1)
            j = s.argmax()
            if s[j] >= FG:
                used[j], tp[i], gid[i] = True, True, G.gt_id.values[j]
                continue
            so, oo = np.where(same[r], ious[r], 0), np.where(~same[r], ious[r], 0)
            if so.max() >= FG:
                err[i] = "dup"
            elif oo.max() >= FG:
                err[i], target[i] = "cls", G.gt_id.values[oo.argmax()]
            elif so.max() >= BG:
                err[i], target[i] = "loc", G.gt_id.values[so.argmax()]
            elif oo.max() >= BG:
                err[i], target[i] = "both", G.gt_id.values[oo.argmax()]
            else:
                err[i] = "bg"
    return preds.assign(tp=tp, gt_id=gid, err=err, target=target)


def ap101(conf, tp, n_gt, w=None):
    """COCO-style 101-point interpolated AP. w = per-prediction weights (bootstrap)."""
    if n_gt <= 0:
        return np.nan
    if len(conf) == 0:
        return 0.0
    o = np.argsort(-conf, kind="stable")
    w = np.ones(len(conf)) if w is None else w[o]
    ctp = np.cumsum(w * tp[o])
    rec, prec = ctp / n_gt, ctp / np.maximum(np.cumsum(w), 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    return np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0).mean()


def per_class_ap(m, gt):
    return np.array([ap101(m.conf.values[m.cls == c], m.tp.values[m.cls == c], (gt.cls == c).sum())
                     for c in range(NC)])


def bootstrap(m, gt, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ims = gt.image.unique()
    ix = {im: k for k, im in enumerate(ims)}
    pi = m.image.map(ix).values
    gi = gt.image.map(ix).values
    out = []
    for _ in range(n):
        cnt = np.bincount(rng.integers(0, len(ims), len(ims)), minlength=len(ims))
        aps = [ap101(m.conf.values[m.cls == c], m.tp.values[m.cls == c],
                     cnt[gi[gt.cls.values == c]].sum(), cnt[pi[m.cls.values == c]].astype(float))
               for c in range(NC)]
        out.append(aps)
    out = np.array(out)
    return np.nanpercentile(np.nanmean(out, 1), [2.5, 97.5]), np.nanpercentile(out, [2.5, 97.5], 0)


def oracle(preds, m, gt, kind):
    """TIDE-style fix of one error type, re-match, return mAP50."""
    p = m.copy()
    g = gt.set_index("gt_id")
    if kind in ("cls", "loc"):
        fix = p.err == kind
        t = g.loc[p.target[fix]]
        if kind == "cls":
            p.loc[fix, "cls"] = t.cls.values
        else:
            p.loc[fix, ["x1", "y1", "x2", "y2"]] = t[["x1", "y1", "x2", "y2"]].values
        p = p.assign(fixed=fix)
        r = match(p[["image", "cls", "conf", "x1", "y1", "x2", "y2", "fixed"]], gt)
        r = r[~(r.fixed & ~r.tp)]  # a fixed prediction that becomes a duplicate is suppressed
    else:
        r = p[p.err != kind]
    return np.nanmean(per_class_ap(r, gt))


def confusion(m, gt, thr):
    """Rows: GT class (+ background). Cols: predicted class (+ missed). Uses preds with conf>=thr."""
    cm = np.zeros((NC + 1, NC + 1), int)
    p = m[m.conf >= thr]
    used = set()
    for im, G in gt.groupby("image"):
        P = p[p.image == im]
        if len(P):
            ious = iou(G[["x1", "y1", "x2", "y2"]].values, P[["x1", "y1", "x2", "y2"]].values)
        for r, c in enumerate(G.cls.values):
            cand = np.where(ious[r] >= FG)[0] if len(P) else []
            if len(cand):
                k = cand[P.conf.values[cand].argmax()]
                cm[c, P.cls.values[k]] += 1
                used.add(P.index[k])
            else:
                cm[c, NC] += 1
    for c in p[~p.index.isin(used) & (p.err == "bg")].cls:
        cm[NC, c] += 1
    return cm


def gt_table(m, gt):
    """Per GT: TP-matched? and the class/conf of the best-scoring prediction overlapping it (IoU>=0.5)."""
    g = gt.copy()
    g["matched"] = g.gt_id.isin(m.gt_id[m.tp])
    g["targeted"] = g.gt_id.isin(m.target[m.err.isin(["cls", "loc", "both"])])
    best_cls, best_conf = [], []
    for im, G in g.groupby("image", sort=False):
        P = m[m.image == im]
        ious = iou(G[["x1", "y1", "x2", "y2"]].values, P[["x1", "y1", "x2", "y2"]].values) if len(P) \
            else np.zeros((len(G), 0))
        for r in range(len(G)):
            cand = np.where(ious[r] >= FG)[0]
            k = cand[P.conf.values[cand].argmax()] if len(cand) else None
            best_cls.append(P.cls.values[k] if k is not None else -1)
            best_conf.append(P.conf.values[k] if k is not None else 0.0)
    g["best_cls"], g["best_conf"] = best_cls, best_conf  # aligned: groupby keeps original order per image
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", default=None)
    ap.add_argument("--cm_thr", type=float, default=0.25)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--iou", type=float, default=0.5, help="diagnostic only: match IoU (reported metric uses 0.5)")
    ap.add_argument("--agnostic", action="store_true", help="diagnostic only: merge all classes into one")
    args = ap.parse_args()
    global FG
    FG = args.iou

    preds = pd.read_csv(args.preds, dtype={"image": str})
    gt = load_gt(args.split, args.root)
    gt = gt.sort_values(["image", "gt_id"]).reset_index(drop=True)
    if args.agnostic:
        preds["cls"], gt["cls"] = 0, 0
    m = match(preds, gt)
    aps = per_class_ap(m, gt)
    ci, ci_c = bootstrap(m, gt, args.boot) if args.boot else (None, None)
    dap = {k: oracle(preds, m, gt, k) - np.nanmean(aps) for k in ("cls", "loc", "both", "dup", "bg")}
    gtt = gt_table(m, gt)
    missed = int((~gtt.matched & ~gtt.targeted).sum())
    cm = confusion(m, gt, args.cm_thr)

    res = dict(name=args.name, mAP50=float(np.nanmean(aps)),
               AP50={CLASSES[c]: float(a) for c, a in enumerate(aps)},
               mAP50_CI95=None if ci is None else [float(x) for x in ci],
               AP50_CI95=None if ci is None else {CLASSES[c]: [float(ci_c[0, c]), float(ci_c[1, c])] for c in range(NC)},
               oracle_dAP=dap, n_err={k: int(v) for k, v in m.err.value_counts().items()},
               n_err_conf_ge_thr={k: int(v) for k, v in m[m.conf >= args.cm_thr].err.value_counts().items()},
               missed_gt=missed, n_gt=len(gt), confusion=cm.tolist(), cm_thr=args.cm_thr)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}_metrics.json").write_text(json.dumps(res, indent=1))
    m.to_csv(OUT / f"{args.name}_matches.csv", index=False)
    gtt.to_csv(OUT / f"{args.name}_gt.csv", index=False)

    print(f"{args.name}: mAP50 = {res['mAP50']:.4f}" + (f"  (95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci is not None else ""))
    for c in range(NC):
        print(f"  {CLASSES[c]:<16} AP50 {aps[c]:.4f}" + (f"  [{ci_c[0, c]:.3f}, {ci_c[1, c]:.3f}]" if ci is not None else ""))
    print("  oracle dAP:", {k: round(v, 4) for k, v in dap.items()}, " missed GT:", missed)
    print(pd.DataFrame(cm, index=CLASSES + ["background"], columns=CLASSES + ["missed"]).to_string())


if __name__ == "__main__":
    main()
