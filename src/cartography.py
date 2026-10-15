"""5.2 training dynamics: score every train GT instance at every saved checkpoint.

collect (GPU): for each checkpoint, sliced inference on train images at conf=0.001, then per GT:
  p_true  = max conf of a correct-class prediction with IoU>=0.5 (0 if none)
  p_other = max conf of a wrong-class prediction with IoU>=0.5 (0 if none) -> AUM-style margin
  other_cls = class of that wrong-class prediction
analyze (CPU): mean/std of p_true over checkpoints (cartography), forgetting events, margin.

Usage:
  python src/cartography.py collect --run /content/drive/MyDrive/auric/runs/E0_s0 --root /content/CV_Assignment_AAL
  python src/cartography.py analyze --name E0_s0
  python src/cartography.py populations --name E0_s0
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import image_paths, image_size, read_labels
from evaluate import FG, iou

OUT = Path(__file__).resolve().parents[1] / "experiments" / "results"
LEARNED = 0.25  # an instance counts as learned at t if p_true >= 0.25 and p_true > p_other


def checkpoints(run):
    ck = sorted(Path(run, "weights").glob("epoch*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    return ck + [Path(run, "weights", "last.pt")]


def score_checkpoint(weights, paths, device=None):
    from ultralytics import YOLO
    from predict import predict_image
    model = YOLO(str(weights))
    rows = []
    for p in paths:
        cls, gt = read_labels(p, image_size(p))
        if not len(gt):
            continue
        b, s, c = predict_image(model, p, 0.001, 640, 32, device)
        I = iou(gt, b) if len(b) else np.zeros((len(gt), 0))
        for k in range(len(gt)):
            hit = I[k] >= FG
            t, o = hit & (c == cls[k]), hit & (c != cls[k])
            j = np.where(o)[0][s[o].argmax()] if o.any() else -1
            rows.append(dict(image=p.stem, idx=k, cls=cls[k], p_true=s[t].max() if t.any() else 0.0,
                             p_other=s[j] if j >= 0 else 0.0, other_cls=c[j] if j >= 0 else -1))
    return pd.DataFrame(rows)


def collect(args):
    paths = image_paths("train", args.root)
    if args.images:
        keep = set(Path(args.images).read_text().split())
        paths = [p for p in paths if p.stem in keep]
    name = Path(args.run).name
    for ck in checkpoints(args.run):
        out = OUT / f"cart_{name}_{ck.stem}.csv"
        if out.exists():  # resumable
            continue
        t0 = time.time()
        score_checkpoint(ck, paths, args.device).to_csv(out, index=False)
        print(f"{ck.name}: {time.time() - t0:.0f}s -> {out.name}", flush=True)
        if args.time_only:
            break


def analyze(args):
    files = sorted(OUT.glob(f"cart_{args.name}_*.csv"),
                   key=lambda f: 10**6 if f.stem.endswith("last") else int(re.findall(r"epoch(\d+)", f.stem)[0]))
    P = [pd.read_csv(f, dtype={"image": str}) for f in files]
    key = ["image", "idx", "cls"]
    pt = np.stack([d.p_true.values for d in P], 1)
    po = np.stack([d.p_other.values for d in P], 1)
    learned = (pt >= LEARNED) & (pt > po)
    df = P[-1][key].copy()
    df["conf_mean"], df["conf_std"] = pt.mean(1), pt.std(1)
    df["forget"] = (learned[:, :-1] & ~learned[:, 1:]).sum(1)  # lower bound: sparse checkpoints
    df["margin_mean"] = (pt - po).mean(1)                       # AUM-style
    df["final_learned"] = learned[:, -1]
    df["other_cls_last"] = P[-1].other_cls.values
    df.to_csv(OUT / f"cart_{args.name}_summary.csv", index=False)
    print(f"{len(files)} checkpoints, {len(df)} instances")
    print(df.groupby("cls")[["conf_mean", "conf_std", "forget", "margin_mean", "final_learned"]].mean().round(3))


def populations(args):
    """Regions + mechanisms on the inspected 80%; prediction test on the untouched 20% holdout and on val.

    Regions (thresholds = quartiles of the inspected set):
      easy: conf_mean >= q75 and conf_std < q75 | ambiguous: conf_std >= q75
      hard-confused: conf_mean < q25 and margin_mean < 0 (another class consistently preferred)
      hard-missed: conf_mean < q25 otherwise (no prediction at the box) | medium: rest
    Prediction test: logistic regression P(not learned at last ckpt) from metadata only
    (log size, aspect, border, crowding, class), and metadata + early dynamics (mean p_true, epochs 0-10).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    s = pd.read_csv(OUT / f"cart_{args.name}_summary.csv", dtype={"image": str})
    e = pd.read_csv(OUT / "eda_instances.csv", dtype={"image": str})
    e = e[e.split == "train"].assign(idx=lambda x: x.groupby("image").cumcount())
    d = s.merge(e.drop(columns="cls"), on=["image", "idx"])
    early = np.mean([pd.read_csv(OUT / f"cart_{args.name}_epoch{k}.csv").p_true.values for k in (0, 5, 10)], 0)
    d["early"] = early
    held = set((OUT.parent / "splits" / "cart_holdout.txt").read_text().split())
    d["held"] = d.image.isin(held)
    I = d[~d.held]
    q = I[["conf_mean", "conf_std"]].quantile([0.25, 0.75])
    lo, hi, shi = q.conf_mean[0.25], q.conf_mean[0.75], q.conf_std[0.75]
    d["region"] = np.select(
        [d.conf_std >= shi, d.conf_mean >= hi, (d.conf_mean < lo) & (d.margin_mean < 0), d.conf_mean < lo],
        ["ambiguous", "easy", "hard-confused", "hard-missed"], "medium")
    d["side"] = np.sqrt(d.w * d.h)
    d.to_csv(OUT / f"cart_{args.name}_populations.csv", index=False)
    I = d[~d.held]
    print(f"thresholds: conf_mean q25 {lo:.3f} q75 {hi:.3f}, conf_std q75 {shi:.3f}; inspected {len(I)}, holdout {d.held.sum()}")
    print(pd.crosstab(I.cls, I.region, margins=True))
    print(I.groupby("region").agg(n=("side", "size"), side_med=("side", "median"), border=("border", "mean"),
                                   crowded=("max_iou_other", lambda x: (x > 0).mean()), forget=("forget", "mean")).round(2))
    hc = I[I.region == "hard-confused"]
    print("hard-confused: class -> preferred class at last ckpt\n", pd.crosstab(hc.cls, hc.other_cls_last))

    def feats(x, dyn):
        f = pd.DataFrame({"lside": np.log(np.sqrt(x.w * x.h)), "aspect": np.log(np.maximum(x.w, x.h) / np.minimum(x.w, x.h)),
                          "border": x.border.astype(float), "crowd": (x.max_iou_other > 0).astype(float)})
        for c in range(1, 5):
            f[f"c{c}"] = (x.cls == c).astype(float)
        if dyn:
            f["early"] = x.early
        return f.values
    H = d[d.held]
    for dyn in (False, True):
        m = LogisticRegression(max_iter=1000).fit(feats(I, dyn), ~I.final_learned)
        auc = roc_auc_score(~H.final_learned, m.predict_proba(feats(H, dyn))[:, 1])
        print(f"holdout AUC (not learned), {'metadata + early dynamics' if dyn else 'metadata only'}: {auc:.3f}")
    auc = roc_auc_score(~H.final_learned, -H.side)
    print(f"holdout AUC, size alone: {auc:.3f}")

    # transfer to val: does the metadata-only hardness score predict which val GT the detector misses?
    m = LogisticRegression(max_iter=1000).fit(feats(I, False), ~I.final_learned)
    v = pd.read_csv(OUT / f"{args.name}_gt.csv", dtype={"image": str})
    ve = pd.read_csv(OUT / "eda_instances.csv", dtype={"image": str})
    ve = ve[ve.split == "val"].reset_index(drop=True)
    assert len(ve) == len(v)
    v = v.assign(w=v.x2 - v.x1, h=v.y2 - v.y1, border=ve.border.values, max_iou_other=ve.max_iou_other.values)
    v["p_hard"] = m.predict_proba(feats(v, False))[:, 1]
    print("val: correct-class recall by predicted-hardness quartile (metadata model fit on train-inspected)")
    print(v.groupby(pd.qcut(v.p_hard, 4, labels=["Q1 easy", "Q2", "Q3", "Q4 hard"])).matched.agg(["mean", "size"]).round(3))
    print(f"val AUC (missed): {roc_auc_score(~v.matched, v.p_hard):.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--run", required=True)
    c.add_argument("--root", default=None)
    c.add_argument("--images", default=None)
    c.add_argument("--device", default=None)
    c.add_argument("--time_only", action="store_true", help="score the first checkpoint only and report time")
    a = sub.add_parser("analyze")
    a.add_argument("--name", required=True)
    sub.add_parser("populations").add_argument("--name", required=True)
    args = ap.parse_args()
    {"collect": collect, "analyze": analyze, "populations": populations}[args.cmd](args)


if __name__ == "__main__":
    main()
