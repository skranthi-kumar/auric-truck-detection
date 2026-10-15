"""Data audit (diagnostic only; no training, no val tuning): image outliers, label outliers, train-val
distribution outliers and annotation coverage.

Inputs: our labels (train + val), images (size, brightness), xView's 60-class labels for our images
(experiments/results/xview_{train,val}_labels.csv; external, provenance in LOG.md), and the RFX0 train dump
(per-query class scores on train tiles) for model-disagreement flags on the 74 checkpoint-unseen train images.

1. Image outliers (train + val): longer side vs train range; scale ratio = median our-GT truck side / median
   xView truck side (images where both exist and boxes align); brightness / contrast; box count; empty images.
2. Label outliers: per-class log box side and aspect z-scores (|z| > 3), boxes < 5 px or touching the border,
   near-duplicate boxes (IoU > 0.7, same or different class), and class disagreement with xView on aligned
   images (our class vs the class of the xView box at IoU >= 0.5); plus, on the 74 checkpoint-unseen train
   images, GT boxes whose best detector query gives >= 0.5 to a different one of our classes.
3. Train-val outliers: each val image's features vs the train 1st-99th percentile (and z-score).
4. Coverage: per aligned image, xView boxes of OUR 5 classes with max IoU < 0.3 to our GT ("omitted"),
   as a share of our GT; train vs val.

Usage: python src/data_audit.py --dump ~/auric_kaggle/DUMP/rfx0_train_dump.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.ops import box_iou

from data import CLASSES, image_paths, image_size, read_labels

R = Path(__file__).resolve().parents[1] / "experiments" / "results"
TB = lambda d: torch.tensor(np.asarray(d, dtype=np.float32).reshape(-1, 4))


def side(b):
    return np.sqrt((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None)
    ap.add_argument("--holdout", default=str(R.parent / "splits" / "rfx_heldout_train_ids.txt"))
    a = ap.parse_args()
    xv = {s: pd.read_csv(R / f"xview_{s}_labels.csv", dtype={"image": str}) for s in ("train", "val")}
    imgs, boxes = [], []
    for split in ("train", "val"):
        X = xv[split]
        for p in image_paths(split):
            W, H = image_size(p)
            cls, b = read_labels(p, (W, H))
            with Image.open(p) as im:
                g = np.asarray(im.convert("L").reduce(8), dtype=np.float32)
            r = dict(split=split, image=p.stem, W=W, H=H, longer=max(W, H), n_gt=len(b), bright=g.mean(), contrast=g.std())
            xi = X[X.image == p.stem]
            r["has_xview"] = len(xi) > 0
            r["aligned"] = False
            if len(b):
                s = side(b)
                r["gt_side_med"] = np.median(s)
                r.update({f"n_{c}": int((cls == i).sum()) for i, c in enumerate(CLASSES)})
            if len(b) and len(xi):
                iou = box_iou(TB(b), TB(xi[["x1", "y1", "x2", "y2"]].values))
                best, j = iou.max(1)
                r["aligned"] = bool((best >= 0.5).float().mean() >= 0.8)
                xt = xi[xi.name.isin(CLASSES)]
                if len(xt):
                    r["scale_ratio"] = np.median(s) / np.median(side(xt[["x1", "y1", "x2", "y2"]].values))
                if r["aligned"]:
                    xname = np.where(best.numpy() >= 0.5, xi.name.values[j.numpy()], "")
                    if len(xt):
                        om = box_iou(TB(xt[["x1", "y1", "x2", "y2"]].values), TB(b)).max(1).values.numpy() < 0.3
                        r["xview_omitted"] = int(om.sum())
                        r.update({f"om_{c}": int((om & (xt.name.values == c)).sum()) for c in CLASSES})
                else:
                    xname = np.full(len(b), "")
            else:
                xname = np.full(len(b), "")
            imgs.append(r)
            for k in range(len(b)):
                boxes.append(dict(split=split, image=p.stem, cls=int(cls[k]), x1=b[k, 0], y1=b[k, 1], x2=b[k, 2], y2=b[k, 3],
                                  W=W, H=H, xview_name=xname[k], aligned=r["aligned"]))
    I = pd.DataFrame(imgs).set_index(["split", "image"])
    B = pd.DataFrame(boxes)
    B["side"] = side(B[["x1", "y1", "x2", "y2"]].values)
    B["aspect"] = np.maximum(B.x2 - B.x1, B.y2 - B.y1) / np.maximum(np.minimum(B.x2 - B.x1, B.y2 - B.y1), 1e-3)
    out = []

    # 1. image outliers
    tr = I.loc["train"]
    lo, hi = tr.longer.min(), tr.longer.max()
    I["size_out"] = (I.longer < lo) | (I.longer > hi)
    I["scale_out"] = (I.scale_ratio - 1).abs() > 0.25
    for f in ("bright", "contrast", "n_gt"):
        q1, q99 = tr[f].quantile([0.01, 0.99])
        I[f"{f}_out"] = (I[f] < q1) | (I[f] > q99)
    flags = ["size_out", "scale_out", "bright_out", "contrast_out", "n_gt_out"]
    I["n_flags"] = I[flags].sum(1)
    t1 = I[I.n_flags > 0].sort_values("n_flags", ascending=False)[["W", "H", "n_gt", "gt_side_med", "scale_ratio", "bright", "contrast", "aligned"] + flags]
    out.append(f"1. IMAGE OUTLIERS (train range of longer side {lo}-{hi} px; scale ratio = our / xView truck size)\n"
               f"train: {int(tr.n_gt.eq(0).sum())} empty images; flagged images per split: {I[I.n_flags > 0].groupby(level=0).size().to_dict()}\n"
               f"scale-ratio outliers (|ratio-1| > 0.25), train: {I.loc['train'].scale_out.sum()} of {I.loc['train'].scale_ratio.notna().sum()} measurable; "
               f"val: {I.loc['val'].scale_out.sum()} of {I.loc['val'].scale_ratio.notna().sum()}\n" + t1.round(2).to_string())

    # 2. label outliers
    Bt = B[B.split == "train"]
    stats = Bt.assign(ls=np.log(Bt.side), la=np.log(Bt.aspect)).groupby("cls")[["ls", "la"]].agg(["mean", "std"])
    B["z_side"] = [(np.log(r.side) - stats.loc[r.cls, ("ls", "mean")]) / stats.loc[r.cls, ("ls", "std")] for r in B.itertuples()]
    B["z_aspect"] = [(np.log(r.aspect) - stats.loc[r.cls, ("la", "mean")]) / stats.loc[r.cls, ("la", "std")] for r in B.itertuples()]
    B["tiny"] = B.side < 5
    B["border"] = (B.x1 <= 1) | (B.y1 <= 1) | (B.x2 >= B.W - 1) | (B.y2 >= B.H - 1)
    B["dup"] = False
    for (sp, im), f in B.groupby(["split", "image"]):
        if len(f) > 1:
            iou = box_iou(TB(f[["x1", "y1", "x2", "y2"]].values), TB(f[["x1", "y1", "x2", "y2"]].values)).numpy()
            np.fill_diagonal(iou, 0)
            B.loc[f.index, "dup"] = iou.max(1) > 0.7
    B["xview_conflict"] = B.aligned & B.xview_name.isin(CLASSES) & (B.xview_name != B.cls.map(dict(enumerate(CLASSES))))
    B["xview_generic"] = B.aligned & B.xview_name.isin(["Truck", "Trailer", "Utility Truck", "Dump Truck", "Bus", "Pickup Truck", "Cargo Car"])
    lab = B.groupby(["split", "cls"]).agg(n=("side", "size"), size_z3=("z_side", lambda z: int((z.abs() > 3).sum())),
                                           aspect_z3=("z_aspect", lambda z: int((z.abs() > 3).sum())), tiny=("tiny", "sum"),
                                           border=("border", "sum"), near_dup=("dup", "sum"))
    lab.index = lab.index.set_levels([lab.index.levels[0], [CLASSES[i] for i in lab.index.levels[1]]])
    al = B[B.aligned & (B.xview_name != "")]
    conf = pd.crosstab([al.split, al.cls.map(dict(enumerate(CLASSES)))], al.xview_name)
    keep_cols = [c for c in CLASSES + ["Truck", "Trailer", "Utility Truck", "Dump Truck", "Bus"] if c in conf.columns]
    conf = conf[keep_cols]
    agree = al.assign(ok=al.xview_name == al.cls.map(dict(enumerate(CLASSES)))).groupby(["split", "cls"]).ok.mean()
    out.append("2. LABEL OUTLIERS\nper class (|z| > 3 vs the train distribution of that class)\n" + lab.to_string()
               + "\n\nour class (rows) vs xView class of the same box (IoU >= 0.5; aligned images)\n" + conf.to_string()
               + "\n\nagreement with xView's class, per split and class\n" + agree.round(3).rename(index=dict(enumerate(CLASSES)), level=1).to_string())
    if a.dump:
        d = np.load(a.dump)
        held = set(Path(a.holdout).read_text().split())
        names = d["names"]
        img = np.array([n.split("_")[0] for n in names])[d["tile"]]
        off = np.array([[int(n.split("_")[1]), int(n.split("_")[2].split(".")[0])] for n in names], dtype=np.float32)[d["tile"]]
        bx = d["box"] + np.tile(off, 2)
        S = 1 / (1 + np.exp(-d["logit"][:, [11, 12, 15, 13, 16]].astype(np.float32)))
        rows = []
        for im, f in B[(B.split == "train") & B.image.isin(held)].groupby("image"):
            k = np.flatnonzero(img == im)
            iou = box_iou(TB(f[["x1", "y1", "x2", "y2"]].values), TB(bx[k])).numpy()
            sc = np.where(iou >= 0.5, S[k].max(1)[None, :], -1)
            j = sc.argmax(1)
            for r, jj, ok in zip(f.itertuples(), j, sc.max(1) >= 0):
                if ok:
                    s = S[k[jj]]
                    rows.append((im, CLASSES[r.cls], CLASSES[int(s.argmax())], float(s.max()), float(s[r.cls]), r.xview_name))
        D = pd.DataFrame(rows, columns=["image", "label", "detector", "det_score", "score_of_label", "xview"])
        sus = D[(D.detector != D.label) & (D.det_score >= 0.5)]
        out.append(f"   detector disagreement on the 74 checkpoint-unseen train images: {len(sus)} of {len(D)} covered GT boxes have a "
                   f"different class scored >= 0.5\n" + pd.crosstab(sus.label, sus.detector).to_string()
                   + (f"\n   of these, xView agrees with the detector: {int((sus.xview == sus.detector).sum())}, with our label: {int((sus.xview == sus.label).sum())}" if len(sus) else ""))
        sus.to_csv(R / "label_suspects_heldout.csv", index=False)

    # 3. train-val outliers
    feats = ["longer", "n_gt", "gt_side_med", "bright", "contrast", "scale_ratio"]
    va = I.loc["val"]
    z = (va[feats] - tr[feats].mean()) / tr[feats].std()
    q = tr[feats].quantile([0.01, 0.99])
    outside = ((va[feats] < q.loc[0.01]) | (va[feats] > q.loc[0.99]))
    t3 = va[feats].round(2).join(outside.sum(1).rename("n_features_outside_train_1-99pct")).join(z.abs().max(1).round(1).rename("max_|z|"))
    cm = pd.DataFrame({"train share": [B[(B.split == "train") & (B.cls == i)].shape[0] / (B.split == "train").sum() for i in range(5)],
                       "val share": [B[(B.split == "val") & (B.cls == i)].shape[0] / (B.split == "val").sum() for i in range(5)]}, index=CLASSES)
    out.append("3. TRAIN-VAL DISTRIBUTION OUTLIERS (val images vs train)\n" + t3.sort_values("max_|z|", ascending=False).to_string()
               + "\nfeature medians train / val: " + ", ".join(f"{f} {tr[f].median():.1f} / {va[f].median():.1f}" for f in feats)
               + "\nclass mix\n" + cm.round(3).to_string())

    # 4. coverage
    cov = I[I.aligned & I.xview_omitted.notna()].copy()
    cov["omitted_share"] = cov.xview_omitted / cov.n_gt
    summ = cov.groupby(level=0).agg(images=("n_gt", "size"), gt=("n_gt", "sum"), omitted=("xview_omitted", "sum"))
    summ["omitted / GT"] = (summ["omitted"] / summ["gt"]).round(3)
    per_cls = cov.groupby(level=0)[[f"om_{c}" for c in CLASSES]].sum()
    out.append("4. ANNOTATION COVERAGE (aligned images; xView boxes of our 5 classes with no GT at IoU >= 0.3)\n" + summ.to_string()
               + "\nomitted by xView class\n" + per_cls.to_string()
               + "\nworst train images by omitted share\n" + cov.loc["train"].sort_values("omitted_share", ascending=False).head(12)[["n_gt", "xview_omitted", "omitted_share"]].round(2).to_string())

    text = "\n\n".join(out)
    print(text)
    (R / "data_audit.txt").write_text(text + "\n")
    I.to_csv(R / "data_audit_images.csv")
    B.to_csv(R / "data_audit_boxes.csv", index=False)


if __name__ == "__main__":
    main()
