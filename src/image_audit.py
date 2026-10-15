"""Image-by-image error attribution for a scored run (diagnostic only; no training, no new inference).

Inputs: <run>_matches.csv / <run>_gt.csv (evaluate.py), val images, xView val labels (scale check), and
RFXDUMP_val.npz (RFX0 per-query sigmoid over all 63 xView outputs) for "what the checkpoint thinks a box is".

1. Leaderboard per image: AP attributed to its FALSE POSITIVES (delete only that image's FP rows, recompute
   per-class AP over all images) and to its UNRECOVERED GT (GT with no TP at any confidence: remove them from
   the per-class GT count). This attributes error populations, not image influence (image size / object count).
2. Scale: image size vs the train range, GT size distribution, our-GT / xView truck-size ratio.
3. Mechanism: each FINAL2 row with conf >= 0.05 is matched (IoU >= 0.5, same image) to an RFX0 query; its
   top xView class names what the checkpoint believes the object is. Cross-tab of false positives by class,
   confidence and scene (scale-anomalous image or not; density tertile).
5. Diagnostic interventions (each a recomputation on the frozen outputs; estimates, not adoptable rules):
   remove FINAL2 rows whose matched RFX0 top class is a car / a non-target truck-like vehicle (TPs removed
   too), or all FPs in scale-anomalous images (oracle upper bound for SC1).

Usage: python src/image_audit.py --run FINAL2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

from data import CLASSES, image_paths, image_size
from evaluate import ap101

REPO = Path(__file__).resolve().parents[1]
R = REPO / "experiments" / "results"
OURS = {11: 0, 12: 1, 15: 2, 13: 3, 16: 4}  # xView id -> our class
CAR = {5, 6}
OTHER_VEH = {7, 8, 9, 10, 14, 17, 19, 20, 21, 23, 34, 40, 41, 43, 44, 45, 46}
TRAIN_RANGE = (2576, 5119)  # longer side of train images (min, max)


def side(d):
    return np.sqrt((d.x2 - d.x1) * (d.y2 - d.y1))


def mean_ap(m, g, drop_rows=None, gt_less=None):
    aps = []
    for c in range(len(CLASSES)):
        mc = m[m.cls == c] if drop_rows is None else m[(m.cls == c) & ~drop_rows]
        n = (g.cls == c).sum() - (0 if gt_less is None else gt_less.get(c, 0))
        aps.append(ap101(mc.conf.values, mc.tp.values, n))
    return np.nanmean(aps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="FINAL2")
    a = ap.parse_args()
    m = pd.read_csv(R / f"{a.run}_matches.csv", dtype={"image": str},
                    usecols=["image", "cls", "conf", "x1", "y1", "x2", "y2", "tp", "gt_id", "err"])
    g = pd.read_csv(R / f"{a.run}_gt.csv", dtype={"image": str})
    xv = pd.read_csv(R / "xview_val_labels.csv", dtype={"image": str})
    base = mean_ap(m, g)
    best_tp = m[m.tp].groupby("gt_id").conf.max()
    g["best_tp_conf"] = g.gt_id.map(best_tp).fillna(0)
    fp = ~m.tp

    # 1 + 2: per-image table
    rows = []
    for p in image_paths("val"):
        im = p.stem
        W, H = image_size(p)
        gi, mi = g[g.image == im], m[m.image == im]
        xt = xv[(xv.image == im) & xv.name.isin(CLASSES)]
        unrec = gi[gi.best_tp_conf == 0]
        d_fp = mean_ap(m, g, drop_rows=(m.image == im) & fp) - base
        d_fn = mean_ap(m, g, gt_less=unrec.cls.value_counts().to_dict()) - base
        s = side(gi)
        hi = mi[(mi.conf >= 0.25) & ~mi.tp]
        rows.append(dict(image=im, W=W, H=H, in_train_range=TRAIN_RANGE[0] <= max(W, H) <= TRAIN_RANGE[1],
                         scale_ratio=round(s.median() / side(xt).median(), 2) if len(xt) else np.nan,
                         n_gt=len(gi), gt_side_p10=round(s.quantile(0.1), 1), gt_side_med=round(s.median(), 1),
                         gt_lt16px=int((s < 16).sum()), unrecovered_gt=len(unrec), low_rank_gt=int(((gi.best_tp_conf > 0) & (gi.best_tp_conf < 0.1)).sum()),
                         hiconf_fp=len(hi), hiconf_tp=int(((mi.conf >= 0.25) & mi.tp).sum()),
                         dAP_if_FPs_removed=round(d_fp, 4), dAP_if_unrecovered_GT_found=round(d_fn, 4)))
    img = pd.DataFrame(rows).set_index("image")
    img["scale_anomaly"] = (~img.in_train_range) | ((img.scale_ratio - 1).abs() > 0.25)
    img["density"] = pd.qcut(img.n_gt, 3, labels=["sparse", "medium", "dense"])
    img = img.sort_values("dAP_if_FPs_removed", ascending=False)

    # 3: checkpoint belief for rows with conf >= 0.05
    d = np.load(R / "RFXDUMP_val.npz")
    dimg = d["ids"][d["img"]].astype(str)
    top = m[m.conf >= 0.05].copy()
    top["belief"] = "no RFX0 match"
    for im, f in top.groupby("image"):
        k = np.flatnonzero(dimg == im)
        if not len(k):
            continue
        iou = box_iou(torch.tensor(f[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32),
                      torch.tensor(d["box"][k], dtype=torch.float32))
        best, j = iou.max(1)
        prob = d["prob"][k][j.numpy()].astype(np.float32)
        cid = prob[:, 1:].argmax(1) + 1  # ignore the background/no-object column 0
        lab = np.where(np.isin(cid, list(OURS)), "one of our 5", np.where(np.isin(cid, list(CAR)), "car",
                       np.where(np.isin(cid, list(OTHER_VEH)), "other vehicle", "other object")))
        same = np.array([OURS.get(int(x), -1) for x in cid]) == f.cls.values
        lab = np.where(same, "same class as predicted", lab)
        top.loc[f.index, "belief"] = np.where(best.numpy() >= 0.5, lab, "no RFX0 match")
    top = top.join(img[["scale_anomaly", "density"]], on="image")
    top["conf_bin"] = pd.cut(top.conf, [0.05, 0.1, 0.25, 1.01], labels=["0.05-0.1", "0.1-0.25", ">=0.25"], right=False)
    fpt = top[~top.tp]
    mech_cls = pd.crosstab([fpt.cls.map(dict(enumerate(CLASSES))), fpt.conf_bin], fpt.belief)
    mech_scene = pd.crosstab([fpt.scale_anomaly.map({True: "scale-anomalous", False: "native scale"}), fpt.density], fpt.belief)

    # 5: diagnostic interventions on the frozen outputs
    row_belief = pd.Series("unmatched", index=m.index)
    row_belief.loc[top.index] = top.belief
    anomalous = m.image.isin(img.index[img.scale_anomaly])
    inter = {
        "remove rows the checkpoint calls a car": row_belief.eq("car"),
        "remove rows the checkpoint calls another vehicle": row_belief.eq("other vehicle"),
        "remove both": row_belief.isin(["car", "other vehicle"]),
        "remove all FPs in scale-anomalous images (oracle)": anomalous & fp,
        "remove all FPs the checkpoint gives another of our 5 (oracle)": row_belief.eq("one of our 5") & fp,
    }
    iv = pd.DataFrame([(k, int(v.sum()), int((v & m.tp).sum()), round(mean_ap(m, g, drop_rows=v) - base, 4))
                       for k, v in inter.items()], columns=["intervention", "rows removed", "TPs removed", "dAP"])

    text = "\n\n".join([
        f"{a.run} mAP50 {base:.4f}",
        "1-2. image leaderboard (sorted by AP attributed to the image's false positives)\n" + img.to_string(),
        "3a. false positives (conf >= 0.05) by class x confidence x what the checkpoint believes\n" + mech_cls.to_string(),
        "3b. same by scene\n" + mech_scene.to_string(),
        "5. diagnostic interventions (recomputed AP on frozen outputs; estimates only)\n" + iv.to_string(index=False)])
    print(text)
    (R / f"{a.run}_image_audit.txt").write_text(text + "\n")
    img.to_csv(R / f"{a.run}_image_audit.csv")


if __name__ == "__main__":
    main()
