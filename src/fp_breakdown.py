"""Where FINAL2's false positives sit: by class, confidence, error type and image; and how they outrank TPs.

Reads the saved evaluate.py outputs (<run>_matches.csv, <run>_gt.csv); no new scoring, no tuning.
Error types are evaluate.py's: tp, cls (right place, wrong subtype), loc (right class, IoU < 0.5), both,
dup (second hit on a matched GT), bg (no GT nearby).

Sections:
  A. per class x confidence bin: counts by error type
  B. per class: AP gained by deleting one error type at a time (rank-aware cost), all types together
  C. outranking: for each class, FPs ranked above the median / 75th-percentile TP, and precision at
     recall 0.25 / 0.50 / 0.75
  D. images: concentration of high-confidence (>= 0.25) FPs, and per-image FP:GT ratio

Usage: python src/fp_breakdown.py --run FINAL2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import CLASSES
from evaluate import ap101

R = Path(__file__).resolve().parents[1] / "experiments" / "results"
TYPES = ["cls", "loc", "both", "dup", "bg"]
BINS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.01]
BLAB = ["<0.05", "0.05-0.1", "0.1-0.25", "0.25-0.5", ">=0.5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="FINAL2")
    a = ap.parse_args()
    m = pd.read_csv(R / f"{a.run}_matches.csv", dtype={"image": str}, usecols=["image", "cls", "conf", "tp", "err"])
    g = pd.read_csv(R / f"{a.run}_gt.csv", dtype={"image": str})
    out = []

    # A
    m["bin"] = pd.cut(m.conf, BINS, labels=BLAB, right=False)
    A = m.pivot_table(index=["cls", "bin"], columns="err", values="conf", aggfunc="size", fill_value=0, observed=False)
    A = A.reindex(columns=["tp"] + TYPES, fill_value=0)
    A.index = A.index.set_levels([[CLASSES[i] for i in A.index.levels[0]], A.index.levels[1]])
    out.append("A. predictions by class x confidence x error type\n" + A.to_string())

    # B
    rows = []
    for c, name in enumerate(CLASSES):
        mc = m[m.cls == c]
        n = int((g.cls == c).sum())
        base = ap101(mc.conf.values, mc.tp.values, n)
        r = {"class": name, "AP": base}
        for t in TYPES + ["ALL_FP"]:
            keep = mc[mc.err == "tp"] if t == "ALL_FP" else mc[mc.err != t]
            r["+" + t] = ap101(keep.conf.values, keep.tp.values, n) - base
        rows.append(r)
    B = pd.DataFrame(rows).set_index("class")
    B.loc["mean"] = B.mean()
    out.append("B. AP gained by deleting one false-positive type (rank-aware cost)\n" + B.round(3).to_string())

    # C
    rows = []
    for c, name in enumerate(CLASSES):
        mc = m[m.cls == c].sort_values("conf", ascending=False, kind="stable")
        n = int((g.cls == c).sum())
        tp = mc.tp.values.astype(bool)
        ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
        tpc = mc.conf.values[tp]
        r = {"class": name, "n_gt": n, "TP_found": int(tp.sum())}
        for q in (50, 75):
            k = int(np.ceil(len(tpc) * q / 100)) - 1  # index of the q-th percentile TP in rank order
            pos = np.flatnonzero(tp)[k]
            r[f"FP above p{q} TP"] = int(cfp[pos])
            r[f"conf of p{q} TP"] = round(float(tpc[k]), 3)
        for rec in (0.25, 0.5, 0.75):
            i = np.searchsorted(ctp / n, rec)
            r[f"prec@R{rec}"] = round(float(ctp[i] / (i + 1)), 3) if i < len(ctp) else np.nan
        # what the FPs above the median TP are
        pos = np.flatnonzero(tp)[int(np.ceil(len(tpc) * 0.5)) - 1]
        top = mc.iloc[:pos + 1]
        top = top[~top.tp]
        for t in TYPES:
            r[f"{t} share above p50"] = round(float((top.err == t).mean()), 3) if len(top) else np.nan
        rows.append(r)
    C = pd.DataFrame(rows).set_index("class")
    out.append("C. outranking: false positives ranked above the median / 75th-percentile true positive\n" + C.T.to_string())

    # D
    hi = m[(m.conf >= 0.25) & ~m.tp]
    per = pd.DataFrame({"hiconf_FP": hi.groupby("image").size(), "hiconf_TP": m[(m.conf >= 0.25) & m.tp].groupby("image").size(),
                        "n_gt": g.groupby("image").size()}).fillna(0).astype(int)
    per["FP_per_GT"] = (per.hiconf_FP / per.n_gt.replace(0, np.nan)).round(2)
    per = per.sort_values("hiconf_FP", ascending=False)
    share = per.hiconf_FP.cumsum() / per.hiconf_FP.sum()
    gtshare = per.n_gt.cumsum() / per.n_gt.sum()
    per["cum_FP_share"], per["cum_GT_share"] = share.round(3), gtshare.round(3)
    by_type = hi.pivot_table(index="image", columns="err", values="conf", aggfunc="size", fill_value=0)
    per = per.join(by_type.reindex(columns=TYPES, fill_value=0))
    out.append("D. images: high-confidence (>= 0.25) false positives, sorted\n" + per.to_string())

    text = "\n\n".join(out)
    print(text)
    (R / f"{a.run}_fp_breakdown.txt").write_text(text + "\n")
    B.to_csv(R / f"{a.run}_fp_cost.csv")
    per.to_csv(R / f"{a.run}_fp_images.csv")


if __name__ == "__main__":
    main()
