"""5.3 / 5.4 analysis from the LC_* runs (E1 recipe, 30 ep; splits frozen in experiments/splits).

5.3: per-class AP50 and mAP50 vs labelled instances (random nested subsets 12.5/25/50/100%); power-law fit
     AP(n) = a - b * n^(-c) on seed means; predicted gain for +500 instances at the current class mix.
5.4: selected (rule in experiments/subset_rule.md) vs random at the same instance budget; 90%-of-full-data
     threshold; characterisation of the selected images.

Usage: python src/learning_curves.py            (expects experiments/results/LC_*_metrics.json)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from data import CLASSES

ROOT = Path(__file__).resolve().parents[1]
RES, SPL = ROOT / "experiments" / "results", ROOT / "experiments" / "splits"
SHORT = ["Cargo", "Box", "Flatbed", "Tractor", "Liquid"]


def subset_counts(name):
    """Images and per-class instance counts of the subset a run trained on."""
    e = pd.read_csv(RES / "eda_instances.csv", dtype={"image": str})
    e = e[e.split == "train"]
    m = re.match(r"LC_(rand|sel|full)_(\d{4})_s(\d)", name)
    kind, frac, seed = m.groups()
    if kind == "full":
        ims = set(e.image)
    else:
        f = SPL / (f"random_s{seed}_{frac}.txt" if kind == "rand" else f"selected_{frac}.txt")
        ims = set(f.read_text().split())
    sub = e[e.image.isin(ims)]
    return len(ims), np.bincount(sub.cls, minlength=5), kind, int(frac) / 1000


def power(n, a, b, c):
    return a - b * np.power(n, -c)


def main():
    rows = []
    for f in sorted(RES.glob("LC_*_metrics.json")):
        m = json.load(open(f))
        n_img, n_cls, kind, frac = subset_counts(m["name"])
        rows.append(dict(run=m["name"], kind=kind, frac=frac, images=n_img, instances=int(n_cls.sum()),
                         mAP50=m["mAP50"], **{SHORT[i]: m["AP50"][CLASSES[i]] for i in range(5)},
                         **{f"n_{SHORT[i]}": int(n_cls[i]) for i in range(5)}))
    d = pd.DataFrame(rows)
    print("## Runs\n", d.drop(columns=[c for c in d.columns if c.startswith("n_")]).round(3).to_string(index=False))
    full = d[d.kind == "full"].mAP50.mean()

    print("\n## 5.3 learning curves (random subsets; seed means)")
    r = d[d.kind.isin(["rand", "full"])].groupby("frac").mean(numeric_only=True)
    print(r[["instances", "mAP50"] + SHORT].round(3).to_string())
    spread = d[d.kind == "rand"].groupby("frac").mAP50.agg(lambda x: x.max() - x.min())
    print("seed spread of mAP50 at each fraction:", spread.round(3).to_dict())
    n_full = r.instances.max()
    share = d[d.kind == "full"][[f"n_{s}" for s in SHORT]].iloc[0].values / n_full
    print(f"\n+500 instances = +{500 / n_full:.1%} of {int(n_full)}; per-class share of the +500 (current mix): "
          + ", ".join(f"{s} {int(500 * sh)}" for s, sh in zip(SHORT, share)))
    for col in ["mAP50"] + SHORT:
        x, y = r.instances.values.astype(float), r[col].values
        try:
            p, _ = curve_fit(power, x, y, p0=[y.max() + 0.05, 1.0, 0.5], bounds=([0, 0, 0.01], [1, 50, 3]), maxfev=20000)
            gain = power(n_full + 500, *p) - power(n_full, *p)
            resid = np.abs(power(x, *p) - y).max()
            print(f"  {col:8s} fit a={p[0]:.3f} b={p[1]:.2f} c={p[2]:.2f}  AP(full)={power(n_full, *p):.3f} -> +500: {gain:+.4f}  (max |resid| {resid:.3f})")
        except Exception as ex:
            print(f"  {col:8s} fit failed: {ex}")
    print("  Caveats: 4 points, 1-2 seeds; extrapolation is a few % beyond the largest point; a flat low curve"
          " points to non-sample limits (see 5.1 ceiling, 5.2 hard-confused).")

    print("\n## 5.4 selected vs random (same instance budget)")
    for frac, g in d[d.kind.isin(["rand", "sel"])].groupby("frac"):
        rr, ss = g[g.kind == "rand"], g[g.kind == "sel"]
        if len(ss) == 0:
            continue
        print(f"  {frac:.3f}: random {rr.mAP50.mean():.3f} (range {rr.mAP50.min():.3f}-{rr.mAP50.max():.3f}, {int(rr.instances.mean())} inst, {int(rr.images.mean())} img)"
              f" | selected {ss.mAP50.mean():.3f} (range {ss.mAP50.min():.3f}-{ss.mAP50.max():.3f}, {int(ss.instances.mean())} inst, {int(ss.images.mean())} img)"
              f" | gap {ss.mAP50.mean() - rr.mAP50.mean():+.3f}; exceeds both seed spreads: {abs(ss.mAP50.mean() - rr.mAP50.mean()) > max(rr.mAP50.max() - rr.mAP50.min(), ss.mAP50.max() - ss.mAP50.min())}"
              f" | >= 90% of full ({0.9 * full:.3f}): random {rr.mAP50.mean() >= 0.9 * full}, selected {ss.mAP50.mean() >= 0.9 * full}")
    passing = d[(d.kind.isin(["rand", "sel"])) & (d.mAP50 >= 0.9 * full)]
    print("  smallest tested subset reaching 90% of full-data mAP50:",
          passing.sort_values("instances").iloc[0][["run", "images", "instances", "mAP50"]].to_dict() if len(passing) else "none of the tested sizes")

    print("\n## 5.4 what survives selection (12.5%)")
    pop = pd.read_csv(RES / "cart_E0_s0_populations.csv", dtype={"image": str})
    e = pd.read_csv(RES / "eda_images.csv", dtype={"image": str}).query("split == 'train'").set_index("image")
    for lab, f in (("selected", "selected_0125.txt"), ("random s0", "random_s0_0125.txt"), ("random s1", "random_s1_0125.txt")):
        ims = set((SPL / f).read_text().split())
        pp = pop[pop.image.isin(ims)]
        print(f"  {lab:10s}: {len(ims)} images, boxes/image {e.loc[list(ims), 'n'].mean():.1f}, "
              f"region shares easy {np.mean(pp.region == 'easy'):.2f} ambiguous {np.mean(pp.region == 'ambiguous'):.2f} "
              f"hard-confused {np.mean(pp.region == 'hard-confused'):.2f} hard-missed {np.mean(pp.region == 'hard-missed'):.2f}, "
              f"median side {np.sqrt(pp.w * pp.h).median():.1f} px")
    d.to_csv(RES / "learning_curves_summary.csv", index=False)


if __name__ == "__main__":
    main()
