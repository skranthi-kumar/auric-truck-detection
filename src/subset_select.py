"""Image-level splits for 5.2-5.4 (frozen before any subset training).

Stratum of an image = rarest class present in it (global train frequency); unlabeled images form
their own stratum. Random subsets shuffle each stratum with a seed and take the first ceil(f * n)
images, so subsets are nested (12.5% is inside 25% is inside 50%) and class-stratified by construction.
The 5.4 selected subsets reuse the same strata and per-stratum counts; only the ranking inside a
stratum differs (rule in experiments/subset_rule.md).

Usage: python src/subset_select.py random
       python src/subset_select.py selected   (rule: experiments/subset_rule.md)
"""
import argparse
import math
import random
from pathlib import Path

import pandas as pd

from data import image_paths, read_labels, image_size

SPLITS = Path(__file__).resolve().parents[1] / "experiments" / "splits"
FRACS = (0.125, 0.25, 0.5)


def strata(root=None):
    rows = []
    for p in image_paths("train", root):
        cls, _ = read_labels(p, image_size(p))
        rows.append((p.stem, set(cls.tolist())))
    freq = pd.Series([c for _, s in rows for c in s]).value_counts()  # images containing each class
    rare = lambda s: min(s, key=lambda c: freq[c]) if s else -1
    return pd.DataFrame({"image": [r[0] for r in rows], "stratum": [rare(r[1]) for r in rows]})


def take(df, frac, order):
    """First ceil(frac * n) images of each stratum, following `order` (dict stratum -> ranked list)."""
    return [im for s, g in df.groupby("stratum") for im in order[s][:math.ceil(frac * len(g))]]


def random_order(df, seed):
    rng = random.Random(seed)
    order = {}
    for s, g in df.groupby("stratum"):
        ims = sorted(g.image, key=int)
        rng.shuffle(ims)
        order[s] = ims
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["random", "selected"])
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    SPLITS.mkdir(parents=True, exist_ok=True)
    df = strata(a.root)
    if a.mode == "selected":  # rule: experiments/subset_rule.md (rev 2, instance-matched)
        pop = pd.read_csv(SPLITS.parent / "results" / "cart_E0_s0_populations.csv", dtype={"image": str})
        tot = pop.groupby("image").size()
        good = pop[pop.region.isin(["easy", "medium", "ambiguous"])].groupby("image").size()
        df["n"] = df.image.map(tot).fillna(0).astype(int)
        df["share"] = (df.image.map(good).fillna(0) / df.n.clip(lower=1))
        for f in FRACS:
            rnd = [set((SPLITS / f"random_s{s}_{int(f * 1000):04d}.txt").read_text().split()) for s in (0, 1)]
            ims = []
            for s, g in df.groupby("stratum"):
                g = g.sort_values(["share", "image"], ascending=[False, True], key=lambda c: c.astype(int) if c.name == "image" else c)
                if s == -1:  # unlabeled stratum: match the random image count
                    ims += g.image[:math.ceil(f * len(g))].tolist()
                    continue
                budget = sum(g[g.image.isin(r)].n.sum() for r in rnd) / 2
                cum = g.n.cumsum().values
                k = int((cum < budget).sum())  # images fully below budget; add the crossing one only if closer
                if k < len(cum) and abs(cum[k] - budget) < abs((cum[k - 1] if k else 0) - budget):
                    k += 1
                ims += g.image[:k].tolist()
            (SPLITS / f"selected_{int(f * 1000):04d}.txt").write_text("\n".join(ims) + "\n")
            print(f"selected frac {f}: {len(ims)} images, {int(df.set_index('image').loc[ims, 'n'].sum())} instances")
        return
    for seed in (0, 1):
        order = random_order(df, seed)
        for f in FRACS:
            ims = take(df, f, order)
            (SPLITS / f"random_s{seed}_{int(f * 1000):04d}.txt").write_text("\n".join(ims) + "\n")
            print(f"seed {seed} frac {f}: {len(ims)} images")
    df.to_csv(SPLITS / "strata.csv", index=False)
    # 5.2 holdout: 20% of images per stratum never inspected (seed 42)
    held = take(df, 0.2, random_order(df, 42))
    (SPLITS / "cart_holdout.txt").write_text("\n".join(sorted(held, key=int)) + "\n")
    print("strata:", df.stratum.value_counts().sort_index().to_dict(), "| 5.2 holdout:", len(held))


if __name__ == "__main__":
    main()
