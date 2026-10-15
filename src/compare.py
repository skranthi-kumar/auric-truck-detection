"""Paired comparison of two evaluated runs on the same val images.

Paired bootstrap: the same resampled image set is applied to both runs, so image-level difficulty
cancels out; reports delta mAP50 with 95% CI and P(delta > 0). Plus recall (correct class, any conf)
by GT box size, within class.

Usage: python src/compare.py E0_s0 E1_up2_s0
       python src/compare.py A_orig A_aug --did B_orig B_aug   # (A_aug - A_orig) - (B_aug - B_orig), same images
Runs scored on different GT for the same images (e.g. GTO) are allowed with --same_images.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import CLASSES
from evaluate import ap101

RES = Path(__file__).resolve().parents[1] / "experiments" / "results"
NC = len(CLASSES)
BINS = [0, 16, 24, 32, 1e9]


def load(name):
    m = pd.read_csv(RES / f"{name}_matches.csv", dtype={"image": str})
    g = pd.read_csv(RES / f"{name}_gt.csv", dtype={"image": str})
    return m, g


def boot_aps(m, g, cnt, ix):
    pi, gi = m.image.map(ix).values, g.image.map(ix).values
    return np.array([ap101(m.conf.values[m.cls == c], m.tp.values[m.cls == c],
                           cnt[gi[g.cls.values == c]].sum(), cnt[pi[m.cls.values == c]].astype(float))
                     for c in range(NC)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--same_images", action="store_true", help="allow different GT on the same image set")
    ap.add_argument("--did", nargs=2, default=None, help="second pair C D: report (b-a)-(D-C)")
    a = ap.parse_args()
    (ma, ga), (mb, gb) = load(a.a), load(a.b)
    if a.same_images or a.did:
        assert set(ga.image) == set(gb.image), "runs must cover the same images"
    else:
        assert len(ga) == len(gb), "runs must share the same GT"
    if a.did:
        (mc, gc), (md, gd) = load(a.did[0]), load(a.did[1])
        assert set(gc.image) == set(ga.image) == set(gd.image), "all four runs must cover the same images"
    ims = ga.image.unique()
    ix = {im: k for k, im in enumerate(ims)}
    one = np.ones(len(ims))
    pa, pb = boot_aps(ma, ga, one, ix), boot_aps(mb, gb, one, ix)
    rng = np.random.default_rng(0)
    d, dc, dd = [], [], []
    for _ in range(a.n):
        cnt = np.bincount(rng.integers(0, len(ims), len(ims)), minlength=len(ims)).astype(float)
        x, y = boot_aps(ma, ga, cnt, ix), boot_aps(mb, gb, cnt, ix)
        d.append(np.nanmean(y) - np.nanmean(x))
        if a.did:
            dd.append(d[-1] - (np.nanmean(boot_aps(md, gd, cnt, ix)) - np.nanmean(boot_aps(mc, gc, cnt, ix))))
        dc.append(y - x)
    d, dc = np.array(d), np.array(dc)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{a.a} mAP50 {np.nanmean(pa):.4f} -> {a.b} {np.nanmean(pb):.4f}: delta {np.nanmean(pb) - np.nanmean(pa):+.4f} "
          f"(paired 95% CI {lo:+.4f} .. {hi:+.4f}), P(delta>0) = {np.mean(d > 0):.3f}")
    if a.did:
        dd = np.array(dd)
        one_c, one_d = boot_aps(mc, gc, one, ix), boot_aps(md, gd, one, ix)
        point = (np.nanmean(pb) - np.nanmean(pa)) - (np.nanmean(one_d) - np.nanmean(one_c))
        print(f"difference in deltas ({a.b}-{a.a}) - ({a.did[1]}-{a.did[0]}): {point:+.4f} "
              f"(paired 95% CI {np.percentile(dd, 2.5):+.4f} .. {np.percentile(dd, 97.5):+.4f}), P(>0) = {np.mean(dd > 0):.3f}")
    clo, chi = np.nanpercentile(dc, [2.5, 97.5], 0)
    print(pd.DataFrame({"AP_" + a.a: pa, "AP_" + a.b: pb, "delta": pb - pa, "ci_lo": clo, "ci_hi": chi,
                        "P(>0)": np.nanmean(dc > 0, 0)}, index=CLASSES).round(3))

    def rec(g):
        s = np.sqrt((g.x2 - g.x1) * (g.y2 - g.y1))
        return g.assign(size=pd.cut(s, BINS, labels=["<16", "16-24", "24-32", ">=32"]))
    ra, rb = rec(ga), rec(gb)
    t = pd.DataFrame({a.a: ra.groupby(["cls", "size"], observed=True).matched.mean(),
                      a.b: rb.groupby(["cls", "size"], observed=True).matched.mean(),
                      "n": ra.groupby(["cls", "size"], observed=True).size()})
    t.index = t.index.set_levels([[CLASSES[i] for i in t.index.levels[0]], t.index.levels[1]])
    print("\nrecall (correct class, any conf) by GT size in px:\n", t.round(3))


if __name__ == "__main__":
    main()
