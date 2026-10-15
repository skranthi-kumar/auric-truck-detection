"""5.3 uncertainty: intervals for the gain from more labels, from the random-subset learning curve.

Two models, both fitted to all random runs (n = train instances):
  log-linear  AP = a + b * log2(n)          (2 params; no saturation, so long extrapolations are optimistic)
  power law   AP = A - B * (n/1000)^(-c)    (3 params, asymptote A; poorly identified from 4 sizes)
Hierarchical bootstrap (2000): resample val images (same resample for every run, as in compare.py) AND, at fractions
with two seeds, resample the seeds with replacement, so seed-to-seed noise enters the interval, not only scatter.
Reports +500 labels and x10 labels (x10 is far outside the data: a bound under each model, not a prediction),
and per-class log-linear slopes for the "who benefits most" ranking.

Usage: python src/lc_interval.py
"""
import warnings

import numpy as np
from scipy.optimize import curve_fit

from compare import boot_aps, load
from data import CLASSES

RUNS = {"LC_rand_0125_s0": 965, "LC_rand_0125_s1": 860, "LC_rand_0250_s0": 1541, "LC_rand_0250_s1": 1637,
        "LC_rand_0500_s0": 3682, "LC_full_1000_s0": 7618}  # train instances, from learning_curves.md
GROUPS = [["LC_rand_0125_s0", "LC_rand_0125_s1"], ["LC_rand_0250_s0", "LC_rand_0250_s1"],
          ["LC_rand_0500_s0"], ["LC_full_1000_s0"]]  # seeds of the same fraction
N, ADD, B = 7618, 500, 2000


def power(n, A, Bc, c):
    return A - Bc * np.power(n / 1000, -c)


def fit_power(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            p, _ = curve_fit(power, x, y, p0=[y.max() + 0.05, 0.05, 0.5], bounds=([0, 0, 0.01], [1, 1, 3]), maxfev=20000)
            return p
        except RuntimeError:
            return np.full(3, np.nan)


def main():
    data = {r: load(r) for r in RUNS}
    ims = data["LC_full_1000_s0"][1].image.unique()
    ix = {im: k for k, im in enumerate(ims)}
    gain500, gain10 = np.log2((N + ADD) / N), np.log2(10)

    def stats(cnt, pick):
        names = [g[k] for g, k in zip(GROUPS, pick)]
        ap = np.array([boot_aps(*data[r], cnt, ix) for r in names])  # runs x classes
        x = np.array([RUNS[r] for r in names], float)
        y = np.nanmean(ap, 1)
        b = np.polyfit(np.log2(x), y, 1)[0]
        bc = np.array([np.polyfit(np.log2(x), ap[:, c], 1)[0] for c in range(len(CLASSES))])
        p = fit_power(x, y)
        return b, bc, p, power(N + ADD, *p) - power(N, *p), power(10 * N, *p) - power(N, *p)

    # point estimate: all 6 runs (duplicated groups expanded), unit image weights
    one = np.ones(len(ims))
    names = list(RUNS)
    ap0 = np.array([boot_aps(*data[r], one, ix) for r in names])
    x0, y0 = np.array(list(RUNS.values()), float), np.nanmean(ap0, 1)
    b0 = np.polyfit(np.log2(x0), y0, 1)[0]
    bc0 = np.array([np.polyfit(np.log2(x0), ap0[:, c], 1)[0] for c in range(len(CLASSES))])
    p0 = fit_power(x0, y0)

    rng = np.random.default_rng(0)
    res = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, len(ims), len(ims)), minlength=len(ims)).astype(float)
        pick = [rng.integers(0, len(g)) for g in GROUPS]
        # two draws per 2-seed group would need 2 points; one draw per group keeps 4 points = seed noise in the fit
        res.append(stats(cnt, pick))
    bs = np.array([r[0] for r in res])
    bcs = np.array([r[1] for r in res])
    p5 = np.array([r[3] for r in res])
    p10 = np.array([r[4] for r in res])
    pA = np.array([r[2][0] for r in res])
    ci = lambda v: np.nanpercentile(v, [2.5, 97.5])

    print("runs mAP50:", {r: round(float(v), 4) for r, v in zip(names, y0)})
    print(f"seed spread: 12.5% {abs(y0[0] - y0[1]):.3f}, 25% {abs(y0[2] - y0[3]):.3f}")
    lo, hi = ci(bs)
    print(f"\nlog-linear slope per doubling: {b0:+.4f} (hierarchical 95% CI {lo:+.4f} .. {hi:+.4f}; "
          f"P(slope>0) {np.mean(bs > 0):.3f})")
    print(f"  +{ADD} labels (+{ADD / N:.1%}): {b0 * gain500:+.4f} (95% CI {lo * gain500:+.4f} .. {hi * gain500:+.4f})")
    print(f"  x10 labels (bound, log-linear): {b0 * gain10:+.3f} (95% CI {lo * gain10:+.3f} .. {hi * gain10:+.3f})")
    l5, h5 = ci(p5)
    l10, h10 = ci(p10)
    la, ha = ci(pA)
    print(f"\npower law A - B(n/1000)^-c: A={p0[0]:.3f} B={p0[1]:.3f} c={p0[2]:.2f}; "
          f"asymptote A 95% CI {la:.3f} .. {ha:.3f} (fits failed {np.isnan(p5).mean():.1%})")
    print(f"  +{ADD} labels: {power(N + ADD, *p0) - power(N, *p0):+.4f} (95% CI {l5:+.4f} .. {h5:+.4f})")
    print(f"  x10 labels: {power(10 * N, *p0) - power(N, *p0):+.3f} (95% CI {l10:+.3f} .. {h10:+.3f})")

    print("\nper-class log-linear slope per doubling (rank = who benefits most):")
    clo, chi = np.nanpercentile(bcs, [2.5, 97.5], 0)
    for c in np.argsort(-bc0):
        print(f"  {CLASSES[c]:16s} {bc0[c]:+.4f}  95% CI {clo[c]:+.4f} .. {chi[c]:+.4f}  P(>0) {np.mean(bcs[:, c] > 0):.2f}"
              f"  AP(full) {ap0[-1, c]:.3f}")


if __name__ == "__main__":
    main()
