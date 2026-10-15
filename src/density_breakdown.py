"""Val AP50 and recall by scene density (GT boxes per image), from existing evaluated runs; no inference.

Buckets: val images split into density tertiles by GT count. AP per bucket = evaluator AP restricted to the bucket's
images (compare.boot_aps with 0/1 image weights); CI = bootstrap over the bucket's IMAGES (boxes within a scene are
correlated). Recall = share of GT boxes matched with the correct class at any confidence (evaluator's 'matched').

Usage: python src/density_breakdown.py E2_last E2_last_agn ENS_E2 ENS_E2_ctx
"""
import argparse

import numpy as np
import pandas as pd

from compare import boot_aps, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    g0 = load(a.runs[0])[1]
    n_img = g0.groupby("image").size().sort_values()
    ims = list(n_img.index)
    ix = {im: k for k, im in enumerate(ims)}
    edges = np.quantile(n_img.values, [1 / 3, 2 / 3])
    bucket = pd.Series(np.digitize(n_img.values, edges, right=True), index=n_img.index)  # 0 sparse .. 2 dense
    names = ["sparse", "medium", "dense"]
    print("val density tertiles (GT boxes/image):")
    for b in range(3):
        s = n_img[bucket == b]
        print(f"  {names[b]:6s} {len(s)} images, {s.min()}-{s.max()} boxes/image, {s.sum()} boxes")
    rng = np.random.default_rng(0)
    rows = []
    for r in a.runs:
        m, g = load(r)
        for b in range(3):
            sel = np.array([bucket[im] == b for im in ims], float)
            apb = np.nanmean(boot_aps(m, g, sel, ix))
            idx = np.flatnonzero(sel)
            bs = []
            for _ in range(a.n):
                cnt = np.zeros(len(ims))
                np.add.at(cnt, rng.choice(idx, len(idx)), 1)
                bs.append(np.nanmean(boot_aps(m, g, cnt, ix)))
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            gb = g[g.image.map(bucket) == b]
            rows.append((r, names[b], round(apb, 3), f"{lo:.3f}-{hi:.3f}", round(gb.matched.mean(), 3), len(gb)))
    print(pd.DataFrame(rows, columns=["run", "density", "mAP50", "95% CI (images)", "recall_any_conf", "gt"])
          .to_string(index=False))


if __name__ == "__main__":
    main()
