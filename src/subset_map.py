"""Secondary analysis: mAP50 on a declared image subset, from saved evaluate.py outputs (no re-matching needed,
matching is per image). Bootstrap CI over the subset's images (2000 resamples, seed 0), as in evaluate.py.

Default subset = "native-scale" val images: images whose longer side lies inside the TRAIN range of longer sides
(2,576-5,119 px). The rule uses image size only (no labels); on val it excludes 2308, 2391 (~7,100 px, upsampled
~2x) and 2384, 2460 (~1,500 px, downsampled ~2x). This number does NOT replace the official 22-image score.

Usage: python src/subset_map.py --runs FINAL2 ENS_E2_ctx
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare import boot_aps, load
from data import CLASSES, image_paths, image_size

R = Path(__file__).resolve().parents[1] / "experiments" / "results"
TRAIN_RANGE = (2576, 5119)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["FINAL2"])
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    keep = [p.stem for p in image_paths("val") if TRAIN_RANGE[0] <= max(image_size(p)) <= TRAIN_RANGE[1]]
    drop = sorted(p.stem for p in image_paths("val") if p.stem not in keep)
    print(f"subset: {len(keep)} native-scale images; excluded {drop}")
    rows = []
    for run in a.runs:
        m, g = load(run)
        for name, ims in (("all 22 (official)", sorted(g.image.unique())), (f"native-scale {len(keep)}", keep)):
            ms, gs = m[m.image.isin(ims)], g[g.image.isin(ims)]
            ix = {im: k for k, im in enumerate(ims)}
            aps = boot_aps(ms, gs, np.ones(len(ims)), ix)
            rng = np.random.default_rng(0)
            bs = [np.nanmean(boot_aps(ms, gs, np.bincount(rng.integers(0, len(ims), len(ims)), minlength=len(ims)).astype(float), ix))
                  for _ in range(a.n)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(run=run, images=name, n_gt=len(gs), mAP50=round(np.nanmean(aps), 4), ci=f"{lo:.3f}-{hi:.3f}",
                             **{c: round(v, 3) for c, v in zip(CLASSES, aps)}))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv(R / "subset_native_scale.csv", index=False)


if __name__ == "__main__":
    main()
