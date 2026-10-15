"""SC1 scale normalisation (declared in LOG.md, 15 Oct): rerun the FINAL2 members on images whose size is outside
the train range, at a power-of-2 scale towards the train median, and splice them into the FINAL2 predictions.

Rule (train image sizes only): longer side outside [2576, 5119] px -> s = 2^round(log2(3228 / longer side)).
RFX0D = 4 flips x 1008 tiles at overlap 504, WBF; RFX3_ep5 at scale 2s (its training scale); WBF of the two.
Usage: python src/sc1.py --root CV_Assignment_AAL --final2 FINAL2_val_preds.csv --work tmp/sc1 --out SC1_val_preds.csv
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import image_paths, image_size

TRAIN_RANGE, TRAIN_MEDIAN = (2576, 5119), 3228
SRC = Path(__file__).resolve().parent


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--final2", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rfx3", default=str(SRC.parent / "weights" / "rfx3_ep5.pt"))
    a = ap.parse_args()
    groups = {}
    for p in image_paths("val", a.root):
        L = max(image_size(p))
        if not TRAIN_RANGE[0] <= L <= TRAIN_RANGE[1]:
            groups.setdefault(float(2.0 ** np.round(np.log2(TRAIN_MEDIAN / L))), []).append(p)
    print({s: [p.stem for p in ps] for s, ps in groups.items()})
    parts = []
    for s, ps in groups.items():
        w = Path(a.work) / f"s{s}"
        (w / "val" / "images").mkdir(parents=True, exist_ok=True)
        for p in ps:
            link = w / "val" / "images" / p.name
            if not link.exists():
                link.symlink_to(p.resolve())
        flips = []
        for f in ("none", "h", "v", "hv"):
            out = w / f"D_{f}.csv"
            run(SRC / "rfx_predict.py", "--split", "val", "--root", w, "--scale", s, "--overlap", 504, "--flip", f, "--out", out)
            flips.append(out)
        run(SRC / "ensemble.py", "--preds", *flips, "--out", w / "RFX0D.csv")
        run(SRC / "rfx_predict.py", "--split", "val", "--root", w, "--weights", a.rfx3, "--scale", 2 * s, "--out", w / "R3.csv")
        run(SRC / "ensemble.py", "--preds", w / "RFX0D.csv", w / "R3.csv", "--out", w / "F.csv")
        parts.append(pd.read_csv(w / "F.csv", dtype={"image": str}))
    f = pd.read_csv(a.final2, dtype={"image": str})
    redone = {p.stem for ps in groups.values() for p in ps}
    pd.concat([f[~f.image.isin(redone)], *parts]).to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
