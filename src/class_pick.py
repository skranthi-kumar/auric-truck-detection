"""Build a prediction CSV by taking each class from a chosen source CSV.

This is mainly a diagnostic for ensembles whose per-class strengths differ.
Example:
  python src/class_pick.py \
    --sources rfx0=experiments/results/RFX0_val_preds.csv ens=experiments/results/ENS_RFX03_val_preds.csv \
    --pick ens,rfx0,rfx0,ens,ens \
    --out experiments/results/RFX_classpick_val_preds.csv

The --pick list is ordered as:
  Cargo, Box, Flatbed, Tractor, Liquid
"""

import argparse

import pandas as pd

from data import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--pick", required=True, help="comma-separated source names for the five classes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sources = {}
    for item in args.sources:
        name, path = item.split("=", 1)
        sources[name] = pd.read_csv(path, dtype={"image": str})

    pick = [x.strip() for x in args.pick.split(",")]
    if len(pick) != len(CLASSES):
        raise SystemExit(f"--pick must contain {len(CLASSES)} names")
    missing = sorted(set(pick) - set(sources))
    if missing:
        raise SystemExit(f"unknown source(s): {missing}; available: {sorted(sources)}")

    parts = []
    for c, name in enumerate(pick):
        part = sources[name][sources[name].cls == c].copy()
        parts.append(part)
        print(f"{CLASSES[c]} <- {name}: {len(part)} rows")

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} rows")


if __name__ == "__main__":
    main()
