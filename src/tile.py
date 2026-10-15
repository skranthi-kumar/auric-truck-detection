"""Cut full-resolution images into overlapping tiles in YOLO format.

Frozen parameters (see DECISIONS.md): tile 640, overlap 128, keep 10% of empty tiles, seed 0.
A box is kept in a tile if >=50% of its area falls inside; it is clipped to the tile.
Writes <out>/<split>/{images,labels}/<img>_<x0>_<y0>.png and <out>/<split>_manifest.csv.

Usage: python src/tile.py --out /content/tiles [--splits train val]
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

from data import image_paths, read_labels

TILE, OVERLAP, BG_KEEP, MIN_VIS, SEED = 640, 128, 0.10, 0.5, 0


def starts(length, tile=TILE, stride=TILE - OVERLAP):
    """Tile origins covering [0, length); the last tile is flush with the border."""
    if length <= tile:
        return [0]
    s = list(range(0, length - tile, stride))
    return s + [length - tile]


def tile_image(p, out, split, rng, writer):
    with Image.open(p) as im:
        im = im.convert("RGB")
        W, H = im.size
        cls, b = read_labels(p, (W, H))
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        for y0 in starts(H):
            for x0 in starts(W):
                x1, y1 = min(x0 + TILE, W), min(y0 + TILE, H)
                cb = np.clip(b, [x0, y0, x0, y0], [x1, y1, x1, y1])
                vis = (cb[:, 2] - cb[:, 0]) * (cb[:, 3] - cb[:, 1]) / np.maximum(area, 1e-9)
                keep = vis >= MIN_VIS
                if not keep.any() and rng.random() > BG_KEEP:
                    continue
                name = f"{p.stem}_{x0}_{y0}"
                im.crop((x0, y0, x1, y1)).save(out / split / "images" / f"{name}.png")
                tw, th = x1 - x0, y1 - y0
                with open(out / split / "labels" / f"{name}.txt", "w") as f:
                    for c, (a1, b1, a2, b2) in zip(cls[keep], cb[keep]):
                        f.write(f"{c} {((a1 + a2) / 2 - x0) / tw:.6f} {((b1 + b2) / 2 - y0) / th:.6f} "
                                f"{(a2 - a1) / tw:.6f} {(b2 - b1) / th:.6f}\n")
                writer.writerow([name, p.stem, x0, y0, int(keep.sum())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()
    out = Path(args.out)
    rng = random.Random(SEED)
    for split in args.splits:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        with open(out / f"{split}_manifest.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tile", "image", "x0", "y0", "n_boxes"])
            paths = image_paths(split, args.root) if args.root else image_paths(split)
            for p in paths:
                tile_image(p, out, split, rng, w)
        print(split, sum(1 for _ in open(out / f"{split}_manifest.csv")) - 1, "tiles")


if __name__ == "__main__":
    main()
