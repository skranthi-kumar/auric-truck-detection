"""Dataset audit. Writes per-instance and per-image CSVs plus a short summary.

Usage: python src/eda.py [--phash]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from data import CLASSES, ROOT, image_paths, image_size, read_labels

OUT = Path(__file__).resolve().parents[1] / "experiments" / "results"


def iou_matrix(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def audit(split):
    inst, imgs = [], []
    for p in image_paths(split):
        w, h = image_size(p)
        cls, b = read_labels(p, (w, h))
        bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        iou = iou_matrix(b, b) if len(b) else np.zeros((0, 0))
        np.fill_diagonal(iou, 0)
        for i in range(len(b)):
            inst.append(dict(split=split, image=p.stem, cls=cls[i], w=bw[i], h=bh[i],
                             cx=(b[i, 0] + b[i, 2]) / 2, cy=(b[i, 1] + b[i, 3]) / 2,
                             border=b[i, 0] <= 1 or b[i, 1] <= 1 or b[i, 2] >= w - 1 or b[i, 3] >= h - 1,
                             max_iou_other=iou[i].max() if len(b) > 1 else 0.0,
                             invalid=bw[i] <= 1 or bh[i] <= 1 or b[i, 0] < -1 or b[i, 1] < -1
                             or b[i, 2] > w + 1 or b[i, 3] > h + 1))
        imgs.append(dict(split=split, image=p.stem, W=w, H=h, n=len(b),
                         **{f"n_c{k}": int((cls == k).sum()) for k in range(len(CLASSES))}))
    return pd.DataFrame(inst), pd.DataFrame(imgs)


def phash_near_duplicates(max_dist=6):
    import imagehash

    def h(p):
        with Image.open(p) as im:
            im.draft("RGB", (512, 512))
            return imagehash.phash(im.convert("L").resize((512, 512)))

    tr = {p.stem: h(p) for p in image_paths("train")}
    rows = [dict(val=v.stem, train=t, dist=hv - ht)
            for v in image_paths("val") for hv in [h(v)] for t, ht in tr.items() if hv - ht <= max_dist]
    return pd.DataFrame(rows, columns=["val", "train", "dist"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phash", action="store_true", help="also run train/val near-duplicate check")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    inst, imgs = zip(*(audit(s) for s in ("train", "val")))
    inst, imgs = pd.concat(inst), pd.concat(imgs)
    inst.to_csv(OUT / "eda_instances.csv", index=False)
    imgs.to_csv(OUT / "eda_images.csv", index=False)

    inst["side"] = np.sqrt(inst.w * inst.h)
    print("== instances per class (train/val) ==")
    print(inst.pivot_table(index="cls", columns="split", values="w", aggfunc="size").rename(index=dict(enumerate(CLASSES))))
    print("\n== box side sqrt(w*h) px, quantiles 5/25/50/75/95 ==")
    print(inst.groupby(["split", "cls"]).side.quantile([.05, .25, .5, .75, .95]).unstack().round(1))
    print("\n== aspect (long/short) median by class ==")
    print((np.maximum(inst.w, inst.h) / np.minimum(inst.w, inst.h)).groupby(inst.cls).median().round(2))
    print("\n== images ==")
    print(imgs.groupby("split")[["W", "H", "n"]].describe().T.round(0))
    print("\n== images containing class (train/val) ==")
    print(imgs.groupby("split")[[f"n_c{k}" for k in range(5)]].apply(lambda d: (d > 0).sum()))
    print(f"\nborder-touching: {inst.border.mean():.1%}  invalid: {int(inst.invalid.sum())}"
          f"  overlapping (IoU>0.3 w/ another GT): {(inst.max_iou_other > 0.3).mean():.1%}")
    print("duplicate GT (IoU>0.9):", int((inst.max_iou_other > 0.9).sum()))

    if args.phash:
        dups = phash_near_duplicates()
        dups.to_csv(OUT / "eda_phash_dups.csv", index=False)
        print(f"\nnear-duplicate train/val pairs (pHash dist<=6): {len(dups)}")
        print(dups.sort_values("dist").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
