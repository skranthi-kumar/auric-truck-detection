"""Section 3 failure analysis on top of evaluate.py outputs (<name>_matches.csv, <name>_gt.csv).

FN/FP as populations, stratified so confounds are visible:
  - recall by size bin *within* class (size vs class confound)
  - recall by crowding (neighbours within 2 box diagonals) within class
  - FP error mix by predicted class and size, at an operating threshold
  - top confusion pairs
Galleries of crops (top-confidence background FPs, missed GT) for visual audit (e.g. H-unlabeled).

Usage: python src/analyze_errors.py --name E0_s0 [--thr 0.25] [--gallery 48]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from data import CLASSES, ROOT

RES = Path(__file__).resolve().parents[1] / "experiments" / "results"
FIG = Path(__file__).resolve().parents[1] / "report" / "figs"
SIZE_BINS = [0, 16, 24, 32, 1e9]
SIZE_LBL = ["<16", "16-24", "24-32", ">=32"]


def side(d):
    return np.sqrt((d.x2 - d.x1) * (d.y2 - d.y1))


def crowding(gt):
    n = np.zeros(len(gt), int)
    for _, G in gt.groupby("image"):
        c = np.stack([((G.x1 + G.x2) / 2).values, ((G.y1 + G.y2) / 2).values], 1)
        diag = np.hypot(G.x2 - G.x1, G.y2 - G.y1).values
        d = np.hypot(*(c[:, None] - c[None]).transpose(2, 0, 1))
        n[G.index] = ((d < 2 * diag[:, None]).sum(1) - 1)
    return n


def gallery(df, root, out, n, title, color):
    rows = df.head(n)
    if not len(rows):
        return
    S, cols = 96, 8
    canvas = Image.new("RGB", (cols * S, ((len(rows) - 1) // cols + 1) * (S + 14)), "white")
    for k, r in enumerate(rows.itertuples()):
        cx, cy, half = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2, max(r.x2 - r.x1, r.y2 - r.y1, 32)
        with Image.open(Path(root) / "val" / "images" / f"{r.image}.png") as im:
            crop = im.convert("RGB").crop((cx - half, cy - half, cx + half, cy + half))
        d = ImageDraw.Draw(crop)
        bx = [r.x1 - cx + half, r.y1 - cy + half, r.x2 - cx + half, r.y2 - cy + half]
        d.rectangle([min(bx[0], bx[2]), min(bx[1], bx[3]), max(bx[0], bx[2]), max(bx[1], bx[3])], outline=color)
        crop = crop.resize((S, S))
        x, y = (k % cols) * S, (k // cols) * (S + 14)
        canvas.paste(crop, (x, y))
        ImageDraw.Draw(canvas).text((x + 2, y + S), r.label[:18], fill="black")
    canvas.save(out)
    print(f"  gallery: {out.name} ({title}, {len(rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--gallery", type=int, default=48)
    a = ap.parse_args()
    root = a.root or ROOT
    m = pd.read_csv(RES / f"{a.name}_matches.csv", dtype={"image": str})
    g = pd.read_csv(RES / f"{a.name}_gt.csv", dtype={"image": str}).reset_index(drop=True)
    g["size"] = pd.cut(side(g), SIZE_BINS, labels=SIZE_LBL, right=False)
    g["crowd"] = pd.cut(crowding(g), [-1, 0, 2, 1e9], labels=["0", "1-2", ">=3"])
    g["cname"] = g.cls.map(dict(enumerate(CLASSES)))
    # outcome per GT at the operating threshold: TP / wrong class / missed
    thr_tp = m[(m.tp) & (m.conf >= a.thr)].gt_id
    g["outcome"] = np.where(g.gt_id.isin(thr_tp), "tp",
                            np.where((g.best_cls >= 0) & (g.best_cls != g.cls) & (g.best_conf >= a.thr), "wrong_cls", "missed"))
    out = []

    def show(title, t):
        print(f"\n== {title} ==\n{t.to_string()}")
        out.append((title, t))

    show(f"GT outcome by class (conf>={a.thr})", pd.crosstab(g.cname, g.outcome, margins=True))
    rec = lambda d: pd.Series({"recall": (d.outcome == "tp").mean(), "n": len(d)})
    show("recall by size within class", g.groupby(["cname", "size"], observed=True).apply(rec).unstack("size").round(2))
    show("recall by crowding within class", g.groupby(["cname", "crowd"], observed=True).apply(rec).unstack("crowd").round(2))
    w = g[g.outcome == "wrong_cls"]
    show("confusion pairs (GT -> predicted)", w.groupby(["cname", w.best_cls.map(dict(enumerate(CLASSES)))]).size()
         .sort_values(ascending=False).head(10).rename("n").to_frame())
    f = m[(m.conf >= a.thr) & ~m.tp].copy()
    f["size"] = pd.cut(side(f), SIZE_BINS, labels=SIZE_LBL, right=False)
    show(f"FP error type by predicted class (conf>={a.thr})", pd.crosstab(f.cls.map(dict(enumerate(CLASSES))), f.err, margins=True))
    show("FP error type by size", pd.crosstab(f["size"], f.err))

    FIG.mkdir(parents=True, exist_ok=True)
    bg = m[(m.err == "bg")].sort_values("conf", ascending=False).copy()
    bg["label"] = bg.cls.map(lambda c: CLASSES[c]) + bg.conf.map(lambda c: f" {c:.2f}")
    gallery(bg, root, FIG / f"{a.name}_fp_bg_top.png", a.gallery, "top-confidence background FPs", "red")
    ms = g[g.outcome == "missed"].sample(frac=1, random_state=0).copy()
    ms["label"] = ms.cname
    gallery(ms, root, FIG / f"{a.name}_fn_missed.png", a.gallery, "random missed GT", "lime")
    with open(RES / f"{a.name}_errors.txt", "w") as fh:
        for t, d in out:
            fh.write(f"== {t} ==\n{d.to_string()}\n\n")


if __name__ == "__main__":
    main()
