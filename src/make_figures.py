"""All report figures from saved results (no inference). Output: report/figs/fig_*.png

Palette: validated 5-slot categorical (dataviz reference palette; CVD and normal-vision checks pass, contrast WARN ->
every chart carries direct labels and the report has the tables). Sequential = one blue ramp.

Usage: python src/make_figures.py --root CV_Assignment_AAL
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from data import CLASSES, image_paths

R = Path(__file__).resolve().parents[1] / "experiments" / "results"
F = Path(__file__).resolve().parents[1] / "report" / "figs"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed order
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("b", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
SHORT = ["Cargo", "Box", "Flatbed", "Tractor", "Liquid"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold"})


def met(n):
    return json.load(open(R / f"{n}_metrics.json"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / name, dpi=160)
    plt.close(fig)
    print("wrote", F / name)


def fig_progression():
    runs = [("E0_s0", "YOLO11s\nbaseline"), ("E1_up2_s0", "+ 2x\nupsample"), ("ENS_E2_ctx", "+ ensemble\n+ context"),
            ("RFX0", "RF-DETR\nxView ckpt"), ("ENS_RFX03", "+ fine-tuned\nmember (WBF)"),
            ("RFX0D", "dense views\n(4 flips, 50%)"), ("FINAL2", "FINAL2 =\nRFX0D + RFX3"),
            ("SC1", "SC1 = FINAL2\n+ scale norm.")]
    m = [met(r) for r, _ in runs]
    y = [d["mAP50"] for d in m]
    lo = [d["mAP50"] - d["mAP50_CI95"][0] for d in m]
    hi = [d["mAP50_CI95"][1] - d["mAP50"] for d in m]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(len(runs))
    ax.bar(x, y, width=0.55, color=[SER[0]] * 3 + [SER[1]] * 5, zorder=2)
    ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    for i, v in enumerate(y):
        ax.text(i + 0.3, v, f"{v:.3f}", va="bottom", ha="left", color=INK, fontsize=9)
    ax.set_xticks(x, [l for _, l in runs], fontsize=8.5)
    ax.set_ylabel("val mAP50 (95% CI over images)")
    ax.set_title("From baseline to final pipeline (blue: YOLO11s from COCO; orange: RF-DETR from xView)")
    ax.text(0.0, -0.3, "Bars: per-model CIs (they overlap). Steps are tested with paired image bootstraps, e.g. "
            "2x upsample +0.034 (CI +0.002..+0.093);\nFINAL2 vs ENS_RFX03 +0.048 (+0.020..+0.070); SC1 vs FINAL2 +0.023 (+0.003..+0.049). "
            "Target 0.75 is off the axis.", transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    ax.grid(axis="y", color=GRID, zorder=0)
    save(fig, "fig_progression.png")


def fig_per_class():
    runs = [("E0_s0", "YOLO baseline"), ("ENS_E2_ctx", "best YOLO pipeline"), ("FINAL2", "final (FINAL2)"),
            ("oracle_cls", "perfect boxes + crop classifier (5.1)")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.2
    for k, (r, lab) in enumerate(runs):
        ap = list(met(r)["AP50"].values())
        xs = np.arange(5) + (k - 1.5) * w
        ax.bar(xs, ap, width=w * 0.9, color=SER[k], label=lab, zorder=2)
    ax.set_xticks(np.arange(5), [f"{s}\n(val n={n})" for s, n in zip(SHORT, [800, 493, 122, 117, 20])])
    ax.set_ylabel("AP50")
    ax.set_title("Per-class AP50: Truck Tractor stays near zero for every detector")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRID, zorder=0)
    save(fig, "fig_per_class.png")


def fig_oracle():
    d = met("FINAL2")["oracle_dAP"]
    keys = [("bg", "background FP"), ("cls", "classification"), ("loc", "localisation"), ("both", "cls + loc"),
            ("dup", "duplicate")]
    vals = [d.get(k, 0) for k, _ in keys]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.barh(range(len(keys))[::-1], vals, color=SER[0], height=0.55, zorder=2)
    for i, v in enumerate(vals):
        ax.text(v + 0.003, len(keys) - 1 - i, f"+{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(keys))[::-1], [l for _, l in keys])
    ax.set_xlabel("mAP50 gained if this error type were fixed (oracle, FINAL2)")
    ax.set_title(f"Where the AP goes (FINAL2): background FPs, then subtype\n"
                 f"({met('FINAL2')['missed_gt']} of 1552 GT never detected)", fontsize=10)
    ax.grid(axis="x", color=GRID, zorder=0)
    save(fig, "fig_oracle_dap.png")


def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (r, t) in zip(axes, [("FINAL2", "Final pipeline: detections (conf >= 0.25) vs GT"),
                                 ("oracle_cls", "Perfect boxes: crop classifier vs GT (5.1)")]):
        d = met(r)
        # evaluate.py confusion: rows = GT classes + background, cols = predicted classes + missed (conf >= cm_thr)
        cm = pd.DataFrame(d["confusion"], index=SHORT + ["background"], columns=SHORT + ["missed"])
        norm = cm.div(cm.sum(1).replace(0, 1), axis=0)
        ax.imshow(norm.values, cmap=BLUES, vmin=0, vmax=1)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, int(cm.values[i, j]), ha="center", va="center", fontsize=8,
                        color="#ffffff" if norm.values[i, j] > 0.55 else INK)
        ax.set_xticks(range(6), list(cm.columns), rotation=30, ha="right")
        ax.set_yticks(range(6), list(cm.index))
        ax.set_xlabel("predicted")
        ax.set_ylabel("ground truth")
        ax.set_title(t, fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Counts; shading = row share. Same Cargo <-> Box and -> Cargo pattern in both", color=INK2, fontsize=9)
    save(fig, "fig_confusion.png")


def fig_recall_size():
    bins, labs = [0, 16, 24, 32, 1e9], ["<16", "16-24", "24-32", ">=32"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
    for k, (r, lab) in enumerate([("E0_s0", "E0 (native)"), ("E1_up2_s0", "E1 (2x upsample)")]):
        g = pd.read_csv(R / f"{r}_gt.csv")
        g["size"] = pd.cut(np.sqrt((g.x2 - g.x1) * (g.y2 - g.y1)), bins, labels=labs)
        for ax, c in zip(axes, range(4)):
            t = g[g.cls == c].groupby("size", observed=False).matched.mean()
            ax.plot(range(4), t.values, marker="o", ms=5, lw=2, color=SER[k], label=lab)
            ax.set_title(SHORT[c], fontsize=10)
            ax.set_xticks(range(4), labs, fontsize=8)
            ax.grid(axis="y", color=GRID)
    axes[0].set_ylabel("recall (correct class)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Recall rises with size within each class; 2x upsampling helps the small bins", color=INK, fontsize=10)
    save(fig, "fig_recall_by_size.png")


def fig_learning_curve():
    pts = {"LC_rand_0125_s0": 965, "LC_rand_0125_s1": 860, "LC_rand_0250_s0": 1541, "LC_rand_0250_s1": 1637,
           "LC_rand_0500_s0": 3682, "LC_full_1000_s0": 7618}
    x = np.array(list(pts.values()), float)
    y = np.array([met(r)["mAP50"] for r in pts])
    b, a = np.polyfit(np.log2(x), y, 1)
    xs = np.geomspace(800, 8200, 50)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(xs, a + b * np.log2(xs), color=SER[0], lw=2, label=f"log-linear fit: +{b:.3f} per doubling")
    for lo_b in (0.0061, 0.0253):  # hierarchical bootstrap slope CI (lc_interval.txt), pivoted at the full-data point
        ax.plot(xs, y[-1] + lo_b * np.log2(xs / 7618), color=SER[0], lw=1, ls="--", alpha=0.6,
                label="slope 95% CI (seed + image bootstrap)" if lo_b < 0.01 else None)
    ax.scatter(x, y, s=40, color=SER[1], zorder=3, label="random-subset runs (2 seeds at 12.5%, 25%)")
    ax.axvline(7618 + 500, color=INK2, lw=1, ls=":")
    ax.text(7618 + 500, 0.03, " +500 labels:\n +0.0014", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("train instances (log scale)")
    ax.set_ylabel("val mAP50")
    ax.set_title("5.3: more labels give log-scale gains")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(color=GRID)
    save(fig, "fig_learning_curve.png")


def fig_subset():
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for k, (kind, lab) in enumerate([("rand", "random"), ("sel", "selected (declared rule)")]):
        for j, f in enumerate(["0125", "0250"]):
            v = [met(f"LC_{kind}_{f}_s{s}")["mAP50"] for s in (0, 1)]
            xc = j + (k - 0.5) * 0.3
            ax.bar(xc, np.mean(v), width=0.28, color=SER[k], label=lab if j == 0 else None, zorder=2)
            ax.scatter([xc, xc], v, color=INK, s=12, zorder=3)
    ax.axhline(0.9 * met("LC_full_1000_s0")["mAP50"], color=INK2, ls="--", lw=1)
    ax.text(1.45, 0.9 * 0.0882, "90% of full", fontsize=8, va="bottom", ha="right", color=INK2)
    ax.set_xticks([0, 1], ["12.5% budget", "25% budget"])
    ax.set_ylabel("val mAP50 (bar = seed mean, dots = seeds)")
    ax.set_title("5.4: the selection rule did not beat random")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=GRID, zorder=0)
    save(fig, "fig_subset.png")


def fig_context_radius():
    val = {50: .913, 100: .893, 150: .890, 200: .876, 300: .841, 400: .810}
    tr = {50: .812, 100: .788, 150: .772, 200: .756, 300: .738, 400: .727}
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k, (d, lab) in enumerate([(val, "val GT"), (tr, "train GT")]):
        ax.plot(list(d), list(d.values()), marker="o", ms=5, lw=2, color=SER[k], label=lab)
    ax.axhline(0.59, color=INK2, ls="--", lw=1)
    ax.text(400, 0.6, "crop classifier on the box itself (0.59)", fontsize=8, ha="right", color=INK2)
    ax.axvline(150, color=INK2, ls=":", lw=1)
    ax.set_xlabel("neighbour radius R (px)")
    ax.set_ylabel("neighbour-majority accuracy")
    ax.set_title("A truck's neighbours predict its class better than its pixels")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID)
    save(fig, "fig_context_radius.png")


def fig_loss():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for k, (r, lab) in enumerate([("E1_up2_s0", "E1 (50 ep)"), ("E2_up2_e100_s0", "E2 (100 ep)")]):
        d = pd.read_csv(R / f"{r}_train_results.csv")
        d.columns = [c.strip() for c in d.columns]
        axes[0].plot(d.epoch, d["val/cls_loss"], lw=2, color=SER[k], label=f"{lab} val")
        axes[0].plot(d.epoch, d["train/cls_loss"], lw=1, ls="--", color=SER[k], label=f"{lab} train")
        axes[1].plot(d.epoch, d["metrics/mAP50(B)"], lw=2, color=SER[k], label=lab)
    axes[0].set_title("Classification loss: val turns up after ~25-40 epochs")
    axes[0].set_xlabel("epoch")
    axes[1].set_title("Tile-level val mAP50 (monitoring only) peaks ~epoch 40")
    axes[1].set_xlabel("epoch")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.grid(color=GRID)
    save(fig, "fig_loss_curves.png")


def fig_overlays(root):
    m = pd.read_csv(R / "FINAL2_matches.csv", dtype={"image": str}, usecols=lambda c: c != "target")
    g = pd.read_csv(R / "FINAL2_gt.csv", dtype={"image": str})
    paths = {p.stem: p for p in image_paths("val", root)}
    m = m[m.conf >= 0.25]
    # dense depot, medium, sparse scene; crop a 900 px window around the densest GT area
    picks = ["1211", "2470", "1181"]
    WIN = 600
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, im in zip(axes, picks):
        gi, mi = g[g.image == im], m[m.image == im]
        hit = set(mi[mi.tp].gt_id)  # GT hit by a true positive at the operating point (conf >= 0.25)
        gx, gy = ((gi.x1 + gi.x2) / 2).values, ((gi.y1 + gi.y2) / 2).values
        with Image.open(paths[im]) as I:
            W, H = I.size
            best = max(((x, y) for x in range(0, max(W - WIN, 1), 100) for y in range(0, max(H - WIN, 1), 100)),
                       key=lambda o: ((gx >= o[0]) & (gx < o[0] + WIN) & (gy >= o[1]) & (gy < o[1] + WIN)).sum())
            x0, y0 = best
            ax.imshow(I.convert("RGB").crop((x0, y0, x0 + WIN, y0 + WIN)))
        n_in = ((gx >= x0) & (gx < x0 + WIN) & (gy >= y0) & (gy < y0 + WIN)).sum()
        def box(r, color, lw=1.4, ls="-"):
            ax.add_patch(plt.Rectangle((r.x1 - x0, r.y1 - y0), r.x2 - r.x1, r.y2 - r.y1, fill=False, ec=color, lw=lw, ls=ls))
        for r in mi[mi.tp].itertuples():
            box(r, SER[0])
        for r in mi[(~mi.tp) & (mi.err == "cls")].itertuples():
            box(r, SER[3])
        for r in mi[(~mi.tp) & (mi.err == "bg")].itertuples():
            box(r, SER[1])
        for r in gi[~gi.gt_id.isin(hit)].itertuples():
            box(r, SER[4], ls="--")
        ax.set_xlim(0, WIN)
        ax.set_ylim(WIN, 0)
        ax.axis("off")
        ax.set_title(f"val image {im}: {n_in} of {len(gi)} GT boxes in this 600 px window", fontsize=10)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=SER[0], lw=2, label="true positive (conf >= 0.25)"),
         Line2D([], [], color=SER[3], lw=2, label="wrong class"),
         Line2D([], [], color=SER[1], lw=2, label="background FP"),
         Line2D([], [], color=SER[4], lw=2, ls="--", label="missed GT")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Final pipeline (FINAL2) on val, densest 600 px window per image", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(F / "fig_overlays.png", dpi=130)
    plt.close(fig)
    print("wrote", F / "fig_overlays.png")


def fig_density():
    t = {"sparse\n(8 imgs, 177 boxes)": (0.202, 0.131, 0.334), "medium\n(7 imgs, 507)": (0.322, 0.220, 0.459),
         "dense\n(7 imgs, 868)": (0.298, 0.158, 0.419)}
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for i, (k, (v, lo, hi)) in enumerate(t.items()):
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="o", ms=7, color=SER[0], ecolor=INK2, capsize=3)
        ax.text(i + 0.08, v, f"{v:.2f}", fontsize=9, va="center")
    ax.set_xticks(range(3), list(t))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("class-agnostic AP50 (E2_last)")
    ax.set_title("No crowding failure in dense depot scenes")
    ax.grid(axis="y", color=GRID)
    save(fig, "fig_density.png")


def fig_fp_forensics():
    t = pd.read_csv(R / "FINAL2_fp_forensics.csv", index_col=0)  # src/fp_forensics.py
    cols = list(t.columns)
    colors = [SER[3], "#b9b8b3", SER[4], SER[1], SER[2], "#d9d8d3", SER[0]]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    left = np.zeros(len(t))
    for k, c in enumerate(cols):
        v = t[c].values
        ax.barh(range(len(t))[::-1], v, left=left, color=colors[k], height=0.6, label=c, edgecolor=SURF, lw=2)
        for i, (l, w) in enumerate(zip(left, v)):
            if w >= 0.08:
                ax.text(l + w / 2, len(t) - 1 - i, f"{w:.0%}", ha="center", va="center", fontsize=8, color=INK)
        left += v
    ax.set_yticks(range(len(t))[::-1], SHORT)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of the top-400 false positives per class (13 xView-aligned val images)")
    ax.set_title("What the damaging false positives sit on (xView's own 60-class labels)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    save(fig, "fig_fp_forensics.png")


def fig_precision_attempts():
    # paired image-bootstrap deltas (src/compare.py), values copied from experiments/LOG.md
    t = [("SC1: test-time scale normalisation (adopted)", "FINAL2", 0.023, 0.0025, 0.0489),
         ("FINAL3: + flip views on fine-tuned member", "FINAL2", 0.002, -0.012, 0.010),
         ("SP1: train-set size/shape prior", "FINAL2", -0.004, -0.0065, -0.0004),
         ("CS1: suppress competing xView vehicle classes", "RFX0", -0.001, -0.0093, 0.0097),
         ("T2: fuse 5-way crop classifier", "RFX0D", -0.128, -0.189, -0.055)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, (lab, base, d, lo, hi) in enumerate(t):
        y = len(t) - 1 - i
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", ms=7, color=SER[2] if lo > 0 and d >= 0.01 else SER[1] if d < 0 else SER[0], ecolor=INK2,
                    capsize=3)
        ax.text(0.06, y, f"{d:+.3f} vs {base}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(0.01, color=INK2, lw=1, ls=":")
    ax.text(0.011, -0.45, "adoption bar +0.01", fontsize=8, color=INK2)
    ax.set_yticks(range(len(t))[::-1], [x[0] for x in t], fontsize=8.5)
    ax.set_xlim(-0.2, 0.13)
    ax.set_ylim(-0.6, len(t) - 0.5)
    ax.set_xlabel("paired delta mAP50 with 95% CI (declared before scoring)")
    ax.set_title("Fixes tried after FINAL2: only SC1 clears the bar")
    ax.grid(axis="x", color=GRID)
    save(fig, "fig_precision_attempts.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    F.mkdir(parents=True, exist_ok=True)
    for f in (fig_progression, fig_per_class, fig_oracle, fig_confusion, fig_recall_size, fig_learning_curve,
              fig_subset, fig_context_radius, fig_loss, fig_density, fig_fp_forensics, fig_precision_attempts):
        f()
    fig_overlays(a.root)


if __name__ == "__main__":
    main()
