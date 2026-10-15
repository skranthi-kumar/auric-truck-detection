"""Report figures that summarise the project (pipeline diagram, timeline, label agreement, ceiling, final round).
Numbers are copied from experiments/LOG.md (each is a scored, logged result); no inference, no scoring.
Same palette and style as src/make_figures.py.

Usage: python src/make_story_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from make_figures import F, GRID, INK, INK2, SER, SHORT, save


def box(ax, x, y, w, h, text, fc, ec=INK2, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, a, b, label=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=INK2, lw=1))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.08, label, ha="center", fontsize=7.5, color=INK2)


def fig_pipeline(rd=False):
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis("off")
    light = ["#e3eefb", "#fde6dc", "#dcf3ea", "#fcefcf"]
    box(ax, 0.1, 1.7, 1.5, 0.9, "Val image\n(~3,000 px,\n0.3 m GSD)", "#ffffff")
    box(ax, 2.0, 1.55, 1.9, 1.2, ("Scale check" if rd else "Scale check (SC1)") + "\nlonger side L in\n2,576-5,119 px?\nno: s = 2^round(log2(3228/L))", light[3], fs=7.8)
    box(ax, 4.4, 2.6, 2.4, 1.3, ("Zero-shot RF-DETR" if rd else "RFX0D: zero-shot RF-DETR") + "\n(public xView checkpoint)\nat scale s; 1008 px tiles,\n50% overlap x 4 flips, WBF", light[0], fs=7.8)
    box(ax, 4.4, 0.4, 2.4, 1.3, ("Fine-tuned RF-DETR: same\ncheckpoint, tuned on 443 train images" if rd else "RFX3: same checkpoint\nfine-tuned on 443 train images")+"\nat 2x, frozen class head,\nrare-class repeat sampling;\nrun at scale 2s", light[1], fs=7.8)
    box(ax, 7.3, 1.6, 1.2, 1.1, "WBF\nIoU 0.55\nequal\nweights", light[2], fs=7.8)
    box(ax, 8.8, 1.6, 1.1, 1.1, "Boxes mapped\nback to the\noriginal image", "#ffffff", fs=7.5)
    arrow(ax, (1.6, 2.15), (2.0, 2.15))
    arrow(ax, (3.9, 2.4), (4.4, 3.2), "s")
    arrow(ax, (3.9, 1.9), (4.4, 1.05), "2s")
    arrow(ax, (6.8, 3.2), (7.3, 2.4))
    arrow(ax, (6.8, 1.05), (7.3, 1.9))
    arrow(ax, (8.5, 2.15), (8.8, 2.15))
    ax.text(5, 4.15, "Final pipeline: two fused RF-DETR detectors + test-time scale normalisation" if rd else "Final pipeline SC1 = FINAL2 + test-time scale normalisation", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(5, 0.05, ("18 of 22 val images keep s = 1 (identical to the pipeline without scale normalisation); 2308" if rd else "18 of 22 val images keep s = 1 (identical to FINAL2); 2308") + " and 2391 get s = 0.5, 2384 and 2460 get s = 2.",
            ha="center", fontsize=8, color=INK2)
    save(fig, "fig_pipeline_main.png" if rd else "fig_pipeline.png")


def fig_timeline(rd=False):
    # best-so-far official score (all 22 val images) by the day it was logged
    pts = [("9 Oct", 0, 0.073, "E0 YOLO11s baseline"), ("9 Oct", 0.6, 0.107, "E1 2x upsampling"),
           ("10 Oct", 1.4, 0.139, "ENS_E2_ctx\nsnapshot WBF + context"), ("11 Oct", 2.3, 0.195, "RFX0 xView\nRF-DETR, zero-shot"),
           ("12 Oct", 3.3, 0.226, "ENS_RFX03\n+ fine-tuned member"), ("13 Oct", 4.3, 0.274, "FINAL2 dense views\n(RFX0D) + RFX3"),
           ("15 Oct", 6.2, 0.297, "SC1 test-time\nscale normalisation")]
    if rd:
        names = ["YOLO11s baseline", "2x upsampling", "checkpoint ensemble\n+ context rule", "zero-shot xView\nRF-DETR",
                 "+ fine-tuned RF-DETR,\nfused", "dense test-time\nviews, fused", "test-time scale\nnormalisation"]
        pts = [(d, x, y, n) for (d, x, y, _), n in zip(pts, names)]
    phases = [(0, 1.0, "Baseline,\ntaxonomy"), (1.0, 2.0, "Squeeze the\nprovided data"), (2.0, 3.0, "xView prior"),
              (3.0, 4.6, "Fine-tune\ndiagnosis, dense\ninference"), (4.6, 5.6, "FP forensics,\nprecision fixes"),
              (5.6, 6.6, "Scale audit, data\naudit, local fixes")]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for k, (a, b, lab) in enumerate(phases):
        ax.axvspan(a, b, color=["#f1f6fd", "#fdf3ee"][k % 2], zorder=0)
        ax.text((a + b) / 2, 0.445, lab, ha="center", va="top", fontsize=7.5, color=INK2)
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    ax.step(xs + [6.6], ys + [ys[-1]], where="post", color=SER[0], lw=2, zorder=2)
    ax.scatter(xs, ys, color=SER[1], s=36, zorder=3)
    for d, x, y, lab in pts:
        ax.text(x + 0.05, y + 0.008, f"{y:.3f}  {lab}", fontsize=7.3, color=INK, va="bottom")
    ax.scatter([6.22], [0.340], marker="D", color=SER[2], s=36, zorder=3)
    ax.text(6.3, 0.348, "0.340 on the 18 images\nat training resolution" if rd else "0.340 on the 18-image\ndiagnostic subset", fontsize=7.3, color=INK, va="bottom", ha="left")
    days = ["9 Oct", "10 Oct", "11 Oct", "12 Oct", "13 Oct", "14 Oct", "15 Oct"]
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6], days)
    ax.set_xlim(-0.1, 6.6); ax.set_ylim(0, 0.45)
    ax.set_ylabel("best mAP50 so far (all 22 val images)")
    ax.set_title("Seven days, one evidence chain: best score by day (target 0.75 is off the axis)")
    ax.grid(axis="y", color=GRID, zorder=0)
    save(fig, "fig_timeline_main.png" if rd else "fig_timeline.png")


def fig_label_agreement():
    train = [0.990, 0.971, 0.898, 0.794, 0.520]
    held = [0.986, 0.984, 0.667, 0.950, 0.350]
    val = [1.0] * 5
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    w = 0.26
    for k, (vals, lab) in enumerate([(train, "train (aligned images, 4,679 boxes)"), (held, "74 train images unseen by the checkpoint"),
                                     (val, "val (aligned images, 0 conflicts)")]):
        xs = [i + (k - 1) * w for i in range(5)]
        ax.bar(xs, vals, width=w * 0.92, color=SER[k], label=lab, zorder=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.3, color=INK)
    ax.set_xticks(range(5), SHORT)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of GT boxes whose class\nmatches xView's label")
    ax.set_title("Subtype labels agree with xView in val, not in the rare train classes")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", color=GRID, zorder=0)
    save(fig, "fig_label_agreement.png")


def fig_ceiling(rd=False):
    rows = [("SC1, class-aware (the reported score)", 0.297, 0.340),
            ("SC1, class-agnostic (every truck one class)", 0.358, 0.380),
            ("SC1 + every subtype error fixed (oracle)", 0.399, None),
            ("Perfect boxes + best crop classifier (5.1)", 0.429, None)]
    if rd:
        rows = [(lab.replace("SC1,", "Final pipeline,").replace("SC1 +", "Final pipeline +"), a, b) for lab, a, b in rows]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, (lab, a, b) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y + 0.17, a, height=0.32, color=SER[0], zorder=2, label="all 22 images" if i == 0 else None)
        ax.text(a + 0.006, y + 0.17, f"{a:.3f}", va="center", fontsize=8, color=INK)
        if b is not None:
            ax.barh(y - 0.17, b, height=0.32, color=SER[2], zorder=2, label=("18 images at training resolution" if rd else "18-image diagnostic subset") if i == 0 else None)
            ax.text(b + 0.006, y - 0.17, f"{b:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.75, color=SER[1], lw=2)
    ax.text(0.745, len(rows) - 0.55, "target 0.75", ha="right", fontsize=8.5, color=SER[1])
    ax.set_yticks(range(len(rows))[::-1], [r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("mAP50")
    ax.set_title("Even a perfect subtype classifier on these detections stays far below 0.75")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=GRID, zorder=0)
    save(fig, "fig_ceiling_main.png" if rd else "fig_ceiling.png")


def fig_final_round(rd=False):
    val = [("FU1_SC: WBF weights 2:1 to zero-shot", 0.0096, -0.0114, 0.0220),
           ("FU2_SC: WBF IoU 0.50", -0.0188, -0.0394, -0.0022),
           ("SC1c: fine-tuned member on cleaned labels", -0.0192, -0.0647, 0.0072),
           ("R8: 8 dihedral views instead of 4 flips", -0.0404, -0.0648, -0.0168)]
    held = [("BX + CB1 (classifier, original labels)", -0.0329, -0.0631, -0.0078),
            ("BX + CB2 (classifier, cleaned labels)", -0.0208, -0.0461, 0.0033)]
    if rd:
        val = [(n,) + v[1:] for n, v in zip(["Fusion re-weighting (2:1 to zero-shot)", "Tighter fusion threshold (IoU 0.50)",
                                             "Clean-label fine-tune", "8-view inference (adds transposed views)"], val)]
        held = [(n,) + v[1:] for n, v in zip(["Box-only re-scoring\n(classifier, original labels)",
                                              "Box-only re-scoring\n(classifier, cleaned labels)"], held)]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1.1, 1]})
    for ax, t, title, xl in ((axs[0], val, "Val, all 22 images: delta mAP50 vs final pipeline" if rd else "Val, all 22 images: delta mAP50 vs SC1", "paired delta, 95% CI"),
                             (axs[1], held, "74 held-out images: delta Box AP" if rd else "Held-out (74 images): delta Box AP", "paired delta, 95% CI")):
        for i, (lab, d, lo, hi) in enumerate(t):
            y = len(t) - 1 - i
            ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", ms=6, color=SER[1] if d < 0 else SER[0], ecolor=INK2, capsize=3)
            ax.text(d, y + 0.2, f"{d:+.3f}", fontsize=7.8, color=INK, ha="center")
        ax.axvline(0, color=INK2, lw=1)
        ax.axvline(0.01, color=INK2, lw=1, ls=":")
        ax.set_yticks(range(len(t))[::-1], [x[0] for x in t], fontsize=8)
        ax.set_xlim(-0.08, 0.045)
        ax.set_xticks([-0.06, -0.03, 0, 0.03])
        ax.set_ylim(-0.6, len(t) - 0.4)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(xl, fontsize=8.5)
        ax.grid(axis="x", color=GRID)
    fig.suptitle("Final round of local fixes (15 Oct): none clears the declared bar (dotted line, +0.01)", fontsize=10.5,
                 fontweight="bold", color=INK)
    save(fig, "fig_final_round_main.png" if rd else "fig_final_round.png")


def fig_subtype_bench():
    det = [0.880, 0.336, 0.314, 0.208, 0.111]
    cb1 = [0.799, 0.545, 0.171, 0.125, 0.111]
    cb2 = [0.748, 0.573, 0.200, 0.250, 0.111]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    w = 0.26
    for k, (vals, lab) in enumerate([(det, "detector's own class (RFX0)"), (cb1, "CB1 crop classifier, original labels"),
                                     (cb2, "CB2 crop classifier, cleaned labels")]):
        xs = [i + (k - 1) * w for i in range(5)]
        ax.bar(xs, vals, width=w * 0.92, color=SER[k], label=lab, zorder=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(5), SHORT)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("subtype recall on located boxes")
    ax.set_title("Held-out subtype benchmark (487 boxes, 74 unseen images): no classifier beats the detector")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", color=GRID, zorder=0)
    save(fig, "fig_subtype_bench.png")


if __name__ == "__main__":
    F.mkdir(parents=True, exist_ok=True)
    for f in (fig_pipeline, fig_timeline, fig_label_agreement, fig_ceiling, fig_final_round, fig_subtype_bench):
        f()
    for f in (fig_pipeline, fig_timeline, fig_ceiling, fig_final_round):  # main report: plain names, no run codes
        f(rd=True)
