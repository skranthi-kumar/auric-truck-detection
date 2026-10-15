"""BX: Box-only re-scoring with a crop classifier (declared in LOG.md). Only rows of class Truck w/Box change, so the
other four class APs cannot move (asserted in val mode).

Rule (fixed): for each Box detection with conf >= 0.05, crop as src/oracle_cls.py (2x context, min 48 px, 128 px,
8-way TTA), p = classifier probability of Truck w/Box, new conf = sqrt(conf * p). Box rows below 0.05 keep their conf.

heldout: Box detections from the RFX0 train dump on the 74 checkpoint-unseen train images (per-tile queries mapped to
image coordinates, class-wise NMS 0.5), scored against the CLEANED labels (CV_clean); Box AP before vs after with a
paired image bootstrap. The classifier must not have trained on these images (CB1/CB2 hold them out).
val: apply the rule to a prediction CSV (e.g. SC1) -> BX_val_preds.csv; scored with evaluate.py by the caller.

Usage: python src/box_rescore.py heldout --cls CB2_ctx2_holdout.pt --dump rfx0_train_dump.npz
       python src/box_rescore.py val --cls CB2_ctx2_holdout.pt --preds SC1_val_preds.csv --out BX_val_preds.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torchvision
from PIL import Image
from torchvision.ops import box_iou

from data import CLASSES, ROOT, image_paths, image_size, read_labels
from evaluate import ap101
from oracle_cls import CTX, MIN_SIDE, SIZE, to_tensor

BOX = CLASSES.index("Truck w/Box")
REPO = Path(__file__).resolve().parents[1]


def classify(model, dev, img_path, boxes):
    X = []
    with Image.open(img_path) as I:
        I = I.convert("RGB")
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            h = max(CTX * max(x2 - x1, y2 - y1), MIN_SIDE) / 2
            X.append(np.asarray(I.crop((cx - h, cy - h, cx + h, cy + h)).resize((SIZE, SIZE), Image.BILINEAR)))
    if not X:
        return np.zeros(0)
    x = to_tensor(np.stack(X))
    with torch.no_grad():
        P = sum(torch.cat([model(torch.rot90(x.flip(-1) if f else x, k, (-2, -1))[i:i + 256].to(dev)).softmax(1).cpu()
                           for i in range(0, len(x), 256)]) for f in (0, 1) for k in range(4)) / 8
    return P[:, BOX].numpy()


def rescore(df, model, dev, path_of):
    df = df.copy()
    k = (df.cls == BOX) & (df.conf >= 0.05)
    for im, f in df[k].groupby("image"):
        p = classify(model, dev, path_of[im], f[["x1", "y1", "x2", "y2"]].values)
        df.loc[f.index, "conf"] = np.sqrt(f.conf.values * p)
    return df


def box_match(det, gt_boxes, ims):
    """Greedy IoU 0.5 matching per image (fixed once); returns conf, tp, image index, GT count per image."""
    ix = {im: i for i, im in enumerate(ims)}
    confs, tps, idx = [], [], []
    for im in ims:
        d = det[det.image == im].sort_values("conf", ascending=False)
        g = gt_boxes.get(im, np.zeros((0, 4)))
        used = np.zeros(len(g), bool)
        tp = np.zeros(len(d), bool)
        if len(d) and len(g):
            iou = box_iou(torch.tensor(d[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32), torch.tensor(g, dtype=torch.float32)).numpy()
            for i in range(len(d)):
                j = np.where(~used, iou[i], -1).argmax()
                if iou[i, j] >= 0.5 and not used[j]:
                    used[j] = tp[i] = True
        confs.append(d.conf.values); tps.append(tp); idx.append(np.full(len(d), ix[im]))
    ngt = np.array([len(gt_boxes.get(im, [])) for im in ims])
    return np.concatenate(confs), np.concatenate(tps), np.concatenate(idx), ngt


def box_ap(M, cnt=None):
    conf, tp, idx, ngt = M
    cnt = np.ones(len(ngt)) if cnt is None else cnt
    return ap101(conf, tp, (ngt * cnt).sum(), cnt[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["heldout", "val"])
    ap.add_argument("--cls", required=True)
    ap.add_argument("--dump", default=str(Path.home() / "auric_kaggle/DUMP/rfx0_train_dump.npz"))
    ap.add_argument("--holdout", default=str(REPO / "experiments/splits/rfx_heldout_train_ids.txt"))
    ap.add_argument("--clean_root", default=str(REPO / "CV_clean"))
    ap.add_argument("--preds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("convnext_tiny.fb_in22k", pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(a.cls, map_location="cpu")); model.to(dev).eval()

    if a.mode == "val":
        df = pd.read_csv(a.preds, dtype={"image": str})
        new = rescore(df, model, dev, {p.stem: p for p in image_paths("val")})
        assert (new[new.cls != BOX].conf.values == df[df.cls != BOX].conf.values).all(), "non-Box rows changed"
        new.to_csv(a.out, index=False); print("wrote", a.out)
        return

    held = sorted(set(Path(a.holdout).read_text().split()))
    d = np.load(a.dump)
    names = d["names"]
    img = np.array([n.split("_")[0] for n in names])[d["tile"]]
    off = np.array([[int(n.split("_")[1]), int(n.split("_")[2].split(".")[0])] for n in names], dtype=np.float32)[d["tile"]]
    s = 1 / (1 + np.exp(-d["logit"][:, 12].astype(np.float32)))
    rows = []
    for im in held:
        k = np.flatnonzero((img == im) & (s >= 0.01))
        if not len(k):
            continue
        b = torch.tensor(d["box"][k] + np.tile(off[k], 2), dtype=torch.float32)
        keep = torchvision.ops.nms(b, torch.tensor(s[k]), 0.5).numpy()
        rows += [(im, BOX, float(s[k][i]), *b[i].tolist()) for i in keep]
    det = pd.DataFrame(rows, columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"])
    paths = {p.stem: p for p in image_paths("train", a.clean_root)}
    gt = {}
    for im in held:
        c, bb = read_labels(paths[im], image_size(paths[im]))
        gt[im] = bb[c == BOX]
    new = rescore(det, model, dev, {im: paths[im] for im in held})
    M0, M1 = box_match(det, gt, held), box_match(new, gt, held)
    base, after = box_ap(M0), box_ap(M1)
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(a.n):
        cnt = np.bincount(rng.integers(0, len(held), len(held)), minlength=len(held)).astype(float)
        deltas.append(box_ap(M1, cnt) - box_ap(M0, cnt))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"held-out Box AP ({len(held)} images, {sum(len(v) for v in gt.values())} Box GT, cleaned labels): "
          f"detector {base:.4f} -> re-scored {after:.4f}, delta {after - base:+.4f} (95% CI {lo:+.4f} .. {hi:+.4f}), P(>0) {np.mean(np.array(deltas) > 0):.3f}")


if __name__ == "__main__":
    main()
