"""Two-stage variant: detector proposes boxes (class ignored), crop classifier assigns the subtype.

1. Merge the detector's per-class boxes class-agnostically (NMS 0.5), keep det_conf = max class conf.
2. Classify each box crop with the oracle_cls model (same crop recipe + 8-way TTA).
3. Emit one row per (box, class) with conf = det_conf * p(class)  -> scored by evaluate.py.

Usage: python src/two_stage.py --preds experiments/results/E1_up2_s0_val_preds.csv \
           --cls experiments/results/oracle_cls.pt --out experiments/results/E1_2stage_val_preds.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision.ops import nms

from data import CLASSES, ROOT
from oracle_cls import CTX, MIN_SIDE, SIZE, to_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--cls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k")
    ap.add_argument("--min_conf", type=float, default=0.01)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model(a.arch, pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(a.cls, map_location="cpu"))
    model.to(dev).eval()

    p = pd.read_csv(a.preds, dtype={"image": str})
    p = p[p.conf >= a.min_conf]
    rows = []
    for im, P in p.groupby("image"):
        b = torch.tensor(P[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32)
        keep = nms(b, torch.tensor(P.conf.values, dtype=torch.float32), 0.5).numpy()
        P = P.iloc[keep]
        with Image.open(Path(a.root or ROOT) / "val" / "images" / f"{im}.png") as I:
            I = I.convert("RGB")
            X = []
            for r in P.itertuples():
                cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
                h = max(CTX * max(r.x2 - r.x1, r.y2 - r.y1), MIN_SIDE) / 2
                X.append(np.asarray(I.crop((cx - h, cy - h, cx + h, cy + h)).resize((SIZE, SIZE), Image.BILINEAR)))
        x = to_tensor(np.stack(X))
        with torch.no_grad():
            pr = sum(torch.cat([model(torch.rot90(x.flip(-1) if f else x, k, (-2, -1))[i:i + 256].to(dev)).softmax(1).cpu()
                                for i in range(0, len(x), 256)]) for f in (0, 1) for k in range(4)).numpy() / 8
        for r, q in zip(P.itertuples(), pr):
            rows += [(im, c, r.conf * q[c], r.x1, r.y1, r.x2, r.y2) for c in range(len(CLASSES))]
    pd.DataFrame(rows, columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"]).to_csv(a.out, index=False)
    print("wrote", a.out, len(rows) // len(CLASSES), "boxes")


if __name__ == "__main__":
    main()
