"""T2 (declared in LOG.md, 13 Oct): two-stage re-scoring that FUSES detector and crop-classifier scores.

Differs from src/two_stage.py (E1-2stage), which replaced the class score:
1. Detector rows with conf >= 0.01, class-agnostic NMS 0.5 -> kept boxes (as two_stage.py).
2. s_k = max detector conf of class-k rows with IoU >= 0.7 to the kept box (all rows; floor 0.001).
3. p_k = oracle_cls ConvNeXt-T probability (same crop recipe + 8-way TTA as two_stage.py).
4. One row per (box, class) with conf = sqrt(s_k * p_k).

Usage: python src/two_stage_fuse.py --preds experiments/results/RFX0D_val_preds.csv \
           --cls experiments/results/oracle_cls.pt --out experiments/results/T2_val_preds.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision.ops import box_iou, nms

from data import CLASSES, ROOT
from oracle_cls import CTX, MIN_SIDE, SIZE, to_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--cls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k")
    ap.add_argument("--ncls", type=int, default=len(CLASSES), help="6 for the T3 reject-class model (p_k = first 5 softmax entries)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model(a.arch, pretrained=False, num_classes=a.ncls)
    model.load_state_dict(torch.load(a.cls, map_location="cpu"))
    model.to(dev).eval()

    allp = pd.read_csv(a.preds, dtype={"image": str})
    out = []
    for im, A in allp.groupby("image"):
        P = A[A.conf >= 0.01]
        b = torch.tensor(P[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32)
        P = P.iloc[nms(b, torch.tensor(P.conf.values, dtype=torch.float32), 0.5).numpy()]
        kb = torch.tensor(P[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32)

        s = np.full((len(P), len(CLASSES)), 0.001, np.float32)
        for c in range(len(CLASSES)):
            C = A[A.cls == c]
            if not len(C):
                continue
            cb = torch.tensor(C[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32)
            cc = torch.tensor(C.conf.values, dtype=torch.float32)
            for i in range(0, len(kb), 512):
                m = (box_iou(kb[i:i + 512], cb) >= 0.7).float() * cc
                s[i:i + 512, c] = np.maximum(s[i:i + 512, c], m.max(1).values.numpy())

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
                                for i in range(0, len(x), 256)]) for f in (0, 1) for k in range(4)).numpy()[:, :len(CLASSES)] / 8
        conf = np.sqrt(s * pr)
        for j, r in enumerate(P.itertuples()):
            out += [(im, c, float(conf[j, c]), r.x1, r.y1, r.x2, r.y2) for c in range(len(CLASSES))]
        print(im, len(P), "boxes", flush=True)
    pd.DataFrame(out, columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"]).to_csv(a.out, index=False)
    print("wrote", a.out, len(out) // len(CLASSES), "boxes")


if __name__ == "__main__":
    main()
