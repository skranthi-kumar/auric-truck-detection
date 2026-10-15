"""T3 (declared in LOG.md, 13 Oct): 6-way crop classifier = our 5 classes + "other vehicle" reject class.

Same recipe as src/oracle_cls.py (5.1b): 2x-context square crops, 128 px, flips + rot90, ConvNeXt-T in22k,
AdamW 2e-4 one-cycle, 15 epochs, label smoothing 0.1, seed 0, last epoch kept (no selection).
Classes 0-4 = our GT crops on all train images; class 5 = xView vehicle boxes of unscored classes on
train images whose xView labels align with our GT (list + labels prepared offline), IoU < 0.3 with our GT,
capped at 6,000 (seed 0). Train images only; nothing from val.

Usage: python src/t3_cls.py --xview xview_train_full_labels_clean.csv --aligned train_aligned_ids.txt \
           --out experiments/results/t3_cls.pt
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import box_iou

from data import image_paths, image_size, read_labels
from oracle_cls import CTX, MIN_SIDE, SIZE, augment, crops, to_tensor

OTHER = ["Truck", "Bus", "Trailer", "Utility Truck", "Cargo Car", "Dump Truck", "Pickup Truck",
         "Cement Mixer", "Crane Truck", "Haul Truck"]


def reject_crops(xv, aligned, root, cap, seed):
    tr = {p.stem: p for p in image_paths("train", root)}
    X = xv[xv.image.isin(aligned) & xv.name.isin(OTHER)]
    keep = []
    for im, g in X.groupby("image"):
        _, b = read_labels(tr[im], image_size(tr[im]))
        if len(b):
            v = box_iou(torch.tensor(g[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32),
                        torch.tensor(b, dtype=torch.float32)).max(1).values.numpy()
            g = g[v < 0.3]
        keep.append(g)
    X = pd.concat(keep).sample(frac=1, random_state=seed).head(cap)
    out = []
    for im, g in X.groupby("image"):
        with Image.open(tr[im]) as I:
            I = I.convert("RGB")
            for r in g.itertuples():
                cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
                h = max(CTX * max(r.x2 - r.x1, r.y2 - r.y1), MIN_SIDE) / 2
                out.append(np.asarray(I.crop((cx - h, cy - h, cx + h, cy + h)).resize((SIZE, SIZE), Image.BILINEAR)))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xview", required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--out", default="experiments/results/t3_cls.pt")
    ap.add_argument("--root", default=None)
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    Xtr, ytr, _ = crops("train", a.root)
    xv = pd.read_csv(a.xview, dtype={"image": str})
    aligned = set(Path(a.aligned).read_text().split())
    Xrej = reject_crops(xv, aligned, a.root, a.cap, a.seed)
    Xtr = np.concatenate([Xtr, Xrej]); ytr = np.concatenate([ytr, np.full(len(Xrej), 5)])
    print(f"crops: ours {len(ytr) - len(Xrej)}, reject {len(Xrej)}; device {dev}", flush=True)
    xtr = to_tensor(Xtr)

    model = timm.create_model(a.arch, pretrained=True, num_classes=6).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * (len(ytr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    ytr_t = torch.from_numpy(ytr)
    for ep in range(a.epochs):
        model.train()
        perm, tot = torch.randperm(len(ytr)), 0.0
        for i in range(len(ytr) // a.bs):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            xb = torch.stack([augment(t) for t in xtr[idx]]).to(dev)
            loss = F.cross_entropy(model(xb), ytr_t[idx].to(dev), label_smoothing=0.1)
            opt.zero_grad(), loss.backward(), opt.step(), sched.step()
            tot += loss.item()
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / (len(ytr) // a.bs):.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
