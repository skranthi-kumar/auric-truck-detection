"""5.1b: how much of the 5-class problem remains if object locations were perfect?

Train a crop classifier on train GT boxes, apply it to val GT boxes, and score it two ways:
  - per-class accuracy / confusion (classification only)
  - "oracle-box mAP50": every val GT box is emitted once per class with the classifier probability
    as confidence, then scored by evaluate.py (localisation error = 0, detection misses = 0).
Fixed schedule, no model selection on val.

Crop: square, side = max(2 * max(w, h), 48 px) centred on the box (context included), resized to 128.
--down k (resolution test): each crop is first area-averaged to 1/k of its native side (simulates a k x coarser
ground sample distance), then resized to 128 as usual; train and val get the same degradation.

Usage: python src/oracle_cls.py --root /content/CV_Assignment_AAL --out experiments/results/oracle_cls
       python src/evaluate.py --preds experiments/results/oracle_cls_val_preds.csv --name oracle_cls
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

from data import CLASSES, image_paths, image_size, read_labels

SIZE, CTX, MIN_SIDE = 128, 2.0, 48


def crops(split, root, keep=None, down=1, ctx=None):
    ctx = CTX if ctx is None else ctx
    X, y, meta = [], [], []
    for p in image_paths(split, root):
        if keep is not None and not keep(p.stem):
            continue
        cls, b = read_labels(p, image_size(p))
        if not len(b):
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            for c, (x1, y1, x2, y2) in zip(cls, b):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                h = max(ctx * max(x2 - x1, y2 - y1), MIN_SIDE) / 2
                cr = im.crop((cx - h, cy - h, cx + h, cy + h))
                if down > 1:
                    side = max(4, round(2 * h / down))
                    cr = cr.resize((side, side), Image.BOX)
                X.append(np.asarray(cr.resize((SIZE, SIZE), Image.BILINEAR)))
                y.append(c)
                meta.append((p.stem, x1, y1, x2, y2))
    return np.stack(X), np.array(y), meta


def augment(x):
    """Overhead imagery has no canonical orientation: random flips + 90-degree rotations."""
    if random.random() < 0.5:
        x = x.flip(-1)
    if random.random() < 0.5:
        x = x.flip(-2)
    return torch.rot90(x, random.randint(0, 3), (-2, -1))


def to_tensor(X):
    x = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255
    m, s = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (x - m) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="experiments/results/oracle_cls")
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_crops", default=None, help="extract train/val crops to this .npz and exit (for remote GPUs)")
    ap.add_argument("--crops", default=None, help="load crops from a --save_crops .npz instead of the images")
    ap.add_argument("--down", type=int, default=1, help="resolution test: area-downsample crops by this factor")
    ap.add_argument("--holdout", default=None, help="txt of train image stems held out of training and scored")
    ap.add_argument("--save_holdout", default=None, help="npz for holdout probabilities (y, P) in data.read_labels order")
    ap.add_argument("--no_val", action="store_true", help="do not crop, predict or score val (held-out benchmark only)")
    ap.add_argument("--ctx", type=float, default=CTX, help="crop side = max(ctx x box long side, 48 px); 2.0 = default context")
    a = ap.parse_args()
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    held = set(Path(a.holdout).read_text().split()) if a.holdout else set()
    if a.crops:
        z = np.load(a.crops, allow_pickle=True)
        assert int(z["down"]) == a.down and not held, "crop file built with another --down (or holdout requested)"
        Xtr, ytr, Xva, yva = z["Xtr"], z["ytr"], z["Xva"], z["yva"]
        mva = [tuple(m) for m in z["mva"]]
    else:
        Xtr, ytr, _ = crops("train", a.root, lambda s: s not in held, a.down, a.ctx)
        Xva, yva, mva = (np.zeros((0, SIZE, SIZE, 3), np.uint8), np.zeros(0, int), []) if a.no_val else crops("val", a.root, down=a.down, ctx=a.ctx)
    if a.save_crops:
        np.savez_compressed(a.save_crops, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, mva=np.array(mva, dtype=object), down=a.down)
        print("saved", a.save_crops, len(ytr), len(yva))
        return
    if held:  # diagnostic: same-distribution held-out train images vs val
        Xho, yho, _ = crops("train", a.root, lambda s: s in held, ctx=a.ctx)
    print(f"crops: train {len(ytr)}, val {len(yva)}; device {dev}", flush=True)
    xtr, xva = to_tensor(Xtr), to_tensor(Xva)

    kw = dict(img_size=SIZE, dynamic_img_size=True, dynamic_img_pad=True) if "vit" in a.arch else {}  # ViT (e.g. DINOv2, patch 14) at 128 px
    model = timm.create_model(a.arch, pretrained=True, num_classes=len(CLASSES), **kw).to(dev)
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

    model.eval()

    def predict(x):  # 8-way test-time augmentation (flips x rotations), averaged
        P = 0
        with torch.no_grad():
            for flip in (False, True):
                for k in range(4):
                    xs = torch.rot90(x.flip(-1) if flip else x, k, (-2, -1))
                    P = P + torch.cat([model(xs[i:i + 256].to(dev)).softmax(1).cpu() for i in range(0, len(xs), 256)])
        return (P / 8).numpy()

    if held:
        Ph = predict(to_tensor(Xho))
        ph = Ph.argmax(1)
        if a.save_holdout:
            np.savez(a.save_holdout, y=yho, P=Ph)
            torch.save(model.state_dict(), Path(a.save_holdout).with_suffix(".pt"))
        print(f"HOLDOUT (train distribution, {len(yho)} crops): accuracy {np.mean(ph == yho):.3f}  macro-recall "
              f"{np.mean([np.mean(ph[yho == c] == c) for c in range(len(CLASSES)) if (yho == c).any()]):.3f}", flush=True)
    if a.no_val:
        return
    P = predict(xva)
    pred = P.argmax(1)
    cm = pd.crosstab(pd.Series(yva, name="GT").map(dict(enumerate(CLASSES))),
                     pd.Series(pred, name="pred").map(dict(enumerate(CLASSES))))
    print(f"\naccuracy {np.mean(pred == yva):.3f}  macro-recall "
          f"{np.mean([np.mean(pred[yva == c] == c) for c in range(len(CLASSES))]):.3f}\n{cm}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [(im, c, P[i, c], x1, y1, x2, y2) for i, (im, x1, y1, x2, y2) in enumerate(mva) for c in range(len(CLASSES))]
    pd.DataFrame(rows, columns=["image", "cls", "conf", "x1", "y1", "x2", "y2"]).to_csv(f"{out}_val_preds.csv", index=False)
    cm.to_csv(f"{out}_confusion.csv")
    torch.save(model.state_dict(), f"{out}.pt")
    print("wrote", f"{out}_val_preds.csv")


if __name__ == "__main__":
    main()
