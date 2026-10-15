"""Architecture-family check: torchvision Faster R-CNN (ResNet-50 FPN v2, COCO weights) on the same tiles.

Same data as the YOLO runs: frozen 640 px tiles, upsampled 2x (min/max size 1280, as E1/S1), h+v flips (as S1).
FPN includes the stride-4 level (P2) by default; anchors shrunk to 16-256 px for ~30-40 px upsampled trucks
(3 aspect ratios kept so the pretrained RPN head is reused). Schedule: standard 1x (12 epochs, LR steps at 8, 11).
Prediction: sliced inference over full images with the frozen tile grid, merged with class-wise NMS 0.5
(same as predict.py), written as the CSV that evaluate.py scores.

Usage: python src/frcnn.py train   --tiles /kaggle/tmp_tiles --out runs/FRCNN_s0
       python src/frcnn.py predict --weights runs/FRCNN_s0/last.pt --split val --out experiments/results/FRCNN_s0_val_preds.csv
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import batched_nms
from torchvision.transforms.functional import to_tensor

from data import CLASSES, image_paths
from tile import TILE, starts

Image.MAX_IMAGE_PIXELS = None


def build(pretrained=True):
    m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
                                   min_size=1280, max_size=1280, box_detections_per_img=300, box_score_thresh=0.001)
    m.rpn.anchor_generator = AnchorGenerator(((16,), (32,), (64,), (128,), (256,)), ((0.5, 1.0, 2.0),) * 5)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, len(CLASSES) + 1)
    return m


class Tiles(torch.utils.data.Dataset):
    def __init__(self, root):
        self.ims = sorted(Path(root, "train", "images").glob("*.png"))

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, i):
        p = self.ims[i]
        img = to_tensor(Image.open(p).convert("RGB"))
        h, w = img.shape[1:]
        lp = p.parent.parent / "labels" / f"{p.stem}.txt"
        a = np.loadtxt(lp, ndmin=2) if lp.exists() and lp.stat().st_size else np.zeros((0, 5))
        b = np.stack([(a[:, 1] - a[:, 3] / 2) * w, (a[:, 2] - a[:, 4] / 2) * h,
                      (a[:, 1] + a[:, 3] / 2) * w, (a[:, 2] + a[:, 4] / 2) * h], 1) if len(a) else np.zeros((0, 4))
        b = torch.tensor(b, dtype=torch.float32)
        if random.random() < 0.5:
            img, b = img.flip(-1), torch.stack([w - b[:, 2], b[:, 1], w - b[:, 0], b[:, 3]], 1) if len(b) else b
        if random.random() < 0.5:
            img, b = img.flip(-2), torch.stack([b[:, 0], h - b[:, 3], b[:, 2], h - b[:, 1]], 1) if len(b) else b
        return img, {"boxes": b.reshape(-1, 4), "labels": torch.tensor(a[:, 0] + 1, dtype=torch.int64)}


def train(a):
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ds = Tiles(a.tiles)
    if a.limit:
        ds.ims = ds.ims[:a.limit]
    dl = torch.utils.data.DataLoader(ds, a.bs, shuffle=True, num_workers=a.workers, collate_fn=lambda x: tuple(zip(*x)))
    m = build().to(dev)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    steps = [int(e * len(dl)) for e in (a.epochs * 8 / 12, a.epochs * 11 / 12)]
    warm = min(500, len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, 0.001 + it / warm) * (0.1 ** sum(it >= s for s in steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start, it = 0, 0
    if (out / "ckpt.pt").exists():  # resume after a disconnect: model, optimiser, schedule, scaler, epoch
        ck = torch.load(out / "ckpt.pt", map_location="cpu")
        m.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, it = ck["epoch"], ck["it"]
        print(f"resuming from epoch {start}", flush=True)
    log = open(out / "log.csv", "a")
    for ep in range(start, a.epochs):
        m.train()
        t0, tot = time.time(), 0.0
        for imgs, tg in dl:
            imgs = [i.to(dev) for i in imgs]
            tg = [{k: v.to(dev) for k, v in t.items()} for t in tg]
            with torch.autocast("cuda", enabled=dev == "cuda"):
                loss = sum(m(imgs, tg).values())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, it = tot + loss.item(), it + 1
            if a.max_iters and it >= a.max_iters:
                break
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / max(1, len(dl)):.3f} {time.time() - t0:.0f}s", flush=True)
        log.write(f"{ep + 1},{tot / max(1, len(dl)):.4f},{time.time() - t0:.0f}\n"), log.flush()
        torch.save(m.state_dict(), out / "last.pt")
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep + 1, "it": it}, out / "ckpt.pt")
        if a.max_iters and it >= a.max_iters:
            break


@torch.no_grad()
def predict(a):
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    m = build(pretrained=False)
    m.load_state_dict(torch.load(a.weights, map_location="cpu"))
    m.to(dev).eval()
    paths = image_paths(a.split, a.root)
    if a.limit:
        paths = paths[:a.limit]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "cls", "conf", "x1", "y1", "x2", "y2"])
        for p in paths:
            with Image.open(p) as im:
                im = im.convert("RGB")
                W, H = im.size
                origins = [(x0, y0) for y0 in starts(H) for x0 in starts(W)]
                B, S, C = [], [], []
                for k in range(0, len(origins), a.bs):
                    chunk = origins[k:k + a.bs]
                    crops = [to_tensor(im.crop((x0, y0, min(x0 + TILE, W), min(y0 + TILE, H)))).to(dev) for x0, y0 in chunk]
                    for (x0, y0), r in zip(chunk, m(crops)):
                        B.append(r["boxes"].cpu() + torch.tensor([x0, y0, x0, y0]))
                        S.append(r["scores"].cpu())
                        C.append(r["labels"].cpu() - 1)
            b, s, c = torch.cat(B), torch.cat(S), torch.cat(C)
            k = batched_nms(b, s, c, 0.5)
            for i in k.tolist():
                w.writerow([p.stem, int(c[i]), f"{s[i]:.5f}", *[f"{v:.1f}" for v in b[i].tolist()]])
            print(p.stem, len(k), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--tiles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images/tiles")
    ap.add_argument("--max_iters", type=int, default=0, help="smoke test: stop after N iterations")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else predict(a)
