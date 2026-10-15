"""Sliced inference on full-resolution images -> one predictions CSV in full-image coordinates.

Same tiling as training (tile.py). Per tile: YOLO predict. Boxes touching an interior tile edge
are dropped (every object <= ~65 px fits whole in some tile because overlap = 128 px), then
class-wise NMS merges the overlaps.

Usage: python src/predict.py --weights weights/final.pt --split val --out experiments/results/final_val_preds.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms
from ultralytics import YOLO

from data import image_paths
from tile import TILE, starts

EDGE = 2  # px


MAX_DET = 300  # per tile; default 1000 with --multi_label (one row per box and class); --max_det overrides
CAP_HITS = [0, 0]  # tiles whose output reached MAX_DET, tiles total (a reached cap can truncate low scores)


def predict_image(model, p, conf, imgsz, batch, device, tta=False):
    """Sliced inference; tta=True adds h/v/hv-flipped passes per tile (overhead imagery has no canonical
    orientation), un-flips the boxes and merges everything in the final class-wise NMS."""
    flips = [(0, 0), (1, 0), (0, 1), (1, 1)] if tta else [(0, 0)]
    with Image.open(p) as im:
        im = im.convert("RGB")
        W, H = im.size
        origins = [(x0, y0) for y0 in starts(H) for x0 in starts(W)]
        boxes, scores, classes = [], [], []
        for i in range(0, len(origins), batch):
            chunk = origins[i:i + batch]
            crops = [im.crop((x0, y0, min(x0 + TILE, W), min(y0 + TILE, H))) for x0, y0 in chunk]
            inputs = [c.transpose(Image.FLIP_LEFT_RIGHT) if fx else c for fx, fy in flips for c in crops]
            inputs = [c.transpose(Image.FLIP_TOP_BOTTOM) if flips[j // len(crops)][1] else c for j, c in enumerate(inputs)]
            res = model.predict(inputs, imgsz=imgsz, conf=conf, iou=0.6, max_det=MAX_DET,
                                device=device, verbose=False)
            for j, r in enumerate(res):
                CAP_HITS[0] += len(r.boxes) >= MAX_DET
                CAP_HITS[1] += 1
                (fx, fy), (x0, y0) = flips[j // len(crops)], chunk[j % len(crops)]
                b = r.boxes.xyxy.cpu().numpy()
                tw, th = min(TILE, W - x0), min(TILE, H - y0)
                if fx:
                    b[:, [0, 2]] = tw - b[:, [2, 0]]
                if fy:
                    b[:, [1, 3]] = th - b[:, [3, 1]]
                # drop boxes cut by an interior tile edge (image edges are kept)
                cut = ((b[:, 0] < EDGE) & (x0 > 0)) | ((b[:, 1] < EDGE) & (y0 > 0)) | \
                      ((b[:, 2] > tw - EDGE) & (x0 + tw < W)) | ((b[:, 3] > th - EDGE) & (y0 + th < H))
                keep = ~cut
                boxes.append(b[keep] + [x0, y0, x0, y0])
                scores.append(r.boxes.conf.cpu().numpy()[keep])
                classes.append(r.boxes.cls.cpu().numpy()[keep].astype(int))
    b, s, c = (torch.from_numpy(np.concatenate(x)) for x in (boxes, scores, classes))
    k = batched_nms(b.float(), s.float(), c, 0.5).numpy()
    return b.numpy()[k], s.numpy()[k], c.numpy()[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", default=None)
    ap.add_argument("--images", default=None, help="optional txt of image stems to restrict to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--tta", action="store_true", help="h/v/hv flip test-time augmentation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_det", type=int, default=None, help="per-tile cap (default 300; 1000 with --multi_label)")
    ap.add_argument("--multi_label", action="store_true",
                    help="emit every class above --conf per box (NMS multi_label), max_det 1000 per tile")
    args = ap.parse_args()
    if args.multi_label:  # the Ultralytics predictor does not expose multi_label; set it on its NMS call
        import functools
        from ultralytics.utils import nms as unms
        unms.non_max_suppression = functools.partial(unms.non_max_suppression, multi_label=True)
    global MAX_DET
    MAX_DET = args.max_det or (1000 if args.multi_label else 300)

    model = YOLO(args.weights)
    paths = image_paths(args.split, args.root) if args.root else image_paths(args.split)
    if args.images:
        keep = set(Path(args.images).read_text().split())
        paths = [p for p in paths if p.stem in keep]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "cls", "conf", "x1", "y1", "x2", "y2"])
        for p in paths:
            b, s, c = predict_image(model, p, args.conf, args.imgsz, args.batch, args.device, args.tta)
            w.writerows([p.stem, int(ci), f"{si:.5f}", *np.round(bi, 1)] for bi, si, ci in zip(b, s, c))
    print("wrote", args.out, f"| max_det {MAX_DET}: {CAP_HITS[0]}/{CAP_HITS[1]} tile passes reached the cap")


if __name__ == "__main__":
    main()
