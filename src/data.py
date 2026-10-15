"""Shared dataset helpers: class names, label reading, image sizes."""
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # val images reach 7072x5932

CLASSES = ["Cargo Truck", "Truck w/Box", "Truck w/Flatbed", "Truck Tractor", "Truck w/Liquid"]
ROOT = Path(__file__).resolve().parents[1] / "CV_Assignment_AAL"


def image_paths(split, root=None):
    return sorted((Path(root or ROOT) / split / "images").glob("*.png"), key=lambda p: int(p.stem))


def label_path(img_path):
    p = Path(img_path)
    return p.parent.parent / "labels" / f"{p.stem}.txt"


def image_size(img_path):
    """(width, height) from the PNG header, without decoding pixels."""
    with Image.open(img_path) as im:
        return im.size


def read_labels(img_path, size=None):
    """Return (cls[N], xyxy[N,4]) in absolute pixels (YOLO normalized xywh on disk)."""
    lp = label_path(img_path)
    arr = np.loadtxt(lp, ndmin=2) if lp.exists() and lp.stat().st_size else np.zeros((0, 5))
    w, h = size or image_size(img_path)
    cls = arr[:, 0].astype(int)
    cx, cy, bw, bh = arr[:, 1] * w, arr[:, 2] * h, arr[:, 3] * w, arr[:, 4] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return cls, xyxy
