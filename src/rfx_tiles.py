"""Cut 1008 px train tiles (stride 880) for RF-DETR fine-tuning. Keeps every tile with >=1 box centre plus
10% of empty tiles (seeded). Boxes kept if >=50% of their area is inside the tile, clipped. Writes PNG +
labels.json {tile: [[cls, x1, y1, x2, y2], ...]} in tile pixels. TRAIN split only."""
import json, sys, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from data import image_paths, read_labels
Image.MAX_IMAGE_PIXELS = None
T, S = 1008, 880
out = Path(sys.argv[1]); (out / 'images').mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0); labels = {}
def starts(n):
    s = list(range(0, max(n - T, 0) + 1, S))
    if s[-1] + T < n: s.append(n - T)
    return s
for p in image_paths('train'):
    im = Image.open(p).convert('RGB'); W, H = im.size
    cls, xyxy = read_labels(p, (W, H)); area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    for y0 in starts(H):
        for x0 in starts(W):
            c = np.clip(xyxy - [x0, y0, x0, y0], 0, T)
            inter = (c[:, 2] - c[:, 0]).clip(0) * (c[:, 3] - c[:, 1]).clip(0)
            k = inter >= 0.5 * area
            if not k.any() and rng.random() > 0.10: continue
            name = f'{p.stem}_{x0}_{y0}.png'
            tile = Image.new('RGB', (T, T)); tile.paste(im.crop((x0, y0, min(x0 + T, W), min(y0 + T, H))))
            tile.save(out / 'images' / name)
            labels[name] = [[int(a), *map(lambda v: round(float(v), 1), b)] for a, b in zip(cls[k], c[k])]
json.dump(labels, open(out / 'labels.json', 'w'))
n = sum(len(v) for v in labels.values())
print(len(labels), 'tiles', n, 'boxes', sum(1 for v in labels.values() if not v), 'empty')
