"""Find which RF-DETR-xView class ids correspond to our 5 trucks, using TRAIN images only.
Tiles 1008x1008 (the model's training crop), matches each GT box to the nearest-centre prediction,
and tabulates GT class x predicted class id."""
import sys, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import rfx_load
from data import image_paths, read_labels, CLASSES
Image.MAX_IMAGE_PIXELS = None
T = 1008
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
model, _, _ = rfx_load.build(device='cpu'); model.to(dev)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

imgs = image_paths('train')
rng = np.random.default_rng(0); pick = rng.choice(len(imgs), int(sys.argv[1]) if len(sys.argv) > 1 else 10, replace=False)
tab = np.zeros((5, 63), int); n_unmatched = np.zeros(5, int)
for i in pick:
    p = imgs[i]; im = np.asarray(Image.open(p).convert('RGB')); H, W = im.shape[:2]
    cls, xyxy = read_labels(p, (W, H))
    if not len(cls): continue
    ctr = (xyxy[:, :2] + xyxy[:, 2:]) / 2
    for y0 in range(0, max(H - T, 0) + 1, T):
        for x0 in range(0, max(W - T, 0) + 1, T):
            inside = (ctr[:, 0] >= x0) & (ctr[:, 0] < x0 + T) & (ctr[:, 1] >= y0) & (ctr[:, 1] < y0 + T)
            if not inside.any(): continue
            crop = np.zeros((T, T, 3), np.uint8); c = im[y0:y0 + T, x0:x0 + T]; crop[:c.shape[0], :c.shape[1]] = c
            x = ((torch.from_numpy(crop).permute(2, 0, 1).float() / 255 - mean) / std)[None].to(dev)
            with torch.no_grad(): out = model(x)
            prob = out['pred_logits'][0].sigmoid().cpu().numpy()          # 300 x 63
            b = out['pred_boxes'][0].cpu().numpy() * T                      # cx cy w h in crop px
            keep = prob.max(1) > 0.05
            for g in np.where(inside)[0]:
                gc = ctr[g] - [x0, y0]
                d = np.hypot(b[:, 0] - gc[0], b[:, 1] - gc[1]); d[~keep] = 1e9
                j = d.argmin()
                if d[j] < 8: tab[cls[g], prob[j].argmax()] += 1
                else: n_unmatched[cls[g]] += 1
    print('done', p.name, flush=True)
np.save(Path(__file__).resolve().parents[1] / 'experiments' / 'results' / 'rfx_classmap_tab.npy', tab)
for k in range(5):
    top = np.argsort(-tab[k])[:4]
    print(f'{CLASSES[k]:16s} matched {tab[k].sum():4d} unmatched {n_unmatched[k]:4d} | top ids', [(int(t), int(tab[k, t])) for t in top])
