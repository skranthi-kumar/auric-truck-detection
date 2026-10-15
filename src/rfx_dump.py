"""Dump RFX0 (frozen xView RF-DETR) per-query features on TRAIN tiles for the probe (protocol in LOG.md).
Per candidate query (max sigmoid over 63 outputs >= 0.01): tile id, xyxy (tile px), 63 logits, 512-d decoder
feature (input to class_embed, last layer, fp16), label (0-4 class, 5 background, -1 ignored) by IoU with tile GT."""
import argparse, json, sys, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import rfx_load
ap = argparse.ArgumentParser(); ap.add_argument('--tiles', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--thr', type=float, default=0.01)
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
model, _, _ = rfx_load.build(device='cpu'); model.to(dev).eval()
feat = {}
model.class_embed.register_forward_hook(lambda m, i, o: feat.__setitem__('h', i[0]))
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
labels = json.load(open(Path(a.tiles) / 'labels.json')); names = sorted(labels)[:a.limit or None]
T = 1008
def iou(A, B):
    lt = np.maximum(A[:, None, :2], B[None, :, :2]); rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    return inter / ((A[:, 2:] - A[:, :2]).prod(1)[:, None] + (B[:, 2:] - B[:, :2]).prod(1)[None] - inter + 1e-9)
out = {k: [] for k in ['tile', 'box', 'logit', 'h', 'label', 'maxiou']}
for ti, n in enumerate(names):
    x = torch.from_numpy(np.asarray(Image.open(Path(a.tiles) / 'images' / n).convert('RGB')).copy()).permute(2, 0, 1).float() / 255
    with torch.no_grad(): o = model(((x - mean) / std)[None].to(dev))
    lg = o['pred_logits'][0].float().cpu().numpy(); b = o['pred_boxes'][0].float().cpu().numpy() * T
    h = feat['h']; h = h[-1] if h.dim() == 4 else h
    if ti < 3:  # final-layer check: hooked features must reproduce the reported logits
        with torch.no_grad(): err = (model.class_embed(h)[0] - o['pred_logits'][0]).abs().max().item()
        assert err < 1e-3, f'hooked features are not the final-layer queries (err {err})'
    h = h[0].float().cpu().numpy()
    keep = 1 / (1 + np.exp(-lg.max(1))) >= a.thr
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)[keep]
    g = np.array(labels[n], dtype=np.float32).reshape(-1, 5)
    if len(g):
        M = iou(xyxy, g[:, 1:]); mi = M.max(1); cls = g[M.argmax(1), 0].astype(int)
        lab = np.where(mi >= 0.5, cls, np.where(mi < 0.1, 5, -1))
    else:
        mi = np.zeros(len(xyxy)); lab = np.full(len(xyxy), 5)
    out['tile'].append(np.full(len(xyxy), ti)); out['box'].append(xyxy); out['logit'].append(lg[keep])
    out['h'].append(h[keep].astype(np.float16)); out['label'].append(lab); out['maxiou'].append(mi)
    if ti % 200 == 0: print(ti, len(names), flush=True)
np.savez_compressed(a.out, names=np.array(names), **{k: np.concatenate(v) for k, v in out.items()})
lab = np.concatenate(out['label']); print('queries', len(lab), 'per label', np.bincount(lab + 1, minlength=7).tolist())
