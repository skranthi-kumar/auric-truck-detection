"""RFX1: fine-tune the xView RF-DETR on the 443 assignment train images (1008 px tiles from rfx_tiles.py).
Keeps the 63-way xView head: our classes are trained as their xView ids (MAP), so no head re-init.
Loss = the checkpoint's own (focal, cls 2, aux). AdamW, lr 5e-5 (encoder 5e-6), cosine, fp16, flips+rot90.
Saves a state_dict at every epoch from --save_from on (for the snapshot ensemble)."""
import argparse, json, math, random, sys, time, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import rfx_load
try:
    import torch_linear_assignment  # noqa: F401  (rfdetr[train] GPU Hungarian solver)
except ImportError:  # fallback: per-problem SciPy Hungarian, same optimum
    from scipy.optimize import linear_sum_assignment
    from rfdetr.models import _assignment
    def _solve_to_indices(stacked, num_matches):
        rows, cols = [], []
        for c in stacked.detach().float().cpu().numpy():
            r, k = linear_sum_assignment(c); o = np.argsort(r)[:num_matches]
            rows.append(r[o]); cols.append(k[o])
        return torch.as_tensor(np.stack(rows), dtype=torch.int64), torch.as_tensor(np.stack(cols), dtype=torch.int64)
    _assignment._solve_to_indices = _solve_to_indices
    print('torch_linear_assignment missing: using SciPy Hungarian fallback', flush=True)

ap = argparse.ArgumentParser()
ap.add_argument('--tiles', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--bs', type=int, default=4)
ap.add_argument('--accum', type=int, default=2); ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--lr_enc', type=float, default=5e-6); ap.add_argument('--save_from', type=int, default=7)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--upscale', type=int, default=1, help='k>1: random 1008/k window (box-biased), nearest-upsampled k x');
ap.add_argument('--freeze_cls', action='store_true', help='freeze class_embed (keep the xView class prior)');
ap.add_argument('--rfs', action='store_true', help='repeat tiles with rare classes: r_c = min(4, sqrt(n_max / n_c)), tile r = max over its classes, stochastic rounding');
ap.add_argument('--init', default=None, help='start from a fine-tuned state_dict (e.g. RFX1) instead of the xView checkpoint'); ap.add_argument('--hours', type=float, default=8.5); ap.add_argument('--max_steps', type=int, default=0, help='smoke test')
a = ap.parse_args()
MAP = [11, 12, 15, 13, 16]   # Cargo, Box, Flatbed, Tractor, Liquid -> xView ids (declared in LOG.md)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
model, _, _ = rfx_load.build(device='cpu')
if a.init: model.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=True))
model.to(dev).train()
crit = rfx_load.criterion().to(dev)
labels = json.load(open(Path(a.tiles) / 'labels.json')); names = sorted(labels)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
T = 1008

def sample(name):
    x = torch.from_numpy(np.asarray(Image.open(Path(a.tiles) / 'images' / name).convert('RGB')).copy()).permute(2, 0, 1).float() / 255
    l = np.array(labels[name], dtype=np.float32).reshape(-1, 5); b = l[:, 1:].copy()
    if a.upscale > 1:  # window of side T/k around a random box (or random), keep boxes >=50% inside, upsample k x
        w = T // a.upscale
        if len(b) and random.random() < 0.9:
            cx, cy = (b[random.randrange(len(b))].reshape(2, 2).mean(0) + np.random.uniform(-w / 2, w / 2, 2))
        else:
            cx, cy = np.random.uniform(w / 2, T - w / 2, 2)
        x0 = int(np.clip(cx - w / 2, 0, T - w)); y0 = int(np.clip(cy - w / 2, 0, T - w))
        c = np.clip(b - [x0, y0, x0, y0], 0, w); area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        k = (c[:, 2] - c[:, 0]).clip(0) * (c[:, 3] - c[:, 1]).clip(0) >= 0.5 * area
        l, b = l[k], c[k] * a.upscale
        x = torch.nn.functional.interpolate(x[:, y0:y0 + w, x0:x0 + w][None], scale_factor=a.upscale, mode='nearest')[0]
    if random.random() < 0.5: x = x.flip(2); b[:, [0, 2]] = T - b[:, [2, 0]]
    if random.random() < 0.5: x = x.flip(1); b[:, [1, 3]] = T - b[:, [3, 1]]
    if random.random() < 0.5:  # rot90 counter-clockwise: (x, y) -> (y, T - x)
        x = torch.rot90(x, 1, (1, 2)); b = np.stack([b[:, 1], T - b[:, 2], b[:, 3], T - b[:, 0]], 1)
    cxcywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) / T
    tgt = {'labels': torch.tensor([MAP[int(c)] for c in l[:, 0]], dtype=torch.long),
           'boxes': torch.tensor(cxcywh, dtype=torch.float32).reshape(-1, 4)}
    return (x - mean) / std, tgt

if a.freeze_cls:
    for n_, p_ in model.named_parameters():
        if 'class_embed' in n_: p_.requires_grad_(False)
    print('frozen:', sum(p_.numel() for n_, p_ in model.named_parameters() if not p_.requires_grad), 'params', flush=True)
base_names = list(names)
if a.rfs:
    cnt = np.bincount([int(b[0]) for n_ in base_names for b in labels[n_]], minlength=5)
    r_c = np.minimum(4.0, np.sqrt(cnt.max() / np.maximum(cnt, 1)))
    tile_r = {n_: max([r_c[int(b[0])] for b in labels[n_]] or [1.0]) for n_ in base_names}
    print('RFS class counts', cnt.tolist(), 'r_c', np.round(r_c, 2).tolist(), flush=True)
def epoch_names():
    if not a.rfs: return list(base_names)
    out = []
    for n_ in base_names:
        r = tile_r[n_]; out += [n_] * (int(r) + (random.random() < r - int(r)))
    return out
enc = [p for n, p in model.named_parameters() if n.startswith('backbone.0.encoder') and p.requires_grad]
rest = [p for n, p in model.named_parameters() if not n.startswith('backbone.0.encoder') and p.requires_grad]
opt = torch.optim.AdamW([{'params': enc, 'lr': a.lr_enc}, {'params': rest, 'lr': a.lr}], weight_decay=1e-4)
steps_per_ep = math.ceil(len(epoch_names()) / (a.bs * a.accum)); total = steps_per_ep * a.epochs
base = [g['lr'] for g in opt.param_groups]
scaler = torch.amp.GradScaler('cuda', enabled=dev == 'cuda')
step = 0; log = open(out / 'log.txt', 'a'); T_START = time.time()
for ep in range(1, a.epochs + 1):
    names = epoch_names(); random.shuffle(names); t0 = time.time(); run = 0.0
    for i in range(0, len(names), a.bs):
        batch = [sample(n) for n in names[i:i + a.bs]]
        x = torch.stack([s[0] for s in batch]).to(dev); tg = [{k: v.to(dev) for k, v in s[1].items()} for s in batch]
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            o = model(x)
        o = {k: (v.float() if torch.is_tensor(v) else v) for k, v in o.items()}
        if 'aux_outputs' in o: o['aux_outputs'] = [{k: v.float() for k, v in d.items()} for d in o['aux_outputs']]
        ld = crit(o, tg); w = crit.weight_dict
        loss = sum(ld[k] * w[k] for k in ld if k in w) / a.accum
        scaler.scale(loss).backward(); run += loss.item() * a.accum
        if (i // a.bs + 1) % a.accum == 0:
            for g, b0 in zip(opt.param_groups, base):  # cosine to 10%
                g['lr'] = b0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            if a.max_steps and step >= a.max_steps: break
    msg = f'epoch {ep} loss {run / max(1, i // a.bs + 1):.4f} time {time.time() - t0:.0f}s'
    print(msg, flush=True); log.write(msg + '\n'); log.flush()
    if ep >= a.save_from or ep == a.epochs or a.max_steps:
        torch.save(model.state_dict(), out / f'rfx1_ep{ep}.pt')
    if a.max_steps and step >= a.max_steps: break
    if time.time() - T_START > a.hours * 3600:
        torch.save(model.state_dict(), out / f'rfx1_ep{ep}.pt'); print('time budget hit', flush=True); break
