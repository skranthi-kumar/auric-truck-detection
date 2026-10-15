"""RF-DETR(xView) sliding-window inference -> predictions CSV in the frozen evaluate.py format.
Tiles 1008 px with overlap; each query emits one row per mapped truck class (multi-label);
merged across tiles by class-wise NMS."""
import argparse, sys, numpy as np, pandas as pd, torch, torchvision
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import rfx_load
from data import image_paths
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument('--split', default='val'); ap.add_argument('--root', default=None); ap.add_argument('--out', required=True)
ap.add_argument('--map', default='11,12,15,13,16', help='model class id for Cargo,Box,Flatbed,Tractor,Liquid')
ap.add_argument('--tile', type=int, default=1008); ap.add_argument('--overlap', type=int, default=128)
ap.add_argument('--conf', type=float, default=0.001); ap.add_argument('--nms', type=float, default=0.5)
ap.add_argument('--scale', type=float, default=1.0, help='upsample (nearest) before tiling; boxes mapped back');
ap.add_argument('--flip', choices=['none', 'h', 'v', 'hv'], default='none', help='test-time flip of each tile (boxes un-flipped)');
ap.add_argument('--res', type=int, default=0, help='model input size: resize each tile to res x res before the model (e.g. 504 for the 504 checkpoint); 0 = tile size');
ap.add_argument('--transpose', action='store_true', help='transpose each tile before the flip (with the 4 flips: the 4 rotation views; boxes mapped back)');
ap.add_argument('--probe', default=None, help='probe .npz from rfx_probe.py: class scores = 6-way probe softmax (declared rule)');
ap.add_argument('--weights', default=None, help='optional fine-tuned state_dict (else xView checkpoint)')
a = ap.parse_args()
ids = [int(x) for x in a.map.split(',')]
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
model, _, _ = rfx_load.build(device='cpu')
if a.weights: model.load_state_dict(torch.load(a.weights, map_location='cpu', weights_only=True), strict=False)
model.to(dev).eval()
PR = None
if a.probe:
    PR = np.load(a.probe); feat = {}
    model.class_embed.register_forward_hook(lambda m, i, o: feat.__setitem__('h', i[0]))
    PW, Pb = torch.tensor(PR['W'], dtype=torch.float32), torch.tensor(PR['b'], dtype=torch.float32)
    Pmu, Psd = torch.tensor(PR['mu'], dtype=torch.float32), torch.tensor(PR['sd'], dtype=torch.float32)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
T, S = a.tile, a.tile - a.overlap

def starts(n):
    s = list(range(0, max(n - T, 0) + 1, S))
    if s[-1] + T < n: s.append(n - T)
    return s

rows = []
for p in image_paths(a.split, a.root):
    img = Image.open(p).convert('RGB'); W0, H0 = img.size
    if a.scale != 1: img = img.resize((round(W0 * a.scale), round(H0 * a.scale)), Image.NEAREST if a.scale > 1 else Image.BOX)  # BOX for downsampling (SC1)
    im = np.asarray(img); H, W = im.shape[:2]
    B, C, Sc = [], [], []
    for y0 in starts(H):
        for x0 in starts(W):
            crop = np.zeros((T, T, 3), np.uint8); c = im[y0:y0 + T, x0:x0 + T]; crop[:c.shape[0], :c.shape[1]] = c
            x = ((torch.from_numpy(crop).permute(2, 0, 1).float() / 255 - mean) / std)[None]
            if a.transpose: x = x.transpose(2, 3)
            if 'h' in a.flip: x = x.flip(3)
            if 'v' in a.flip: x = x.flip(2)
            if a.res and a.res != T: x = torch.nn.functional.interpolate(x, size=(a.res, a.res), mode='bilinear', antialias=True)  # boxes are normalised
            x = x.to(dev)
            with torch.no_grad(): out = model(x)
            if PR is None:
                prob = out['pred_logits'][0].sigmoid()[:, ids].cpu()            # 300 x 5
            else:
                h = feat['h']; h = (h[-1] if h.dim() == 4 else h)
                with torch.no_grad():  # final-layer check
                    assert (model.class_embed(h)[0] - out['pred_logits'][0]).abs().max().item() < 1e-3
                h = h[0]
                f = out['pred_logits'][0] if str(PR['feat']) == 'logit' else h
                prob = torch.softmax(((f.float().cpu() - Pmu) / Psd) @ PW.T + Pb, 1)[:, :5]
            b = out['pred_boxes'][0].cpu() * T
            if 'h' in a.flip: b[:, 0] = T - b[:, 0]   # un-flip centre x
            if 'v' in a.flip: b[:, 1] = T - b[:, 1]   # un-flip centre y
            if a.transpose: b = b[:, [1, 0, 3, 2]]    # un-transpose: swap (cx, cy) and (w, h)
            xyxy = torch.stack([b[:, 0] - b[:, 2] / 2 + x0, b[:, 1] - b[:, 3] / 2 + y0,
                                b[:, 0] + b[:, 2] / 2 + x0, b[:, 1] + b[:, 3] / 2 + y0], 1)
            q, k = torch.nonzero(prob > a.conf, as_tuple=True)
            B.append(xyxy[q]); C.append(k); Sc.append(prob[q, k])
    B, C, Sc = torch.cat(B), torch.cat(C), torch.cat(Sc)
    B[:, [0, 2]] = B[:, [0, 2]].clamp(0, W); B[:, [1, 3]] = B[:, [1, 3]].clamp(0, H)
    keep = torchvision.ops.batched_nms(B, Sc, C, a.nms)
    B = B / a.scale
    for i in keep.tolist():
        rows.append((int(p.stem), int(C[i]), round(float(Sc[i]), 5), *[round(float(v), 1) for v in B[i]]))
    print(p.name, len(keep), flush=True)
pd.DataFrame(rows, columns=['image', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2']).to_csv(a.out, index=False)
