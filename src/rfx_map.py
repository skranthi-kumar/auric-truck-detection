"""Apply a class-id mapping to a cached dump (src/rfx_dump.py) and write a predictions CSV in the
frozen evaluate.py format. Supports multi-id MAX fusion per class: a class channel = max sigmoid over
its id set. --map "6+10,12,15,13,16" means Cargo = max(id6,id10), others single-id.

Replicates rfx_predict.py exactly for single-id maps (conf floor, class-wise NMS, /scale), so
--map 11,12,15,13,16 on the RFX0 dump reproduces RFX0 (0.195)."""
import argparse, numpy as np, pandas as pd, torch, torchvision

ap = argparse.ArgumentParser()
ap.add_argument('--dump', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--map', default='11,12,15,13,16', help='id set per class Cargo,Box,Flatbed,Tractor,Liquid; "+" fuses by max')
ap.add_argument('--suppress', default='', help='competitor ids, e.g. "5,6,7": s_k -> s_k*s_k/(s_k+max competitor sigmoid)')
ap.add_argument('--conf', type=float, default=0.001); ap.add_argument('--nms', type=float, default=0.5)
a = ap.parse_args()
sets = [[int(x) for x in grp.split('+')] for grp in a.map.split(',')]   # 5 lists of ids
d = np.load(a.dump); box, prob, img, ids, W, H = d['box'], d['prob'].astype(np.float32), d['img'], d['ids'], d['W'], d['H']
scale = float(d['scale'])
rows = []
for k, iid in enumerate(ids):
    m = img == k
    b = torch.from_numpy(box[m]); pr = prob[m]                              # (n,4),(n,63)
    chan = np.stack([pr[:, s].max(1) for s in sets], 1)                     # (n,5) fused per class
    if a.suppress:                                                          # competing-class suppression (CS1)
        mc = pr[:, [int(x) for x in a.suppress.split(',')]].max(1, keepdims=True)
        chan = chan * chan / (chan + mc + 1e-9)
    bx = b.clone(); bx[:, [0, 2]] = bx[:, [0, 2]].clamp(0, int(W[k])); bx[:, [1, 3]] = bx[:, [1, 3]].clamp(0, int(H[k]))
    q, c = np.nonzero(chan > a.conf)                                        # q outer, c inner (= rfx_predict order)
    if not len(q): continue
    B = bx[q]; C = torch.from_numpy(c); Sc = torch.from_numpy(chan[q, c])
    keep = torchvision.ops.batched_nms(B, Sc, C, a.nms)
    B = B / scale
    for i in keep.tolist():
        rows.append((int(iid), int(C[i]), round(float(Sc[i]), 5), *[round(float(v), 1) for v in B[i]]))
pd.DataFrame(rows, columns=['image', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2']).to_csv(a.out, index=False)
print('wrote', a.out, len(rows), 'rows')
