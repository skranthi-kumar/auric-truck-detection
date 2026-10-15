"""Class-conditional fusion of two RFX prediction CSVs: keep model A's class labels + scores
(A = RFX0, good discrimination) but take model B's boxes (B = RFX1, better localization) wherever
B overlaps. Unlike symmetric WBF (ENS_RFX), B's collapsed class scores never enter the ranking.

Per image, for each A detection: find the highest-IoU B box (any class). If IoU >= --iou, replace
A's box (mode=replace) or average A+B boxes (mode=wbf) while KEEPING A's class and score; else keep A
as-is. With --add, B boxes that matched no A detection are re-added with the class borrowed from the
nearest A detection centre (recovers B-only recall; off by default). Class-wise NMS merges duplicates.

Usage: python src/rfx_fuse.py --a RFX0_val_preds.csv --b RFX1_ep10_val_preds.csv --out OUT.csv [--iou .3] [--mode replace|wbf] [--add]"""
import argparse, numpy as np, pandas as pd, torch, torchvision

ap = argparse.ArgumentParser()
ap.add_argument('--a', required=True, help='class source (RFX0 preds csv)')
ap.add_argument('--b', required=True, help='box source (RFX1 preds csv)')
ap.add_argument('--out', required=True)
ap.add_argument('--iou', type=float, default=0.3, help='min IoU for an A det to adopt a B box')
ap.add_argument('--mode', default='replace', choices=['replace', 'wbf'])
ap.add_argument('--nms', type=float, default=0.5)
ap.add_argument('--bconf', type=float, default=0.05, help='only use B boxes above this conf as the localization source')
ap.add_argument('--add', action='store_true', help='also add B-only boxes, class borrowed from nearest A det')
a = ap.parse_args()
A = pd.read_csv(a.a); B = pd.read_csv(a.b); B = B[B.conf >= a.bconf]
cols = ['x1', 'y1', 'x2', 'y2']
rows = []
for img in sorted(A.image.unique()):
    da = A[A.image == img]; db = B[B.image == img]
    ba = torch.tensor(da[cols].values, dtype=torch.float32)
    C = torch.tensor(da.cls.values); S = torch.tensor(da.conf.values, dtype=torch.float32)
    out_b = ba.clone()
    matched_b = np.zeros(len(db), bool)
    if len(db):
        bb = torch.tensor(db[cols].values, dtype=torch.float32)
        iou = torchvision.ops.box_iou(ba, bb)                    # (na, nb)
        best = iou.max(1); bj = best.indices; bv = best.values
        m = bv >= a.iou                                          # A dets that adopt a B box (vectorized)
        if m.any():
            sel = bj[m]
            out_b[m] = bb[sel] if a.mode == 'replace' else (ba[m] + bb[sel]) / 2
            matched_b[sel.numpy()] = True
    Boxes = out_b
    if a.add and len(db) and (~matched_b).any():
        bu = bb[~matched_b]                                      # B-only boxes
        su = torch.tensor(db.conf.values[~matched_b], dtype=torch.float32)
        ctr_a = (ba[:, :2] + ba[:, 2:]) / 2; ctr_u = (bu[:, :2] + bu[:, 2:]) / 2
        k = torch.cdist(ctr_u, ctr_a).argmin(1)                  # nearest A det per B-only box
        Boxes = torch.cat([Boxes, bu]); C = torch.cat([C, C[k]]); S = torch.cat([S, su * S[k]])
    keep = torchvision.ops.batched_nms(Boxes, S, C, a.nms)
    for i in keep.tolist():
        rows.append((int(img), int(C[i]), round(float(S[i]), 5), *[round(float(v), 1) for v in Boxes[i]]))
pd.DataFrame(rows, columns=['image', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2']).to_csv(a.out, index=False)
print('wrote', a.out, len(rows), 'rows')
