"""Fit the declared probe on TRAIN-tile RFX0 features (protocol in experiments/LOG.md, 'PROBE protocol').
6-way multinomial LR (5 classes + background) on standardised features; C from {0.01,0.1,1,10} by 5-fold
GroupKFold over source images, criterion = mean one-vs-rest out-of-fold AP over the 5 classes; refit on all train.
Saves mu, sd, W (6 x d), b and the CV table to an .npz used by rfx_predict.py --probe."""
import argparse, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
ap = argparse.ArgumentParser(); ap.add_argument('--dump', required=True); ap.add_argument('--feat', choices=['logit', 'h'], default='logit')
ap.add_argument('--out', required=True); ap.add_argument('--bg_ratio', type=float, default=0, help='P2: subsample background to this x positives (seed 0)')
a = ap.parse_args()
d = np.load(a.dump)
y = d['label']; keep = y >= 0
X = d[a.feat][keep].astype(np.float32); y = y[keep]
groups = np.array([n.split('_')[0] for n in d['names']])[d['tile'][keep]]
if a.bg_ratio:
    rng = np.random.default_rng(0); pos = np.where(y < 5)[0]; bg = np.where(y == 5)[0]
    sel = np.concatenate([pos, rng.choice(bg, min(len(bg), int(a.bg_ratio * len(pos))), replace=False)])
    X, y, groups = X[sel], y[sel], groups[sel]
print('samples', len(y), 'per class', np.bincount(y, minlength=6).tolist(), 'images', len(set(groups)), flush=True)
mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
cv = {}
for C in [0.01, 0.1, 1, 10]:
    oof = np.zeros((len(y), 6))
    for tr, te in GroupKFold(5).split(Xs, y, groups):
        m = LogisticRegression(C=C, max_iter=2000).fit(Xs[tr], y[tr])
        p = np.zeros((len(te), 6)); p[:, m.classes_] = m.predict_proba(Xs[te]); oof[te] = p
    aps = [average_precision_score(y == k, oof[:, k]) for k in range(5)]
    cv[C] = (float(np.mean(aps)), [round(float(v), 4) for v in aps]); print('C', C, 'mean OOF AP', round(cv[C][0], 4), cv[C][1], flush=True)
Cb = max(cv, key=lambda c: cv[c][0]); print('chosen C', Cb, flush=True)
if a.feat == 'logit' and not a.bg_ratio:  # declared OOF probe-vs-raw check (train only, indicative)
    raw = 1 / (1 + np.exp(-X[:, [11, 12, 15, 13, 16]]))
    raw_ap = [round(float(average_precision_score(y == k, raw[:, k])), 4) for k in range(5)]
    print('RAW name-mapped AP per class', raw_ap, 'mean', round(float(np.mean(raw_ap)), 4))
    print('PROBE OOF AP per class (chosen C)', cv[Cb][1], 'mean', round(cv[Cb][0], 4), flush=True)
m = LogisticRegression(C=Cb, max_iter=2000).fit(Xs, y)
W = np.zeros((6, Xs.shape[1])); b = np.full(6, -1e4); W[m.classes_] = m.coef_; b[m.classes_] = m.intercept_
np.savez(a.out, mu=mu, sd=sd, W=W, b=b, feat=a.feat, C=Cb, cv=json.dumps({str(k): v for k, v in cv.items()}))
print('saved', a.out)
