"""S1-gate: 5.1 oracle-box protocol with frozen DINOv3 SAT-493M ViT-L features + logistic regression (see LOG.md).
Crops are exactly src/oracle_cls.py crops(); feature = [CLS, mean patch tokens], averaged over 8-way flip x rot90."""
import argparse, json, sys, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from oracle_cls import crops
from data import CLASSES
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from transformers import AutoModel
ap = argparse.ArgumentParser(); ap.add_argument('--root', default=None); ap.add_argument('--out', required=True)
ap.add_argument('--model', default='facebook/dinov3-vitl16-pretrain-sat493m'); ap.add_argument('--crops', default=None, help='npz from --save_crops (Kaggle)'); ap.add_argument('--save_crops', default=None); ap.add_argument('--bs', type=int, default=32)
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
m = None if a.save_crops else AutoModel.from_pretrained(a.model).to(dev).eval()
mean = torch.tensor([0.430, 0.411, 0.296]).view(1, 3, 1, 1); std = torch.tensor([0.213, 0.156, 0.143]).view(1, 3, 1, 1)
try:
    assert m is not None
    from transformers import AutoImageProcessor
    pp = AutoImageProcessor.from_pretrained(a.model); mean = torch.tensor(pp.image_mean).view(1, 3, 1, 1); std = torch.tensor(pp.image_std).view(1, 3, 1, 1)
except Exception as e: print('processor not loaded, using SAT defaults', e)
print('norm mean', mean.flatten().tolist(), 'std', std.flatten().tolist(), flush=True)
nreg = getattr(m.config, 'num_register_tokens', 4) if m is not None else 4
def feats(X):
    out = []
    for i in range(0, len(X), a.bs):
        x = torch.from_numpy(X[i:i + a.bs]).permute(0, 3, 1, 2).float() / 255
        x = torch.nn.functional.interpolate(x, size=224, mode='bilinear', align_corners=False)
        x = ((x - mean) / std)
        acc = 0
        for k in range(4):
            for fl in (False, True):
                xx = torch.rot90(x, k, (2, 3)); xx = xx.flip(3) if fl else xx
                with torch.no_grad(): h = m(pixel_values=xx.to(dev)).last_hidden_state.float().cpu()
                acc = acc + torch.cat([h[:, 0], h[:, 1 + nreg:].mean(1)], 1)
        out.append((acc / 8).numpy()); print(i, len(X), flush=True) if i % (a.bs * 50) == 0 else None
    return np.concatenate(out)
if a.crops:
    z = np.load(a.crops, allow_pickle=True); Xtr, ytr, Xva, yva = z['Xtr'], z['ytr'], z['Xva'], z['yva']; mtr, mva = list(z['mtr']), list(z['mva'])
else:
    Xtr, ytr, mtr = crops('train', a.root); Xva, yva, mva = crops('val', a.root)
if a.save_crops:
    np.savez(a.save_crops, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, mtr=np.array(mtr, dtype=object), mva=np.array(mva, dtype=object)); print('saved crops'); sys.exit()
print('crops', len(ytr), len(yva), 'device', dev, flush=True)
Ftr, Fva = feats(Xtr), feats(Xva)
np.savez(a.out + '_feats.npz', Ftr=Ftr, ytr=ytr, gtr=np.array([t[0] for t in mtr]), Fva=Fva, yva=yva)
mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6; Ztr, Zva = (Ftr - mu) / sd, (Fva - mu) / sd
g = np.array([t[0] for t in mtr]); cv = {}
for C in [0.01, 0.1, 1, 10]:
    oof = np.zeros((len(ytr), 5))
    for tr, te in GroupKFold(5).split(Ztr, ytr, g):
        oof[te] = LogisticRegression(C=C, max_iter=3000).fit(Ztr[tr], ytr[tr]).predict_proba(Ztr[te])
    cv[C] = float(np.mean([average_precision_score(ytr == k, oof[:, k]) for k in range(5)])); print('C', C, 'OOF macro AP', round(cv[C], 4), flush=True)
Cb = max(cv, key=cv.get); clf = LogisticRegression(C=Cb, max_iter=3000).fit(Ztr, ytr); P = clf.predict_proba(Zva)
pred = P.argmax(1); print('chosen C', Cb, 'val accuracy', round(float(np.mean(pred == yva)), 3), 'macro-recall', round(float(np.mean([np.mean(pred[yva == c] == c) for c in range(5)])), 3))
print(pd.crosstab(pd.Series(yva, name='GT').map(dict(enumerate(CLASSES))), pd.Series(pred, name='pred').map(dict(enumerate(CLASSES)))))
rows = [(im, c, P[i, c], x1, y1, x2, y2) for i, (im, x1, y1, x2, y2) in enumerate(mva) for c in range(5)]
pd.DataFrame(rows, columns=['image', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2']).to_csv(a.out + '_val_preds.csv', index=False)
json.dump({'C': Cb, 'cv': {str(k): v for k, v in cv.items()}}, open(a.out + '_cv.json', 'w')); print('wrote', a.out + '_val_preds.csv')
