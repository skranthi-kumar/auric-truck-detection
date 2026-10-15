"""Blind visual audits (H5 missing labels; H2 vs H3 label noise vs ambiguity) as one self-contained HTML page.

Part R  reference: 6 labelled train examples per class (cartography 'easy'/'medium'), labels shown.
Part A  H5: top-50 highest-confidence E1 background FPs on val (class-agnostic NMS 0.3 among them)
        + 15 random val GT boxes + 15 random detections (conf>=0.25) as controls; shuffled; no class or source shown.
Part B  H2/H3: 30 'hard-confused' train boxes + 30 controls with the same class counts from
        'easy'/'medium'; shuffled; no label shown.
Answers are saved from the page as audit_answers.csv; the key (audit_key.csv) is joined afterwards.

Usage: python src/audit.py build  [--root DIR]
       python src/audit.py score --answers experiments/audit/audit_answers.csv
"""
import argparse
import base64
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms

from data import CLASSES, ROOT, read_labels

OUT = Path(__file__).resolve().parents[1] / "experiments" / "audit"
RES = Path(__file__).resolve().parents[1] / "experiments" / "results"
SEED, SHOW = 0, 192
SHORT = ["Cargo", "Box", "Flatbed", "Tractor", "Liquid"]


def crop(root, split, image, box):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = max(4 * max(x2 - x1, y2 - y1), 64) / 2
    with Image.open(Path(root) / split / "images" / f"{image}.png") as im:
        c = im.convert("RGB").crop((cx - h, cy - h, cx + h, cy + h))
    s = SHOW / (2 * h)
    c = c.resize((SHOW, SHOW), Image.LANCZOS)
    ImageDraw.Draw(c).rectangle([(x1 - cx + h) * s - 2, (y1 - cy + h) * s - 2,
                                 (x2 - cx + h) * s + 2, (y2 - cy + h) * s + 2], outline=(255, 0, 255), width=1)
    buf = io.BytesIO()
    c.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def train_boxes():
    pop = pd.read_csv(RES / "cart_E0_s0_populations.csv", dtype={"image": str})
    boxes = {}
    for im in pop.image.unique():
        _, b = read_labels(ROOT / "train" / "images" / f"{im}.png")
        boxes[im] = b
    pop["box"] = [boxes[r.image][r.idx] for r in pop.itertuples()]
    return pop[~pop.held]  # holdout images stay untouched


def build(a):
    rng = random.Random(SEED)
    root = a.root or ROOT
    OUT.mkdir(parents=True, exist_ok=True)
    pop = train_boxes()
    items = []  # (part, source, image, box, gt_cls, extra)

    for c in range(len(CLASSES)):
        P = pop[(pop.cls == c) & pop.region.isin(["easy", "medium"])]
        for r in P.sample(min(6, len(P)), random_state=SEED).itertuples():
            items.append(("R", "train_ref", r.image, r.box, c, ""))

    m = pd.read_csv(RES / "E1_up2_s0_matches.csv", dtype={"image": str})
    bg = m[m.err == "bg"].sort_values("conf", ascending=False)
    keep = []
    for im, G in bg.groupby("image"):
        b = torch.tensor(G[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32)
        keep += list(G.index[nms(b, torch.tensor(G.conf.values, dtype=torch.float32), 0.3).numpy()])
    bg = bg.loc[keep].sort_values("conf", ascending=False).head(50)
    for r in bg.itertuples():
        items.append(("A", "fp_bg", r.image, (r.x1, r.y1, r.x2, r.y2), -1, f"pred={SHORT[r.cls]} conf={r.conf:.2f}"))
    gt = pd.read_csv(RES / "E1_up2_s0_gt.csv", dtype={"image": str}).sample(15, random_state=SEED)
    for r in gt.itertuples():
        items.append(("A", "gt_control", r.image, (r.x1, r.y1, r.x2, r.y2), r.cls, ""))
    rnd = m[m.conf >= 0.25].sample(15, random_state=SEED)  # base rate: any confident detection
    for r in rnd.itertuples():
        items.append(("A", "pred_random", r.image, (r.x1, r.y1, r.x2, r.y2), -1, f"pred={SHORT[r.cls]} conf={r.conf:.2f} tp={r.tp}"))

    hc = pop[pop.region == "hard-confused"].sample(30, random_state=SEED)
    ctrl_pool = pop[pop.region.isin(["easy", "medium"])]
    ctrl = pd.concat([ctrl_pool[ctrl_pool.cls == c].sample(n, random_state=SEED)
                      for c, n in hc.cls.value_counts().items()])
    for src, D in (("hard_confused", hc), ("control", ctrl)):
        for r in D.itertuples():
            items.append(("B", src, r.image, r.box, r.cls, f"other_cls={r.other_cls_last}"))

    key, html = [], []
    for part in "RAB":
        P = [it for it in items if it[0] == part]
        if part != "R":
            rng.shuffle(P)
        for i, (pt, src, im, box, c, extra) in enumerate(P):
            iid = f"{part}{i + 1:02d}"
            split = "val" if src in ("fp_bg", "gt_control", "pred_random") else "train"
            key.append(dict(id=iid, part=pt, source=src, split=split, image=im, x1=box[0], y1=box[1],
                            x2=box[2], y2=box[3], gt_cls=c, extra=extra))
            html.append((part, iid, crop(root, split, im, box), SHORT[c] if part == "R" else ""))
    pd.DataFrame(key).to_csv(OUT / "audit_key.csv", index=False)
    (OUT / "audit.html").write_text(page(html))
    print(f"wrote {OUT / 'audit.html'} ({len(html)} crops) and audit_key.csv (do not open before finishing)")


def page(html):
    opts = {"A": ["Cargo", "Box", "Flatbed", "Tractor", "Liquid", "other vehicle", "not a vehicle", "can't tell"],
            "B": ["Cargo", "Box", "Flatbed", "Tractor", "Liquid", "not a truck", "can't tell"]}
    head = {"R": "Part R: reference (labelled examples, just look)",
            "A": "Part A: what is inside the magenta box? (80 crops)",
            "B": "Part B: which truck type is inside the magenta box? (60 crops)"}
    s = ["<html><head><meta charset='utf-8'><style>body{font-family:sans-serif}"
         ".g{display:flex;flex-wrap:wrap;gap:10px}.c{width:200px;font-size:12px}img{image-rendering:auto}"
         "</style></head><body><h2>Blind audit</h2><p>Magenta box = the object. Answer from the image only; "
         "use 'can't tell' freely. Save with the button at the bottom.</p>"]
    for part in "RAB":
        s.append(f"<h3>{head[part]}</h3><div class='g'>")
        for p, iid, b64, lab in (h for h in html if h[0] == part):
            ctl = (f"<b>{lab}</b>" if part == "R" else
                   f"<select id='{iid}'><option value=''>--</option>"
                   + "".join(f"<option>{o}</option>" for o in opts[part]) + "</select>")
            s.append(f"<div class='c'><img src='data:image/png;base64,{b64}' width=192><br>{iid} {ctl}</div>")
        s.append("</div>")
    s.append("""<p><button onclick="save()">Save answers (CSV)</button> <span id='n'></span></p><script>
function save(){const r=[['id','answer']];let n=0;document.querySelectorAll('select').forEach(e=>{r.push([e.id,e.value]);if(e.value)n++;});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([r.map(x=>x.join(',')).join('\\n')],{type:'text/csv'}));
a.download='audit_answers.csv';a.click();document.getElementById('n').textContent=n+' answered';}
</script></body></html>""")
    return "".join(s)


def score(a):
    k = pd.read_csv(OUT / "audit_key.csv", dtype={"image": str})
    d = k.merge(pd.read_csv(a.answers), on="id")
    d = d[d.answer.notna() & (d.answer != "")]
    A = d[d.part == "A"]
    print("Part A (H5): answer share by source\n", pd.crosstab(A.source, A.answer, normalize="index").round(2))
    B = d[d.part == "B"].copy()
    B["agree"] = B.answer == B.gt_cls.map(dict(enumerate(SHORT)))
    B["cant"] = B.answer == "can't tell"
    print("Part B (H2 vs H3): by source\n", B.groupby("source")[["agree", "cant"]].mean().round(2),
          "\n", B.groupby("source").size())
    d.to_csv(OUT / "audit_scored.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--answers", default=str(OUT / "audit_answers.csv"))
    a = ap.parse_args()
    build(a) if a.cmd == "build" else score(a)
