"""Object-level view of the GTO test: which false positives become true positives when omitted trucks are added?

Both GTO runs score identical predictions; only the GT differs. Rows are matched on (image, cls, conf, box).
Reported per class for (a) the top-N false positives by confidence under the original GT and (b) false positives
at the operating point conf >= 0.25. Diagnostic only.

Usage: python src/gto_objects.py --run GTO_FINAL2 [--top 400]
"""
import argparse
from pathlib import Path

import pandas as pd

from data import CLASSES

R = Path(__file__).resolve().parents[1] / "experiments" / "results"
KEY = ["image", "cls", "conf", "x1", "y1", "x2", "y2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="GTO_FINAL2")
    ap.add_argument("--top", type=int, default=400)
    a = ap.parse_args()
    o = pd.read_csv(R / f"{a.run}_orig_matches.csv", dtype={"image": str}, usecols=KEY + ["tp", "err"])
    g = pd.read_csv(R / f"{a.run}_aug_matches.csv", dtype={"image": str}, usecols=KEY + ["tp", "err"])
    m = o.merge(g, on=KEY, suffixes=("_o", "_a"), validate="one_to_one")
    assert len(m) == len(o) == len(g), "prediction rows must match one to one"
    rows = []
    for c, name in enumerate(CLASSES):
        fp = m[(m.cls == c) & ~m.tp_o]
        for label, sel in ((f"top{a.top} FP", fp.nlargest(a.top, "conf")), ("FP conf>=0.25", fp[fp.conf >= 0.25])):
            n, flip = len(sel), int(sel.tp_a.sum())
            rows.append((name, label, n, flip, flip / n if n else float("nan")))
    t = pd.DataFrame(rows, columns=["class", "set", "n_fp", "become_tp", "share"])
    for label in t.set.unique():
        s = t[t.set == label]
        rows.append(("ALL", label, s.n_fp.sum(), s.become_tp.sum(), s.become_tp.sum() / s.n_fp.sum()))
    t = pd.DataFrame(rows, columns=t.columns)
    print(t.round(3).to_string(index=False))
    t.to_csv(R / f"{a.run}_objects.csv", index=False)


if __name__ == "__main__":
    main()
