"""Train a YOLO detector on tiles from a config YAML. Resumes automatically after a disconnect.

Usage: python src/train.py --config configs/E0.yaml --tiles /content/tiles --project /content/drive/MyDrive/auric/runs
"""
import argparse
import csv
import platform
from pathlib import Path

import yaml
from ultralytics import YOLO

from data import CLASSES


def tile_list(tiles, stems, out):
    """Write the tile paths that come from the given source images (subsets are chosen per image)."""
    keep = set(stems)
    with open(Path(tiles) / "train_manifest.csv") as f:
        rows = [r["tile"] for r in csv.DictReader(f) if r["image"] in keep]
    out.write_text("\n".join(str(Path(tiles) / "train" / "images" / f"{t}.png") for t in rows) + "\n")
    return out


def tile_list_repeat(tiles, out, cap=4):
    """Oversample tiles from dense source images (H11: val is depot scenes, train is mostly sparse).
    Each tile is listed r times, r = clip(round(sqrt(n_img / median_n)), 1, cap), n_img = labelled boxes of its
    source image (from the manifest); duplicates in the list are kept by Ultralytics."""
    import math
    with open(Path(tiles) / "train_manifest.csv") as f:
        rows = list(csv.DictReader(f))
    n_img = {}
    for r in rows:
        n_img[r["image"]] = n_img.get(r["image"], 0) + int(r["n_boxes"])
    med = sorted(n_img.values())[len(n_img) // 2]
    rep = lambda im: min(cap, max(1, round(math.sqrt(n_img[im] / max(med, 1)))))
    paths = [str(Path(tiles) / "train" / "images" / f"{r['tile']}.png") for r in rows for _ in range(rep(r["image"]))]
    out.write_text("\n".join(paths) + "\n")
    print(f"repeat list: {len(rows)} tiles -> {len(paths)} entries (median boxes/image {med})")
    return out


def tile_list_rfs(tiles, out, t=0.25, cap=4, keep=None):
    """Repeat-factor sampling (Gupta et al., LVIS 2019) at tile level: a tile is listed r times with
    r = clip(round(max_c sqrt(t / f_c)), 1, cap) over the classes c present in it, f_c = class share of all
    train instances (from the tile labels). Rare-class tiles are seen more often; others once."""
    import math
    import numpy as np
    with open(Path(tiles) / "train_manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if keep is None or r["image"] in keep]  # keep: subset of source images
    labs = {}
    for r in rows:
        lp = Path(tiles) / "train" / "labels" / f"{r['tile']}.txt"
        labs[r["tile"]] = np.loadtxt(lp, ndmin=2)[:, 0].astype(int) if lp.exists() and lp.stat().st_size else np.zeros(0, int)
    cnt = np.bincount(np.concatenate(list(labs.values())), minlength=len(CLASSES)).astype(float)
    f_c = cnt / cnt.sum()
    rep = lambda cls: min(cap, max(1, round(max(math.sqrt(t / f_c[c]) for c in set(cls.tolist()))))) if len(cls) else 1
    paths = [str(Path(tiles) / "train" / "images" / f"{r['tile']}.png") for r in rows for _ in range(rep(labs[r["tile"]]))]
    out.write_text("\n".join(paths) + "\n")
    print(f"rfs list: {len(rows)} tiles -> {len(paths)} entries; class shares {np.round(f_c, 3).tolist()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--project", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run = Path(args.project) / cfg["name"]
    last = run / "weights" / "last.pt"

    if last.exists():
        print(f"resuming {last}")
        YOLO(str(last)).train(resume=True)
        return

    # data files live outside the run dir (Ultralytics DDP did not see files written inside it)
    meta = Path(args.project) / "_data" / cfg["name"]
    meta.mkdir(parents=True, exist_ok=True)
    train = str(Path(args.tiles) / "train" / "images")
    stems = None
    if cfg.get("subset"):  # txt file of source-image stems, relative to repo root
        stems = (Path(__file__).resolve().parents[1] / cfg["subset"]).read_text().split()
    if cfg.get("subset_complement"):  # train on the images NOT in this list (cross-fitting halves)
        drop = set((Path(__file__).resolve().parents[1] / cfg["subset_complement"]).read_text().split())
        with open(Path(args.tiles) / "train_manifest.csv") as f:
            stems = sorted({r["image"] for r in csv.DictReader(f)} - drop)
    if cfg.get("repeat_rfs"):  # rare-class oversampling (see tile_list_rfs), within the subset if one is given
        train = str(tile_list_rfs(args.tiles, meta / "train_tiles.txt", keep=set(stems) if stems else None))
    elif cfg.get("repeat_dense"):  # H11 oversampling (see tile_list_repeat)
        train = str(tile_list_repeat(args.tiles, meta / "train_tiles.txt"))
    elif stems is not None:
        train = str(tile_list(args.tiles, stems, meta / "train_tiles.txt"))
    epochs = cfg["epochs"]
    if cfg.get("iter_match") and train.endswith(".txt"):  # same optimiser steps as an unrepeated list of the same images
        entries = Path(train).read_text().split()
        epochs = max(1, round(cfg["epochs"] * len(set(entries)) / len(entries)))
        print(f"iteration-matched: {len(entries)} entries / {len(set(entries))} tiles -> {epochs} epochs")
    data = dict(path=str(args.tiles), train=train, val=str(Path(args.tiles) / "val" / "images"),
                names=dict(enumerate(CLASSES)))
    (meta / "data.yaml").write_text(yaml.safe_dump(data))
    (meta / "env.txt").write_text(f"python {platform.python_version()}\n")

    extra = {}
    if cfg.get("nwd"):  # NWD assignment + loss for tiny objects (src/nwd.py; patch applies on import, also in DDP workers)
        from nwd import NWDTrainer
        extra["trainer"] = NWDTrainer
    YOLO(cfg["model"]).train(data=str(meta / "data.yaml"), project=args.project, name=cfg["name"],
                             exist_ok=True, imgsz=cfg.get("imgsz", 640), epochs=epochs,
                             batch=cfg.get("batch", 16), seed=cfg.get("seed", 0), deterministic=True,
                             save_period=cfg.get("save_period", -1), workers=cfg.get("workers", 2),
                             **cfg.get("train_args", {}), **extra)


if __name__ == "__main__":
    main()
