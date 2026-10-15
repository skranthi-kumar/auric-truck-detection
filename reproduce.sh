#!/usr/bin/env bash
# Reproduce the reported validation result (ENS_E2_ctx, mAP50 0.139) from the 22 val images. No training.
#   1. sliced prediction with each of the 4 E2 checkpoints (conf 0.001, iou 0.6, max_det 300/tile, class-wise NMS 0.5)
#   2. Weighted Boxes Fusion of the 4 prediction sets (IoU 0.55, per class; library default)
#   3. context smoothing (R = 150 px, alpha = 0.5; declared before running)
#   4. evaluate.py (COCO 101-point AP50 at IoU 0.5, class-aware, bootstrap CI over images)
# Usage: bash reproduce.sh [DATA_ROOT]   (default: ./CV_Assignment_AAL with val/images and val/labels)
# Steps 2-4 are deterministic (byte-identical outputs verified); step 1 runs on GPU (CUDA/MPS), see README.
set -euo pipefail
ROOT=${1:-CV_Assignment_AAL}
OUT=experiments/results/repro
mkdir -p "$OUT"
(cd weights && shasum -a 256 -c SHA256SUMS)

for ck in E2_epoch40 E2_epoch60 E2_epoch80 E2_last; do   # order matters only for byte-identity, not for the metric
  python src/predict.py --weights weights/$ck.pt --split val --root "$ROOT" --imgsz 1280 --out "$OUT/${ck}_val_preds.csv"
done
python src/ensemble.py --preds "$OUT"/E2_epoch40_val_preds.csv "$OUT"/E2_epoch60_val_preds.csv \
    "$OUT"/E2_epoch80_val_preds.csv "$OUT"/E2_last_val_preds.csv --out "$OUT/ENS_E2_val_preds.csv"
python src/context_smooth.py --preds "$OUT/ENS_E2_val_preds.csv" --out "$OUT/final_val_preds.csv" --R 150 --alpha 0.5

python src/evaluate.py --preds "$OUT/E2_last_val_preds.csv" --name repro_E2_last --root "$ROOT"     # single checkpoint
python src/evaluate.py --preds "$OUT/ENS_E2_val_preds.csv" --name repro_ENS_E2 --root "$ROOT"       # + ensemble
python src/evaluate.py --preds "$OUT/final_val_preds.csv" --name repro_final --root "$ROOT"         # + context = headline
# expected: repro_E2_last 0.107, repro_ENS_E2 0.129, repro_final 0.139
