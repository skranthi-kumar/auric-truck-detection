#!/usr/bin/env bash
# Reproduce the RF-DETR (xView-pretrained) validation results from the 22 val images. No training.
#   0. download the public xView-trained RF-DETR checkpoint (Hugging Face ShotaroFukunaga/rfdetr_xview,
#      1008/checkpoint0099.pth; 810 MB) and verify its SHA-256
#   1. sliced prediction: 1008 px tiles, 128 px overlap, multi-label over the 5 mapped xView class ids
#      (Cargo 11, Box 12, Flatbed 15, Tractor 13, Liquid 16; mapping fixed on TRAIN images, src/rfx_classmap.py),
#      conf 0.001, class-wise NMS 0.5 after merging
#   2. evaluate.py (same single evaluator as every other number in the report)
# Usage: bash reproduce_rfx.sh [DATA_ROOT]   (environment: pip install -r requirements-rfx.txt)
set -euo pipefail
ROOT=${1:-CV_Assignment_AAL}
OUT=experiments/results/repro_rfx
CK=weights/rfdetr_xview_ckpt1008.pth
SHA=cd5f2d2cff000189906d84681aa92c4f900411a1528f32f9b6d7c6d5830214ed
mkdir -p "$OUT" weights
[ -f "$CK" ] || curl -L -o "$CK" https://huggingface.co/ShotaroFukunaga/rfdetr_xview/resolve/main/1008/checkpoint0099.pth
echo "$SHA  $CK" | shasum -a 256 -c -

# RFX0: zero-shot xView RF-DETR (headline)
python src/rfx_predict.py --split val --root "$ROOT" --out "$OUT/RFX0_val_preds.csv"
python src/evaluate.py --preds "$OUT/RFX0_val_preds.csv" --name repro_RFX0 --root "$ROOT"
# expected: repro_RFX0 mAP50 0.195 (95% CI 0.116-0.279)

# HEADLINE: ENS_RFX03 = WBF(RFX0, RFX3_ep5). RFX3_ep5 = fine-tuned checkpoint (weights/rfx3_ep5.pt, SHA256SUMS_RFX),
# predicted at 2x (nearest upsample before tiling). The 2x pass is ~4x the tiles: use a GPU (~10 min on a T4).
(cd weights && shasum -a 256 -c SHA256SUMS_RFX)
python src/rfx_predict.py --split val --root "$ROOT" --weights weights/rfx3_ep5.pt --scale 2 --out "$OUT/RFX3_ep5_val_preds.csv"
python src/evaluate.py --preds "$OUT/RFX3_ep5_val_preds.csv" --name repro_RFX3_ep5 --root "$ROOT"
python src/ensemble.py --preds "$OUT/RFX0_val_preds.csv" "$OUT/RFX3_ep5_val_preds.csv" --out "$OUT/ENS_RFX03_val_preds.csv"
python src/evaluate.py --preds "$OUT/ENS_RFX03_val_preds.csv" --name repro_ENS_RFX03 --root "$ROOT"
# expected: repro_RFX3_ep5 0.113, repro_ENS_RFX03 0.226 (95% CI 0.133-0.317)

# HEADLINE (D1): RFX0D = RFX0 at 50% tile overlap (504 px) x 4 flip passes, fused by WBF; FINAL2 = WBF(RFX0D, RFX3_ep5).
for f in none h v hv; do
  python src/rfx_predict.py --split val --root "$ROOT" --overlap 504 --flip $f --out "$OUT/D1_${f}_val_preds.csv"
done
python src/ensemble.py --preds "$OUT"/D1_none_val_preds.csv "$OUT"/D1_h_val_preds.csv "$OUT"/D1_v_val_preds.csv "$OUT"/D1_hv_val_preds.csv --out "$OUT/RFX0D_val_preds.csv"
python src/ensemble.py --preds "$OUT/RFX0D_val_preds.csv" "$OUT/RFX3_ep5_val_preds.csv" --out "$OUT/FINAL2_val_preds.csv"
python src/evaluate.py --preds "$OUT/RFX0D_val_preds.csv" --name repro_RFX0D --root "$ROOT"
python src/evaluate.py --preds "$OUT/FINAL2_val_preds.csv" --name repro_FINAL2 --root "$ROOT"
# expected: repro_RFX0D 0.264, repro_FINAL2 0.274 (95% CI 0.175-0.357). GPU recommended (16x the RFX0 tile passes).

# HEADLINE (SC1, 15 Oct): scale normalisation. Images whose longer side is outside the TRAIN range (2,576-5,119 px)
# are rescaled by s = 2^round(log2(3228 / longer side)) before inference (val: 2308, 2391 -> 0.5; 2384, 2460 -> 2).
# RFX0D is rerun at scale s and RFX3 at 2s for those images; the other images keep their FINAL2 predictions.
python src/sc1.py --root "$ROOT" --final2 "$OUT/FINAL2_val_preds.csv" --work "$OUT/sc1" --out "$OUT/SC1_val_preds.csv"
python src/evaluate.py --preds "$OUT/SC1_val_preds.csv" --name repro_SC1 --root "$ROOT"
# expected: repro_SC1 0.2975 (95% CI 0.203-0.370)
