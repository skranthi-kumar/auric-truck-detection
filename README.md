# Auric AI – Computer Vision Researcher assignment

5-class overhead truck detection (Cargo Truck, Truck w/Box, Truck w/Flatbed, Truck Tractor, Truck w/Liquid).

**Result: 0.340 mAP50 on the 18 validation images at the training resolution (95% CI 0.240–0.429) and 0.297 on all 22 (CI 0.203–0.370), a fourfold improvement over the 0.073 YOLO11s baseline.**
The 18-image subset excludes four validation images rescaled about 2× relative to every training image (2308, 2391, 2384, 2460; `src/subset_map.py`). The pretrained checkpoint's training data includes the validation imagery, so generalisation to external imagery is not yet verified; the best detector trained only on the provided images scores 0.139 (0.082–0.214).
SC1 = FINAL2 plus test-time scale normalisation: images whose longer side is outside the train range (2,576–5,119 px)
are rescaled by a power of 2 towards the train median before inference. FINAL2 (0.274) is a Weighted-Boxes-Fusion of two
RF-DETR models built on a public xView-pretrained checkpoint: RFX0D (the zero-shot model run with dense native-scale
views, 50% tile overlap × 4 flips, 0.264 alone) plus RFX3 (fine-tuned at 2× with a frozen class head and rare-class
sampling). Reproduce with `bash reproduce_rfx.sh` (GPU recommended).

**Best model trained only on the provided images: 0.139 val mAP50** (95% CI 0.082–0.214): YOLO11s (COCO init),
2× tiles, 4-checkpoint WBF, fixed context rule. Reproduce with `bash reproduce.sh`.
Evaluation on full validation images uses a single in-repo evaluator (`src/evaluate.py`, verified identical to pycocotools).

Report: `report/report.pdf` (main report, 6 pages; also `.docx` and `REPORT.md`) · technical appendix `report/report_full.pdf` (30 pages; also `.docx` and `REPORT_FULL.md`) · Experiment records: `experiments/LOG.md` · Decisions with evidence: `DECISIONS.md`

## Layout
```
src/        rfx_load.py  rfx_predict.py  rfx_finetune.py  rfx_tiles.py  rfx_classmap.py  (RF-DETR path)
            data.py  eda.py  tile.py  train.py  predict.py  ensemble.py  context_smooth.py  context_model.py  evaluate.py  compare.py  learning_curves.py
            analyze_errors.py  oracle_cls.py  two_stage.py  cartography.py  dataset_forensics.py  audit.py
            subset_select.py  frcnn.py  nwd.py  lc_interval.py  density_breakdown.py  disagreement.py  context_radius_train.py
configs/    one YAML per experiment (+ yolo11s-p2.yaml model definition)
experiments/LOG.md  subset_rule.md  splits/ (frozen subsets)  results/ (all metric CSV/JSON)  audit/
report/     report.pdf  literature.md  figs/
weights/    rfx3_ep5.pt (RF-DETR fine-tuned, headline) + E2_epoch40/60/80/last.pt (YOLO ensemble) + SHA256SUMS, SHA256SUMS_RFX;
            shipped separately as Auric_CV_weights.zip (unzip into weights/); rfdetr_xview_ckpt1008.pth is downloaded by reproduce_rfx.sh
            E2_epoch40.pt E2_epoch60.pt E2_epoch80.pt E2_last.pt (4 snapshots of one YOLO11s run, 100 ep) + SHA256SUMS
DECISIONS.md  dated decision log with evidence, including corrections
```

## Setup
```bash
pip install -r requirements.txt        # torch/torchvision: use the platform build (Colab/Kaggle preinstalled)
```
Place the dataset as `CV_Assignment_AAL/{train,val}/{images,labels}` in the repo root, or pass `--root <dir>` to every script.

## Reproduce the reported validation metrics (no training needed)
```bash
bash reproduce.sh            # or: bash reproduce.sh /path/to/CV_Assignment_AAL
```
It verifies the weight checksums, runs sliced prediction with the 4 checkpoints, fuses them with Weighted Boxes
Fusion, applies context smoothing and scores every stage with the single evaluator:

| Stage | Output name | Val mAP50 (95% CI over images) |
|---|---|---|
| single checkpoint `E2_last.pt` (headline single model) | `repro_E2_last` | 0.107 |
| + 4-checkpoint WBF ensemble | `repro_ENS_E2` | 0.129 |
| + context smoothing (**reported result**) | `repro_final` | **0.139** (0.082-0.214) |

Fixed parameters (none tuned on val labels): conf 0.001, NMS IoU 0.6 per tile and 0.5 class-wise after merging,
max_det 300 per tile, 640 px tiles with 128 px overlap at imgsz 1280; WBF IoU 0.55 per class (library default);
context R = 150 px, alpha = 0.5. Steps after prediction are deterministic (byte-identical outputs verified).
All models are single-seed; the only seed-variance evidence is from the reduced learning-curve runs (12.5%: 0.024).

**Validation hygiene.** There is no test set. Every configuration ever scored on val is listed in
`experiments/LOG.md`; the reported best is therefore *selected on val* and is an optimistic estimate of
deployment performance. Inference-side steps (ensemble, context smoothing) used parameters declared in
`DECISIONS.md` before they were run; sensitivity tables are reported but were not used for selection.
One exception is disclosed: the context radius R = 150 px was first chosen by looking at val ground truth
(neighbour-majority accuracy for 150 / 400 px / whole image). The same comparison on train ground truth alone
selects the same R (`src/context_radius_train.py`, `experiments/results/context_radius_train.txt`).

## RF-DETR pretrained on xView (RFX series, final result)
External weights (permitted by the brief; documented for reproduction): public RF-DETR checkpoint trained on
xView, Hugging Face `ShotaroFukunaga/rfdetr_xview`, file `1008/checkpoint0099.pth`, SHA-256
`cd5f2d2cff000189906d84681aa92c4f900411a1528f32f9b6d7c6d5830214ed`. `reproduce_rfx.sh` downloads and verifies it.
The checkpoint was trained on xView, whose training split includes the 22 validation images.

```bash
pip install -r requirements-rfx.txt     # separate environment from requirements.txt
bash reproduce_rfx.sh                   # downloads + verifies the checkpoint, predicts val, scores it
```

| Run | Val mAP50 (95% CI) | Notes |
|---|---|---|
| RFX0, zero-shot (best single model) | 0.195 (0.116–0.279) | reproduced byte-identically by `reproduce_rfx.sh` |
| RFX3, fine-tuned (2×, frozen class head, rare-class sampling, epoch 5) | 0.113 (0.060–0.198) | `weights/rfx3_ep5.pt` (`SHA256SUMS_RFX`), `--scale 2` |
| ENS_RFX03 = WBF(RFX0, RFX3) | 0.226 (0.133–0.317) | Δ vs RFX0 +0.031 (−0.010, +0.079), P 0.93 |
| RFX0D = RFX0, 50% overlap × 4 flips (WBF) | 0.264 (0.163–0.352) | Δ vs RFX0 +0.069 (+0.019, +0.097) |
| **FINAL2 = WBF(RFX0D, RFX3)** (**best**) | **0.274** (0.175–0.357) | Δ vs ENS_RFX03 +0.048 (+0.020, +0.070), P 1.00 |

Per class (headline FINAL2): Cargo 0.173, Box 0.325, Flatbed 0.390, Tractor 0.038, Liquid 0.445.

Fixed inference settings: 1008 px tiles, 128 px overlap, multi-label over the 5 mapped xView class ids (Cargo 11,
Box 12, Flatbed 15, Tractor 13, Liquid 16; mapping fixed on train images by `src/rfx_classmap.py`), conf 0.001,
class-wise NMS 0.5 after merging; `--scale 2` = nearest-neighbour 2x before tiling (boxes mapped back).

Fine-tuning (`src/rfx_finetune.py`; Kaggle T4, ~10 min/epoch):
```bash
python src/rfx_tiles.py tiles1008                         # 1008 px train tiles, stride 880 (+10% empty, seed 0)
python src/rfx_finetune.py --tiles tiles1008 --out runs/rfx3 --epochs 5 --upscale 2 --freeze_cls --rfs \
    --bs 4 --accum 2 --save_from 1                        # RFX3; drop the last three flags for RFX1
python src/rfx_predict.py --weights runs/rfx3/rfx1_ep5.pt --scale 2 --split val --out experiments/results/RFX3_val_preds.csv
python src/evaluate.py --preds experiments/results/RFX3_val_preds.csv --name RFX3
```

## Full pipeline
```bash
python src/eda.py --phash                                   # dataset audit
python src/tile.py --out tiles                              # 640 px tiles, 128 px overlap, seed 0 (frozen)
python src/train.py --config configs/E0.yaml --tiles tiles --project runs
python src/predict.py --weights runs/E0_s0/weights/last.pt --split val --out experiments/results/E0_s0_val_preds.csv
python src/evaluate.py --preds experiments/results/E0_s0_val_preds.csv --name E0_s0
python src/compare.py E0_s0 E1_up2_s0                       # paired bootstrap between two evaluated runs
python src/analyze_errors.py --name E0_s0                   # failure taxonomy tables + galleries
python src/oracle_cls.py                                    # 5.1b oracle-box classifier (--arch vit_small_patch14_dinov2.lvd142m for 5.1c)
python src/cartography.py collect --run runs/E0_s0 && python src/cartography.py analyze --name E0_s0 && python src/cartography.py populations --name E0_s0   # 5.2
python src/dataset_forensics.py                             # label/structure audit; context oracle
python src/audit.py build                                   # blind visual audit page (answers scored with: audit.py score)
python src/subset_select.py random && python src/subset_select.py selected   # frozen 5.3/5.4 splits (rule: experiments/subset_rule.md)
python src/frcnn.py train --tiles tiles --out runs/FRCNN_s0   # architecture-family check (torchvision Faster R-CNN)
```

## Resuming after a Colab disconnect
1. Re-mount Drive and re-run the setup cells of `notebooks/colab_runner.ipynb` (tiles are restored from `tiles.zip`).
2. Re-run the same `train.py` command: it detects `runs/<name>/weights/last.pt` and resumes (optimizer, epoch, LR schedule).
3. Resumed runs are not bit-identical to uninterrupted ones; they are marked in `experiments/LOG.md`.

## Hardware / runtime
Colab T4 (16 GB) and Kaggle 2×T4 for training; Apple M5 (MPS) for evaluation and analysis. Per-run `pip freeze` is saved next to each run.
