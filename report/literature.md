# Literature notes: benchmark expectation for xView truck subtypes
*9–10 Oct 2026: 23 web searches, snippet/abstract level only. Exact numbers should be checked against the PDFs before they go into the report.*

## What the published xView numbers suggest
Our five classes are xView classes: Cargo Truck and Truck w/Box appear under Vehicle in Table 2 of Lam et al. 2018.

| Source | Setting | Reported | Relevance |
|---|---|---|---|
| Lam et al. 2018, *xView: Objects in Context in Overhead Imagery*, [arXiv 1802.07856](https://arxiv.org/pdf/1802.07856) | Official baseline (SSD per secondary summary, to verify), 60 classes | Over 80% of the classes are fine-grained. Pickup truck scored <1% mAP despite >1000 instances, and small cars did worse than rarer but larger classes | Fine-grained, small vehicle classes are very hard even with far more data. Informs H1 (resolution) and H3 (ambiguity); does not test them on our data |
| Sergievskiy & Ponamarev 2019, *Reduced Focal Loss: 1st Place Solution to xView*, [arXiv 1903.01347](https://ar5iv.arxiv.org/html/1903.01347) | Challenge winner, 60 classes, full xView training set | **28.32 mAP (validation)**, recall 0.612; the paper says the public leaderboard is 1–2% lower | The winning system's average over all classes is about 0.28 |
| *Oriented Bounding Boxes for Small and Freely Rotated Objects*, [arXiv 2104.11854](https://arxiv.org/pdf/2104.11854) | Later academic method, xView | 0.5315 mAP over 60 classes; 0.5218 on 19 tiny-object classes incl. trucks, buses, small cars (metric/protocol to verify) | Upper range of later work; uses oriented boxes, which we don't have |

**Not found:** per-class AP50 for our five exact truck classes in any of these sources. The baseline table in Lam et al. may contain it; the PDF has not been checked yet.

## How this bears on our hypotheses (rule: literature informs, does not conclude)
- **The 0.75 target vs the field:** the best published xView results we found are about 0.28 (challenge winner) and about 0.53 (later work), averaged over 60 classes. They were trained on the full xView set (~1M instances), versus our ~7.6k.
  - This makes ≥0.75 on five of xView's hardest fine-grained vehicle classes **implausible by external comparison**.
  - It is not proof for our split: different classes in the average, different protocol, and our val is only 22 images.
  - The report can state it as context.
- **H4 (too little data):** fine-grained vehicle classes stay poor in xView even with thousands of instances (pickup truck <1% mAP). This weakly argues against "more data alone fixes it". Our 5.3 learning curves remain the actual test.
- **H1 (tiny objects):** the source behind the 0.5218 tiny-object number works on small, freely rotated objects. It does not test our specific levers (P2 head, upsampling).

## Methods we use (citations from memory; still to verify before the report)
| Reference | Used for |
|---|---|
| TIDE (Bolya et al., ECCV 2020) | error oracle |
| Dataset Cartography (Swayamdipta et al., EMNLP 2020) | 5.2 |
| Forgetting events (Toneva et al., ICLR 2019) | 5.2 |
| Confident Learning (Northcutt et al., JAIR 2021) | label-noise flags |
| SAHI (Akyon et al., ICIP 2022) | sliced inference |
| Deep learning scaling is predictable (Hestness et al., 2017) | 5.3 |
| Repeat factor sampling (Gupta et al., LVIS, CVPR 2019) | E8 |

## Actionable levers found (pass 2): each maps to a hypothesis, none concludes anything for our data
| Lever | Evidence (as reported) | Hypothesis | Fit for us |
|---|---|---|---|
| **P2 / stride-4 head** | Controlled A/B, P2 the only change: +49% relative AP_small (0.070 vs 0.047), single run ([LibreYOLO9P2s card](https://huggingface.co/LibreYOLO/LibreYOLO9P2s-visdrone)). 3-seed ablation: AP_small +31% ([arXiv 2606.09081](https://arxiv.org/pdf/2606.09081)). Combined P2 + 1280 input + Wasserstein loss: +16.6 mAP50 over YOLOv8s on VisDrone ([DroneScan-YOLO](https://www.alphaxiv.org/abs/2604.13278)). The card notes P2 is *worse* on COCO, so it only pays off for tiny objects | H1 | Strong support for E4; `yolov8-p2.yaml` ships with Ultralytics |
| **NWD (Gaussian Wasserstein) instead of IoU in assignment/loss** | +6.7 AP on AI-TOD (tiny objects) vs baseline ([Wang et al., arXiv 2110.13389](https://arxiv.org/pdf/2110.13389)); NWD-RKA +4.3 AP on AI-TOD-v2 ([arXiv 2206.13996](https://arxiv.org/abs/2206.13996v1)); RFLA +4.0 AP ([code](https://github.com/Chasel-Tsui/mmdet-rfla)) | H1: IoU is very sensitive to a 1–2 px shift at 15 px, which hurts positive assignment | **New candidate E9.** Not built into Ultralytics; needs a small patch to the assigner/loss. Code exists for mmdet ([NWD](https://github.com/jwwangchn/NWD)) |
| **Learned super-resolution vs interpolation** | On xView-type data, SR from 30 cm to 15 cm: +13–36% mAP (CVPRW) / +16–20% (arXiv v2) ([Shermeyer & Van Etten, arXiv 1812.04098](https://arxiv.org/pdf/1812.04098)). SR vs nearest-neighbour comparison: [arXiv 1907.05283](https://arxiv.org/pdf/1907.05283) (result not yet read) | H1 | Our E1 uses plain interpolation (imgsz resize). SR is a candidate but costly (an SR model plus re-tiling); read 1907.05283 first, since it may show interpolation is about as good |
| **Training with missing annotations** | Unlabelled objects become negatives and weaken the classifier; fixes are teacher pseudo-labels ([BigDetection, arXiv 2203.13249](https://arxiv.org/pdf/2203.13249)), SparseDet (ICCV 2023, [arXiv 2201.04620](https://ar5iv.labs.arxiv.org/html/2201.04620)), and erasing/covering unlabelled objects (patent [US 11354793](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11354793)) | H5 | **New candidate E10 (cheap):** take high-confidence detections on train that match no GT and *grey them out* in the train tiles, so they stop acting as negatives. Val untouched |
| **Label noise in tiny-object benchmarks** | AI-TOD authors found "prominent noisy labels" and relabelled the dataset (AI-TOD-v2) | H2 | Makes H2 plausible for this kind of data; does not test it on ours (E7 does) |
| **Reduced focal loss** | xView 1st place ([arXiv 1903.01347](https://ar5iv.arxiv.org/html/1903.01347)) | imbalance | Lower priority; YOLO11 already uses BCE + TAL |
| **Context for fine-grained subtypes** | FAIR1M ([arXiv 2103.05569](https://arxiv.org/pdf/2103.05569)): fine-grained AP well below coarse AP (aircraft table seen; vehicle table not read) | H3 | No evidence found yet that context fixes subtype confusion |

## Similar problems and algorithms (pass 3)
| Work | Reported | Hypothesis | Fit for us |
|---|---|---|---|
| **SAHI** (Akyon et al., ICIP 2022, [arXiv 2202.06934](https://arxiv.org/pdf/2202.06934)) | On VisDrone + xView: sliced inference +5–7% AP; sliced fine-tuning + sliced inference up to +12.7–14.5% AP (FCOS/VFNet/TOOD). On xView, training on full images gives "poor" results. 25% overlap helps small AP | H1 | Our pipeline already does both (640 tiles for training, sliced inference, 20% overlap). Supports the design; nothing new to run |
| **Coarse-to-fine / hierarchical detection**: CHODNet on FAIR1M ([arXiv 2103.05569](https://arxiv.org/pdf/2103.05569)) adds a coarse-category training stage; Balanced Hierarchical Contrastive loss (CVPR 2026) balances gradients per hierarchy level; SSD-ML hierarchical loss ([scitepress](https://www.scitepress.org/Papers/2020/89024)) | Gains reported for FAIR1M/ShipRSImageNet (numbers not read) | H3 / detection weakness | **Candidate E11 (cheap):** train 1-class "truck" first (`single_cls`), then fine-tune 5-class from those weights. Targets the 0.296 class-agnostic detection weakness separately from subtype confusion |
| **Augmentation for small objects** (Kisantal et al. 2019, [arXiv 1902.07296](https://arxiv.org/abs/1902.07296v1)) | Oversampling images with small objects + copy-pasting small objects: +7.1% relative small-object detection AP on COCO | H1, imbalance | Supports E8 (oversampling). Copy-paste: Ultralytics `copy_paste` may need segment masks; **verify** before planning |
| **Noisy-label detection**: per-object co-teaching with two YOLOs ([arXiv 2511.09955](https://arxiv.org/pdf/2511.09955)): KITTI mAP50 31.1% → 46.6% with noisy pseudo-labels. Co-teaching for detectors (Chadwick & Newman, [arXiv 1905.07202](https://arxiv.org/pdf/1905.07202)). Label-error finding by loss inspection ([arXiv 2303.06999](https://arxiv.org/pdf/2303.06999)) | Large gains, but under *synthetic/pseudo-label* noise | H2 | Only relevant if E7 shows noise matters. Co-teaching costs 2× training; loss inspection could replace our cartography flags as a cross-check |
| **Fine-grained aerial vehicles**: COFGA ([arXiv 2105.12786](https://arxiv.org/pdf/2105.12786)), truck subtypes at 5–15 cm GSD, "insufficient" results from state-of-the-art classifiers; Ground2Sky: annotators "often cannot identify the object" at low resolution and extreme viewpoint ([ICIP 2017](https://web.eecs.umich.edu/~stellayu/publication/2017aerialCarICIP.html)) | Fine-grained aerial vehicle classification is unsolved even at 2–6× finer GSD than ours | H2, H3 | Makes annotator-level ambiguity plausible for our ~30 cm data; does not test it. Reporting per hierarchy level (truck vs subtype) is recommended |

## Architecture choice (pass 4)
| Evidence | Reported | Takeaway |
|---|---|---|
| Hurt et al. 2025, 11 detectors (5 transformer, 6 CNN) on xView, RarePlanes and a third dataset ([arXiv 2508.02871](https://arxiv.org/pdf/2508.02871)) | xView: **best of all 11 = Swin, AP50 17.59%**; YOLOX 2nd; Co-DETR has much higher recall (AR50 +20%) but lower F1 | **No architecture "solves" xView.** The best AP50 across 11 modern detectors is ~0.18. Strong framing evidence for the 0.75 target (different class set and protocol) |
| DQ-DETR (2024) on AI-TOD-v2 (mean object ~12.8 px) ([AI-TOD](https://cs.paperswithcode.com/paper/tiny-object-detection-in-aerial-images)) | 30.2% mAP (SOTA claim; metric likely AP@.5:.95) | Tiny-object specialist DETRs exist; not in Ultralytics; costly on a T4 |
| RT-DETR vs YOLO on VisDrone ([UFO-DETR](https://arxiv.org/pdf/2602.22712), [RT-DETR++](https://arxiv.org/pdf/2509.09157), [ISPRS 2025](https://isprs-annals.copernicus.org/articles/X-4-W8-2025/721/2026/)) | RT-DETR-L 43.5 vs YOLOv8-m 40.7 mAP50; RT-DETR-R18 AP_s 12.5 vs YOLOv8-N 10.6. ISPRS: RT-DETR higher recall, YOLOv12 higher precision and faster to train | DETR-family gains are modest (+2–3 points) and compute-heavy; mixed for small remote-sensing objects |
| YOLO26 ([arXiv 2509.25164](https://arxiv.org/pdf/2509.25164), [Ultralytics blog](https://www.ultralytics.com/blog/how-ultralytics-yolo26-trains-smarter-with-progloss-stal-and-musgd)) | STAL (small-target-aware label assignment) + ProgLoss claimed to help small objects; **no isolated ablation found**. `yolo26-p2.yaml` ships with Ultralytics 8.4 | Drop-in in our pipeline; its assigner change addresses the same mechanism as NWD (H1b). Evidence is vendor/preprint level |

**Recommendation (constraints: T4 GPUs, 6 days, frozen Ultralytics pipeline and evaluator):**
- YOLO is the default for practical reasons (T4s, 6 days, working tiling/training code), **not** because it is proven best: on xView, Swin and YOLOX ranked above the YOLOv3 baseline among 11 models (Hurt 2025). The evaluator reads a predictions CSV, so any framework can be scored identically. The cost of switching is training code + COCO-format tiles + slower training, not evaluator integrity.
- **Add one out-of-family check:** RT-DETR (transformer, already in Ultralytics, zero pipeline cost). If architecture turns out to matter, a Swin-backbone detector via mmdetection is the next candidate.
- **E4 = YOLO11s + P2** (isolates P2; the best-evidenced change).
- **E12 = YOLO26s + P2 vs E4** (isolates YOLO26's STAL/ProgLoss; verify STAL is active in Ultralytics 8.4 first).
- RT-DETR (in Ultralytics) only if time remains, as an "architecture family" check; expected gain is small and compute is high.

## Context refinement and ensembling (pass 5, 10 Oct) — supports the two adopted inference-side gains
| Work | Reported | Fit for us |
|---|---|---|
| **Affinity-Aware Relation Network** (Fang et al., ACCV 2022, [paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Fang_Affinity-Aware_Relation_Network_for_Oriented_Object_Detection_in_Aerial_Images_ACCV_2022_paper.pdf)) | Builds an affinity graph over detected objects and attends over similar neighbours; motivated by "aerial objects tend to appear in groups with similar orientations"; SOTA on DOTA/HRSC2016/UCAS-AOD | Published version of the mechanism behind our context smoothing (H12). Ours is a training-free post-hoc rule; theirs is learned inside a two-stage detector — the natural "next experiment" |
| **Context-Likelihood Graphs and Graph Refinement** (Bhowmik et al., ICCV 2023, [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Bhowmik_Detecting_Objects_with_Context-Likelihood_Graphs_and_Graph_Refinement_ICCV_2023_paper.pdf)) | Graph from inter-object relation priors + initial class predictions, refined by an energy-based model; detector-agnostic; "especially beneficial for rare object classes" (COCO/Visual Genome) | Exactly our setting (initial class predictions refined by relations); their rare-class benefit matches our Liquid/Flatbed gains |
| **Context-Aware Aerial Object Detection** ([arXiv 2404.04140](https://arxiv.org/pdf/2404.04140)) | Models inter-object and background relationships, arguing they are especially stable in aerial images | Supports treating neighbourhood consistency as a real property of overhead scenes, not only an annotation artefact (does not settle it for our labels) |
| **SOHR network** for fine-grained airplane detection (Beihang, [page](https://research.buaa.edu.cn/zh/publications/scene-object-holistic-relation-network-for-fine-grained-airplane-/)) | Box-only features are insufficient for subtly different classes; scene-object relations help | Same argument for truck subtypes |
| **Snapshot Ensembles** (Huang et al., ICLR 2017, [page](https://mlanthology.org/iclr/2017/huang2017iclr-snapshot)) | Ensemble checkpoints of one training run (classification results) | Our 4-checkpoint ensemble is this idea without the cyclic schedule; gain +0.022 is our own measurement |
| **Weighted Boxes Fusion** (Solovyev et al., Image and Vision Computing 2021, [arXiv 1910.13302](https://arxiv.org/pdf/1910.13302)) | Confidence-weighted box averaging across models; used for top COCO ensembles | The fusion rule in `src/ensemble.py` (re-implemented, 40 lines) |

Not found: a published number for snapshot ensembling + WBF in detection, or an ablation showing spatial neighbour context improving fine-grained *vehicle* classification in aerial imagery. Both are therefore reported as our measurements, not as literature-backed expectations.

## Pretraining for aerial tiny objects (pass 6, 10 Oct)
| Source | Reported | Fit for us |
|---|---|---|
| **Ultralytics YOLO11-OBB weights** (trained on DOTA v1; large-/small-vehicle classes) | — | **Usable now**: 100% parameter transfer into our detector (local check); imagery sources differ from xView -> no val overlap. Queued as E13 |
| RemoteCLIP ([arXiv 2306.11029](https://arxiv.org/pdf/2306.11029)) | Web-pretrained CLIP is "surprisingly powerful" for RS; in-domain continued pretraining helps but data quantity is the bottleneck | Supports trying aerial-pretrained init over COCO; no tiny-vehicle number |
| RS foundation models (SatMAE, Scale-MAE, SatDINO [arXiv 2508.21402](https://arxiv.org/pdf/2508.21402), billion-scale ViT [arXiv 2304.05215](https://ar5iv.labs.arxiv.org/html/2304.05215)) | Classification / rotated-detection benchmarks (fMoW, DOTA v2, DIOR-R); no per-size vehicle AP | ViT-L backbones (300M+) are out of T4 budget; overlap risk: several pretrain on fMoW (DigitalGlobe, like xView) -> would need checking before use |
| "Bridging the Scale Gap: tiny and general object detection in RS" ([arXiv 2512.01665](https://arxiv.org/abs/2512.01665v1)) | Even a satellite-pretrained DINOv3-7B backbone is beaten by a tiny-object-specific detector on AI-TOD-v2; "application of foundation models to tiny object detection remains unexplored" | Backbone scale alone does not solve tiny objects; detector design (resolution, assignment) matters more |
| FAIR1M methods (PETDet 2nd/220 in the 2021 Gaofen fine-grained track; OBBStacking ensembles [arXiv 2209.13369](https://arxiv.org/pdf/2209.13369): +1-2 mAP over the best member) | Fine-grained aerial SOTA = oriented two-stage detectors + strong backbones + ensembles | Ensembling gain magnitude matches ours (+0.022); oriented boxes are not available here |

## External review document (ChatGPT "deep research" PDF supplied by the user, 10 Oct) — assessed against our constraints
A 9-week, 2–4-GPU research proposal (Oriented R-CNN + context/graph modules + SSL pretraining + metric learning + pseudo-labelling, oriented mAP). Judged on evidence and feasibility:
- **Not applicable here:** oriented detection needs angle labels (ours are axis-aligned YOLO boxes; the metric is axis-aligned mAP50); SSL pretraining needs unlabeled imagery we do not have (443 images); "24–48 GPU-h per experiment" is 10–20× our total budget.
- **Already done or queued:** tiling + upsampling (E1), context reasoning (context smoothing; learned model declared), ensembles, aerial pretraining (E13 DOTA init), coarse/fine (E11 candidate), pseudo-labelling of unannotated objects (E10 candidate; forensics show few confident bg FPs on val, so low expected gain).
- **Speculative for 5 days:** supervised-contrastive / prototype losses for subtypes — changes the objective, not the information in a 15 px crop; two different pretrained classifiers already plateau at ~58% on GT crops.
- **Useful for the report:** citations to verify — Oriented R-CNN (Xie et al., ICCV 2021), CAD-Net (Zhang et al., TGRS 2019), Relation Networks (Hu et al., CVPR 2018), Graph R-CNN, ST-FSOD (incompletely annotated novel objects); and the xView taxonomy list (Pickup, Utility, Dump, Haul, Trailer, Truck Tractor w/ Trailer…) that supports the H5 nuance: unboxed vehicles in our images may be xView classes *outside* our five, i.e. correctly unlabeled.
- Its citation markers (【n†L..】) are tool-internal and unresolvable; nothing from it is cited without an independent source.
Net effect on the plan: none.

## Implication for the plan
- Keep improving the score, but **reframe the target** in the report against the published xView range.
- **Raise E4 (P2 head)** to the first detection experiment after E2/E3: it has the most consistent tiny-object evidence and the lowest cost.
- **Add E10 (grey out unlabelled confident detections in train tiles)**: cheap, and the only lever aimed at H5.
- **Add E9 (NWD assignment)** as a candidate only if E4 helps detection but small-object recall is still low; it needs a code patch.
- **SR**: read arXiv 1907.05283 before deciding; skip if interpolation is shown to be competitive.
- **Add E11 (coarse-to-fine: 1-class pretrain → 5-class fine-tune)** as a cheap candidate aimed at the detection weakness.
- Report metrics at two levels: truck (class-agnostic) and subtype.
- Context and xView framing are as above. Reduced focal loss stays a low-priority note.
