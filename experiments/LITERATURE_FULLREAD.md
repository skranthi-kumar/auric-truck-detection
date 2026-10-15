# Full-text reading notes (13 Oct 2026)

Each paper below was read in full text: PDF from arXiv, converted with pdftotext, read directly by me rather than through a summariser. Numbers are quoted from the paper's tables or text, with the section given. **Bold** marks what matters for us.

## 1812.04098 Shermeyer & Van Etten, Effects of Super-Resolution on Object Detection in Satellite Imagery (CVPRW 2019)
- xView WorldView-3 at 30 cm. Classes are aggregated, so all truck subtypes fall into one "Bus/Truck" class (Sec 3). The authors note: "We chose an aggregation schema due to inconsistent labeling... many objects are mislabeled or simply missed... Truck w/Liquid has only 149 examples, poorly differentiated from... Truck w/Box (3653)... looks very similar."
- Metric: IoU **0.25**, not 0.5 (Sec 6). Their detectors are trained AND tested on the SR imagery.
- Table 4 (YOLT): native 30 cm mAP 0.53±0.03 -> VDSR/RFSR 2x (15 cm) 0.60±0.03 (**+1.7 sigma, i.e. not strongly significant**). SSD: 0.30 -> 0.36-0.41 (+3.5-4.2 sigma). Abstract: 30->15 cm "13-36% improvement".
- The gain is largest for boats, large aircraft and buses/trucks (Sec 8). "Performance increases significantly once objects are greater than ~20 pixels in extent."
- **Relevance:** the gain needs a detector RETRAINED on SR imagery, with VDSR training taking 55.9 h on a Titan Xp. Our RF-DETR breaks under test-time upscaling (0.038). Our YOLO E1 2x nearest-neighbour upsampling (+0.034) is the cheap version of this. The paper's earlier "16-20%" snippet was inaccurate: the true YOLT figure is +13% at 1.7 sigma. **Not actionable in 1.5 days.** It does give citable support for xView truck-label noise.

## 1907.05283 Koester & Sahin, Super-Resolution vs Nearest-Neighbour Interpolation for Object Detection on Satellite Data (2019)
- xView, 846 labelled images split 676/170. The 22 truck/vehicle classes are merged into one "vehicle" class. YOLOv2, ImageNet init. Metric: AP at IoU 0.5 (Sec III).
- Pipeline: 208 px tiles (overlap 50) -> upscale x4 -> 416 px tiles (overlap 50), with intersection-over-area (IOA) merging at 0.75 to keep adjacent trucks.
- Table I: 1-stage tiling **0.4548**; NN x2 **0.4889** (+3.4 pts); 2-stage tiling + MDSR x4 **0.6833**; 2-stage + **NN x4 0.6835**.
- Conclusion: **a learned SR model gives NO advantage over nearest-neighbour upsampling** (difference 0.0002 AP). The gain comes from making objects bigger for an ImageNet-pretrained backbone.
- **Relevance:** it settles the SR question. Using a super-resolution model is pointless, and NN upsampling is what matters. Our YOLO E1 used NN x2 (+0.034). This paper's x4 gave a much larger jump than x2 (+22.8% vs +3.4 pts), but for a single merged class and with retraining. **YOLO at NN x4 has never been tried here.** It is costly (16x the pixels, retraining) and would help only the independent YOLO line, whose ceiling is classification. Not for 1.5 days.

## 2404.03507 DQ-DETR (ECCV 2024)
- AI-TOD-V2 (mean object 12.7 px, 1 to 2,267 objects per image). ResNet50 and deformable DETR, trained from scratch on that dataset.
- Table 6, **fixed 900-query DINO-DETR by objects per image, AP50:** N<=10: 53.1; 10-100: 58.8; 100-500: 67.3; **N>500: 27.9**. With dynamic queries (1,500 for N>500), DQ-DETR reaches 35.6. Overall AP50 61.3 -> 68.6.
- Table 5 ablation: counting module + dynamic query selection alone gives +2.2 AP (25.9 -> 28.1); feature enhancement adds the rest (-> 30.2).
- Text (Sec 4.4): "In dense images, the detection limitation of DINO-DETR with only 900 queries leads to many objects undetected (FN)."
- Correction: the earlier search snippet "LRP FN 75.1 vs 40.7" does NOT appear in this version (v6). Treat it as unverified.
- **Relevance:** direct evidence that a fixed query budget collapses recall in images with far more objects than queries (AP50 67 -> 28). Our checkpoint has 300 queries per 1008 tile, shared across ALL 60 xView classes (cars, buildings, etc.), so dense depot tiles saturate. **This supports D4 (smaller tiles = more queries per m^2) and explains why denser views helped (+0.069).** DQ-DETR itself needs retraining, so it is not usable directly.

## 2202.06934 SAHI, Slicing Aided Hyper Inference (ICIP 2022)
- xView (60 classes, random 75/25 split), MMDetection detectors. Inference slices are 400x400 for xView. PO = 25% patch overlap.
- Table 2 (xView val AP50): full-image inference with no slicing 2.1-2.2; **slicing + sliced fine-tuning 15.8-19.4**; **+25% overlap gives a further +1.3-1.7 AP50** (TOOD 19.4 -> 20.6, VFNet 16.0 -> 17.7, FCOS 15.8 -> 17.1); adding the full-image pass alongside slices slightly HURTS small objects (TOOD 20.6 -> 20.4).
- Sec 4: "major room for improvement after fixing category confusions and false positives" on xView (error analysis, Fig 4).
- **Relevance:** independent confirmation that patch overlap gives a small but consistent gain on xView, consistent with our D1. A full-image or coarse pass adds nothing for small objects, so a coarser tile size is not worth trying. On xView, category confusion is again named as the main error.

## 2601.22164 Do Open-Vocabulary Detectors Transfer to Aerial Imagery? (Jan 2026, single-author preprint)
- LAE-80C (validation splits of DOTA-v2, DIOR, FAIR1M **and xView**), strict zero-shot. Metric: precision/recall/F1 at **IoA 0.7, score threshold 0.1**, NOT AP50.
- Table II: OWLv2 P 0.313 / R 0.247 / F1 0.276 (21,408 TP vs 47,058 FP); LLMDet F1 0.125; Grounding DINO variants F1 0.005-0.074; YOLO-E 0.039; YOLO-World 0.028.
- Sec IV-C: going from 80 prompts to ~3.2 prompts (oracle mode) gives 15x F1 for Grounding DINO (0.5% -> 7.4%). An "Aerial view of" prefix and synonym expansion both hurt.
- Sec IV-D / Fig 7: performance drops sharply on the FAIR1M and xView parts ("extremely small GSD and tiny objects, often <20 px").
- **Relevance:** zero-shot open-vocabulary detectors fail worst on exactly our data (xView, small objects). OWLv2's "recall" uses a lenient IoA metric. **Rejected as a recall source.** The paper reaches the same conclusion as ours: semantic confusion is the bottleneck.

## 2401.08056 DN-TOD, Robust Tiny Object Detection amidst Label Noise (ISPRS preprint 2024)
- AI-TOD-v2 (tiny objects), synthetic noise. Sec 3.3 / Fig 3: **missing and extra labels barely hurt (<2 mAP at 60% noise), while class shifts and inaccurate boxes hurt a lot** (30% box noise roughly halves mAP).
- Table 1 (class-shift noise): frequent classes barely move (vehicle 24.5 -> 24.2 at 40%), but **rare classes collapse to ~0** (storage-tank-like SP 4.7 -> 0.0, airplane 0.9 -> 0.0, windmill 1.1 -> 0.0). Method CLC (class-aware label correction) recovers part of that: Faster R-CNN 14.1 -> 16.6 mAP at 40%.
- Table 4 (real-world noisy AI-TOD labels): FCOS* AP50 32.6 -> DN-TOD 34.9 (= the clean-label model).
- **Relevance:** gives a mechanism for our observations. Label class-shifts (which xView has; see 1812.04098) disproportionately destroy rare-class AP. That matches Tractor/Flatbed/Liquid being the classes that collapsed when we fine-tuned. DN-TOD is a training-time method (12 epochs, MMDet), so **not feasible now**. It is good citable support for the report's "label ambiguity limits rare classes" argument.

## 2407.17738 Orthogonal Mapping for fine-grained aerial detection (GRSL letter 2024)
- Table I (FAIR1M-v1.0 online validation, AP0.5, ResNet-50, 8 detectors): **Cargo Truck 33.2-42.8; Truck Tractor (TT) 0.24-0.72; agricultural Tractor (TRC) 0.96-2.61; Trailer 3.5-15.5**; overall AP0.5 34.1 -> 41.2 (PETDet w/ OM). The abbreviations are not defined in this paper; the mapping TRC = tractor, TT = truck tractor comes from PETDet (2312.10515, Sec IV-A). An earlier version of this note had them swapped.
- Sec III-C: OM gives "less substantial improvement" for Tractor because of "limited presence in the dataset (only 0.07%)". The method needs retraining (MMRotate, 12 epochs).
- **Relevance:** verified first-hand (my earlier summary was correct). Even state-of-the-art fine-grained detectors trained on ~16k FAIR1M images reach **Truck Tractor <= 2.6 AP50**. Our Tractor of 0.04 (4 AP50) is in the same range. **Strong report evidence that 0.75 is unreachable for these classes.**

## 1802.07856 xView: Objects in Context in Overhead Imagery (Lam et al. 2018)
- Sec 3.2: "If the labeler can confidently determine an instance to be of a fine grained child category, then that labeler will classify the instance at the child level. However, if the labeler is unable to make a confident determination, then they will fall back to the more general parent category." So ambiguous trucks become the parent "Truck" class, which is NOT in our 5 classes. Clusters "could be labeled as singular due to human error."
- **Sec 3.3 (QC): a batch passed expert quality control if it had "precision of 0.75 and recall of 0.95 at 0.5 IoU when compared to the gold standard."** The labels themselves were only required to agree with expert gold labels at 75% precision. The text does not say whether this was class-aware.
- Sec 4: SSD; 300 px non-overlapping chips (vanilla) vs 300/400/500 multi-resolution chips; 70/30 split; 7 days on 4 M60s.
- Table 1 (AP, IoU 0.5), vanilla / multires / aug: Cargo Truck 0.049/0.097/0.045; Truck w/Box 0.136/0.086/0.119; Flatbed 0.032/0.126/0.034; **Truck Tractor 0.005/0.213/0.000; Liquid 0.018/0.474/0.000**; Truck 0.153/0.047/0.136. Total mAP 0.146/0.259/0.155. Mean of our 5 classes: vanilla 0.048, multires 0.199.
- **Relevance:**
  1. A citable fact for the report: xView's own QC accepted labels at 0.75 precision against gold. A detector scored against such labels cannot be expected to exceed that consistency on confusable subtypes, which directly frames the 0.75 target.
  2. The multires gain (mixed chip sizes in TRAINING) is the literature analogue of our D4 (a different tile size at inference). Their per-class jumps are noisy, since one class goes up while "Truck" goes down 0.153 -> 0.047, consistent with heavy class confusion.
  3. Our 0.274 (in-sample) and 0.139 (independent) can be set against their 0.048 / 0.199 on the same 5 classes, with the caveat of different splits and protocol.

## 2312.10515 PETDet (IEEE TGRS 2024)
- Sec IV-A defines the abbreviations: **tractor (TRC), truck tractor (TT)**. FAIR1M v1.0 has 16,488 train / 8,137 test images; v2.0 adds a val set and a larger test set. AP50 comes from the online server (VOC 2012).
- Table II (FAIR1M-v2.0, AP50, 12 ResNet-50 detectors): **Cargo Truck 8.5-55.6; Truck Tractor (TT) 0.0-33.1** (PETDet 33.1, ORCNN 5.4, FRCNN-O 13.5); Trailer 0.0-16.0; agricultural Tractor (TRC) 0.0-1.9. Overall AP50 26.1-48.8.
- **Relevance:** Truck Tractor results swing wildly between methods and dataset versions (v1.0 server <= 0.72 vs v2.0 up to 33), a sign of a very hard, low-sample, label-sensitive class. Even the best result (33 AP50) is far below 0.75.

## 2103.05569 FAIR1M (ISPRS 2022)
- Sec 3.1: Gaofen + Google Earth, **0.3-0.8 m**, more than 15k images. Vehicle classes: small car, bus, cargo truck, dump truck, van, trailer, tractor, truck tractor, excavator. Objects under 16 px are not annotated (Sec 3.3.1). Three-stage annotation QC (Sec 3.3.2).
- Benchmarks crop 1024x1024 with **stride 256 (75% overlap)** (Sec 4.1.2).
- Table 2 (their own test, oriented boxes, R-101, AP%): **Cargo Truck 44.2-54.0; Truck Tractor 29.0-46.7; Tractor 0.9-16.4; Trailer 13.7-19.2**; mAP 29.5-34.7. Their fine-grained metric mAP_F (FIoU + score-weighted) is much lower: 7.9-19.1.
- **Relevance:**
  1. Per-class truck APs differ a lot between the paper's own test and the server versions, so these numbers should be cited as ranges.
  2. A standard practice in this literature is **75% tile overlap**, while we use 50% in D1. That makes a 75%-overlap test (or the rot90 views) a well-precedented inference change.
  3. FAIR1M stops at 0.8 m GSD for vehicles, a coarser resolution than ours.

## 2505.05741 Dome-DETR (ACM MM 2025)
- AI-TOD-V2 / VisDrone, D-FINE-based, trained on 8x 4090. Table 3 ablation (AP, AI-TOD-V2): baseline 30.1; +DeFE+MWAS 31.2; +DeFE+PAQI (adaptive query allocation) 32.1; all three 33.3. Table 1: AP50 for Dome-DETR-S/M/L is 67.5/68.4/69.2, vs DQ-DETR 68.6 and DINO 61.3.
- Sec 3 motivation: fixed-query DETRs have "insufficient queries in dense" scenes. Dynamic NMS (Eq 14) raises the IoU threshold in dense regions; Table 4 shows (0.4, 0.9) is best.
- **Relevance:** a second independent source that query allocation is a real limitation for dense tiny objects (density-adaptive queries give about +1-2 AP). Like DQ-DETR it requires training a new architecture. **Not usable directly. It supports the tile-size / query-saturation hypothesis behind D4.**

## 2604.19233 ASAHI, Adaptive Slicing-Assisted Hyper Inference (arXiv Apr 2026, preprint)
- TPH-YOLOv5 with adaptive slice count. Table 3 (xView-test): SAHI 512 px mAP50 20.4 -> ASAHI 22.7. Table 5 (xView-val): full image 14.8 -> sliced 24.4 mAP50; +25% patch overlap +0.1.
- **Table 7 (VisDrone-val, post-processing): NMS 33.9 mAP / Soft-NMS 32.3 / WBF 33.4 / Cluster-CIoU-NMS 35.9 / CDN 36.0.** Soft-NMS is WORSE than plain NMS.
- Sec 4.8: on xView, "localization errors (14%)" and category confusion remain the dominant failure sources.
- Reliability caveat: some Table 3 columns look misaligned (e.g. "4 slices" mAP 7.5 but mAP75 18.4). It is a recent preprint; use with caution.
- **Relevance:** evidence AGAINST switching to Soft-NMS (ChatGPT's suggestion). The slicing gains are again about recall.

## 2303.17937 STFAR, test-time self-training (2023)
- The setting is synthetic CORRUPTION shift (COCO-C, Pascal-C, Foggy Cityscapes). Table 1 (COCO-C, R-50): direct test 15.9 mAP -> STFAR 24.7. **BN adaptation 4.7 and TENT 3.6 collapse far below direct test.** Table 9 (DETR, snow): 12.1 -> 21.7. Table 10: too many update steps start to overfit.
- **Relevance:** test-time adaptation targets large appearance shift such as fog, noise or blur. Our val images are clean imagery from the same source, and for RF-DETR they are even in-sample, so the mechanism does not apply. Naive variants (BN/TENT) are catastrophic. **Rejected.**

## Off-domain vehicle papers (read: abstract, method, conclusion)
- **2606.05149** (two-stage RT-DETR + ViT-B/16, 2026): ground-level roadside video, 6 body types, 0.94 accuracy in-distribution, 0.89 OOD, with confidence abstention at 0.60. It is not satellite imagery. **The only transferable idea is two-stage detect-then-classify, which we already test (E1-2stage; planned T2).**
- **2608.17917** (TNO et al., out-of-the-box ATR, 2026): YOLO and DETR on a military air-to-ground set, fine-tuned on VisDrone. Conclusion (Sec IV): bigger models do better; DETRs beat YOLO after fine-tuning; "all models still struggle with detecting small objects"; "in-domain training still is crucial." No satellite data. **Not actionable.**
- **2402.00703** (Vehicle Perception from Satellite videos, 2024): Jilin-1 satellite video and GTA-V synthetic data, a single "vehicle" class, a few pixels per object, focused on counting and tiny-object detection. No fine-grained classes and no weights for our setting. **Not actionable.**

## Remote-sensing CLIP models: 2306.11029 RemoteCLIP, 2312.12856 SkyScript/SkyCLIP, 2409.00698 RS-VLM zero-shot
- SkyScript Table 2 (zero-shot top-1): the "fine-grained" tests are **roof shape, road smoothness and surface material**, not vehicles. SkyCLIP-50 ViT-L reaches 46.8 / 35.8 / 67.5%. **RemoteCLIP ViT-L is WORSE than original CLIP on 5 of 8 scene benchmarks** (e.g. SkyScript 34.4 vs 55.1; fMoW 16.8 vs 26.2).
- RemoteCLIP: evaluated on retrieval and scene classification (RSICD, RSITMD, UCM, AID...). Vehicles appear only in captions ("4 large vehicles"); there is no vehicle-subtype benchmark.
- 2409.00698: scene-level zero-shot classification only.
- **Relevance:** none of these models has been evaluated on vehicle-subtype recognition, let alone on ~30 px crops. Our own frozen DINOv3-SAT probe (0.351 oracle) already tested a stronger satellite foundation model on exactly our crops. **Rejected.**

## Geospatial VLM benchmarks: 2411.19325 GEOBench-VLM, 2412.15190 EarthDial
- GEOBench-VLM: 31 tasks; **the best model (LLaVA-OneVision) reaches only 41.7% MCQ accuracy**, "slightly above double the random guess" (Sec 4, Fig 4); "almost all models struggled" with counting (Appendix).
- EarthDial: a conversational EO assistant. Detection and grounding output is coarse ("1 vehicle at the bottom left"); there is no vehicle-subtype evaluation.
- **Relevance:** multimodal LLMs cannot localise or classify 30 px truck subtypes. **Rejected.**

## 2408.09110 LAE-DINO / LAE-1M (AAAI 2025)
- Table 6: LAE-1M's fine-grained part includes **xView (1,129 images, 1,024 px crops, 26,543 crops, 60 classes)**, sampled at 0.2 per class (Sec "LAE-1M", p.4). Its category list explicitly contains **Cargo Truck, Truck w/Box, Truck Tractor, Truck w/Flatbed, Truck w/Liquid**. Also DIOR, FAIR1M, DOTAv2, NWPU, RSOD, HRSC2016, etc.
- **Relevance:** 1,129 images is about the whole xView image set, which would include our val images, and our exact 5 classes were in its training labels. Using it would repeat the RF-DETR overlap. **Rejected on overlap grounds (now verified from the paper itself).**

## 2408.12246 RT-OVAD and 2505.03334 OS-W2S (open-vocabulary aerial detection)
- RT-OVAD (Sec 4.1): "the model is first pretrained on LAE-1M", which contains xView with our 5 truck classes (see LAE-DINO). It also uses xView for its zero-shot split. **Same overlap problem; rejected.**
- OS-W2S (Sec 5): zero-shot Grounding DINO detection **AP50 4.0%**, YOLO-World sentence-level 1.4%. Training on their MI-OAD data lifts Grounding DINO to 9.8% detection AP50. **Confirms that zero-shot open-vocabulary detectors are useless at our scale.**

## 2403.13430 MTP, multi-task pretraining for RS foundation models (2024)
- Sec IV-B, xView horizontal detection: "we randomly select **700 images as the training set and 146 images for testing**" from the 846 labelled xView images, with RetinaNet and 416 px clips. Table V (60-class AP50): random init 10.8; IN1K R-50 14.4; best MAE+MTP ViT-L+RVSA **19.4**.
- **Relevance:**
  1. MTP's released xView detectors were trained on a RANDOM 700 of the 846 labelled xView images, so they very likely saw most of our 22 val images. That rules them out on overlap grounds.
  2. MTP's FAIR1M detectors (no xView) remain independent, but they need heavy MMRotate setup.
  3. A 60-class xView AP50 of ~19 with large foundation backbones is useful context for our numbers.

## 2102.12219 DOTA benchmark (TPAMI 2021)
- Sources: Google Earth, Gaofen-2 and Jilin-1 satellites, plus CycloMedia airborne (Rotterdam, including OBLIQUE 45-degree views) (Sec 3.2). The GSD is mixed.
- Table 8 (DOTA-v1.5, class-wise AP): **Large Vehicle (LV) 58.5-81.2 AP** across 8 baselines; Small Vehicle 71-79.
- **Relevance:** a DOTA-trained detector is a competent COARSE large-vehicle detector. That supports the YOLO11-OBB idea as a recall-only proposal source (no subtypes, mixed GSD). It contains no xView imagery, so there is no overlap.

## 2303.02404 Fine-Grained Classification with Noisy Labels (CVPR 2023)
- Abstract and Sec 1: in fine-grained data, "large inter-class ambiguities among fine-grained classes cause more noisy labels", and generic noisy-label methods "fail to achieve satisfying performance". The proposed SNSCL (noise-tolerant contrastive loss) is a training-time method.
- **Relevance:** supports the label-ambiguity framing in the report. Not actionable in the time left.

---
## SYNTHESIS (all 27 papers)
**What the full texts change compared with the earlier snippet-level claims:**
1. Super-resolution: the true gain is +13% (YOLT, +1.7 sigma) and needs detector retraining. **NN upsampling performs identically to learned SR** (0.6835 vs 0.6833, Koester 2019). A super-resolution model is pointless.
2. FAIR1M Truck Tractor: **<= 0.72 AP50 on the v1.0 server**, up to 33 on v2.0, and 29-47 on the paper's own split. My earlier "~2" swapped two columns (TRC is agricultural tractor).
3. The DQ-DETR "LRP FN 75 vs 41" snippet is not in the paper. The real evidence is AP50 67.3 -> 27.9 for images with N>500 under 900 fixed queries.
4. **Soft-NMS is worse than NMS** (ASAHI Table 7: 32.3 vs 33.9 mAP).
5. **xView QC accepted labels at precision 0.75 / recall 0.95 against gold** (Lam 2018 Sec 3.3). This is the most direct framing of the 0.75 target.
6. **LAE-DINO, RT-OVAD (LAE-1M) and MTP-xView detectors were all trained on xView**, including our 5 classes or a random 700/846 split. They are overlap-equivalent to the RF-DETR checkpoint.
7. Zero-shot open-vocabulary detectors and geospatial VLMs are useless at this scale (AP50 1-5%; MCQ 41.7%). Test-time adaptation targets corruption shift and does not apply.
8. Class-shift label noise destroys RARE-class AP while frequent classes barely move (DN-TOD Table 1). This is the mechanism behind our fine-tune collapse.

**Inference-only levers that remain supported:**
- smaller tiles / more queries per area (DQ-DETR, Dome-DETR, xView multires);
- more views per object (SAHI overlap +1.3-1.7; 75% overlap is standard in the FAIR1M benchmarks; rot90 views as in the xView winner);
- a coarse large-vehicle proposal source (DOTA LV AP 58-81).

**Ruled out:** Soft-NMS, SR models, open-vocabulary detectors and VLMs, test-time adaptation, and xView-trained third-party weights.

## 2512.24074 BHCL: Balanced Hierarchical Contrastive Learning with Decoupled Queries (CVPR 2026); code github.com/chncdx/BHCL
- DETR-based (built on OrientedFormer / RHINO, MMRotate 1.x, Apache-2.0). Decoupled classification vs localisation queries + a hierarchical contrastive loss over the label tree, balanced across classes. **No pretrained BHCL weights released** (repo README, checked 13 Oct). FAIR1M preprocessing script "lost". No xView.
- Table 1 (FAIR1M-v1.0 AP50): OrientedFormer 41.31; +Decoupling 41.38; +Decoupling+HCL 41.14; +Decoupling+BHCL **41.66** (+0.35). ShipRSImageNet AP50:95: 63.17 -> 64.32.
- Table 3: FAIR1M-v1.0 AP50 36.83 (FRCNN) to 41.66 (BHCL); v2.0 39.37 to 47.53. No per-class vehicle numbers in the main text.
- Sec 3/4.1: "Other*" instances are REASSIGNED TO THE PARENT CLASS (e.g. Other Vehicle -> Vehicle) inside the hierarchical loss, not used as a reject class.
- **Relevance:** a training-time method, gains of +0.35 AP50 on FAIR1M-v1.0, no weights. Not usable in our time frame. The hierarchical idea (ambiguous objects -> parent class) parallels xView's own labeller fallback to "Truck".
- NOTE: a pasted AI summary claimed BHCL contains "NorCal" and "reject-class verification". The paper text contains neither word (grep: 0 hits). NorCal is a separate paper (Pan et al., NeurIPS 2021).

## 2107.02170 NorCal: On Model Calibration for Long-Tailed Object Detection (NeurIPS 2021); code github.com/tydpan/NorCal
- Post-processing only, no retraining. Softmax detectors (Eq 4): divide exp(logit_c) by a_c = N_c^gamma (N_c = training images of class c), keep the BACKGROUND logit intact, renormalise over all classes incl. background. One hyper-parameter gamma, tuned on TRAINING data (Fig 5: train and val AP-vs-gamma trends match).
- Table 1 (LVIS v1, Mask R-CNN RFS R-50): AP 22.58 -> 25.22; APr 12.30 -> 19.33 (+7.0); APf +0.43. Table 3: WITHOUT normalisation, results collapse (e.g. 17.52), so normalisation is the key ingredient (Sec 4.3).
- **Sigmoid detectors (Sec 3.4, Eq 7):** s_c = (e^phi_c / a_c) / (e^phi_c / a_c + 1). The authors state the drawback: "the infeasibility of normalization across classes", and the gains for sigmoid models come from re-ranking ACROSS classes under the per-image cap of 300 detections. Table 1: EQL (sigmoid) frequent classes -0.1.
- **Relevance (our analysis, not from the paper):**
  1. Our RF-DETR uses per-class sigmoids, has no background logit, and our evaluator has NO per-image cap with per-class AP. Eq 7 is monotone within each class, so applying it to the detector leaves every per-class AP UNCHANGED, i.e. zero effect by construction. CS1 (a cross-class normalisation) already measured ~0 (0.194 vs 0.195).
  2. NorCal's normalised form CAN apply to the softmax classifiers in T2/T3 (ConvNeXt 5-/6-way softmax): p_k / N_k^gamma, renormalised, with the 6th "other vehicle" class of T3 playing the role of the untouched background. That changes within-class ranking. gamma would have to be chosen on train data only. Untested.
