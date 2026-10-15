# 5.4 subset-selection rule (declared before any subset training)
*Written 2026-10-10 ~00:20 IST (rev 1 image-matched, replaced within minutes by rev 2 instance-matched), before any 5.3/5.4 run; file mtime is the authoritative timestamp. Any later change goes in DECISIONS.md with a reason.*

## Question
What is the smallest training subset that reaches ≥90% of the full-data mAP50, and does a declared selection rule beat random selection at the same size?

## Fixed setup (shared with 5.3)
- **Unit:** whole source images. All tiles of an image go together.
- **Strata:** the rarest class present in each image (`experiments/splits/strata.csv`); unlabeled images form their own stratum.
- **Budget = labeled instances (rev 2).** Per stratum, the selected subset matches the mean instance count of the two random subsets at the same fraction (seeds 0 and 1). The unlabeled stratum matches the random image count.
  - Rev 1 matched image counts instead. That let the rule pick dense images: 3568 instances at 12.5% vs ~910 for random, so it would have "won" on label quantity, not selection quality.
- **Sizes:** f = 12.5% and 25% (plus one extra size in the needed direction if both pass or both fail, as in the plan).
- **Random baseline:** the frozen nested random subsets `random_s{0,1}_*.txt` (created 9 Oct, before this rule).
- **Training:** the same reduced recipe for all 5.3/5.4 runs (E1 recipe: YOLO11s, 640 tiles at imgsz 1280, 30 epochs), seeds 0 and 1.
- **Evaluation:** `evaluate.py` on full val images; `last.pt`.
- **Reference:** the 90% target is relative to the **full-data run under the same reduced recipe**, not the final model.

## Selection rule
Within each stratum, rank images by

> **score = share of an image's labeled instances that fall in the 5.2 regions {easy, medium, ambiguous}**

(from `experiments/results/cart_E0_s0_populations.csv`), descending. Ties are broken by image id. Images are then taken in rank order until the stratum's instance budget is reached (the image that crosses the budget is included only if that brings the count closer to the budget).

It **excludes** two kinds of instance from the count:
- **hard-confused:** another class is consistently preferred, a candidate label-noise or ambiguity population.
- **hard-missed:** never predicted, which suggests these are unlearnable at this resolution.

Rationale (stated before seeing any result):
- At small budgets, the data-pruning literature (Sorscher et al. 2022, "Beyond neural scaling laws") finds that keeping *easier, cleaner* examples beats keeping hard ones. Hard examples help only when data is plentiful.
- Our subsets are 55–111 images, a small-budget regime, so the rule prefers images rich in learnable, consistently labeled instances.
- The rule uses a proxy trained on full data (E0 cartography). This is standard in data-pruning work and is stated as a limitation: such a proxy is not available before data collection.

## Read-out
- **Primary:** mAP50 of selected vs random at each size, mean ± range over 2 training seeds; and whether the gap exceeds both seed spreads.
- **Report:**
  - instance and image counts per class for every subset (instances are matched by construction; image counts will differ)
  - the paired-bootstrap Δ vs random
- **Smallest passing size:** the smallest *tested* size where mean mAP50 ≥ 0.9 × full-data mAP50.
