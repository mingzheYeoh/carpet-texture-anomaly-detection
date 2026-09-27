# Clean test evaluation

Measured locally on 2026-09-26. All three configurations and normal-calibration
thresholds were fixed before test scoring. No test-based parameter or threshold
adjustment was performed. These are image anomaly detection results, not defect
classification accuracy or a validated deployment guarantee.

## Dataset and protocol

Original MVTec AD carpet: 224 normal fitting images, 56 normal calibration images,
117 untouched official test images (28 normal, 89 anomalous). Shared manifest
SHA-256: `ac41e4abf3f98d7cfc2805480d1ad283094dfa28ecbacb98d0b4db6fab8ae2bc`.
Each decision uses score > that model's frozen 95th-percentile calibration threshold;
ties are normal. Higher scores are more anomalous; they are not probabilities.

## Image-level results

| Model | AUROC | Precision | Recall | F1 | Normal false-positive rate | TN / FP / FN / TP |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LBP + OCSVM | 0.6344 | 0.9000 | 0.4045 | 0.5581 | 14.29% | 24 / 4 / 53 / 36 |
| LBP + GLCM + OCSVM | 0.5353 | 0.7200 | 0.4045 | 0.5180 | 50.00% | 14 / 14 / 53 / 36 |
| PatchCore | 0.9860 | 0.9355 | 0.9775 | 0.9560 | 21.43% | 22 / 6 / 2 / 87 |

Exact values are in [clean_comparison.csv](tables/clean_comparison.csv).
ROC and confusion-matrix figures: [LBP](figures/lbp_roc_confusion.png),
[LBP + GLCM](figures/lbp_glcm_roc_confusion.png),
[PatchCore](figures/patchcore_roc_confusion.png).

PatchCore ranked anomalies much better and detected 87 of 89 anomalous test images
at the frozen threshold, but also flagged 6 of 28 normal images. The calibration
rule does not guarantee a 5% deployment false-positive rate. The normal test subset
is small, so these rates should not be generalized to arbitrary textiles.

Adding global GLCM features did not improve this fixed baseline: both handcrafted
models detected 36 anomalies, while the GLCM combination produced more false alarms.
This does not establish that GLCM is generally inferior; no search for alternative
feature settings or SVM parameters was performed.

## Descriptive breakdown by defect type

Entries are detected anomalous images / total. These types are used only for
reporting, not as trained classification targets.

| Type | LBP | LBP + GLCM | PatchCore |
| --- | --- | --- | --- |
| color | 8 / 19 | 9 / 19 | 19 / 19 |
| cut | 12 / 17 | 11 / 17 | 17 / 17 |
| hole | 8 / 17 | 6 / 17 | 17 / 17 |
| metal_contamination | 1 / 17 | 0 / 17 | 16 / 17 |
| thread | 7 / 19 | 10 / 19 | 18 / 19 |

The two PatchCore misses are `test/metal_contamination/000.png` (score
25.36200714111328) and `test/thread/018.png` (30.483518600463867), both below its
30.58691692352295 threshold. Normal false positives are good/003, 006, 009, 012,
014, and 026. The inspected normal examples show strong edge responses in their
relative heatmaps; this is an observation, not a demonstrated causal explanation.

Every error is listed in each run's `evaluation/clean/failures.csv`. Local example
figures select the first two false positives and first two misses in manifest
order, avoiding discretionary selection. Dataset images and heatmaps remain under
ignored run directories with MVTec attribution. Their relative display colors do
not change scores or decisions. No threshold was adjusted after inspecting them.

## PatchCore pixel evaluation

Global pixel AUROC: **0.9905906062731301**. All 117 raw 256x256 maps were concatenated
with nearest-neighbor-resized binary ground-truth masks; normal images use zero
masks. There were 122,808 positive pixels among 7,667,712 pixels. No per-image
min-max scaling was applied before evaluation. An independent rank-sum calculation
reproduced the AUROC to absolute tolerance 1e-12.

This metric measures pixel ranking, not calibrated binary segmentation quality.
The substantial background majority and single-category benchmark limit its
interpretation. There is no binary-mask accuracy claim.

## Measured cost

Same Windows laptop: Intel Core i5-14500HX, RTX 4060 Laptop GPU, NVIDIA driver
610.62, PyTorch CUDA runtime 12.6. Environment details are in environment.md.

| Model | Device | Fit seconds | Median inference ms | p95 inference ms |
| --- | --- | ---: | ---: | ---: |
| LBP + OCSVM | CPU | 13.4851 | 26.1178 | 38.5160 |
| LBP + GLCM + OCSVM | CPU | 13.6712 | 29.6748 | 85.7695 |
| PatchCore | CUDA | 206.8878 | 82.0218 | 88.2930 |

Inference used batch 1, five untimed warm-ups, then one timed pass per test image.
Images were decoded into RAM before timing; EXIF orientation, RGB conversion,
resizing, feature extraction/scoring, and returned output transfer were timed.
File I/O, hashing, model loading, plotting, and UI were excluded. CUDA was
synchronized around timings. PatchCore also produces a map, unlike the baselines.
These CPU/GPU results compare practical deployment paths, not equal-compute
algorithm performance. One pass on an active laptop does not quantify run-to-run
latency uncertainty. Fit-time boundaries are documented in each run's metadata.

## Artifacts and verification

Runs: `lbp_seed42`, `lbp_glcm_seed42`, `patchcore_seed42`. Each
`runs/<id>/evaluation/clean/` contains predictions.csv (including per-image timing),
metrics.json, failures.csv, by_defect.csv, roc.csv, ROC/confusion plot, and local
example images. PatchCore also has pixel_maps.npz with raw maps, masks, and paths.
Metrics bind to prediction hashes and frozen model/config/calibration/manifest
hashes, mask hashes, package versions, and evaluation-source hashes.

Verified identical ordered test paths/labels across all models, recomputed image
metrics and latency quantiles from CSVs, checked all saved decisions, cross-checked
global pixel AUROC, and confirmed frozen artifacts stayed unchanged. Unit and
regression suite: 9 passed in final verification. Scientific protocol was unchanged.
Full evaluation outputs remain local; small measured tables/plots are retained
under reports/. No public deployment or Git push was performed.

The paired brightness experiment below is now complete.


## Paired brightness study (Task 7)

Measured on 2026-09-27. Each condition uses the same 117 parent images,
labels, masks, models and original normal-calibration thresholds. RGB pixels
are EXIF-oriented, multiplied in float32 by 0.8 or 1.2, clipped to [0,255],
and truncated to uint8 before model-specific resizing. No fitting or calibration
was repeated. Delta columns are absolute changes from clean, not percentages.

| Model | Condition | AUROC | Delta AUROC | F1 | Delta F1 | FPR | Delta FPR | Changed decisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lbp_ocsvm | clean | 0.634430 | +0.000000 | 0.558140 | +0.000000 | 0.142857 | +0.000000 | 0 |
| lbp_ocsvm | brightness_0.8 | 0.668138 | +0.033708 | 0.602941 | +0.044802 | 0.214286 | +0.071429 | 11 |
| lbp_ocsvm | brightness_1.2 | 0.659310 | +0.024880 | 0.594203 | +0.036063 | 0.285714 | +0.142857 | 11 |
| lbp_glcm_ocsvm | clean | 0.535313 | +0.000000 | 0.517986 | +0.000000 | 0.500000 | +0.000000 | 0 |
| lbp_glcm_ocsvm | brightness_0.8 | 0.550562 | +0.015249 | 0.864078 | +0.346092 | 1.000000 | +0.500000 | 67 |
| lbp_glcm_ocsvm | brightness_1.2 | 0.398876 | -0.136437 | 0.847291 | +0.329305 | 1.000000 | +0.500000 | 64 |
| patchcore | clean | 0.985955 | +0.000000 | 0.956044 | +0.000000 | 0.214286 | +0.000000 | 0 |
| patchcore | brightness_0.8 | 0.985554 | -0.000401 | 0.956044 | +0.000000 | 0.214286 | +0.000000 | 2 |
| patchcore | brightness_1.2 | 0.986356 | +0.000401 | 0.956522 | +0.000478 | 0.250000 | +0.035714 | 2 |

PatchCore retains image AUROC near 0.986 at both multipliers. At 0.8 its
confusion counts remain [22,6;2,87], but two individual decisions change; equal
aggregate metrics do not imply identical predictions. At 1.2 they become
[21,7;1,88]: one more normal false positive and one fewer missed anomaly.
Its existing clean false-positive problem is therefore not resolved.

LBP false positives increase from 4/28 to 6/28 and 8/28. LBP+GLCM marks all
28 normal images anomalous under both multipliers. At 0.8 it marks every image
anomalous ([0,28;0,89]); at 1.2 it misses only three anomalies ([0,28;3,86]).
Its apparently higher F1 is misleading in isolation on this anomaly-heavy test
set: false-positive rate is 1.0, and AUROC at 1.2 drops from 0.5353 to 0.3989.
No thresholds or feature settings were changed after observing these results.

This supports brightness sensitivity comparisons only for these two synthetic
conditions on the original carpet test set. The 702 transformed predictions
are paired repeated observations across three models, not 702 independent
images. Clipping and integer quantization are part of the perturbation. No
claim is made about other lighting changes, cameras, materials, or deployments.

The exact table is `tables/brightness_comparison.csv`. Each local condition
folder retains predictions, metrics, errors, ROC/confusion and example figures;
PatchCore additionally retains raw maps and unchanged masks. Frozen hashes,
parent IDs/labels and mask hashes match clean. All image metrics are recomputed
by `compare`; raw map hashes, finite shapes and identical masks were separately
verified. CSV loading uses round-trip float precision to preserve frozen thresholds.

Brightness generation is excluded from scoring latency. These later timings
are much lower than the earlier clean run across all models; they are separate
laptop sessions, so no speedup from brightness is inferred. Use Task 6's primary
latency comparison; session-to-session load/thermal effects were not controlled.

Next: Task 8, the local React + Tailwind interface and thin FastAPI inference API.
