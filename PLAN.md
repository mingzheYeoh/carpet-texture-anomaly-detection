# Carpet Texture Anomaly Detection — Implementation Plan

Version: 1.1 | Date: 2026-09-26 | Status: implementation in progress; see reports/progress.md. UI revised at the owner's request.

## 1. Project purpose

Build a small, reproducible academic computer-vision project comparing handcrafted texture features against pretrained deep features for carpet surface anomaly detection.

**Research question:** When fitting uses only normal carpet images, how do LBP, LBP + GLCM, and PatchCore differ in image-level anomaly detection, inference cost, and sensitivity to brightness changes?

**Repository name:** `carpet-texture-anomaly-detection`

**Suggested GitHub description:** Reproducible carpet defect detection comparing LBP/GLCM + One-Class SVM with PatchCore, including anomaly heatmaps and an interactive local web demo.

The application accepts a carpet image and returns a normal/anomalous decision, an anomaly score, and (for PatchCore) a spatial anomaly heatmap. Scores are not probabilities. This is a study of one dataset category, not a validated detector for arbitrary textiles.

## 2. Scope and decisions

| Area | Decision |
| --- | --- |
| Dataset | Original MVTec AD, `carpet` category only; not MVTec AD 2 |
| Learning setting | Normal-only fitting with held-out normal threshold calibration |
| Experiment A | Spatial LBP histograms + StandardScaler + One-Class SVM |
| Experiment B | Spatial LBP histograms + global GLCM statistics + StandardScaler + One-Class SVM |
| Experiment C | PatchCore with frozen pretrained Wide ResNet-50-2 |
| Main comparison | Image-level anomaly detection |
| Localization | PatchCore only, evaluated separately |
| Hardware | User's Predator Helios Neo 16, Intel i5, RTX 4060 Laptop GPU; measure actual RAM/VRAM |
| Platform | Windows local development, Python 3.11 preferred subject to dependency compatibility |
| Interface | React + Tailwind CSS single-page frontend and thin FastAPI inference API, local only |
| Credentials | No LLM API key, paid API, database, or cloud subscription required |
| Exclusions | Defect-type classification, webcam pipeline, cloud deployment, CNN fine-tuning, extra datasets |

PatchCore fits a memory bank using frozen pretrained features. Do not describe it as fine-tuning or invent epochs, optimizer settings, or training-loss curves. One-Class SVM is different from a supervised binary SVM: it uses only normal fitting samples.

## 3. Owner: create and clone the repository

1. In GitHub, create `carpet-texture-anomaly-detection` under your account.
2. Choose the desired visibility. Initialize with a README and the Python `.gitignore` template.
3. Add an MIT license for your original code if desired; it does not relicense the dataset or third-party model assets.
4. Copy the repository's HTTPS clone URL from GitHub.
5. In PowerShell, clone into the parent directory where you keep projects:

```powershell
git clone https://github.com/YOUR_USERNAME/carpet-texture-anomaly-detection.git
cd carpet-texture-anomaly-detection
code .
```

Replace `YOUR_USERNAME` before executing. If `code` is unavailable, open this folder from VS Code's File menu.

6. Put this file at the repository root as `PLAN.md`.
7. Optionally commit the plan:

```powershell
git add PLAN.md
git commit -m "docs: add research and implementation plan"
git push origin main
```

Use the actual default branch if it is not `main`. Authentication is handled by GitHub/Git Credential Manager; never put access tokens in files or commands committed to Git.

## 4. Prerequisites and environment setup

Install Git, VS Code, Python, and a working NVIDIA driver. Use a project-local virtual environment. Start with Python 3.11, but verify the supported intersection of the selected Anomalib, PyTorch, and torchvision releases before installation.

```powershell
git --version
py -3.11 --version
nvidia-smi
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Using the explicit virtual-environment interpreter avoids changing PowerShell execution policy. If the Python launcher is unavailable, use the verified installed Python executable.

The implementing agent must:

1. Inspect GPU name, actual VRAM, available RAM, driver, Python, and disk space. Record the observations in `reports/environment.md`.
2. Select mutually compatible stable versions of PyTorch, torchvision, and Anomalib using their official documentation. Do not blindly install independent latest releases.
3. Use the official PyTorch Windows/Pip/CUDA installation command appropriate to the driver. Record its exact index URL and versions in the setup guide. Do not assume the CUDA version printed by `nvidia-smi` is a locally installed toolkit.
4. Install NumPy, pandas, Pillow, OpenCV, scikit-image, scikit-learn, SciPy, joblib, PyYAML, Matplotlib, Seaborn, pytest, and the selected deep-learning dependencies. Install UI dependencies during Task 8; Streamlit from the original environment is no longer part of the planned application.
5. Verify imports, run a small CUDA tensor operation, and run a PatchCore API smoke check before processing the full dataset.
6. Freeze the verified environment into a reproducible lock/constraints file and document the PyTorch wheel source. Ensure `pip check` passes. Do not commit machine-specific editable paths from an unfiltered freeze.

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

GPU unavailable: continue CPU-compatible data and baseline tasks; explicitly report the limitation. Do not claim GPU validation. Do not install a CUDA toolkit unless a verified dependency actually needs it.

Initial deep configuration: batch size 4, FP32, `num_workers=0`, CUDA when available. On GPU OOM, retry 2 then 1 and record the change. Memory-bank construction can consume substantial host memory; reducing batch size alone may not solve it. If necessary use chunked feature extraction/distance computation. A lower coreset ratio is a separately named, documented fallback experiment, not a silent replacement.

## 5. Dataset acquisition and integrity

Download the original MVTec AD carpet category from the official provider. Follow any form or access requirements; if manual download is required, provide exact instructions and resume after the owner places the files locally.

Expected category layout under `data/raw/carpet/`:

- `train/good/`: normal training images.
- `test/good/`: normal test images.
- `test/<defect_type>/`: defective test images.
- `ground_truth/<defect_type>/`: corresponding defect masks.

Verify the actual layout and mask naming rather than assuming every extension matches. Inventory actual counts, dimensions, channels, missing masks, unreadable files, and duplicate hashes in `reports/data_audit.md`. Do not fabricate counts from memory.

Create `data/manifests/carpet_split.csv` containing relative path, SHA-256, original split, assigned split, label (0 normal, 1 anomaly), defect type, and optional mask path.

Split only official `train/good` images: sort paths, shuffle with NumPy's documented seeded generator using seed 42, take `floor(0.8*N)` for fitting and the remainder for calibration. Preserve the entire official test set. Record generator and library version. If cross-split duplicates exist, report and resolve leakage before claiming a clean evaluation; preserve the original audit and document any benchmark deviation.

All augmentations and derived crops inherit their parent image split. Calibration images cannot enter the SVM fit, scaler fit, or PatchCore memory bank. Ground-truth masks and test labels cannot influence fitting, thresholds, normalization, or hyperparameters.

The dataset uses CC BY-NC-SA 4.0. Keep data out of Git; provide acquisition instructions and attribution. Check applicable terms before distributing dataset-derived figures, weights, or assets. Include `DATA_LICENSE.md` distinguishing original code licensing from data and pretrained weights.

## 6. Shared preprocessing

- Decode consistently, apply EXIF orientation, convert to RGB, reject unreadable/unsupported images.
- Resize full images to 256 x 256 with bilinear interpolation; do not center-crop.
- Grayscale baselines: convert RGB using one documented implementation; uint8 in [0,255].
- PatchCore: ImageNet normalization appropriate to the selected pretrained weights, applied exactly once.
- Resize masks with nearest-neighbor interpolation; preserve binary labels.
- No augmentation, histogram equalization, denoising, or adaptive contrast enhancement in the primary experiment.
- Use identical preprocessing during fitting, calibration, evaluation, and demo inference.

## 7. Model specifications

### A. LBP + One-Class SVM

Compute uniform LBP on the entire 256 x 256 grayscale image with `P=8`, `R=1`. Partition the resulting map into a 4 x 4 grid; compute a 10-bin histogram in each cell, L1-normalize each histogram, and concatenate to 160 features. Keep feature order fixed.

Fit a scikit-learn pipeline: StandardScaler then OneClassSVM with `kernel='rbf'`, `nu=0.05`, `gamma='scale'`. Fit both only on fitting images. Define anomaly score as negative `decision_function` so higher is more anomalous. Use the project calibration threshold, not the estimator's built-in `predict` threshold.

### B. LBP + GLCM + One-Class SVM

Reuse the exact 160 LBP features. Quantize grayscale using `floor(gray * 32 / 256)` with floating-point or widened arithmetic (avoid uint8 overflow). Compute GLCMs on the full image with levels=32, distances=[1,3], angles=[0, pi/4, pi/2, 3*pi/4], symmetric=True, normed=True.

Extract contrast, homogeneity, energy, and correlation for all distance-angle combinations: 32 additional features, 192 total. Document feature order. Handle degenerate correlation consistently with finite output and test a constant image. Concatenate before fitting a separate StandardScaler and the same SVM settings.

### C. PatchCore

Use the maintained Anomalib implementation through a thin project adapter:

- Backbone: `wide_resnet50_2`, pretrained weights with exact identity recorded.
- Layers: `layer2`, `layer3`.
- Coreset sampling ratio: 0.1.
- Nearest-neighbor setting: explicitly configure 9 where supported by the pinned API.
- Batch size: 4 initially; FP32; workers=0.
- Frozen feature extractor, evaluation mode, no gradient optimization.
- Fit only on the fitting split, with no calibration images in the memory bank.

Verify version-specific constructors and preprocessing behavior. Disable/replace automatic validation splitting, test-derived threshold fitting, and score normalization that would violate this protocol. Use raw model image scores and raw anomaly maps for evaluation; do not reconstruct an alternative score formula if the library provides the algorithm's score.

Persist enough state to reproduce inference, including the memory bank, backbone/weight identity, configuration, transform definition, score semantics, and calibration threshold. Validate the actual supported save/load path rather than assuming a generic Lightning checkpoint retains every required buffer.

## 8. Calibration and evaluation protocol

For each fitted model, score the held-out normal calibration images. Set its image threshold to the 95th percentile using `numpy.quantile(scores, 0.95, method='linear')`. Predict anomaly when score > threshold (ties are normal). Save calibration scores and the threshold. This small-sample calibration rule does not guarantee a 5% deployment false-positive rate.

Lock configuration and thresholds before final test evaluation. All three models use the same manifest. Report:

| Metric | Scope |
| --- | --- |
| Image AUROC | All models; continuous raw scores |
| Anomaly precision, recall, F1 | All models; frozen calibration threshold |
| Confusion matrix and normal false-positive rate | All models |
| Pixel AUROC | PatchCore only; aggregate flattened test masks/maps using one documented method |
| Fit time and inference latency | All models, with hardware and timing boundaries |

For normal test images, pixel ground truth is all zeros. Spatial maps must align with 256 x 256 masks. Do not independently min-max normalize each image's map before aggregated pixel evaluation. Pixel AUROC is not a measure of calibrated segmentation quality; the first version makes no binary-mask accuracy claim.

Latency: batch size 1, same hardware, model loaded, at least 5 warm-up passes, one timed pass per test image; report median and p95. Include preprocessing + model scoring, exclude file I/O, model loading, plotting, and UI. Synchronize CUDA around timings. State that baseline CPU vs PatchCore GPU latency is a practical deployment comparison, not an equal-compute algorithm benchmark.

Persist per-image predictions (path, label, defect type, raw score, threshold, decision), aggregate JSON/CSV, confusion matrices, ROC curves, timing records, and representative failures. No invented numbers or target accuracy promises.

## 9. Controlled brightness experiment

After locking the primary pipeline, evaluate each test image at multipliers 0.8 and 1.2, applied to RGB uint8 via float multiplication and clipping before model-specific preprocessing. Keep the clean version as the reference. Labels and masks stay unchanged.

Use the original models and thresholds; do not recalibrate or refit. Record each transformed image under its original image ID and condition. Report AUROC, F1, false-positive rate, and change from clean for each condition. These are paired variants, not independent extra samples. This evaluates brightness sensitivity only, not general real-world robustness.

## 10. Repository structure

Create these paths as work progresses (a structure specification, not existing files):

```text
PLAN.md
README.md
DATA_LICENSE.md
pyproject.toml
requirements-lock.txt
.gitignore
configs/{lbp_ocsvm,lbp_glcm_ocsvm,patchcore}.yaml
src/carpet_ad/{data,preprocessing,features,models,calibration,evaluation,inference,cli}.py
app/api.py                        # thin local FastAPI adapter, created in Task 8
frontend/                         # React + Tailwind single page, created in Task 8
tests/
data/raw/                         # ignored
data/manifests/carpet_split.csv
artifacts/                        # ignored: models, memory banks, caches
runs/                             # ignored: complete local experiments
reports/{environment,data_audit,results,decisions,progress}.md
reports/tables/                   # small measured result tables
reports/figures/                  # selected figures with attribution
notebooks/01_data_exploration.ipynb
```

Use modules or subpackages as needed; do not overengineer. Notebooks call reusable package code rather than containing the only implementation.

Ignore `.venv/`, caches, raw data, model binaries, checkpoints, feature caches, full run outputs, secrets, and local uploads. Commit manifests, source, configs, small measured summaries, and documentation. Inspect `git diff --stat` and staged paths before any commit. Do not push datasets or weights to GitHub.

## 11. Command interface to implement

These are target commands, not commands that exist yet. Implement them and keep README synchronized. On Windows, `python` below means `.\.venv\Scripts\python.exe` unless the environment is activated.

```powershell
python -m carpet_ad.cli doctor
python -m carpet_ad.cli prepare-data --data-root data/raw/carpet --seed 42
python -m carpet_ad.cli fit --config configs/lbp_ocsvm.yaml --run-id lbp_seed42
python -m carpet_ad.cli calibrate --run-id lbp_seed42
python -m carpet_ad.cli evaluate --run-id lbp_seed42 --condition clean
python -m carpet_ad.cli robustness --run-id lbp_seed42
python -m carpet_ad.cli compare --runs lbp_seed42 lbp_glcm_seed42 patchcore_seed42
# Task 8 will document the verified local API/frontend launch commands.
```

Repeat fit/calibrate/evaluate/robustness for B and C with their corresponding config and run ID. `fit` cannot consume test images. `calibrate` cannot consume test labels. `evaluate` and `robustness` cannot mutate models or thresholds. Reject reused run IDs unless an explicit safe overwrite option is given. Log manifest/config hashes, code commit, package versions, seed, device, timestamps, and artifact paths for every run.

## 12. Sequential agent tasks and acceptance checks

### Task 1 — Inspect and scaffold

Read this entire file and existing repository instructions. Inspect existing work before editing. Create the package, configs, ignore rules, README setup section, and progress/decision records. Do not overwrite unrelated owner work.

**Done when:** editable installation and CLI help work, paths are portable, planned outputs are distinguished from measured results.

### Task 2 — Validate environment

Resolve and pin dependencies, verify CPU imports and CUDA, implement `doctor`, record actual hardware and exact setup commands. Smoke-test the selected PatchCore API.

**Done when:** environment report and reproducible dependency instructions exist; missing capabilities are explicitly recorded.

### Task 3 — Audit and split data

Implement inventory, integrity checks, mask mapping, deterministic manifests, and a small exploration notebook. Produce illustrative samples without using test results to tune methods.

**Done when:** split overlap tests pass, all expected masks align, counts come from actual files, and the manifest is fixed.

### Task 4 — Implement baselines

Implement A and B, fitting, persistence, score direction, and calibration. Add focused tests for LBP/GLCM dimensions, finite constant-image features, split isolation, and quantile/tie behavior.

**Done when:** both baselines load and reproduce scores within documented numeric tolerance; fitting excludes calibration/test data.

### Task 5 — Implement PatchCore

Implement the adapter and exact transforms, build memory bank, calibrate separately, save/load artifacts, and generate raw aligned maps. Start with a small normal-only smoke subset, then execute the full fitting split. Smoke results are not final results.

**Done when:** frozen weights, split isolation, persistence, map dimensions, and absence of test-based calibration are verified. Record real resource usage and any fallback.

### Task 6 — Run primary evaluation

Freeze configs, evaluate all models on the same test manifest, produce measured metrics and timing tables, and inspect failure cases. Report anomalies by defect type as descriptive analysis, not trained defect classification.

**Done when:** all summary numbers trace to stored predictions; metrics recompute correctly; no model is selected or tuned using test outcomes. If later exploratory changes occur, label their reuse of this test set honestly.

### Task 7 — Run robustness study

Execute the two brightness conditions with frozen artifacts. Produce clean-vs-brightness comparison tables and explanations bounded by evidence.

**Done when:** original thresholds are unchanged and all variants retain the same parent IDs and labels.

### Task 8 — Build the local web demo

Use React + Tailwind CSS for a dark industrial inspection interface, with a thin
FastAPI adapter calling the existing Python inference code. Keep one page and
local-only operation; no database, accounts, or cloud deployment. Include an
original/heatmap comparison slider and restrained transitions. Confirm the visual
design at this stage, without adding frontend scaffolding during model work.

Provide image upload, original image, model selector, decision, raw score, threshold, and PatchCore heatmap overlay with opacity control. Show "Localization unavailable" for the baselines. Add a results tab reading measured reports.

Allow PNG/JPEG up to 10 MB and 20 megapixels decoded. Handle corrupt files, unsupported images, missing artifacts, and unavailable GPU clearly. Cache loaded models and use local inference; do not persist uploads by default. No fabricated fallback predictions. Describe expected carpet-image input and out-of-domain limitations.

Heatmap display scaling is separate from evaluation; label any relative visualization and never let per-image color scaling change predictions. Do not show raw scores as comparable probabilities across models.

**Done when:** real artifact inference works from a fresh process and error states are usable.

### Task 9 — Document and hand over

Write the methodology, exact run commands, environment, measured results, error analysis, brightness study, limitations, data attribution, and a short demo walkthrough. Include a concise explanation of LBP, GLCM, One-Class SVM, and PatchCore for the owner.

**Done when:** another developer can reproduce the workflow from the README, and the final report distinguishes performed verification from anything blocked or unrun.

## 13. Execution rules for AI agents

- Work through the tasks in dependency order; continue routine authorized local work without requesting approval after every task.
- This document authorizes implementation and local experiments in the owner's project, not paid services or public deployment.
- Do not create a second repository, force-push, rewrite history, or change account settings. Follow the owner's separate instructions for commits and remote pushes.
- Do not silently broaden scope, change the dataset, replace models, or introduce a database/LLM. The owner-authorized thin local FastAPI inference adapter is the only planned backend.
- Record necessary implementation-level adjustments with reasons in `reports/decisions.md`; preserve the scientific protocol.
- Maintain `reports/progress.md` with task status, actual commands, outputs, blockers, and the next step so another agent can resume.
- If downloads or hardware access are blocked, complete independent work, then give the exact remaining owner action. Do not claim experiments ran elsewhere when they did not.
- Add meaningful tests for leakage, preprocessing parity, calibration, metrics, and persistence. A smoke test does not establish benchmark performance.
- Never invent results, citations, images of results, training curves, runtime measurements, or accuracy claims.
- Prefer one sequential worker by default; this plan does not require simultaneous agents.

## 14. Final completion checklist

- [ ] Verified environment and pinned installation instructions.
- [ ] Dataset audit, licensing notes, and immutable split manifest.
- [ ] Three fitted and reloadable local model artifacts.
- [ ] Calibration scores and thresholds for each model.
- [ ] Clean per-image predictions and aggregate metrics for all models.
- [ ] PatchCore pixel evaluation and representative heatmaps.
- [ ] Brightness study using unchanged models and thresholds.
- [ ] Reproducible timings with actual device information.
- [ ] Focused tests and a working local React + FastAPI demo.
- [ ] README, methodology, measured report, limitations, and demo walkthrough.
- [ ] Git excludes datasets, weights, caches, and personal uploads.

## 15. Primary references

Consult the version-matched API documentation during implementation; links below were consulted while planning on 2026-09-26.

- MVTec AD dataset and license: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- Anomalib PatchCore documentation: https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html
- Anomalib source: https://github.com/open-edge-platform/anomalib
- PyTorch installation selector: https://pytorch.org/get-started/locally/

## 16. Copy-paste kickoff prompt

> Read PLAN.md completely and inspect any existing repository instructions and files. Implement this project following Tasks 1–9 in order. Start by verifying my Windows/Python/NVIDIA environment and scaffolding the repository. Keep all model fitting normal-only, preserve the shared split manifest, and never use test data for calibration or hyperparameter selection. Use real local experiments, record versions and commands, and keep reports/progress.md current. Continue routine work autonomously. If dataset download or hardware access requires my action, finish independent work and tell me exactly what to do. Do not fabricate metrics, replace the chosen methods, deploy publicly, or push to GitHub without my instruction. Explain completed work and the next step in Chinese; keep source code and project documentation in English.
