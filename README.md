# carpet-texture-anomaly-detection
Comparing handcrafted texture features (LBP/GLCM + One-Class SVM) with PatchCore for carpet defect detection on MVTec AD, with anomaly heatmaps and a React + Tailwind / FastAPI local demo.

## Status

Environment diagnostics and the real carpet dataset audit/split are verified.
The fixed manifest contains 224 fitting, 56 calibration, and 117 test images.
All three models are fitted, calibrated, reload-verified, and evaluated on the
same 117-image test set. Measured results are in [reports/results.md](reports/results.md).
The paired brightness study and English local web demo are complete.
See [PLAN.md](PLAN.md) for the protocol and [progress](reports/progress.md) for evidence.

## Environment setup (Windows x64 / PowerShell)

Run from the repository root. Use Python 3.11; the recorded environment uses
3.11.16. If Python 3.11 is already installed, create `.venv` with that interpreter.
Alternatively, the following uses an existing `uv` installation to keep Python
inside the project without changing the registry or global Python:

```powershell
uv python install 3.11.16 --install-dir .python --no-bin --no-registry
.\.python\cpython-3.11.16-windows-x86_64-none\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m carpet_ad.cli doctor
.\.venv\Scripts\python.exe -m carpet_ad.cli --help
.\.venv\Scripts\python.exe tests/check_scaffold.py
.\.venv\Scripts\python.exe -m pytest -q
```

The lock targets this Windows/Python/CUDA combination. Install the PyTorch wheels
from the specified official index first; all other packages use PyPI. The CUDA
runtime is bundled in the wheels; a separate CUDA toolkit is not required here.
Do not reuse a Python 3.13 virtual environment for this installation.

`doctor` reports versions, import failures, CPU/CUDA checks, and available disk.
It returns a nonzero exit code on dependency or tensor-operation failures. If CUDA
is absent, it explicitly reports that limitation and permits CPU-compatible work.
It does not download weights. After fitting and calibrating PatchCore, run its
optional fresh-process artifact check:

```powershell
.\.venv\Scripts\python.exe tests/check_patchcore.py
```

That check loads the local saved weights and memory bank without downloading
pretrained assets. It verifies frozen parameters, raw score/map persistence, and
all calibration scores and decisions. It does not evaluate test images.

Help, `doctor`, `prepare-data`, `fit`, `calibrate`, `evaluate`, `robustness`, and
`compare` and the local web application are available.
Configuration paths are repository-relative; run experiment commands from
the repository root. Fitting validates and snapshots the selected configuration.

Raw data belongs under `data/raw/carpet/`; model artifacts and complete run outputs
belong under `artifacts/` and `runs/`. These paths and local uploads are ignored by Git.
Create remaining modules and output directories as their tasks are implemented.

All fitting will use normal fitting images only, with a separate normal calibration
split. Test data must not select thresholds or hyperparameters. Anomaly scores
are not probabilities. See [environment](reports/environment.md) and
[decisions](reports/decisions.md) for current constraints.

## Prepare the dataset

1. Open the [official MVTec AD page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
   and complete its download form. Choose the original dataset's carpet category,
   not MVTec AD 2. If only the full archive is offered, extract its carpet category.
2. Put the downloaded archive under ignored `data/raw/`, or extract carpet there.
   The resulting paths must be `data/raw/carpet/train/good/`,
   `data/raw/carpet/test/good/`, `data/raw/carpet/test/<defect_type>/`, and
   `data/raw/carpet/ground_truth/<defect_type>/`.
3. Run:

```powershell
.\.venv\Scripts\python.exe -m carpet_ad.cli prepare-data --data-root data/raw/carpet --seed 42
```

The command writes `reports/data_audit.md` and, only on a clean audit,
`data/manifests/carpet_split.csv`. Paths in the CSV are relative to the carpet
category root. It maps masks by defect directory and `<image_stem>_mask`, allowing
different image/mask extensions. It checks decoding, dimensions, binary masks,
orphan masks, and byte-identical image hashes. It rejects cross-split duplicates.

The sorted official normal training paths are shuffled with NumPy Generator(PCG64),
seed 42; floor(80%) become `fit`, the remainder `calibration`. Official test images
remain `test`. Repeating an identical run leaves the manifest unchanged; changes
to data or seed cannot silently replace it. Failed audits retain any existing
manifest, which must not be used until the audit passes again. Audit history is
appended when observations change. Unexpected files inside image/mask directories
are reported rather than silently skipped.

`notebooks/01_data_exploration.ipynb` reuses the package audit and displays fitting
samples only. It is optional; a notebook frontend and `ipykernel` are not part of
the current locked CLI environment. Keep image outputs local and clear notebook
outputs before committing. See [DATA_LICENSE.md](DATA_LICENSE.md) for attribution
and licensing distinctions.

## Fit and calibrate the baselines

Run from the repository root after the data audit passes:

```powershell
.\.venv\Scripts\python.exe -m carpet_ad.cli fit --config configs/lbp_ocsvm.yaml --run-id lbp_seed42
.\.venv\Scripts\python.exe -m carpet_ad.cli calibrate --run-id lbp_seed42
.\.venv\Scripts\python.exe -m carpet_ad.cli fit --config configs/lbp_glcm_ocsvm.yaml --run-id lbp_glcm_seed42
.\.venv\Scripts\python.exe -m carpet_ad.cli calibrate --run-id lbp_glcm_seed42
```

These run IDs already exist in the current local checkout; use new IDs for a new
run. Commands refuse to overwrite runs or calibration outputs. Only the fixed
PLAN.md baseline settings are supported; differing config fields are rejected.

Each run contains `model.joblib`, exact config and manifest snapshots,
`metadata.json`, `calibration_scores.csv`, and `calibration.json`. Metadata records
source hashes and Git commit/dirty state, versions, seed, device, timings, and
artifact hashes. Load only trusted locally created joblib artifacts.

Both feature paths use EXIF-oriented RGB, full-frame 256x256 Pillow bilinear resize,
and Pillow `L` grayscale. Uniform LBP is computed before dividing the map into
row-major 4x4 cells, each with L1-normalized bins 0–9 (160 features).
GLCM adds 32 values ordered by property (contrast, homogeneity, energy, correlation),
distance (1, 3), then angle (0, 45, 90, 135 degrees). Quantization uses widened
integer arithmetic. The verified scikit-image version returns correlation 1 for
constant images.

The scaler and One-Class SVM see only the 224 fitting images. Calibration uses the
56 held-out normal images and negative `decision_function` scores. Its 95th
percentile uses linear interpolation; scores strictly above the threshold are
anomalous and ties are normal. Scores are not probabilities or comparable across
models. Frozen-threshold test metrics are reported in reports/results.md.

The English UI uses React + Tailwind CSS with a thin local FastAPI adapter.
Streamlit has been removed from the environment, project dependencies and lock.
See the local demo instructions below.

## Fit and calibrate PatchCore

```powershell
.\.venv\Scripts\python.exe -m carpet_ad.cli fit --config configs/patchcore.yaml --run-id patchcore_seed42
.\.venv\Scripts\python.exe -m carpet_ad.cli calibrate --run-id patchcore_seed42
.\.venv\Scripts\python.exe tests/check_patchcore.py --run-id patchcore_seed42
```

Use a new run ID when one already exists. The adapter calls Anomalib's raw
PatchcoreModel with tagged `wide_resnet50_2.racm_in1k` weights, layer2/layer3,
FP32, 9 neighbors, and a 10% coreset. It verifies the pretrained revision and
SHA-256 recorded in reports/environment.md; changed weights fail explicitly.
First fitting may download weights to ignored `artifacts/cache/`.

The shared Pillow full-image 256x256 bilinear RGB transform is followed by float32
division by 255 and one ImageNet normalization. Raw library image scores and
256x256 anomaly maps are returned without image-wise min-max normalization.
Only fitting-split patches enter the bank. Automatic validation splitting,
thresholding, normalization, and Lightning hooks are not involved.

`model.pt` stores the complete state dictionary including the memory bank;
`metadata.json` records preprocessing, weight identity, fit timing, resource use,
and reload tolerances. Loading uses weights_only=True, strict matching, and
Anomalib's dynamic-buffer loader, without a pretrained download. Calibration is
separate, uses normal calibration images only, and saves its threshold beside
the immutable model. `reload_reference.npz` preserves a normal fitting image's
raw output for verification; it is not a test prediction.

Feature extraction starts at batch 4 and retries remaining inputs at 2 then 1
on CUDA OOM, recording any fallback. Coreset ratio never changes automatically.
Coreset construction is distinct from extraction and may require additional
memory; an unrecoverable error leaves the incomplete run available for inspection.
CUDA is used when available, otherwise CPU. No gradient optimization occurs.

## Evaluate frozen models

```powershell
.\.venv\Scripts\python.exe -m carpet_ad.cli evaluate --run-id lbp_seed42 --condition clean
.\.venv\Scripts\python.exe -m carpet_ad.cli evaluate --run-id lbp_glcm_seed42 --condition clean
.\.venv\Scripts\python.exe -m carpet_ad.cli evaluate --run-id patchcore_seed42 --condition clean
.\.venv\Scripts\python.exe -m carpet_ad.cli compare --runs lbp_seed42 lbp_glcm_seed42 patchcore_seed42
```

The clean evaluations already exist locally; evaluate refuses to overwrite them.
Comparison reads saved results and verifies their shared manifest, prediction
hashes, and frozen run artifacts. It writes reports/tables/clean_comparison.csv.
Detailed local outputs live under each run's evaluation/clean/ directory.

Scoring goes through the same inference functions for every image. Model,
configuration, manifest, calibration scores, and thresholds are verified before
evaluation and remain unchanged. Pixel AUROC uses globally flattened raw maps and
nearest-neighbor-resized masks, including zero masks for normal images. No per-image
map normalization is used in metrics. Latency uses five warm-ups and batch 1,
excluding decode/file I/O and including preprocessing/scoring, with CUDA synchronized.
CPU versus GPU timings are practical deployment measurements, not equal-compute
comparisons. See the results report for limitations and actual failure cases.

## Paired brightness study

```powershell
.\.venv\Scripts\python.exe -m carpet_ad.cli robustness --run-id lbp_seed42
.\.venv\Scripts\python.exe -m carpet_ad.cli robustness --run-id lbp_glcm_seed42
.\.venv\Scripts\python.exe -m carpet_ad.cli robustness --run-id patchcore_seed42
.\.venv\Scripts\python.exe -m carpet_ad.cli compare --runs lbp_seed42 lbp_glcm_seed42 patchcore_seed42
```

The two conditions multiply original RGB pixels by 0.8 or 1.2 using float32,
clip to [0,255], then truncate to uint8 before resizing. They keep the clean
models, thresholds, parent paths, labels, and masks. No refitting or recalibration.
Brightness generation is outside the scoring timer. These are paired variants
of 117 images, not additional independent samples or a general robustness benchmark.

Outputs live in each run's evaluation/brightness_0.8/ and brightness_1.2/ folders;
existing evaluations are never overwritten. To resume after a partially completed
study, use `evaluate --run-id <id> --condition brightness_1.2` (or the missing
condition). `compare` writes reports/tables/brightness_comparison.csv from available
completed conditions, checking paired IDs/labels, frozen artifacts and decisions,
and recomputing AUROC, F1 and false-positive rate from saved predictions.

## Run the local demo

Requires Node.js 22.12+ (verified with 22.19.0 and npm 11.6.0), the Python
environment above, and the three fitted/calibrated runs. From the repository root:

```powershell
npm --prefix frontend ci
npm --prefix frontend run build
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Leave this terminal running; Ctrl+C stops the
service. After the first build, only the Python command is needed to launch.
Rebuild after frontend edits; restart the API after changing model artifacts.
Use one worker, bound to loopback. This is a local research demo, not a hosted service.

Upload a PNG/JPEG carpet image or choose Sample A/B from the locally downloaded
MVTec data. Select a model and click **Run inspection**. PatchCore provides a
draggable original/heatmap divider and opacity control. Baselines explicitly say
localization is unavailable. The **Experiment results** tab reads measured CSVs,
including both brightness conditions. Sample files are not bundled or downloaded
by the application; attribution: MVTec Software GmbH, CC BY-NC-SA 4.0.

Limits: 10 MiB file size and 20 million decoded pixels. Animated images and corrupt
files are rejected. Upload bytes remain in memory and are never written to disk.
Loaded models are cached until process restart; inference is serialized to avoid
concurrent GPU workloads. Missing/invalid artifacts return an actionable error,
never substitute predictions. With CUDA unavailable, PatchCore loads on CPU and
the footer states that inference will be slower.

Heatmap colors are relative to each image and only affect display. Scores, strict
threshold decisions and preprocessing use the frozen inference implementation.
Scores are not probabilities or comparable across models. Reported interactive
scoring time excludes model loading, upload/decode and rendering; it is not the
warm-start benchmark. Images outside this carpet category are not validated.

Optional frontend development (keep the API running in another terminal):

```powershell
npm --prefix frontend run dev
# Open http://127.0.0.1:5173; Vite proxies /api to the local Python service.
```

Verification, with the built app running at port 8000:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tests/check_webapp.py
```

The opt-in browser check uses Playwright and installed Microsoft Edge (headless).
Playwright is pinned in the full lock; minimal installations can use
`python -m pip install -e ".[browser-test]"`. It checks real model scores against
saved clean predictions, UI controls, upload errors and mobile layout. Screenshots
are local under ignored runs/task8/ because they can include dataset imagery.
