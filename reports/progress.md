# Progress

## Task 1 — Inspect and scaffold

Status: verified locally on 2026-09-26. No release or remote changes.

Read PLAN.md and inspected the repository. Initial tracked content was README.md;
PLAN.md was untracked. Created package metadata, CLI help, planned configurations,
ignore rules, setup documentation, and hardware/decision records.

Commands and outcomes:

- `git switch -c scaffold/task-1`: created the working branch.
- `py -3.13 -m venv .venv`: created an isolated scaffold environment.
- `.\.venv\Scripts\python.exe -m pip install -e .`: built and installed version 0.1.0.
- `.\.venv\Scripts\python.exe -m carpet_ad.cli --help`: exit 0; scaffold status displayed.
- `.\.venv\Scripts\python.exe tests/check_scaffold.py`: PASS; installed module
  works from a temporary directory, and unavailable `fit` returns exit 2.
- `.\.venv\Scripts\python.exe -m pip check`: no broken requirements found
  (scaffold only; research dependencies are not installed).
- `.\.venv\Scripts\python.exe -m compileall -q src tests`: exit 0.
- `git diff --check`: exit 0; Git reported only its existing LF/CRLF conversion warning.
- `git check-ignore` on virtual environment, raw image, model artifact, run output,
  upload, `.env`, and Streamlit secret paths: all seven paths ignored.
- Reviewed tracked README diff and newly created package, CLI, configurations,
  check, and documentation. PLAN.md remains unchanged and untracked.

Hardware observations are in [environment.md](environment.md). Initial sandbox
access failures were resolved after the owner changed session permissions.
No outstanding Task 1 blocker. Local verification is not independent CI evidence.

## Task 2 — Validate environment

Status: verified locally (2026-09-26). No benchmark experiment or release.

- Selected Python 3.11.16, Anomalib 2.3.2, torch 2.8.0 and torchvision 0.23.0
  using PyPI dependency metadata and the official PyTorch version pairing.
- Installed Python locally with `uv python install 3.11 --install-dir .python
  --no-bin --no-registry`; preserved the old environment as `.venv-scaffold`.
  Created `.venv` with `.python/cpython-3.11.16-windows-x86_64-none/python.exe`.
- CUDA wheel source: `https://download.pytorch.org/whl/cu126`.
- Implemented `doctor`: package import errors, versions, CPU/CUDA arithmetic,
  interpreter/platform, free disk, GPU identity/memory. It downloads nothing.
- `test_doctor.py`: observed RED before implementation; GREEN after implementation.
  A real pre-install CLI run exited 1 and recorded missing packages correctly.
- Added an opt-in pretrained PatchCore check using synthetic images, not dataset
  samples. It checks coreset construction and finite raw score/map shapes.
- The guessed GitHub source tag URLs returned 404; inspected the actual downloaded
  Anomalib 2.3.2 wheel source instead. No API assumptions based on those missing URLs.

Verification and outcomes:

- `python -m pip install -e . lightning==2.6.1 pytorch-lightning==2.6.1`:
  completed after installing official CUDA wheels. The intermediate torch-only
  install warned about not-yet-installed project dependencies; final pip check passed.
- `python -m carpet_ad.cli doctor`: exit 0; all 16 imports and CPU/CUDA arithmetic
  passed. Actual GPU: RTX 4060 Laptop GPU. Output: `runs/task2/doctor.json`.
- Repeated doctor with process-local `CUDA_VISIBLE_DEVICES=-1`: exit 0; CPU passed,
  CUDA explicitly unavailable. This verifies the documented CPU continuation path.
- `python tests/check_patchcore.py`: exit 0; real pretrained model, synthetic
  2-image coreset fit and 1-image inference; finite raw scores/maps. Weight revision
  and hash recorded in environment.md. No fitting artifact was saved.
- `python -m pip freeze --exclude-editable`: generated requirements-lock.txt.
  Verified all 102 pins equal installed distribution versions and contain no local paths.
- `python -m pip install --dry-run -r requirements-lock.txt`: exit 0 against the
  installed environment. A second clean-environment reinstall was not performed.
- `python -m pip check`: no broken requirements found.
- `python -m pytest -q`: 1 passed; `python tests/check_scaffold.py`: PASS.
- `git diff --check`: passed, with existing LF/CRLF conversion warning.
- `git check-ignore`: local Python, preserved old venv, model assets, and run logs ignored.

The commands above use `.\.venv\Scripts\python.exe`. Setup instructions now match
the verified environment. No outstanding Task 2 blocker. Next: Task 3 dataset
acquisition, integrity audit, and deterministic fitting/calibration manifest.

## Task 3 — Audit and split data

Status: verified on real local data on 2026-09-26. The initial download blocker
described below was resolved when the owner supplied the extracted carpet directory.

- Owner confirmed deletion of the old `.venv-scaffold/`; removed its ignore entry
  and updated the environment record.
- Official MVTec AD download page checked on 2026-09-26: form required. Requested
  that the owner download original carpet data into `data/raw/`. No third-party
  source or legacy access route was substituted.
- Added one `data.py` module and `prepare-data` CLI command: file inventory,
  decode/mask checks, SHA-256 duplicate checks, normal-only fitting/calibration
  allocation, and an immutable manifest. Changed audit snapshots are appended.
- Added `DATA_LICENSE.md` and a small exploration notebook that calls package
  code and displays fitting samples only. No extra dependencies installed.
- Observed the new data test fail before implementation because the data module
  was absent. Final `python -m pytest -q`: 3 passed. Checks cover repeatability,
  explicit seed-42 calibration membership, split isolation, unchanged test data,
  different mask extensions, missing/mismatched masks, corrupt files, cross-split
  duplicates, missing dataset, and refusal to replace an existing manifest.
- `python tests/check_scaffold.py`: PASS. Notebook code cells compile; real
  exploration and image rendering remain unrun because the dataset is absent.
- `python -m carpet_ad.cli prepare-data --data-root data/raw/carpet --seed 42`:
  exit 1, correctly reports missing data. `reports/data_audit.md` records the failed
  observation; no `carpet_split.csv` was generated. No dataset counts are claimed.
- `git diff --check`: passed with the existing LF/CRLF warning.

Real-data completion evidence:

- Owner supplied `data/raw/carpet/`, already extracted, with train/test/ground_truth,
  readme.txt, and license.txt. Read the supplied attribution and license text.
- `python -m carpet_ad.cli prepare-data --data-root data/raw/carpet --seed 42`:
  exit 0. Actual inventory: 397 RGB images, all 1024x1024; 89 binary masks,
  each matching its source image dimensions. No decoding, missing/orphan mask,
  binary-mask, path, or byte-identical image duplicate issue found.
- Official train/good: 280, split into fit 224 and calibration 56. Official test:
  117 unchanged (28 normal, 89 anomalous). Defect counts are in data_audit.md.
- Manifest: `data/manifests/carpet_split.csv`; SHA-256
  `ac41e4abf3f98d7cfc2805480d1ad283094dfa28ecbacb98d0b4db6fab8ae2bc`.
- Executed both notebook code cells headlessly with the current interpreter and
  Matplotlib Agg, without adding a Jupyter dependency. This reran the real audit
  and left manifest bytes unchanged. Independently checked unique paths/hashes,
  normal-only fit/calibration, preservation of the test split, and anomaly mask paths.
- Saved the rendered fitting-only illustration to ignored
  `runs/task3/fitting_samples.png` and visually inspected it. Notebook outputs
  remain empty. Agg emitted the expected noninteractive-show warning; saving worked.
- `python -m pytest -q`: 3 passed. No integrity-related benchmark deviation or
  sample removal was needed. Earlier missing-data audit snapshots are preserved.

No remaining Task 3 blocker. Next: Task 4, shared preprocessing, LBP/GLCM baselines,
normal-only fitting, persistence, and held-out normal threshold calibration.

## Task 4 — Baselines and normal calibration

Status: verified locally on 2026-09-26. Both baselines fitted and calibrated;
official test images remain unscored.

- Added flat preprocessing.py, features.py, and models.py modules, plus fit and
  calibrate CLI commands. Existing manifests/configs reused; no new dependencies.
- Added focused baseline checks. Initial run failed because implementation was
  absent. A later test setup hit a Windows pytest temp-directory permission error;
  switching its workspace to stdlib TemporaryDirectory resolved it. Final suite:
  `python -m pytest -q` → 6 passed. `python tests/check_scaffold.py` → PASS.
- Coverage: 160/192 feature dimensions, identical LBP prefix, histogram sums,
  finite constant-image statistics, overflow-sensitive GLCM contrast, EXIF/RGB
  preprocessing, quantile/ties/nonfinite inputs, normal-only scaler fitting,
  no test-image/test-label consumption, persistence, unchanged model during
  calibration, run-ID validation, and refusal to overwrite runs/calibration.
- Actual commands (all using `.\.venv\Scripts\python.exe`):
  `-m carpet_ad.cli fit --config configs/lbp_ocsvm.yaml --run-id lbp_seed42`,
  `-m carpet_ad.cli fit --config configs/lbp_glcm_ocsvm.yaml --run-id lbp_glcm_seed42`,
  then `-m carpet_ad.cli calibrate --run-id <each-run-id>`; all exit 0.

| Run | Fitting images | Features | Fit seconds | Calibration images | Raw-score threshold |
| --- | --- | --- | --- | --- | --- |
| lbp_seed42 | 224 | 160 | 13.485081800026819 | 56 | 0.1838646488750951 |
| lbp_glcm_seed42 | 224 | 192 | 13.671176799980458 | 56 | 0.16614510826979537 |

These are measured local fit/calibration records, not accuracy or inference-latency
benchmarks. Fit timing includes decode, preprocessing, feature extraction, scaler
and SVM fitting, excluding prior hash validation and persistence. CPU execution.

- Each `runs/<id>/` contains model.joblib, config.yaml, manifest.csv, metadata.json,
  calibration_scores.csv, and calibration.json. Raw scores and exact threshold rule
  are preserved. The shared manifest hash remains
  `ac41e4abf3f98d7cfc2805480d1ad283094dfa28ecbacb98d0b4db6fab8ae2bc`.
- A fresh Python process loaded both pipelines, recomputed every calibration score
  within absolute tolerance 1e-12 (rtol=0), recomputed thresholds/decisions, checked
  scaler fit count 224 and unchanged model SHA-256. All passed.
- Source hashes, Git commit and dirty state record the uncommitted implementation;
  no commit or push was performed. Config comment wording was subsequently updated
  to reflect fitted status; exact original bytes remain in each run snapshot.
- Owner's UI change recorded in PLAN.md v1.1. Frontend work remains Task 8.
- `git diff --check` passed with the existing LF/CRLF conversion warning.

Next: Task 5, the PatchCore adapter and real normal-only memory bank, independent
calibration, aligned raw maps, and verified artifact persistence.

## Task 5 — PatchCore

Status: verified locally, 2026-09-26. Complete normal-only fitting, calibration,
and fresh-process persistence checks passed. No test image scored.

- Added a thin patchcore.py adapter over Anomalib PatchcoreModel. Reused the shared
  run/config/manifest validation and held-out normal calibration path.
- Pinned the timm weight tag explicitly and verify the downloaded revision and
  SHA-256 recorded in Task 2. State dictionaries include frozen backbone buffers
  and the memory bank; load with weights_only=True and strict state matching.
- Preprocessing reuses Pillow full-image bilinear RGB resize and applies ImageNet
  normalization exactly once. Automatic Lightning preprocessing/postprocessing,
  validation splitting, thresholding, and normalization are not used.
- Added a preprocessing test; observed RED before implementation, then GREEN.
  Full fast suite: 7 passed.
- Four normal fitting images were used for a real adapter smoke under
  runs/patchcore_smoke_task5; memory bank [409,1536], raw maps [256,256], score/map
  reload within rtol=atol=1e-5. Batch 4, no OOM or ratio reduction.
- Started `python -m carpet_ad.cli fit --config configs/patchcore.yaml --run-id
  patchcore_seed42` on all 224 normal fitting images. Progress log:
  runs/task5-fit.log. Planned coreset size: 22,937 from 229,376 patch embeddings.
- Updated tests/check_patchcore.py into a reusable fresh-process check for the
  final fitted and calibrated run; it replaces the earlier Task 2 synthetic-only
  API smoke script. Task 2 historical logs remain under runs/task2/.

Full-run evidence:

- `python -m carpet_ad.cli fit --config configs/patchcore.yaml --run-id
  patchcore_seed42`: exit 0. CUDA, all 224 fitting images, batch 4, workers 0,
  FP32, unchanged coreset ratio 0.1, no OOM or other fallback.
- `python -m carpet_ad.cli calibrate --run-id patchcore_seed42`: exit 0; only the
  56 normal calibration images were scored. Threshold: 30.58691692352295, linear
  95th percentile, strictly greater scores are anomalous, ties normal.
- `python tests/check_patchcore.py --run-id patchcore_seed42`: exit 0 in a fresh
  process. Verified model/manifest/score hashes, frozen/eval parameters, dynamic
  memory-bank loading, raw score/map reload (rtol=atol=1e-5), calibration membership,
  all 56 raw scores, recomputed threshold, and identical decisions.
- `python -m pytest -q`: 7 passed; `python tests/check_scaffold.py`: PASS.
- Rendered and visually inspected a normal fitting-image raw map under ignored
  runs/patchcore_seed42/reference_map.png. The aligned 256x256 reference map is
  also stored numerically in reload_reference.npz. Color scaling is display-only,
  explicitly relative, and changes neither raw values nor predictions.

| Measurement | Actual result |
| --- | --- |
| Memory bank | 22,937 x 1,536 FP32 values |
| Fit duration | 206.88776750001125 seconds |
| CUDA peak allocated | 2,920,305,664 bytes |
| CUDA peak reserved | 3,571,449,856 bytes |
| Windows process peak working set | 2,313,945,088 bytes |
| Complete model.pt size | 240,610,275 bytes |
| model.pt SHA-256 | 6881f36af8359f19e9cb636deb8c7ba6426a9fe9d72fc44305ecc0e322adccd1 |

Timing excludes model construction and persistence; includes decode, shared
preprocessing, frozen feature extraction, and coreset selection. CUDA allocator
peaks cover fitting and the first reference prediction; process working-set peak
covers construction through reload. Neither is whole-machine resource consumption.
This is not inference latency or accuracy evidence. CPU fitting and forced OOM
fallback branches were not exercised because the full CUDA run succeeded.

Exact run configuration, manifest, weight identity, source/package versions, and
timing are stored in runs/patchcore_seed42/. Calibration leaves the model untouched.
The subsequent config comment update is cosmetic; exact original config bytes
are retained in the run. The shared manifest hash remains unchanged.

No remaining Task 5 blocker. Next: Task 6, frozen-model test evaluation of all
three methods, pixel AUROC for PatchCore, timing measurements, and failure analysis.

## Task 6 — Primary test evaluation

Status: verified locally, 2026-09-26. No model or threshold changes after observing
test outcomes. Results and limitations are in reports/results.md.

- Added inference.py for verified artifact loading/shared scoring and evaluation.py
  for clean evaluation/comparison. Extended CLI; no new dependencies.
- Metrics tests failed before implementation and passed afterward. They verify
  raw-score AUROC, strict ties, hand-counted classification metrics, finite inputs,
  and global pixel ranking without per-image scaling. Baseline tests also verify
  inference parity and rejection of a tampered threshold.
- Ran `python -m carpet_ad.cli evaluate --run-id <id> --condition clean` sequentially
  for lbp_seed42, lbp_glcm_seed42, patchcore_seed42; all exit 0. Each scored the same
  117 test images using its frozen artifacts, five warm-ups, and batch size 1.
- Ran `python -m carpet_ad.cli compare --runs lbp_seed42 lbp_glcm_seed42 patchcore_seed42`;
  exit 0. Output: reports/tables/clean_comparison.csv. Exact metrics derive from
  runs/<id>/evaluation/clean/predictions.csv and metrics.json.
- Image AUROC: LBP 0.634430176565008; LBP+GLCM 0.5353130016051364;
  PatchCore 0.9859550561797752. PatchCore F1 0.9560439560439561, with 6 false positives
  among 28 normal images and 2 misses among 89 anomalies. No tuning followed.
- Confusion matrices [TN,FP;FN,TP]: LBP [24,4;53,36], LBP+GLCM [14,14;53,36],
  PatchCore [22,6;2,87]. By-defect counts and all failure paths are retained.
- PatchCore pixel AUROC 0.9905906062731301 over 7,667,712 pixels, 122,808 positive.
  Saved raw maps/masks/paths in pixel_maps.npz. Independent scipy rank-sum AUROC
  agrees to absolute tolerance 1e-12; normal-image masks were confirmed zero.
- Independently reloaded all predictions and recomputed decisions, precision,
  recall, F1, FPR, AUROC, median/p95 latency, identical ordered paths/labels, and
  frozen artifact hashes. All checks passed.
- Visually inspected PatchCore ROC/confusion and deterministic error illustrations.
  Small ROC/confusion figures copied to reports/figures; attributed dataset
  illustrations remain under ignored run outputs. No test-derived image is needed
  to reproduce the numeric summaries.

- Final verification on 2026-09-27: 9 tests passed, scaffold check passed, and
  `git diff --check` passed (existing line-ending warning only).

Next: Task 7, paired brightness multipliers 0.8 and 1.2 using unchanged models and
thresholds. No UI implementation, commit, or push performed.

## Remaining tasks

Task 9 remains. Tasks 1-8 are complete. The owner authorized GitHub push and
merge on 2026-09-27; integration verification is recorded below.


## Task 7 - paired brightness study (2026-09-27)

- Reused evaluation.py; added fixed 0.8/1.2 RGB perturbations, robustness CLI,
  transformed error illustrations and paired comparison. No new dependencies.
- Transform test failed before implementation, then passed. Comparison regression
  covers exact threshold CSV round trips and rejection of mismatched parent IDs.
- Ran robustness sequentially for lbp_seed42, lbp_glcm_seed42 and patchcore_seed42;
  all exited 0, producing six complete 117-image evaluations (702 predictions).
- Ran compare for all three models, recomputing AUROC/F1/FPR and clean deltas.
  Initial comparison exposed pandas default float parsing losing a threshold digit;
  round-trip parsing fixes precision without loosening the frozen-threshold check.
- Verified all parent IDs/labels, decisions, manually counted confusion matrices,
  model/config/calibration hashes against clean, and PatchCore map hashes, finite
  shapes and unchanged masks. No refit, recalibration or clean-output overwrite.
- Results and limitations recorded in results.md and brightness_comparison.csv.
  PatchCore AUROC remains about 0.986; brighter FPR rises to 7/28. Both perturbed
  LBP+GLCM conditions have 28/28 normal false positives despite higher F1.
- Timing differs substantially across sessions; no brightness-induced speedup claim.
- Final verification: 11 pytest tests passed; installed CLI check passed;
  git diff --check passed. The new test uses stdlib TemporaryDirectory like the
  existing tests because this machine's pytest shared temp root denies access.

Next: Task 8, local React + Tailwind / FastAPI demo. No commit or push performed.

## Task 8 - local English web demo (2026-09-27)

- Owner selected an all-English interface. Added one React page, one stylesheet,
  Vite/Tailwind configuration, and app/api.py. Graphite, warm white and amber
  inspection workspace; responsive layout and reduced-motion support.
- Reused load_run/score_image; cached all three frozen local models. Uploaded
  PNG/JPEG bytes stay in memory; streamed 10 MiB cap, 20 MP decoded cap, corrupt,
  unsupported and animated-image rejection. Serialized local inference.
- Original/relative-heatmap divider and opacity controls only change display.
  Raw scores, strict frozen-threshold decisions and baseline localization limits
  remain explicit. Model changes and new uploads clear stale results.
- Results tab reads existing measured CSV reports. Two optional dataset samples
  are served from fixed local paths; no data assets bundled in the frontend.
- Removed Streamlit from project, doctor, environment and lock. Added FastAPI;
  full lock has 114 exact distribution pins, without local editable paths.
  React/Vite/Tailwind dependencies have a package-lock.json.
- API boundary tests failed before implementation and now pass. Final suite:
  14 pytest tests passed; pip check passed; CLI scaffold check passed; doctor
  reports ok with CPU and CUDA tensor checks. npm production build passed.
- Started a fresh loopback-only API process; headless Edge browser test verified
  all three real model scores/decisions against test/cut/000.png saved clean
  predictions (PatchCore rtol/atol 1e-5, baselines atol 1e-12). Thresholds exact.
- Browser checks passed divider keyboard input, opacity, result conditions,
  corrupt/unsupported uploads, no horizontal overflow at 390px, and overlay
  alignment for a non-square image. No browser runtime exceptions.
- Visually inspected desktop empty, inference, results and mobile screenshots
  under ignored runs/task8/. Full frozen model/config/calibration hashes still
  match Task 6. Missing artifacts/report and CPU-status errors covered by tests;
  CUDA-unavailable status is simulated, not a hardware outage measurement.
- README records one-server startup, development mode, browser test and limits.
  No cloud deployment or dataset/model upload is part of this work.

Next: Task 9, final methodology explanations and handover. GitHub integration
of completed Tasks 1-8 is authorized by the owner.

### Merge review

- Independent read-only reviewer checked Tasks 1-8 and found no Critical or
  Important findings. One minor malformed-metadata error path was fixed: missing
  JSON fields now return the same actionable 503 as missing model files.
  Regression failed before the fix and passed afterward with the complete suite.
- Review scope excludes unfinished Task 9, a fresh-machine installation and full
  experiment reruns. These were not claimed by this integration; recorded local
  experiments and current test/build/browser verification are the evidence.
- Final docs and Python/Node locks were checked by the implementing agent. No
  outstanding reviewer findings; no datasets, weights, caches or uploads staged.
- Keep text checkout line endings at LF so Windows Git conversion cannot alter
  frozen manifest/config hashes. Verified staged bytes against local originals.
