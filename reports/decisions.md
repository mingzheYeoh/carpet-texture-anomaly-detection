# Decisions

## 2026-09-26: Task 1 scope

- Authority: PLAN.md Task 1 and the owner's instruction to execute the first step.
- Preserve the existing README description and untracked PLAN.md. No other source
  or repository AGENTS.md was found; checked accessible ancestor instruction paths.
- Use a minimal setuptools package and stdlib argparse help. Do not implement
  placeholder experiment commands that could falsely report successful work.
- Use installed Python 3.13.7 for scaffold verification only. Do not imply that
  the deep-learning dependency combination is compatible until Task 2 verifies it.
- Three YAML files transcribe planned method settings; exact PatchCore weights
  and version-specific adapter behavior remain unresolved until Task 2.
- No empty future modules, fake result files, unverified dependency lock, or model
  downloads. Add them in the corresponding implementation tasks.
- Work on `scaffold/task-1`; no commits, pushes, or releases are requested.

## Controls and recovery

Project tier: T1. Task risk: green for local scaffold and documentation.
Enforcement: advisory-only; no independent CI/ownership enforcement is present
locally and remote branch rules have not been inspected. No governance bootstrap
or platform changes are included in this task. Repository naming review found
no additional security-sensitive path rules needed for this scaffold.

Acceptance: editable install succeeds; module help works from outside the checkout;
unimplemented commands return an error; ignored paths exclude data and artifacts;
documentation separates observations from planned experiments.

Verification is local implementer evidence, not independent review. Scientific
tests, dependency compatibility/security checks, and UI checks belong to later
tasks. Recovery is to revert the Task 1 additions and README edits while retaining
the owner's PLAN.md; no user data, global Python packages, or remote state is changed.

## 2026-09-26: Task 2 environment and API

- Owner authorized Task 2. Risk: yellow (dependencies and model asset download);
  project tier remains T1, enforcement advisory-only. No CI/branch rules changed.
- Install Python 3.11.16 inside `.python/` with uv, no registry/bin registration.
  Preserve the scaffold environment as `.venv-scaffold/`; the active `.venv` uses
  3.11. Narrow package Python metadata to the validated 3.11 series.
- Select Anomalib 2.3.2 with its declared cu126-compatible torch 2.8.0 range;
  use official paired torchvision 0.23.0 wheels and pin Lightning packages to 2.6.1.
  The verified complete resolution is frozen in requirements-lock.txt. No paid
  services, credentials, or global package installations are needed.
- `doctor` uses importlib, metadata, stdlib platform/disk inspection, and actual
  PyTorch arithmetic. Missing imports or failed operations fail the command;
  absent CUDA is explicitly reported so CPU-compatible tasks remain usable.
- Inspect shipped wheel source when remote tag URLs are unavailable. Use
  Anomalib's own memory-bank and raw scoring APIs, with automatic hooks disabled.
- Preserve the chosen backbone, and record its actual timm pretrained identity
  `wide_resnet50_2.racm_in1k`, downloaded revision, and SHA-256. Task 5 must enforce
  that identity rather than silently accepting a different timm default.
- The smoke has no dataset, threshold, score normalization, saved fitted artifact,
  or benchmark result. Synthetic input verifies API execution only. Model weights
  and logs stay in ignored artifacts/ and runs/ directories.
- Recovery: recreate the virtual environment from the documented exact pins;
  reverse Task 2 source changes if needed. The preserved scaffold environment can
  be restored to its original path; moving a venv does not make its scripts portable.

## 2026-09-26: Task 3 data preparation

- The owner deleted `.venv-scaffold/`; remove its obsolete ignore entry. Recovery
  now uses environment recreation from README and requirements-lock.txt.
- Tier T1, risk yellow (dataset acquisition and immutable evaluation split),
  advisory-only enforcement. Owner authorized the next sequential plan task.
- Keep one data.py module, one focused synthetic test file, and the requested
  exploration notebook. No download framework or new runtime dependency.
- The official page requires a download form. Request owner completion instead
  of guessing legacy download links or using third-party mirrors.
- Match masks by directory and stem, not file extension. Decode files and inspect
  oriented dimensions, channel modes, binary masks, missing/orphan masks, and
  exact-file hash duplicates. Report unexpected paths explicitly.
- Refuse a manifest when integrity checks fail or byte-identical images cross
  assigned splits. Do not silently drop samples or alter the official test set.
- Manifest paths are category-root-relative POSIX paths. Use explicit PCG64 with
  NumPy's Generator, sorted paths, seed 42, floor(0.8*N) fitting images.
- Keep manifests immutable on rerun; preserve earlier audit observations by
  appending changed snapshots. A failed audit means any prior manifest is not
  currently verified. No automatic overwrite or repair option is introduced.
- Exploration displays fitting samples only, with source/license attribution;
  no dataset-derived image is stored in tracked notebook outputs.
- The owner subsequently supplied the already-extracted official carpet folder.
  Its real audit passed with no duplicate/leakage or integrity issue, so no sample
  removal or benchmark deviation was necessary. Verified notebook code headlessly
  and saved its sample illustration under ignored runs/task3 instead of adding
  notebook execution dependencies or tracked image output.

## 2026-09-26: Task 4 baselines and owner-requested UI change

- Owner requested a more visually polished UI instead of Streamlit and then
  authorized the next step. PLAN.md v1.1 records the proposed React + Tailwind
  single page and thin local FastAPI adapter. No frontend scaffolding, database,
  accounts, or deployment was added during model implementation. UI dependencies
  will replace the historical Streamlit dependency at Task 8.
- Continue sequentially with Task 4. Tier T1, risk yellow (model artifacts and
  calibration), advisory-only enforcement. No remote/CI changes.
- Keep shared preprocessing, feature extraction, and baseline run operations in
  three flat modules. Calibration lives with baseline operations instead of
  introducing a one-function module or model class hierarchy.
- Reject configurations differing from the fixed primary baseline protocol;
  do not silently ignore settings or introduce hyperparameter tuning.
- Use Pillow RGB-to-L conversion and uint16 GLCM quantization. Feature order is
  documented in README and model metadata. Constant-image correlation follows
  the verified scikit-image implementation (1.0).
- Fit and calibrate select their own normal-only split before image loading.
  Validate selected image hashes, full-manifest path uniqueness and cross-split
  hash separation. Never parse test labels or open test images in these commands.
- Store local trusted joblib pipelines alongside config/manifest snapshots and
  provenance. Reused run IDs and repeated calibration are errors; failed partial
  runs are left for inspection, with a new ID required for retry. No overwrite flag.
- Quantile calibration uses exactly 0.95, linear interpolation, strict greater-than
  decisions. Save raw calibration scores and the model hash; do not alter the model.
- A pytest tmp_path setup failed due to an existing Windows temp directory ACL.
  Use stdlib TemporaryDirectory for that test's isolated workspace; no assertions
  or production behavior were weakened. The final suite passes.

## 2026-09-26: Task 5 PatchCore adapter

- Tier T1, risk yellow; same authorized local experiment scope and advisory-only
  enforcement. Reuse existing normal-split selection and run/calibration records.
- Use Anomalib's lower-level PatchcoreModel directly. This keeps its feature,
  coreset, nearest-neighbor, raw score, and raw map algorithms while avoiding
  automatic preprocessing, validation, normalization, and threshold hooks.
- Explicitly request the timm racm_in1k tag and verify the known downloaded
  revision/hash. Save all state (including memory_bank) rather than relying on a
  generic Lightning checkpoint. Strict weights-only load uses the installed
  DynamicBufferMixin; no pretrained download is required for reload.
- Fix TF32 off and cuDNN benchmarking off in model construction so fitting and
  fresh-process inference use the same arithmetic policy. Seed Python, NumPy,
  and PyTorch before memory-bank construction. No cross-hardware bitwise
  reproducibility claim; score/map reload tolerance is rtol=atol=1e-5.
- Keep the 10% coreset and original library sampler. Batch OOM retry affects
  only extraction batch size; a coreset failure does not silently change the
  experiment. No custom nearest-neighbor or sampler implementation added.
- Report PyTorch peak CUDA allocated/reserved bytes through fitting and the first
  reference inference, plus the Windows process peak working set through reload.
  These are process/allocator measurements, not whole-machine RAM/VRAM totals.
- The full 224-image run succeeded on CUDA at batch 4 and coreset ratio 0.1.
  No fallback, sample removal, or protocol change was necessary. Independent
  normal calibration and a fresh-process score/map/threshold check passed.

## 2026-09-26: Task 6 frozen test evaluation

- T1/yellow/advisory-only. Add one shared inference module and one evaluation
  module, using installed sklearn metrics and matplotlib. No new dependency.
- Verify frozen run hashes and reconstruct the threshold from its saved normal
  calibration scores before loading. Evaluate rejects existing output folders;
  it never refits or recalibrates. Retain hashes before and after test scoring.
- Decode files outside timing but retain EXIF/RGB/resize inside the scoring path.
  Use five warm-ups, batch 1, one timed prediction per test image, CUDA synchronization,
  and record each latency. Sequential model evaluations avoid concurrent test loads.
- Global pixel AUROC concatenates all raw 256x256 maps and binary masks; nearest
  neighbor for masks, zero masks for normal samples, no per-image map scaling.
- Preserve every prediction, latency, error, ROC point, raw pixel map/mask, and
  per-defect count locally. Commit-sized summaries include only small tables and
  ROC/confusion figures. Dataset example figures remain ignored and attributed.
- Deterministic error illustrations use the first two false positives and first
  two misses in manifest order. Observations do not justify changing a frozen
  threshold or tuning against this test set.
- Independently recomputed image metrics/confusion counts and timing quantiles
  from CSVs, checked identical test paths/labels, and verified pixel AUROC via
  a tie-aware rank-sum calculation. Frozen artifacts remained unchanged.


## Task 7 - brightness protocol

- Reuse evaluation.py and the frozen inference path; no extra module or dependency.
- Apply EXIF orientation/RGB first, float32 multiply, clip, truncate to uint8 before
  resize. Record exact transform and multiplier. Generate variants outside timing.
- Retain parent IDs/labels and masks, original thresholds and all model hashes.
  Saved clean results are the reference; compare recomputes metrics and paired deltas.
- Use round-trip CSV float parsing: default pandas parsing lost the final digit of
  the LBP+GLCM threshold. Exact frozen comparisons remain enforced.
- Preserve full condition outputs locally, including transformed failure examples.
  Conditions are paired observations and cannot increase the independent sample size.
- Higher F1 alone is not robustness: report normal false positives alongside it.

## Task 8 - local web interface

- Follow the approved React/Tailwind + FastAPI scope; owner chose all-English
  copy. Use one component file and stylesheet, no router, state library, database
  or extra component framework. Decorative weave is CSS, not a fabricated result.
- Build frontend assets with Vite; FastAPI serves the build and API on loopback
  from one process. Development proxy is optional. No CDN fonts or runtime cloud
  calls. Use installed system fonts and inline SVG interface icons.
- Stream the raw image request body with a hard size cap, avoiding multipart
  temporary files. Check decoded format and dimensions before pixel allocation.
  Exif/RGB conversion matches shared inference; upload data is not persisted.
- Cache trusted, checksum-verified fixed run IDs. Serialize inference with one
  lock for this local user; return busy/errors clearly instead of fake outputs.
  Changing artifacts requires a restart; do not expose arbitrary filesystem paths.
- Rescale maps only for display. Full-frame inverse mapping aligns overlays with
  non-square originals. Both display sliders leave raw scores untouched.
- Keep interactive scoring time separate from primary benchmark timings.
  First use includes cold model execution; loading/upload/rendering are excluded.
- Fixed local sample routes aid demonstrations without redistributing images.
  Node dependencies are locked; Streamlit is removed. Browser testing is an
  optional development extra, included in the full verified Python lock.
