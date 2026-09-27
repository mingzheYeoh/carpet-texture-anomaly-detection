# Environment observations

Observed locally on 2026-09-26 (Asia/Kuala_Lumpur). Hardware discovery only;
PyTorch CUDA execution and PatchCore have not been verified.

| Item | Observation |
| --- | --- |
| OS | Windows 11 Home Single Language, 10.0.26200 |
| CPU | Intel Core i5-14500HX |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | 8188 MiB reported by nvidia-smi |
| NVIDIA driver | 610.62 |
| OS-visible RAM | 33,251,384 KiB (about 31.71 GiB) |
| Available RAM at query | 13,919,896 KiB (about 13.28 GiB) |
| C: free disk at query | 161,492,369,408 bytes (about 150.4 GiB) |
| Git | 2.53.0.windows.3 |
| Python launcher inventory | Python 3.13 only |
| Executed Python | 3.13.7 |
| Global pip | 26.2.1 |

Commands: `py -0p`, `py -3.13 --version`, `py -3.13 -m pip --version`,
`git --version`, `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`,
`Get-CimInstance Win32_OperatingSystem`, `Get-CimInstance Win32_Processor`,
and `Get-PSDrive C`.

The original restricted sandbox denied Python execution and CIM queries; after
the session permissions changed, these queries succeeded. Free RAM and disk
are point-in-time observations. No CUDA toolkit version is inferred from the driver.

At the end of Task 1, Task 2 still needed to select a compatible Python/PyTorch/torchvision/Anomalib combination,
record exact wheel sources and versions, verify imports and a CUDA operation,
smoke-test PatchCore, and generate a verified lock file. Python 3.11 was not
listed by the launcher; the scaffold's Python 3.13 environment was provisional.

## Task 2 validation — 2026-09-26

Status: verified locally for imports, tensor operations, and a synthetic pretrained
PatchCore API smoke. This is not dataset evaluation or full artifact persistence verification.

| Component | Verified version / result |
| --- | --- |
| Python | 3.11.16, uv-managed standalone CPython, project-local |
| torch | 2.8.0+cu126 |
| torchvision | 0.23.0+cu126 |
| Anomalib | 2.3.2 |
| timm | 1.0.30 |
| Lightning / pytorch-lightning | 2.6.1 / 2.6.1 |
| NumPy / pandas | 2.4.6 / 2.3.3 |
| scikit-image / scikit-learn | 0.26.0 / 1.9.1 |
| CPU tensor operation | Passed |
| CPU-only diagnostic path | Passed with CUDA_VISIBLE_DEVICES=-1; CUDA explicitly unavailable |
| CUDA tensor operation | Passed; RTX 4060 Laptop GPU, 8,585,216,000 bytes VRAM |
| CUDA runtime supplied by wheels | 12.6 |
| Full requested import set | Passed (16 modules, including anomalib.models) |
| pip check | No broken requirements found |
| Dependency lock | 102 exact installed version pins, no editable or machine-local paths |
| C: free disk during doctor | 154,130,096,128 bytes |

Exact setup commands are in [README.md](../README.md); all transitive runtime
versions are in [requirements-lock.txt](../requirements-lock.txt). Initial resolution
used `python -m pip install -e . lightning==2.6.1 pytorch-lightning==2.6.1` after
installing torch/torchvision from `https://download.pytorch.org/whl/cu126`.
The old Python 3.13 environment was initially retained under `.venv-scaffold/`;
the owner deleted that backup before Task 3. The active environment is `.venv/`.
No system Python, driver, or CUDA toolkit was modified.

### PatchCore smoke evidence

Executed `.\.venv\Scripts\python.exe tests/check_patchcore.py` successfully on CUDA.
The check used the real pretrained Wide ResNet-50-2, layer2/layer3, FP32,
coreset ratio 0.1, and 9 neighbors. Two synthetic 256x256 RGB inputs produced
a memory bank of shape `[204, 1536]`; one separate synthetic inference input
produced score shape `[1]` and map shape `[1, 1, 256, 256]`, both finite.
The smoke batch of 2 is solely a small API check; the planned experiment batch
size remains 4. No OOM fallback was needed.

Weights were explicitly frozen; the feature extractor stayed in evaluation mode.
Anomalib's training mode collected embeddings, `fit()` selected the coreset, and
`configure_optimizers()` returned None. Preprocessing and postprocessing hooks,
evaluator, and visualizer were disabled; ImageNet normalization was applied once
to synthetic input. The returned score/map were raw library outputs.

Recorded pretrained identity:

- timm identifier: `wide_resnet50_2.racm_in1k`.
- Hugging Face repository: `timm/wide_resnet50_2.racm_in1k`.
- Downloaded revision: `30f73aceaaa1911830a9795b83ab1908dba18719`.
- `model.safetensors` SHA-256: `03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000`.
- Model metadata license: Apache-2.0 (dataset licensing remains separate).
- ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

The pretrained model's default classification resize/crop recipe is not the
project's preprocessing: PLAN.md requires full-image 256x256 bilinear resize
without center crop. Task 5 must preserve that protocol and verify image decoding
and preprocessing parity; this smoke used already-sized synthetic tensors.

Local logs are under ignored `runs/task2/`: `install.log`, `doctor.json`,
`doctor.stderr`, `patchcore-smoke.json`, and `patchcore-smoke.stderr`.
The HF download emitted an unauthenticated-request rate-limit warning but
succeeded without a token. Coreset progress messages are normal stderr output.

### Sources checked

- [Official PyTorch version pairings and CUDA wheel commands](https://pytorch.org/get-started/previous-versions/).
- [Anomalib 2.3.2 release](https://pypi.org/project/anomalib/2.3.2/)
  and [release dependency metadata](https://pypi.org/pypi/anomalib/2.3.2/json):
  Python >=3.10; cu126 extra permits torch >=2.6.0, <=2.8.0.
- Version-matched implementation inspected directly from the downloaded 2.3.2
  wheel: `patchcore/lightning_model.py`, `patchcore/torch_model.py`, and
  `components/feature_extractors/timm.py`. Guessed GitHub tag URLs returned 404.

Remaining verification belongs to Tasks 3–9: real data integrity, preprocessing
parity, fitting/calibration isolation, complete persistence, resource use on the
full fitting split, measured evaluation, and UI behavior.

## Task 5 measured full PatchCore run

The full 224-image CUDA fit and 56-image normal calibration passed on this machine.
Batch size remained 4, FP32, coreset ratio 0.1; no memory fallback was required.
Fit time was 206.88776750001125 seconds (decode/preprocess/extract/coreset only).
Peak allocated/reserved CUDA memory was 2,920,305,664 / 3,571,449,856 bytes through
fitting and the first reference inference. Windows process peak working set
through save/reload was 2,313,945,088 bytes. The memory bank contains 22,937 vectors
of dimension 1,536. Model and raw score/map reload passed in a fresh process.
See progress.md for exact commands, artifact hashes, and scope. Remaining work
is test evaluation, robustness, and the local web interface; those are not yet verified.

## Task 8 environment update (2026-09-27)

The earlier sections above describe their historical task state. Primary
evaluation, brightness experiments and the local web interface are now verified;
see progress.md and results.md for evidence.

- Node.js 22.19.0; npm 11.6.0; Vite 7.3.6. React and Tailwind exact resolved
  versions are retained in frontend/package-lock.json.
- FastAPI 0.141.1; uvicorn 0.54.0; httpx 0.28.1; Playwright 1.63.0.
  Streamlit 1.64.0 was uninstalled and removed from project requirements and lock.
- requirements-lock.txt now contains 114 exact installed distribution pins,
  excluding the editable project. Existing scientific package versions are
  unchanged. Historical transitive packages remain pinned if still installed;
  no broad package cleanup was performed.
- API and built frontend ran at http://127.0.0.1:8000 with one worker. Browser
  checks used installed Microsoft Edge in headless mode; no browser download.
- pip check passed; doctor ok, including CPU/CUDA tensor checks. Full tests and
  real-artifact browser checks passed. Task 8 logs and screenshots remain local
  in ignored runs/task8/.

Official integration references checked for this implementation:
[Vite requirements](https://vite.dev/guide/),
[Tailwind Vite integration](https://tailwindcss.com/docs/installation/using-vite),
[FastAPI request access](https://fastapi.tiangolo.com/advanced/using-request-directly/),
[FastAPI static files](https://fastapi.tiangolo.com/tutorial/static-files/).
