"""Local-only web adapter. Start from the repository root with one worker."""

import base64
import csv
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter
import warnings

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
import matplotlib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from carpet_ad.inference import load_run, score_image

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
MODELS = {
    "patchcore": {"name": "PatchCore", "run_id": "patchcore_seed42", "artifact": "model.pt"},
    "lbp": {"name": "LBP", "run_id": "lbp_seed42", "artifact": "model.joblib"},
    "lbp_glcm": {"name": "LBP + GLCM", "run_id": "lbp_glcm_seed42", "artifact": "model.joblib"},
}
SAMPLES = {"a": "test/good/000.png", "b": "test/cut/000.png"}
# ponytail: one local inference at a time; add a worker queue only for multi-user use.
INFERENCE_LOCK = Lock()
app = FastAPI(title="Carpet inspection", docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@lru_cache(maxsize=3)
def cached_run(model):
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Start the API from the repository root")
    return load_run(MODELS[model]["run_id"])


def decode_upload(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise HTTPException(415, "Only PNG and JPEG images are supported.")
                if image.width * image.height > MAX_PIXELS:
                    raise HTTPException(413, "Image exceeds 20 megapixels. Resize it and try again.")
                if getattr(image, "n_frames", 1) != 1:
                    raise HTTPException(415, "Animated images are unsupported. Choose a static PNG or JPEG.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(413, "Image dimensions are too large. Resize it and try again.") from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(400, "Cannot read this image. It may be corrupt. Choose another PNG or JPEG.") from error


def png_url(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def predict_bytes(data, model_id):
    image = decode_upload(data)
    if not INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(503, "Another inspection is running. Please try again shortly.")
    try:
        try:
            metadata, calibration, model = cached_run(model_id)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            raise HTTPException(503, "Model artifacts are missing, invalid or could not load. Check the local runs folder and restart the server.") from error
        device = str(model.memory_bank.device) if model_id == "patchcore" else "cpu"
        try:
            started = perf_counter()
            score, raw_map = score_image(model, metadata["model"], image)
            elapsed = (perf_counter() - started) * 1000
        except RuntimeError as error:
            raise HTTPException(503, "Inference failed; RAM or GPU memory may be unavailable. Close other workloads and retry.") from error
        except (ValueError, OSError) as error:
            raise HTTPException(422, "The model could not process this image. Try another carpet texture image.") from error
        heatmap, low, high = None, None, None
        if raw_map is not None:
            low, high = float(raw_map.min()), float(raw_map.max())
            relative = (raw_map - low) / (high - low) if high > low else np.zeros_like(raw_map)
            heatmap = png_url(Image.fromarray(matplotlib.colormaps["inferno"](relative, bytes=True)[..., :3]))
        preview = image.copy()
        preview.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
        return dict(model=model_id, run_id=MODELS[model_id]["run_id"], score=score,
                    threshold=calibration["threshold"], decision="anomaly" if score > calibration["threshold"] else "normal",
                    device=device, inference_ms=elapsed, width=image.width, height=image.height,
                    original=png_url(preview), heatmap=heatmap, map_min=low, map_max=high)
    finally:
        INFERENCE_LOCK.release()


@app.post("/api/predict")
async def predict(request: Request, model: str = "patchcore"):
    if model not in MODELS:
        raise HTTPException(400, "Unknown model. Select one of the available models.")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, "File exceeds 10 MB. Compress or resize it and try again.")
        data.extend(chunk)
    return await run_in_threadpool(predict_bytes, bytes(data), model)


@app.get("/api/status")
def status():
    import torch
    gpu = torch.cuda.is_available()
    return dict(device=torch.cuda.get_device_name(0) if gpu else "CPU: CUDA unavailable; inference will be slower",
                gpu=gpu, models=[dict(id=key, name=value["name"],
                    available=all((ROOT / "runs" / value["run_id"] / name).is_file()
                        for name in (value["artifact"], "metadata.json", "config.yaml", "manifest.csv", "calibration.json", "calibration_scores.csv")))
                    for key, value in MODELS.items()])


@app.get("/api/results")
def results():
    tables = {}
    for name in ("clean", "brightness"):
        try:
            with (ROOT / f"reports/tables/{name}_comparison.csv").open(encoding="utf-8", newline="") as stream:
                tables[name] = list(csv.DictReader(stream))
        except OSError as error:
            raise HTTPException(503, "Measured reports are missing. Run evaluation and compare first.") from error
    return tables


@app.get("/api/samples/{sample}")
def sample(sample: str):
    if sample not in SAMPLES:
        raise HTTPException(404, "Sample not found.")
    path = ROOT / "data/raw/carpet" / SAMPLES[sample]
    if not path.is_file():
        raise HTTPException(404, "Local sample is missing. Upload your own carpet image.")
    return FileResponse(path, media_type="image/png")


if (ROOT / "frontend/dist").is_dir():
    app.mount("/", StaticFiles(directory=ROOT / "frontend/dist", html=True), name="frontend")
