"""Thin adapter over Anomalib's raw PatchcoreModel; no automatic calibration."""

import gc
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from carpet_ad.preprocessing import load_rgb, resize_rgb


WEIGHTS = "wide_resnet50_2.racm_in1k"
REVISION = "30f73aceaaa1911830a9795b83ab1908dba18719"
WEIGHT_SHA256 = "03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000"


def to_tensor(image):
    rgb = np.asarray(resize_rgb(image), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
    return (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def create_model(pretrained=True, device=None):
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["HF_HOME"] = str(Path("artifacts/cache/huggingface").resolve())
    os.environ["TORCH_HOME"] = str(Path("artifacts/cache/torch").resolve())
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    # The tag and pinned timm version make the architecture/weight choice explicit.
    model = PatchcoreModel(backbone=WEIGHTS, layers=["layer2", "layer3"],
                          pre_trained=pretrained, num_neighbors=9)
    if pretrained:
        cache = Path(os.environ["HF_HOME"]) / "hub/models--timm--wide_resnet50_2.racm_in1k"
        revision = (cache / "refs/main").read_text().strip()
        weights = cache / "snapshots" / revision / "model.safetensors"
        if revision != REVISION or hashlib.sha256(weights.read_bytes()).hexdigest() != WEIGHT_SHA256:
            raise ValueError("Downloaded pretrained weight identity changed; refusing to fit")
    model.requires_grad_(False)
    return model.float().to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()


def load_model(path, device=None):
    model = create_model(pretrained=False, device="cpu")
    # Anomalib DynamicBufferMixin resizes memory_bank when loading the state dict.
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    if model.memory_bank.ndim != 2 or model.memory_bank.shape[0] < 9:
        raise ValueError("Missing or invalid PatchCore memory bank")
    return model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()


def predict(model, image):
    device = model.memory_bank.device
    with torch.no_grad():
        output = model(to_tensor(image).unsqueeze(0).to(device))
    score = float(output.pred_score[0].cpu())
    anomaly_map = output.anomaly_map[0, 0].cpu().numpy()
    if anomaly_map.shape != (256, 256) or not np.isfinite(score) or not np.isfinite(anomaly_map).all():
        raise ValueError("Invalid raw PatchCore output")
    return score, anomaly_map


def fit_bank(rows, root, run, config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = create_model()
    device = model.memory_bank.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = perf_counter()
    batch_size, fallbacks = config["batch_size"], []
    model.train()
    model.feature_extractor.eval()
    position = 0
    while position < len(rows):
        batch = None
        try:
            batch = torch.stack([to_tensor(load_rgb(root / r["path"])) for r in rows[position:position + batch_size]]).to(device)
            with torch.no_grad():
                model(batch)
            position += len(batch)
            del batch
        except torch.cuda.OutOfMemoryError:
            del batch
            if batch_size == 1:
                raise
            batch_size //= 2
            fallbacks.append(f"CUDA extraction OOM: retry remaining images with batch {batch_size}")
            torch.cuda.empty_cache()
    model.subsample_embedding(config["coreset_sampling_ratio"])
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    fit_seconds = perf_counter() - started
    assert not model.feature_extractor.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    reference_image = load_rgb(root / rows[0]["path"])
    reference_score, reference_map = predict(model, reference_image)
    np.savez_compressed(run / "reload_reference.npz", score=reference_score, anomaly_map=reference_map)
    info = dict(device=str(device), fit_seconds=fit_seconds, batch_size=batch_size, fallbacks=fallbacks,
                memory_bank_shape=list(model.memory_bank.shape), pretrained_weights=WEIGHTS,
                pretrained_revision=REVISION, pretrained_sha256=WEIGHT_SHA256,
                score_semantics="Anomalib raw pred_score; higher is more anomalous",
                preprocessing="Pillow EXIF -> RGB -> full 256x256 bilinear -> float32 /255 -> ImageNet mean/std once",
                fit_timing_scope="decode, preprocessing, frozen feature extraction, 10% coreset; excludes model construction and persistence",
                reload_reference_path=rows[0]["path"], reload_rtol=1e-5, reload_atol=1e-5,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else None)
    torch.save({name: tensor.cpu() for name, tensor in model.state_dict().items()}, run / "model.pt")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded = load_model(run / "model.pt", str(device))
    score, anomaly_map = predict(reloaded, reference_image)
    np.testing.assert_allclose(score, reference_score, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(anomaly_map, reference_map, rtol=1e-5, atol=1e-5)
    info["reload_verified"] = True
    if sys.platform == "win32":
        measurement = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {os.getpid()}).PeakWorkingSet64"], capture_output=True, text=True, check=True)
        info["process_peak_working_set_bytes"] = int(measurement.stdout.strip())
    return info
