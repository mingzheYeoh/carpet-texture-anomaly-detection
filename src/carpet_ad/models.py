"""Normal-only baseline fitting and separate calibration; local trusted artifacts."""

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import re
import platform
import subprocess
from time import perf_counter

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import yaml

from carpet_ad.features import extract_features
from carpet_ad.preprocessing import load_rgb


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_path(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run-id must contain only letters, digits, underscores, or hyphens")
    path = Path("runs") / run_id
    if path.is_symlink():
        raise ValueError("Run directory cannot be a symlink")
    return path


def _config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("model") not in {"lbp_ocsvm", "lbp_glcm_ocsvm", "patchcore"}:
        raise ValueError("Unsupported model configuration")
    if config["model"] == "patchcore":
        expected = dict(model="patchcore", seed=42, manifest="data/manifests/carpet_split.csv",
                        image_size=[256, 256], backbone="wide_resnet50_2", pretrained=True,
                        pretrained_weights="wide_resnet50_2.racm_in1k", layers=["layer2", "layer3"],
                        coreset_sampling_ratio=0.1, num_neighbors=9, batch_size=4,
                        precision="float32", num_workers=0, device="auto",
                        calibration=dict(quantile=0.95, method="linear", comparison="greater"))
        if config != expected:
            raise ValueError("Configuration differs from the approved PatchCore protocol")
        return config
    expected = dict(model=config["model"], seed=42, manifest="data/manifests/carpet_split.csv",
                    image_size=[256, 256], lbp=dict(points=8, radius=1, method="uniform", grid=[4, 4], bins=10, normalization="l1"),
                    scaler="standard", svm=dict(kernel="rbf", nu=0.05, gamma="scale"),
                    calibration=dict(quantile=0.95, method="linear", comparison="greater"))
    if config["model"] == "lbp_glcm_ocsvm":
        expected["glcm"] = dict(levels=32, distances=[1, 3], angles_degrees=[0, 45, 90, 135],
                                symmetric=True, normed=True, properties=["contrast", "homogeneity", "energy", "correlation"])
    if config != expected:
        raise ValueError("Configuration differs from the approved fixed baseline protocol")
    return config


def _normal_rows(manifest, split, root):
    with Path(manifest).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    paths, hashes = set(), defaultdict(set)
    selected = []
    for row in rows:
        if row["path"] in paths:
            raise ValueError("Duplicate manifest path")
        paths.add(row["path"])
        hashes[row["sha256"]].add(row["split"])
        if row["split"] != split:
            continue  # Never parse test labels or open images outside the selected split.
        if row["original_split"] != "train" or row["label"] != "0" or row["defect_type"] != "good":
            raise ValueError("Fitting and calibration require official normal training images")
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or not row["path"].startswith("train/good/"):
            raise ValueError("Selected image path must stay inside train/good")
        if _hash(path) != row["sha256"]:
            raise ValueError(f"Image hash changed: {row['path']}")
        selected.append(row)
    if any(len(splits) > 1 for splits in hashes.values()):
        raise ValueError("Cross-split duplicate hashes")
    if not selected:
        raise ValueError(f"No normal images in {split}")
    return selected


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit(config_path, run_id):
    run = _run_path(run_id)
    if run.exists():
        raise FileExistsError(f"Run already exists: {run}; use a new run-id")
    config = _config(config_path)
    root = Path("data/raw/carpet").resolve()
    manifest = Path(config["manifest"])
    rows = _normal_rows(manifest, "fit", root)
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.yaml").write_bytes(Path(config_path).read_bytes())
    (run / "manifest.csv").write_bytes(manifest.read_bytes())
    source = Path(__file__).parent
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=source, capture_output=True, text=True)
    metadata = dict(status="fitting", model=config["model"], seed=config["seed"], device="cpu",
                    created_at=datetime.now(timezone.utc).isoformat(), manifest_sha256=_hash(manifest),
                    config_sha256=_hash(config_path), code_commit=git.stdout.strip() if git.returncode == 0 else None,
                    code_dirty=bool(status.stdout.strip()), python=platform.python_version(),
                    source_sha256={p.name: _hash(p) for p in sorted(source.glob("*.py"))},
                    packages={d.metadata["Name"]: d.version for d in distributions()}, fit_images=len(rows),
                    score_semantics="negative decision_function; higher is more anomalous",
                    preprocessing="Pillow EXIF transpose -> RGB -> 256x256 bilinear -> Pillow L uint8; no crop",
                    feature_order="LBP: row-major 4x4 cells, bins 0..9; GLCM: property, distance, angle",
                    artifacts={"model": "model.joblib", "manifest": "manifest.csv", "config": "config.yaml"})
    _write_json(run / "metadata.json", metadata)
    if config["model"] == "patchcore":
        from carpet_ad.patchcore import fit_bank

        metadata.update(fit_bank(rows, root, run, config))
        metadata.pop("feature_order")
        metadata["artifacts"]["model"] = "model.pt"
        metadata.update(status="fitted", model_sha256=_hash(run / "model.pt"),
                        completed_at=datetime.now(timezone.utc).isoformat())
        _write_json(run / "metadata.json", metadata)
        return {"run": str(run), "fit_images": len(rows), "memory_bank_shape": metadata["memory_bank_shape"],
                "fit_seconds": metadata["fit_seconds"], "device": metadata["device"]}
    started = perf_counter()
    features = np.stack([extract_features(load_rgb(root / row["path"]), config["model"]) for row in rows])
    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", OneClassSVM(**config["svm"]))])
    pipeline.fit(features)
    metadata["fit_seconds"] = perf_counter() - started
    metadata["fit_timing_scope"] = "image decode, preprocessing, feature extraction, scaler and SVM fit; excludes hash validation and persistence"
    joblib.dump(pipeline, run / "model.joblib")
    reloaded = joblib.load(run / "model.joblib")
    np.testing.assert_allclose(pipeline.decision_function(features), reloaded.decision_function(features), rtol=0, atol=1e-12)
    metadata.update(status="fitted", feature_count=features.shape[1], model_sha256=_hash(run / "model.joblib"),
                    reload_verified=True, completed_at=datetime.now(timezone.utc).isoformat())
    _write_json(run / "metadata.json", metadata)
    return {"run": str(run), "fit_images": len(rows), "feature_count": features.shape[1], "fit_seconds": metadata["fit_seconds"]}


def calibrate_scores(scores):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be a nonempty finite vector")
    return float(np.quantile(values, 0.95, method="linear"))


def calibrate(run_id):
    run = _run_path(run_id)
    if (run / "calibration.json").exists() or (run / "calibration_scores.csv").exists():
        raise FileExistsError("Calibration already exists; use a new run for any change")
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "fitted":
        raise ValueError("Run has not completed fitting")
    model_filename = metadata["artifacts"]["model"]
    for filename, key in (("manifest.csv", "manifest_sha256"), ("config.yaml", "config_sha256"), (model_filename, "model_sha256")):
        if _hash(run / filename) != metadata[key]:
            raise ValueError(f"Run artifact hash changed: {filename}")
    config = _config(run / "config.yaml")
    if _hash(config["manifest"]) != metadata["manifest_sha256"]:
        raise ValueError("Current manifest differs from fitted manifest")
    root = Path("data/raw/carpet").resolve()
    rows = _normal_rows(run / "manifest.csv", "calibration", root)
    device = "cpu"
    if config["model"] == "patchcore":
        from carpet_ad.patchcore import load_model, predict

        model = load_model(run / model_filename)
        device = str(model.memory_bank.device)
        scores = np.array([predict(model, load_rgb(root / row["path"]))[0] for row in rows])
    else:
        pipeline = joblib.load(run / model_filename)  # Only locally created trusted artifacts.
        features = np.stack([extract_features(load_rgb(root / row["path"]), config["model"]) for row in rows])
        scores = -pipeline.decision_function(features)
    threshold = calibrate_scores(scores)
    with (run / "calibration_scores.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "score", "threshold", "decision"])
        writer.writerows((row["path"], float(score), threshold, int(score > threshold)) for row, score in zip(rows, scores, strict=True))
    result = dict(run=str(run), calibration_images=len(rows), threshold=threshold, quantile=0.95,
                  method="linear", decision_rule="score > threshold; ties are normal",
                  created_at=datetime.now(timezone.utc).isoformat(), model_sha256=metadata["model_sha256"],
                  manifest_sha256=metadata["manifest_sha256"], config_sha256=metadata["config_sha256"],
                  device=device, seed=config["seed"], code_commit=metadata["code_commit"],
                  source_sha256={p.name: _hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                  packages={d.metadata["Name"]: d.version for d in distributions()},
                  scores_sha256=_hash(run / "calibration_scores.csv"))
    _write_json(run / "calibration.json", result)
    return result
