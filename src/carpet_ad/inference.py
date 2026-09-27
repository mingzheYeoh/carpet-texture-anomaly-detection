"""Load frozen local runs and share the same scoring path across consumers."""

import csv
import json

import joblib
import numpy as np

from carpet_ad.features import extract_features
from carpet_ad.models import _config, _hash, _run_path, calibrate_scores


def load_run(run_id):
    run = _run_path(run_id)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads((run / "calibration.json").read_text(encoding="utf-8"))
    if metadata["status"] != "fitted":
        raise ValueError("Run is not fitted")
    config = _config(run / "config.yaml")
    for filename, key in ((metadata["artifacts"]["model"], "model_sha256"),
                          ("manifest.csv", "manifest_sha256"), ("config.yaml", "config_sha256")):
        if _hash(run / filename) != metadata[key] or calibration[key] != metadata[key]:
            raise ValueError(f"Frozen artifact mismatch: {filename}")
    if _hash(config["manifest"]) != metadata["manifest_sha256"]:
        raise ValueError("Current manifest differs from the fitted manifest")
    if _hash(run / "calibration_scores.csv") != calibration["scores_sha256"]:
        raise ValueError("Calibration scores changed")
    with (run / "calibration_scores.csv").open(encoding="utf-8", newline="") as stream:
        scores = [float(row["score"]) for row in csv.DictReader(stream)]
    if calibration["threshold"] != calibrate_scores(scores):
        raise ValueError("Threshold differs from frozen calibration scores")
    if metadata["model"] == "patchcore":
        from carpet_ad.patchcore import load_model
        model = load_model(run / metadata["artifacts"]["model"])
    else:
        model = joblib.load(run / metadata["artifacts"]["model"])
    return metadata, calibration, model


def score_image(model, model_name, image):
    if model_name == "patchcore":
        from carpet_ad.patchcore import predict
        return predict(model, image)
    features = extract_features(image, model_name)
    score = float(-model.decision_function(features.reshape(1, -1))[0])
    if not np.isfinite(score):
        raise ValueError("Nonfinite anomaly score")
    return score, None
