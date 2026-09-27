"""Opt-in fresh-process validation of a fitted, calibrated local PatchCore run."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from carpet_ad.models import _hash, _run_path, calibrate_scores
from carpet_ad.patchcore import load_model, predict
from carpet_ad.preprocessing import load_rgb

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-id", default="patchcore_seed42")
run = _run_path(parser.parse_args().run_id)
metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
calibration = json.loads((run / "calibration.json").read_text(encoding="utf-8"))
assert metadata["status"] == "fitted" and metadata["model"] == "patchcore"
assert _hash(run / "model.pt") == metadata["model_sha256"] == calibration["model_sha256"]
assert _hash(run / "manifest.csv") == metadata["manifest_sha256"]
assert _hash(run / "calibration_scores.csv") == calibration["scores_sha256"]
model = load_model(run / "model.pt")
assert list(model.memory_bank.shape) == metadata["memory_bank_shape"]
assert not model.training and not model.feature_extractor.training
assert not any(parameter.requires_grad for parameter in model.parameters())
root = Path("data/raw/carpet")
with np.load(run / "reload_reference.npz") as reference:
    score, anomaly_map = predict(model, load_rgb(root / metadata["reload_reference_path"]))
    np.testing.assert_allclose(score, reference["score"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(anomaly_map, reference["anomaly_map"], rtol=1e-5, atol=1e-5)
with (run / "calibration_scores.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
with (run / "manifest.csv").open(encoding="utf-8", newline="") as stream:
    manifest = list(csv.DictReader(stream))
assert {row["path"] for row in rows} == {row["path"] for row in manifest if row["split"] == "calibration"}
actual = np.array([predict(model, load_rgb(root / row["path"]))[0] for row in rows])
np.testing.assert_allclose(actual, [float(row["score"]) for row in rows], rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(calibrate_scores(actual), calibration["threshold"], rtol=1e-5, atol=1e-5)
assert [int(score > calibration["threshold"]) for score in actual] == [int(row["decision"]) for row in rows]
print("PASS: frozen weights, memory bank, raw score/map reload, calibration membership/scores/threshold/decisions.")
