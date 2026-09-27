"""Baseline dimensions, calibration, isolation, and persistence contracts."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
from PIL import Image
import pytest

from carpet_ad.features import extract_features
from carpet_ad.models import fit, calibrate, calibrate_scores
from carpet_ad.preprocessing import load_rgb
from carpet_ad.inference import load_run, score_image


def test_features_and_calibration():
    image = Image.new("RGB", (300, 200), "white")
    a = extract_features(image, "lbp_ocsvm")
    b = extract_features(image, "lbp_glcm_ocsvm")
    assert a.shape == (160,) and b.shape == (192,)
    np.testing.assert_array_equal(a, b[:160])
    np.testing.assert_allclose(a.reshape(16, 10).sum(axis=1), 1)
    assert np.isfinite(b).all()
    np.testing.assert_allclose(b[160:168], 0)  # Constant-image contrast.
    np.testing.assert_allclose(b[168:], 1)  # Homogeneity, energy, correlation.
    edge = np.zeros((256, 256), dtype=np.uint8)
    edge[:, 128:] = 255
    # One 0-to-31 transition per 255 horizontal pairs; catches uint8 overflow.
    contrast = extract_features(Image.fromarray(edge), "lbp_glcm_ocsvm")[160]
    assert contrast == pytest.approx(31 ** 2 / 255)
    assert calibrate_scores([0, 1, 2, 3, 4]) == 3.8
    threshold = calibrate_scores([2, 2, 2])
    assert not (2 > threshold)  # Ties remain normal.
    with pytest.raises(ValueError):
        calibrate_scores([float("nan")])
    with pytest.raises(ValueError):
        calibrate_scores([])


def test_preprocessing_applies_exif_and_keeps_full_frame():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "oriented.png"
        image = Image.new("RGB", (4, 2), "red")
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif)
        loaded = load_rgb(path)
        assert loaded.size == (2, 4)
        from carpet_ad.preprocessing import resize_rgb
        resized = resize_rgb(loaded)
        assert resized.size == (256, 256) and resized.mode == "RGB"
        assert resized.getpixel((0, 0)) == (255, 0, 0)


def test_fit_calibrate_are_isolated_and_reloadable():
    with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as monkeypatch:
        tmp_path = Path(directory)
        monkeypatch.chdir(tmp_path)
        root = Path("data/raw/carpet")
        root.mkdir(parents=True)
        rows = []
        rng = np.random.default_rng(12)
        for index in range(6):
            path = f"train/good/{index}.png"
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)).save(target)
            rows.append(dict(path=path, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                             original_split="train", split="fit" if index < 4 else "calibration",
                             label="0", defect_type="good", mask_path=""))
        # An absent test image proves neither command opens test images or parses test labels.
        rows.append(dict(path="test/cut/absent.png", sha256="f" * 64, original_split="test",
                         split="test", label="NOT_CONSUMABLE", defect_type="cut", mask_path=""))
        manifest = Path("data/manifests/carpet_split.csv")
        manifest.parent.mkdir(parents=True)
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        project = Path(__file__).resolve().parents[1]
        for model in ("lbp_ocsvm", "lbp_glcm_ocsvm"):
            config = tmp_path / f"{model}.yaml"
            config.write_bytes((project / "configs" / f"{model}.yaml").read_bytes())
            result = fit(config, model)
            artifact = Path(result["run"]) / "model.joblib"
            pipeline = joblib.load(artifact)
            features = np.stack([extract_features(load_rgb(root / r["path"]), model) for r in rows[:4]])
            np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, features.mean(axis=0))
            np.testing.assert_allclose(pipeline.decision_function(features), joblib.load(artifact).decision_function(features), rtol=0, atol=1e-12)
            before = artifact.read_bytes()
            calibrated = calibrate(model)
            assert calibrated["calibration_images"] == 2
            assert artifact.read_bytes() == before
            metadata, frozen_calibration, restored = load_run(model)
            score, anomaly_map = score_image(restored, metadata["model"], load_rgb(root / rows[0]["path"]))
            assert anomaly_map is None
            assert score == pytest.approx(-pipeline.decision_function(features[:1])[0])
            calibration_path = artifact.parent / "calibration.json"
            calibration_bytes = calibration_path.read_bytes()
            frozen_calibration["threshold"] += 1
            calibration_path.write_text(json.dumps(frozen_calibration), encoding="utf-8")
            with pytest.raises(ValueError, match="Threshold"):
                load_run(model)
            calibration_path.write_bytes(calibration_bytes)
            with pytest.raises(FileExistsError):
                fit(config, model)
            with pytest.raises(FileExistsError):
                calibrate(model)
        with pytest.raises(ValueError):
            fit(config, "../escape")
        rows[0]["label"] = "1"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with pytest.raises(ValueError, match="normal"):
            fit(config, "bad-label")
