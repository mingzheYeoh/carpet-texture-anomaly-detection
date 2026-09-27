"""Hand-counted metrics and strict thresholds; no model tuning."""

import numpy as np
import pytest


def test_brightness_clips_before_resize_and_preserves_input():
    from PIL import Image
    from carpet_ad.evaluation import apply_condition

    pixels = np.array([[[0, 101, 255], [200, 250, 10]]], dtype=np.uint8)
    image = Image.fromarray(pixels)
    assert apply_condition(image, "clean") is image
    np.testing.assert_array_equal(np.asarray(apply_condition(image, "brightness_0.8")),
                                  [[[0, 80, 204], [160, 200, 8]]])
    np.testing.assert_array_equal(np.asarray(apply_condition(image, "brightness_1.2")),
                                  [[[0, 121, 255], [240, 255, 12]]])
    np.testing.assert_array_equal(np.asarray(image), pixels)
    with pytest.raises(ValueError):
        apply_condition(image, "unknown")


def test_metrics_use_raw_scores_and_strict_threshold():
    from carpet_ad.evaluation import image_metrics

    result = image_metrics([0, 0, 1, 1], [.1, .8, .6, .9], .6)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["auroc"] == .75
    assert result["precision"] == result["recall"] == result["f1"] == .5
    assert result["false_positive_rate"] == .5
    with pytest.raises(ValueError):
        image_metrics([0, 1], [np.nan, .4], .5)


def test_pixel_auc_is_global_not_per_image_scaled():
    from carpet_ad.evaluation import pixel_auroc

    # Two maps whose global ordering is deliberately different from local scaling.
    masks = [np.array([[0, 1]]), np.array([[0, 1]])]
    maps = [np.array([[10., 11.]]), np.array([[0., 1.]])]
    assert pixel_auroc(masks, maps) == .75


def test_comparison_preserves_threshold_precision_and_rejects_unpaired_ids():
    import json
    import pandas as pd
    from carpet_ad.evaluation import compare, image_metrics
    from carpet_ad.models import _hash

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as monkeypatch:
        tmp_path = Path(directory)
        monkeypatch.chdir(tmp_path)
        run = tmp_path / "runs/example"
        threshold = 0.16614510826979537  # Default CSV parsing loses its last digit.
        frame = pd.DataFrame(dict(path=["test/good/a.png", "test/cut/b.png"],
            label=[0, 1], defect_type=["good", "cut"], condition=["clean"] * 2,
            raw_score=[0., 1.], threshold=[threshold] * 2, decision=[0, 1]))
        summary = dict(image_metrics(frame.label, frame.raw_score, threshold), run_id="example",
            model="lbp_glcm_ocsvm", images=2, threshold=threshold, manifest_sha256="same",
            frozen_hashes={}, mask_sha256={}, pixel_auroc=None, fit_seconds=1., device="cpu",
            latency_median_ms=1., latency_p95_ms=1.)
        for condition in ("clean", "brightness_0.8"):
            folder = run / "evaluation" / condition
            folder.mkdir(parents=True)
            frame.assign(condition=condition).to_csv(folder / "predictions.csv", index=False)
            summary["predictions_sha256"] = _hash(folder / "predictions.csv")
            (folder / "metrics.json").write_text(json.dumps(summary))
        compare(["example"])
        table = pd.read_csv("reports/tables/brightness_comparison.csv")
        assert len(table) == 2 and (table.delta_f1 == 0).all()
        folder = run / "evaluation/brightness_0.8"
        frame.loc[0, "path"] = "test/good/different.png"
        frame.assign(condition="brightness_0.8").to_csv(folder / "predictions.csv", index=False)
        summary["predictions_sha256"] = _hash(folder / "predictions.csv")
        (folder / "metrics.json").write_text(json.dumps(summary))
        with pytest.raises(ValueError, match="paired"):
            compare(["example"])
