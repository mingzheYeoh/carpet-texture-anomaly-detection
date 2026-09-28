"""Upload boundary and frozen inference contract; no model download required."""

from io import BytesIO
from unittest.mock import patch

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient


def test_upload_validation_and_frozen_response():
    from app.api import app, MODELS

    client = TestClient(app)
    def post(data, model="lbp"):
        return client.post(f"/api/predict?model={model}", content=data,
                           headers={"content-type": "image/png"})
    assert post(b"not an image").status_code == 400
    assert post(b"", "../escape").status_code == 400
    assert post(b"x" * (10 * 1024 * 1024 + 1)).status_code == 413
    gif = BytesIO()
    Image.new("RGB", (2, 2)).save(gif, format="GIF")
    assert post(gif.getvalue()).status_code == 415
    large = BytesIO()
    Image.new("1", (5000, 4001)).save(large, format="PNG")
    assert post(large.getvalue()).status_code == 413
    data = BytesIO()
    Image.new("RGB", (24, 16), "red").save(data, format="PNG")
    metadata = {"model": "lbp_ocsvm"}
    with patch("app.api.cached_run", return_value=(metadata, {"threshold": .5}, object())), \
         patch("app.api.score_image", return_value=(.5, None)):
        result = post(data.getvalue()).json()
        assert result["score"] == result["threshold"] == .5
        assert result["decision"] == "normal" and result["heatmap"] is None
        assert result["width"] == 24 and result["height"] == 16
        assert result["run_id"] == MODELS["lbp"]["run_id"]
    with patch("app.api.cached_run", side_effect=FileNotFoundError("missing")):
        assert post(data.getvalue()).status_code == 503
    with patch("app.api.cached_run", side_effect=KeyError("status")):
        assert post(data.getvalue()).status_code == 503


def test_heatmap_does_not_change_score():
    from app.api import app
    client = TestClient(app)
    data = BytesIO()
    Image.new("RGB", (10, 10)).save(data, format="JPEG")
    model = type("Model", (), {"memory_bank": type("Bank", (), {"device": "cpu"})()})()
    with patch("app.api.cached_run", return_value=({"model": "patchcore"}, {"threshold": 30.}, model)), \
         patch("app.api.score_image", return_value=(31., np.full((256, 256), 7.))):
        result = client.post("/api/predict?model=patchcore", content=data.getvalue()).json()
        assert result["score"] == 31. and result["decision"] == "anomaly"
        assert result["heatmap"].startswith("data:image/png;base64,")
        assert result["map_min"] == result["map_max"] == 7.


def test_cpu_status_and_missing_report_are_explicit():
    from pathlib import Path
    import tempfile
    from app.api import app

    client = TestClient(app)
    with patch("torch.cuda.is_available", return_value=False):
        result = client.get("/api/status").json()
        assert not result["gpu"] and "CUDA unavailable" in result["device"]
    with tempfile.TemporaryDirectory() as directory, patch("app.api.ROOT", Path(directory)):
        assert client.get("/api/results").status_code == 503
        assert client.get("/api/samples/a").status_code == 404


def test_space_hostname_is_allowed_only_when_configured():
    from app.api import app
    client = TestClient(app)
    assert client.get("/api/results", headers={"host": "untrusted.example"}).status_code == 400
    # The configured Space host is populated by Hugging Face's SPACE_HOST variable.
    from app.api import allowed_hosts
    with patch.dict("os.environ", {"SPACE_HOST": "owner-carpet.hf.space"}):
        assert "owner-carpet.hf.space" in allowed_hosts()
        assert "*" not in allowed_hosts()
