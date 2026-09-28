"""Opt-in container/Space smoke check against saved clean predictions."""

import argparse
import csv
from io import BytesIO
import base64
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:7860")
parser.add_argument("--private", action="store_true", help="Use saved Hugging Face login for a private Space")
args = parser.parse_args()
url = args.url.rstrip("/")
headers = {}
if args.private:
    from urllib.parse import urlsplit
    from huggingface_hub import get_token
    target = urlsplit(url)
    assert target.scheme == "https" and target.hostname.endswith(".hf.space")
    token = get_token()
    assert token, "Log in with hf auth login first"
    headers["Authorization"] = f"Bearer {token}"
with httpx.Client(base_url=url, timeout=180, headers=headers) as client:
    status = client.get("/api/status")
    status.raise_for_status()
    assert status.json()["cloud"] and all(model["available"] for model in status.json()["models"])
    assert not status.json()["samples"]  # No dataset images are shipped.
    response = client.get("/api/results")
    response.raise_for_status()
    assert len(response.json()["clean"]) == 3
    assert client.post("/api/predict", content=b"corrupt").status_code == 400
    for model, run in (("lbp", "lbp_seed42"), ("lbp_glcm", "lbp_glcm_seed42"), ("patchcore", "patchcore_seed42")):
        response = client.post(f"/api/predict?model={model}",
            content=Path("data/raw/carpet/test/cut/000.png").read_bytes(), headers={"Content-Type": "image/png"})
        response.raise_for_status()
        result = response.json()
        with Path(f"runs/{run}/evaluation/clean/predictions.csv").open() as stream:
            saved = next(row for row in csv.DictReader(stream) if row["path"] == "test/cut/000.png")
        np.testing.assert_allclose(result["score"], float(saved["raw_score"]), rtol=1e-5, atol=1e-5)
        assert result["threshold"] == float(saved["threshold"])
        assert (result["decision"] == "anomaly") == bool(int(saved["decision"]))
        if model == "patchcore":
            with Image.open(BytesIO(base64.b64decode(result["heatmap"].split(",", 1)[1]))) as image:
                assert image.size == (256, 256)
        else:
            assert result["heatmap"] is None
        print(f"PASS {model}: score={result['score']:.6f}, device={result['device']}, first scoring={result['inference_ms']:.1f}ms")
print("PASS: cloud status, report data, upload validation, frozen scores/thresholds and heatmap")
