"""Fixed PLAN.md features: row-major cells/bins, then property/distance/angle."""

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from carpet_ad.preprocessing import resize_rgb


def extract_features(image, model):
    if model not in {"lbp_ocsvm", "lbp_glcm_ocsvm"}:
        raise ValueError(f"Unsupported baseline: {model}")
    gray = np.asarray(resize_rgb(image).convert("L"), dtype=np.uint8)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    histograms = []
    for y in range(0, 256, 64):
        for x in range(0, 256, 64):
            counts = np.bincount(lbp[y:y + 64, x:x + 64].astype(np.uint8).ravel(), minlength=10)
            histograms.append(counts / counts.sum())
    features = np.concatenate(histograms)
    if model == "lbp_glcm_ocsvm":
        quantized = (gray.astype(np.uint16) * 32 // 256).astype(np.uint8)
        glcm = graycomatrix(quantized, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                           levels=32, symmetric=True, normed=True)
        statistics = [graycoprops(glcm, name).ravel() for name in ("contrast", "homogeneity", "energy", "correlation")]
        features = np.concatenate([features, *statistics])
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite texture features")
    return features
