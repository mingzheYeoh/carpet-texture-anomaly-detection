"""PatchCore preprocessing follows the shared full-frame RGB transform."""

import numpy as np
from PIL import Image
import torch


def test_patchcore_normalizes_once_without_cropping():
    from carpet_ad.patchcore import to_tensor

    image = Image.new("RGB", (400, 200), "white")
    image.paste((0, 0, 0), (0, 0, 200, 200))
    tensor = to_tensor(image)
    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype == torch.float32
    np.testing.assert_allclose(tensor[:, 0, 0], -np.array([.485, .456, .406]) / [.229, .224, .225], rtol=1e-6)
    np.testing.assert_allclose(tensor[:, -1, -1], (1 - np.array([.485, .456, .406])) / [.229, .224, .225], rtol=1e-6)
