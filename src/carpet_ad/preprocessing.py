"""Shared full-image RGB preprocessing; no crop or contrast adjustment."""

from PIL import Image, ImageOps


def load_rgb(path):
    with Image.open(path) as image:
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def resize_rgb(image):
    return ImageOps.exif_transpose(image).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
