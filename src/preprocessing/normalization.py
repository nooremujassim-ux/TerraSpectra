"""Normalization utilities for hyperspectral image data."""

import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize hyperspectral image values to the range [0, 1].

    Expected input shape:
        (bands, height, width)
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3:
        raise ValueError(
            "Expected a 3D image with shape (bands, height, width)."
        )

    if image.size == 0:
        raise ValueError("Image cannot be empty.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values.")

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum == minimum:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - minimum) / (maximum - minimum)

    return normalized.astype(np.float32)