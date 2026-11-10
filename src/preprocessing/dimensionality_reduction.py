"""Dimensionality reduction utilities for hyperspectral images."""

import numpy as np
from sklearn.decomposition import PCA


def apply_pca(
    image: np.ndarray,
    n_components: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce hyperspectral image spectral dimensions using PCA.

    Input shape:
        (bands, height, width)

    Output shape:
        (components, height, width)
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 3:
        raise ValueError("Expected image shape (bands, height, width).")

    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values.")

    bands, height, width = image.shape
    pixels = height * width

    if n_components < 1:
        raise ValueError("n_components must be at least 1.")

    if n_components > min(bands, pixels):
        raise ValueError(
            f"n_components cannot exceed {min(bands, pixels)} "
            "for this image."
        )

    pixel_data = image.reshape(bands, pixels).T

    pca = PCA(
        n_components=n_components,
        random_state=random_state,
    )

    transformed = pca.fit_transform(pixel_data)

    return transformed.T.reshape(n_components, height, width).astype(
        np.float32
    )