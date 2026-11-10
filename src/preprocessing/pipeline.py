"""End-to-end hyperspectral image preprocessing pipeline."""

from pathlib import Path

import numpy as np

from src.preprocessing.data_loader import load_hyperspectral_image
from src.preprocessing.normalization import normalize_image
from src.preprocessing.dimensionality_reduction import apply_pca


def preprocess_image(
    file_path: str | Path,
    apply_dimensionality_reduction: bool = True,
    n_components: int = 30,
) -> np.ndarray:
    """
    Load, normalize, and optionally reduce a hyperspectral image.

    Returns:
        Preprocessed image with shape:
        (bands or components, height, width).
    """
    image = load_hyperspectral_image(file_path)

    image = normalize_image(image)

    if apply_dimensionality_reduction:
        image = apply_pca(
            image,
            n_components=n_components,
        )

    return image