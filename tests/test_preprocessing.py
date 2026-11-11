
import numpy as np
import pytest

from src.preprocessing.normalization import normalize_image
from src.preprocessing.dimensionality_reduction import apply_pca


def test_normalization_output_range():
    image = np.array(
        [[[10, 20], [30, 40]]],
        dtype=np.float32,
    )

    result = normalize_image(image)

    assert result.min() == 0.0
    assert result.max() == 1.0


def test_normalization_constant_image():
    image = np.full((3, 4, 4), 5.0, dtype=np.float32)

    result = normalize_image(image)

    assert np.all(result == 0.0)


def test_normalization_rejects_invalid_dimensions():
    image = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        normalize_image(image)


def test_normalization_rejects_nan_values():
    image = np.array([[[1.0, np.nan]]])

    with pytest.raises(ValueError):
        normalize_image(image)


def test_pca_output_shape():
    rng = np.random.default_rng(42)
    image = rng.random((40, 10, 10), dtype=np.float32)

    result = apply_pca(image, n_components=10)

    assert result.shape == (10, 10, 10)
    assert np.isfinite(result).all()


def test_pca_rejects_excessive_components():
    image = np.random.default_rng(42).random(
        (5, 4, 4),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        apply_pca(image, n_components=10)
