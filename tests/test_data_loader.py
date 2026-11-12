
"""Tests for hyperspectral GeoTIFF data loading."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from src.preprocessing.data_loader import (
    get_image_metadata,
    load_hyperspectral_image,
)


def create_test_geotiff(file_path):
    """Create a small GeoTIFF for testing."""
    image = np.ones((4, 5, 6), dtype=np.float32)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=5,
        width=6,
        count=4,
        dtype="float32",
        transform=Affine.identity(),
    ) as dataset:
        dataset.write(image)


def test_load_hyperspectral_image(tmp_path):
    file_path = tmp_path / "test_image.tif"
    create_test_geotiff(file_path)

    result = load_hyperspectral_image(file_path)

    assert result.shape == (4, 5, 6)
    assert result.dtype == np.float32


def test_get_image_metadata(tmp_path):
    file_path = tmp_path / "test_image.tif"
    create_test_geotiff(file_path)

    metadata = get_image_metadata(file_path)

    assert metadata["width"] == 6
    assert metadata["height"] == 5
    assert metadata["bands"] == 4
    assert metadata["dtype"] == "float32"


def test_loader_rejects_missing_file(tmp_path):
    file_path = tmp_path / "missing.tif"

    with pytest.raises(FileNotFoundError):
        load_hyperspectral_image(file_path)


def test_loader_rejects_unsupported_format(tmp_path):
    file_path = tmp_path / "image.txt"
    file_path.write_text("not a raster image")

    with pytest.raises(ValueError):
        load_hyperspectral_image(file_path)
