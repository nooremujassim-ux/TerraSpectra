
"""Tests for the synthetic hyperspectral sample dataset."""

import numpy as np
import rasterio

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


def test_sample_image_has_200_bands():
    image_path = SAMPLE_DIR / "mock_hyperspectral.tif"

    with rasterio.open(image_path) as dataset:
        assert dataset.count == 200
        assert dataset.width == 64
        assert dataset.height == 64


def test_sample_labels_have_matching_dimensions():
    image_path = SAMPLE_DIR / "mock_hyperspectral.tif"
    labels_path = SAMPLE_DIR / "mock_labels.tif"

    with rasterio.open(image_path) as image:
        with rasterio.open(labels_path) as labels:
            assert image.width == labels.width
            assert image.height == labels.height


def test_sample_labels_have_expected_classes():
    labels_path = SAMPLE_DIR / "mock_labels.tif"

    with rasterio.open(labels_path) as dataset:
        labels = dataset.read(1)

    assert set(np.unique(labels)) == {0, 1}


def test_sample_label_pixel_counts():
    labels_path = SAMPLE_DIR / "mock_labels.tif"

    with rasterio.open(labels_path) as dataset:
        labels = dataset.read(1)

    assert np.count_nonzero(labels == 0) == 3520
    assert np.count_nonzero(labels == 1) == 576
