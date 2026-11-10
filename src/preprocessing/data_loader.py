"""Load and validate hyperspectral raster datasets."""

from pathlib import Path

import numpy as np
import rasterio


def load_hyperspectral_image(file_path: str | Path) -> np.ndarray:
    """
    Load a hyperspectral image from a GeoTIFF file.

    Returns:
        NumPy array with shape (bands, height, width).
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    if file_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Only GeoTIFF (.tif or .tiff) files are supported.")

    with rasterio.open(file_path) as dataset:
        image = dataset.read()

    if image.ndim != 3 or image.shape[0] < 1:
        raise ValueError("Invalid hyperspectral image dimensions.")

    return image


def get_image_metadata(file_path: str | Path) -> dict:
    """Return basic metadata about a GeoTIFF image."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    with rasterio.open(file_path) as dataset:
        return {
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": str(dataset.crs) if dataset.crs else None,
            "transform": tuple(dataset.transform),
        }