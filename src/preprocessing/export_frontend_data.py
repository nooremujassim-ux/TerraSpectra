
"""Export synthetic hyperspectral pixels for the TerraSpectra React dashboard."""

import json
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = PROJECT_ROOT / "data" / "sample" / "mock_hyperspectral.tif"
LABEL_PATH = PROJECT_ROOT / "data" / "sample" / "mock_labels.tif"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "public" / "crop_sample.json"


def export_frontend_data():
    """Convert the sample hyperspectral image and labels into JSON."""

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Sample image not found: {IMAGE_PATH}")

    if not LABEL_PATH.exists():
        raise FileNotFoundError(f"Sample labels not found: {LABEL_PATH}")

    with rasterio.open(IMAGE_PATH) as image_dataset:
        image = image_dataset.read().astype(np.float32)
        metadata = {
            "width": image_dataset.width,
            "height": image_dataset.height,
            "bands": image_dataset.count,
            "dtype": str(image_dataset.dtypes[0]),
            "crs": str(image_dataset.crs),
        }

    with rasterio.open(LABEL_PATH) as label_dataset:
        labels = label_dataset.read(1)

    if image.shape[1:] != labels.shape:
        raise ValueError("Image and label dimensions do not match.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values.")

    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must contain only classes 0 and 1.")

    mean_spectral_values = image.mean(axis=0)

    pixels = []

    for y in range(metadata["height"]):
        for x in range(metadata["width"]):
            pixels.append({
                "position": [x, y],
                "stressed": bool(labels[y, x] == 1),
                "value": round(float(mean_spectral_values[y, x]), 6),
            })

    result = {
        "dataset_type": "SYNTHETIC_MOCK_DATA",
        "description": (
            "Artificially generated hyperspectral sample. "
            "Not real crop disease observations."
        ),
        "metadata": metadata,
        "total_pixels": len(pixels),
        "healthy_pixels": int(np.count_nonzero(labels == 0)),
        "stressed_pixels": int(np.count_nonzero(labels == 1)),
        "pixels": pixels,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("Frontend data exported successfully.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total pixels: {len(pixels)}")
    print(f"Healthy pixels: {result['healthy_pixels']}")
    print(f"Stressed pixels: {result['stressed_pixels']}")


if __name__ == "__main__":
    export_frontend_data()
