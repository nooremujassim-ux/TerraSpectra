
"""Generate clearly labelled synthetic hyperspectral data for pipeline testing."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "sample"

HEIGHT = 64
WIDTH = 64
BANDS = 200
RANDOM_SEED = 42


def generate_sample_data() -> None:
    """Create a synthetic hyperspectral cube and its class-label raster."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    wavelengths = np.linspace(400, 1000, BANDS, dtype=np.float32)

    # Synthetic healthy spectral signature.
    healthy_signature = (
        0.25
        + 0.35 * np.exp(-((wavelengths - 750) / 100) ** 2)
        - 0.10 * np.exp(-((wavelengths - 680) / 25) ** 2)
    )

    # Synthetic chemically stressed signature.
    stressed_signature = (
        healthy_signature
        - 0.08 * np.exp(-((wavelengths - 750) / 120) ** 2)
        + 0.08 * np.exp(-((wavelengths - 680) / 35) ** 2)
    )

    cube = np.zeros((BANDS, HEIGHT, WIDTH), dtype=np.float32)

    labels = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    # Define a synthetic stressed patch.
    labels[20:44, 20:44] = 1

    for band in range(BANDS):
        noise = rng.normal(
            loc=0.0,
            scale=0.005,
            size=(HEIGHT, WIDTH),
        )

        cube[band] = healthy_signature[band] + noise

        cube[band, labels == 1] = (
            stressed_signature[band]
            + noise[labels == 1]
        )

    cube = np.clip(cube, 0.0, 1.0)

    # Pixel-grid transform only; these are not real geographic coordinates.
    transform = Affine(1, 0, 0, 0, -1, HEIGHT)

    image_path = OUTPUT_DIR / "mock_hyperspectral.tif"
    labels_path = OUTPUT_DIR / "mock_labels.tif"

    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=BANDS,
        dtype="float32",
        transform=transform,
        compress="deflate",
    ) as dataset:
        dataset.write(cube)

        for band in range(BANDS):
            dataset.set_band_description(
                band + 1,
                f"Band_{band + 1:03d}",
            )

        dataset.update_tags(
            data_type="SYNTHETIC_MOCK_DATA",
            description="Artificial hyperspectral data for pipeline testing",
            spectral_bands=BANDS,
            random_seed=RANDOM_SEED,
            geographic_reference="None - pixel grid only",
        )

    with rasterio.open(
        labels_path,
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=1,
        dtype="uint8",
        transform=transform,
        compress="deflate",
    ) as dataset:
        dataset.write(labels, 1)
        dataset.update_tags(
            data_type="SYNTHETIC_MOCK_LABELS",
            class_0="healthy_mock",
            class_1="chemically_stressed_mock",
        )

    print("Synthetic sample dataset created successfully.")
    print(f"Hyperspectral image: {image_path}")
    print(f"Label raster: {labels_path}")
    print(f"Image shape: {cube.shape}")
    print(f"Spectral bands: {BANDS}")
    print(f"Healthy pixels: {np.count_nonzero(labels == 0)}")
    print(f"Stressed pixels: {np.count_nonzero(labels == 1)}")


if __name__ == "__main__":
    generate_sample_data()
