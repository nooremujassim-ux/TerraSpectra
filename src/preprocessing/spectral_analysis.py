
"""Analyze and visualize synthetic hyperspectral spectral signatures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = PROJECT_ROOT / "data" / "sample" / "mock_hyperspectral.tif"
LABEL_PATH = PROJECT_ROOT / "data" / "sample" / "mock_labels.tif"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def generate_spectral_plot() -> Path:
    """Plot mean spectral signatures for synthetic mock classes."""

    with rasterio.open(IMAGE_PATH) as dataset:
        image = dataset.read()

    with rasterio.open(LABEL_PATH) as dataset:
        labels = dataset.read(1)

    healthy_pixels = image[:, labels == 0]
    stressed_pixels = image[:, labels == 1]

    healthy_mean = healthy_pixels.mean(axis=1)
    stressed_mean = stressed_pixels.mean(axis=1)

    wavelengths = np.linspace(400, 1000, image.shape[0])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "synthetic_spectral_signatures.png"

    plt.figure(figsize=(11, 6))

    plt.plot(
        wavelengths,
        healthy_mean,
        label="Healthy mock pixels",
    )

    plt.plot(
        wavelengths,
        stressed_mean,
        label="Stressed mock pixels",
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean synthetic reflectance")
    plt.title("Synthetic Hyperspectral Spectral Signatures")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print("Spectral signature plot generated successfully.")
    print(f"Plot location: {output_path}")

    return output_path


if __name__ == "__main__":
    generate_spectral_plot()
