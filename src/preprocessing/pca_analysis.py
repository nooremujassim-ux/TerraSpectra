
"""Analyze and visualize PCA explained variance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.preprocessing.data_loader import load_hyperspectral_image
from src.preprocessing.normalization import normalize_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = PROJECT_ROOT / "data" / "sample" / "mock_hyperspectral.tif"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def generate_pca_analysis() -> Path:
    """Calculate and plot PCA explained variance."""

    image = load_hyperspectral_image(IMAGE_PATH)
    image = normalize_image(image)

    bands, height, width = image.shape

    pixel_data = image.reshape(bands, height * width).T

    pca = PCA(n_components=30, random_state=42)
    pca.fit(pixel_data)

    individual_variance = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(individual_variance)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "pca_explained_variance.png"

    components = np.arange(1, 31)

    plt.figure(figsize=(11, 6))

    plt.bar(
        components,
        individual_variance,
        alpha=0.6,
        label="Individual explained variance",
    )

    plt.plot(
        components,
        cumulative_variance,
        marker="o",
        label="Cumulative explained variance",
    )

    plt.xlabel("Principal component")
    plt.ylabel("Explained variance (%)")
    plt.title("PCA Explained Variance - Synthetic Hyperspectral Data")
    plt.xticks(np.arange(1, 31, 2))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print("PCA analysis completed successfully.")
    print(f"Original bands: {bands}")
    print(f"Reduced components: {pca.n_components_}")
    print(
        f"Cumulative explained variance: "
        f"{cumulative_variance[-1]:.2f}%"
    )
    print(f"Plot location: {output_path}")

    return output_path


if __name__ == "__main__":
    generate_pca_analysis()
