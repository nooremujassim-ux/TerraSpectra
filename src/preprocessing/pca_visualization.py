
"""Visualize the first three PCA components spatially."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.pipeline import preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = PROJECT_ROOT / "data" / "sample" / "mock_hyperspectral.tif"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def generate_pca_visualization() -> Path:
    """Generate a spatial visualization of the first three PCA components."""

    components = preprocess_image(
        IMAGE_PATH,
        apply_dimensionality_reduction=True,
        n_components=30,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "pca_component_visualization.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for index, axis in enumerate(axes):
        component = components[index]

        limit = np.percentile(np.abs(component), 98)

        if limit == 0:
            limit = 1.0

        image = axis.imshow(
            component,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )

        axis.set_title(f"Principal Component {index + 1}")
        axis.set_xlabel("Pixel column")
        axis.set_ylabel("Pixel row")

        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Spatial Visualization of PCA Components",
        fontsize=14,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("PCA component visualization generated successfully.")
    print("Components visualized: 3")
    print(f"Output shape: {components.shape}")
    print(f"Plot location: {output_path}")

    return output_path


if __name__ == "__main__":
    generate_pca_visualization()
