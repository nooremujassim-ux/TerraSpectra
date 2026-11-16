
from pathlib import Path
import csv

import numpy as np
import rasterio
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.hyperspectral_dataset_v2 import (
    StandardizedHyperspectralDataset,
)
from src.models.hyperspectral_hybrid import HyperspectralHybridViT
from src.evaluation.hybrid_explainability import HybridGradCAM


def main():
    root = Path(__file__).resolve().parents[2]
    device = torch.device("cpu")

    checkpoint_path = (
        root / "models/hyperspectral_hybrid_spatial.pth"
    )

    output_dir = root / "reports/hybrid_explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    image_path = root / "data/sample/mock_hyperspectral.tif"
    label_path = root / "data/sample/mock_labels.tif"

    with rasterio.open(image_path) as src:
        height, width = src.shape

    with rasterio.open(label_path) as src:
        labels = src.read(1).astype(np.int64)

    rows, cols = np.indices((height, width))

    test_mask = (
        (rows >= 32)
        & (cols >= 32)
        & np.isin(labels, [0, 1])
    )

    test_coordinates = np.argwhere(test_mask)

    if len(test_coordinates) == 0:
        raise ValueError("No valid test samples found.")

    test_indices = np.ravel_multi_index(
        (
            test_coordinates[:, 0],
            test_coordinates[:, 1],
        ),
        (height, width),
    )

    dataset = StandardizedHyperspectralDataset(
        image_path=str(image_path),
        label_path=str(label_path),
        patch_size=checkpoint["patch_size"],
        band_mean=checkpoint["band_mean"].cpu().numpy(),
        band_std=checkpoint["band_std"].cpu().numpy(),
    )

    model = HyperspectralHybridViT(
        num_bands=checkpoint["num_bands"],
        patch_size=checkpoint["patch_size"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        depth=checkpoint["depth"],
        num_classes=checkpoint["num_classes"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Select the first spatial test sample.
    sample_index = int(test_indices[0])

    features, target = dataset[sample_index]
    features = features.unsqueeze(0).to(device)

    explainer = HybridGradCAM(model)

    try:
        heatmap = explainer.generate(features)

        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = outputs.argmax(dim=1).item()

        cam = heatmap[0].cpu().numpy()

        # Average activation across spatial positions.
        spectral_importance = cam.mean(axis=(1, 2))

        # Average activation across spectral bands.
        spatial_importance = cam.mean(axis=0)

        actual_class = int(target)

        row, col = test_coordinates[0]

        # Save complete numerical heatmap.
        np.save(
            output_dir / "hybrid_gradcam_3d.npy",
            cam,
        )

        # Save spectral importance CSV.
        with open(
            output_dir / "spectral_importance.csv",
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["Spectral Band", "Importance"])

            for band, importance in enumerate(
                spectral_importance, start=1
            ):
                writer.writerow([band, float(importance)])

        # Spectral importance plot.
        plt.figure(figsize=(12, 5))

        plt.plot(
            np.arange(1, len(spectral_importance) + 1),
            spectral_importance,
        )

        plt.xlabel("Spectral Band")
        plt.ylabel("Mean Grad-CAM Activation")
        plt.title(
            "TerraSpectra - Hybrid Model Spectral Importance"
        )
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_dir / "spectral_importance.png",
            dpi=300,
        )
        plt.close()

        # Spatial Grad-CAM heatmap.
        plt.figure(figsize=(7, 6))

        plt.imshow(
            spatial_importance,
            cmap="inferno",
            interpolation="nearest",
        )

        plt.colorbar(label="Mean Grad-CAM Activation")
        plt.title(
            f"Spatial Grad-CAM | Actual: {actual_class} "
            f"| Predicted: {prediction}"
        )
        plt.xlabel("Patch Column")
        plt.ylabel("Patch Row")
        plt.tight_layout()

        plt.savefig(
            output_dir / "spatial_gradcam.png",
            dpi=300,
        )
        plt.close()

        # Save sample information.
        with open(
            output_dir / "sample_metadata.txt",
            "w",
        ) as file:
            file.write("TerraSpectra Hybrid Explainability\n")
            file.write("=" * 45 + "\n")
            file.write(f"Sample index: {sample_index}\n")
            file.write(f"Pixel location: ({row}, {col})\n")
            file.write(f"Actual class: {actual_class}\n")
            file.write(f"Predicted class: {prediction}\n")
            file.write(
                f"Prediction confidence: "
                f"{probabilities[0, prediction].item():.4f}\n"
            )
            file.write(f"Heatmap shape: {cam.shape}\n")
            file.write("Data: SYNTHETIC_MOCK_DATA\n")
            file.write(
                "Interpretation: Gradient-based model "
                "activation, not causal feature importance.\n"
            )

        print("\nTerraSpectra Hybrid Explainability")
        print("=" * 45)
        print("Actual class:", actual_class)
        print("Predicted class:", prediction)
        print("Heatmap shape:", cam.shape)
        print("\nGenerated files:")

        for path in sorted(output_dir.iterdir()):
            print(path.name)

        print("\nExplainability generation completed.")

    finally:
        explainer.close()


if __name__ == "__main__":
    main()
