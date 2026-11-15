
from pathlib import Path

import numpy as np
import rasterio
import torch

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.models.hyperspectral_vit import HyperspectralViT
from src.models.hyperspectral_dataset_v2 import (
    StandardizedHyperspectralDataset,
)


def main():
    project_root = Path(__file__).resolve().parents[2]
    device = torch.device("cpu")

    checkpoint_path = (
        project_root / "models" / "hyperspectral_vit_spatial.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    with rasterio.open(
        project_root / "data/sample/mock_hyperspectral.tif"
    ) as src:
        image_shape = src.shape

    with rasterio.open(
        project_root / "data/sample/mock_labels.tif"
    ) as src:
        labels = src.read(1).astype(np.int64)

    height, width = image_shape

    rows, columns = np.indices((height, width))

    test_mask = (
        (rows >= 32)
        & (columns >= 32)
        & np.isin(labels, [0, 1])
    )

    test_coordinates = np.argwhere(test_mask)

    test_indices = np.ravel_multi_index(
        (test_coordinates[:, 0], test_coordinates[:, 1]),
        (height, width),
    )

    band_mean = checkpoint["band_mean"].cpu().numpy()
    band_std = checkpoint["band_std"].cpu().numpy()

    dataset = StandardizedHyperspectralDataset(
        image_path=str(
            project_root / "data/sample/mock_hyperspectral.tif"
        ),
        label_path=str(
            project_root / "data/sample/mock_labels.tif"
        ),
        patch_size=checkpoint["patch_size"],
        band_mean=band_mean,
        band_std=band_std,
    )

    test_dataset = Subset(dataset, test_indices.tolist())

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = HyperspectralViT(
        num_bands=checkpoint["input_bands"],
        patch_size=checkpoint["patch_size"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        depth=checkpoint["depth"],
        num_classes=checkpoint["num_classes"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():
        for features, targets in test_loader:
            outputs = model(features.to(device))
            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.cpu().numpy())
            actual.extend(targets.numpy())

    predictions = np.asarray(predictions)
    actual = np.asarray(actual)

    print("\nSpatial ViT Test Evaluation")
    print("=" * 45)
    print("Training samples:", checkpoint["training_samples"])
    print("Testing samples:", len(test_dataset))

    print("\nEvaluation Metrics")
    print("-" * 45)
    print(f"Accuracy:  {accuracy_score(actual, predictions):.4f}")
    print(
        f"Precision: {precision_score(actual, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(actual, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1-score:  {f1_score(actual, predictions, zero_division=0):.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predictions, labels=[0, 1]))

    print("\nClass order: 0 = Healthy, 1 = Stressed")
    print(
        "Predicted class counts:",
        np.bincount(predictions, minlength=2).tolist(),
    )

    print("\nLimitations:")
    print("- Evaluation uses synthetic mock data.")
    print("- Training and test centers are spatially separated.")
    print("- 5x5 patches near the training boundary may overlap the test region.")
    print("- This is not independent real-world validation.")


if __name__ == "__main__":
    main()
