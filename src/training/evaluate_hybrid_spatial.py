
from pathlib import Path

import numpy as np
import rasterio
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from torch.utils.data import DataLoader, Subset

from src.models.hyperspectral_dataset_v2 import (
    StandardizedHyperspectralDataset,
)
from src.models.hyperspectral_hybrid import HyperspectralHybridViT


def main():
    device = torch.device("cpu")

    project_root = Path(__file__).resolve().parents[2]

    image_path = project_root / "data/sample/mock_hyperspectral.tif"
    label_path = project_root / "data/sample/mock_labels.tif"

    checkpoint_path = (
        project_root / "models/hyperspectral_hybrid_spatial.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    with rasterio.open(image_path) as src:
        height, width = src.shape
        num_bands = src.count

    with rasterio.open(label_path) as src:
        labels = src.read(1).astype(np.int64)

    rows, columns = np.indices((height, width))

    train_mask = (
        ((rows < 28) | (columns < 28))
        & np.isin(labels, [0, 1])
    )

    test_mask = (
        (rows >= 32)
        & (columns >= 32)
        & np.isin(labels, [0, 1])
    )

    train_coordinates = np.argwhere(train_mask)
    test_coordinates = np.argwhere(test_mask)

    test_indices = np.ravel_multi_index(
        (test_coordinates[:, 0], test_coordinates[:, 1]),
        (height, width),
    )

    band_mean = checkpoint["band_mean"].cpu().numpy()
    band_std = checkpoint["band_std"].cpu().numpy()

    dataset = StandardizedHyperspectralDataset(
        image_path=str(image_path),
        label_path=str(label_path),
        patch_size=checkpoint["patch_size"],
        band_mean=band_mean,
        band_std=band_std,
    )

    test_dataset = Subset(
        dataset,
        test_indices.tolist(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
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

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)

            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_targets.extend(
                targets.numpy()
            )

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_predictions)

    print("\nHybrid 3D-CNN + ViT Spatial Test Evaluation")
    print("=" * 50)

    print("Training samples:", len(train_coordinates))
    print("Testing samples:", len(test_coordinates))

    print("\nEvaluation Metrics")
    print("-" * 40)

    print(
        f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}"
    )
    print(
        f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print("\nConfusion Matrix")
    print("-" * 40)
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nPredicted Class Distribution")
    print("-" * 40)
    print(
        "Healthy:",
        int(np.sum(y_pred == 0)),
    )
    print(
        "Stressed:",
        int(np.sum(y_pred == 1)),
    )

    print("\nEvaluation Notes")
    print("-" * 40)
    print("Data type: SYNTHETIC_MOCK_DATA")
    print("Evaluation: Spatial holdout")
    print("Test region was not used during training.")
    print("Results are not independent real-world validation.")


if __name__ == "__main__":
    main()
