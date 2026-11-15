
from pathlib import Path
import random

import numpy as np
import rasterio
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from src.models.hyperspectral_vit import HyperspectralViT
from src.models.hyperspectral_dataset_v2 import (
    StandardizedHyperspectralDataset,
)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")
    project_root = Path(__file__).resolve().parents[2]

    image_path = project_root / "data/sample/mock_hyperspectral.tif"
    label_path = project_root / "data/sample/mock_labels.tif"

    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)

    with rasterio.open(label_path) as src:
        labels = src.read(1).astype(np.int64)

    bands, height, width = image.shape

    if labels.shape != (height, width):
        raise ValueError("Image and label dimensions do not match.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains invalid values.")

    if not np.isin(np.unique(labels), [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    rows, columns = np.indices((height, width))

    # Match the existing spatial Logistic Regression experiment.
    train_mask = (rows < 28) | (columns < 28)
    test_mask = (rows >= 32) & (columns >= 32)

    train_coordinates = np.argwhere(
        train_mask & np.isin(labels, [0, 1])
    )

    test_coordinates = np.argwhere(
        test_mask & np.isin(labels, [0, 1])
    )

    train_indices = np.ravel_multi_index(
        (train_coordinates[:, 0], train_coordinates[:, 1]),
        (height, width),
    )

    test_indices = np.ravel_multi_index(
        (test_coordinates[:, 0], test_coordinates[:, 1]),
        (height, width),
    )

    # Training-only per-band statistics.
    train_spectra = image[
        :,
        train_coordinates[:, 0],
        train_coordinates[:, 1],
    ]

    band_mean = train_spectra.mean(axis=1)
    band_std = np.maximum(train_spectra.std(axis=1), 1e-6)

    print("Spatial ViT V3 Training")
    print("=" * 45)
    print("Training samples:", len(train_indices))
    print("Testing samples:", len(test_indices))
    print(
        "Excluded buffer/unlabelled samples:",
        height * width - len(train_indices) - len(test_indices),
    )

    print("\nTraining class distribution:")
    print(
        "Healthy:",
        int(np.sum(labels[train_coordinates[:, 0],
                          train_coordinates[:, 1]] == 0)),
    )
    print(
        "Stressed:",
        int(np.sum(labels[train_coordinates[:, 0],
                          train_coordinates[:, 1]] == 1)),
    )

    dataset = StandardizedHyperspectralDataset(
        image_path=str(image_path),
        label_path=str(label_path),
        patch_size=5,
        band_mean=band_mean,
        band_std=band_std,
    )

    train_dataset = Subset(dataset, train_indices.tolist())
    test_dataset = Subset(dataset, test_indices.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    train_labels = labels[
        train_coordinates[:, 0],
        train_coordinates[:, 1],
    ]

    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = len(train_labels) / (2.0 * class_counts)

    weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    model = HyperspectralViT(
        num_bands=bands,
        patch_size=5,
        embed_dim=64,
        num_heads=4,
        depth=2,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Training Loss: {running_loss / len(train_loader):.4f}"
        )

    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "hyperspectral_vit_spatial.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["healthy", "stressed"],
            "input_bands": bands,
            "patch_size": 5,
            "embed_dim": 64,
            "num_heads": 4,
            "depth": 2,
            "num_classes": 2,
            "epochs": epochs,
            "training_samples": len(train_dataset),
            "testing_samples": len(test_dataset),
            "training_data": "synthetic mock data",
            "evaluation_type": "spatial holdout",
            "random_seed": 42,
            "band_mean": torch.tensor(band_mean),
            "band_std": torch.tensor(band_std),
        },
        model_path,
    )

    print("\nSpatial ViT checkpoint saved:", model_path)
    print("Training completed.")
    print("The spatial test region was not used for training.")


if __name__ == "__main__":
    main()
