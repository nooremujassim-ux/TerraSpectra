
from pathlib import Path
import random

import numpy as np
import rasterio
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from src.models.hyperspectral_dataset_v2 import (
    StandardizedHyperspectralDataset,
)
from src.models.hyperspectral_hybrid import HyperspectralHybridViT


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")

    project_root = Path(__file__).resolve().parents[2]

    image_path = project_root / "data/sample/mock_hyperspectral.tif"
    label_path = project_root / "data/sample/mock_labels.tif"

    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / "hyperspectral_hybrid_spatial.pth"

    # Load labels and image dimensions.
    with rasterio.open(image_path) as src:
        height, width = src.shape
        num_bands = src.count

    with rasterio.open(label_path) as src:
        labels = src.read(1).astype(np.int64)

    rows, columns = np.indices((height, width))

    # Spatial training and testing regions.
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

    train_indices = np.ravel_multi_index(
        (train_coordinates[:, 0], train_coordinates[:, 1]),
        (height, width),
    )

    # Calculate normalization statistics using training pixels only.
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)

    train_spectra = image[
        :,
        train_coordinates[:, 0],
        train_coordinates[:, 1],
    ].T

    band_mean = train_spectra.mean(axis=0)
    band_std = train_spectra.std(axis=0)
    band_std = np.maximum(band_std, 1e-6)

    # Dataset standardizes each band using training statistics.
    dataset = StandardizedHyperspectralDataset(
        image_path=str(image_path),
        label_path=str(label_path),
        patch_size=5,
        band_mean=band_mean,
        band_std=band_std,
    )

    train_dataset = Subset(
        dataset,
        train_indices.tolist(),
    )

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

    if np.any(class_counts == 0):
        raise ValueError("Both classes must be present in training data.")

    class_weights = len(train_labels) / (
        2.0 * class_counts
    )

    weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    model = HyperspectralHybridViT(
        num_bands=num_bands,
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

    print("\nHybrid 3D-CNN + ViT Spatial Training")
    print("=" * 50)
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_coordinates))
    print(
        "Excluded buffer/unlabelled samples:",
        height * width - len(train_dataset) - len(test_coordinates),
    )
    print("Training class distribution:", class_counts.tolist())
    print("Device:", device)
    print(
        "Model parameters:",
        sum(parameter.numel() for parameter in model.parameters()),
    )

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

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(train_dataset)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Training Loss: {epoch_loss:.4f}"
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_bands": num_bands,
        "patch_size": 5,
        "embed_dim": 64,
        "num_heads": 4,
        "depth": 2,
        "num_classes": 2,
        "training_samples": len(train_dataset),
        "testing_samples": len(test_coordinates),
        "epochs": epochs,
        "seed": 42,
        "band_mean": torch.tensor(band_mean, dtype=torch.float32),
        "band_std": torch.tensor(band_std, dtype=torch.float32),
        "class_names": ["Healthy", "Stressed"],
        "data_type": "SYNTHETIC_MOCK_DATA",
        "evaluation_type": "spatial_holdout",
    }

    torch.save(checkpoint, checkpoint_path)

    print("\nHybrid checkpoint saved:", checkpoint_path)
    print("Training completed.")
    print("The spatial test region was not used for training.")


if __name__ == "__main__":
    main()
