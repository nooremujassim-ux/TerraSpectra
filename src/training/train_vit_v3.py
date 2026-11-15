
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.models.hyperspectral_vit import HyperspectralViT
from src.models.hyperspectral_dataset_v2 import (
    StandardizedHyperspectralDataset,
)


def main():
    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")
    print("Training device:", device)

    # Load image and labels to establish the training split.
    with rasterio.open("data/sample/mock_hyperspectral.tif") as src:
        image = src.read().astype(np.float32)

    with rasterio.open("data/sample/mock_labels.tif") as src:
        raw_labels = src.read(1)

    coordinates = np.argwhere(np.isin(raw_labels, [0, 1]))
    labels = raw_labels[coordinates[:, 0], coordinates[:, 1]].astype(
        np.int64
    )

    indices = np.arange(len(coordinates))

    # Stratified 60/20/20 split.
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.4,
        random_state=42,
        stratify=labels,
    )

    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.5,
        random_state=42,
        stratify=labels[temporary_indices],
    )

    # Calculate statistics using training pixel spectra only.
    train_coordinates = coordinates[train_indices]

    train_spectra = image[
        :,
        train_coordinates[:, 0],
        train_coordinates[:, 1],
    ]

    band_mean = train_spectra.mean(axis=1)
    band_std = train_spectra.std(axis=1)

    # Avoid division by zero for constant bands.
    band_std = np.maximum(band_std, 1e-6)

    print("\nTraining-only spectral statistics")
    print("-" * 40)
    print("Mean shape:", band_mean.shape)
    print("Standard deviation range:",
          float(band_std.min()), float(band_std.max()))

    # Build standardized dataset using training statistics.
    dataset = StandardizedHyperspectralDataset(
        image_path="data/sample/mock_hyperspectral.tif",
        label_path="data/sample/mock_labels.tif",
        patch_size=5,
        band_mean=band_mean,
        band_std=band_std,
    )

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    print("\nDataset Split")
    print("-" * 40)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Testing samples:", len(test_dataset))

    # Class weights from training labels only.
    train_labels = labels[train_indices]
    class_counts = np.bincount(train_labels, minlength=2)

    class_weights = len(train_labels) / (2.0 * class_counts)

    weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    print("Training class distribution:", class_counts.tolist())
    print("Class weights:", weights.tolist())

    # Initialize the original ViT architecture.
    model = HyperspectralViT(
        num_bands=200,
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

    # Train using only the training partition.
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

        average_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Training Loss: {average_loss:.4f}"
        )

    # Save a separate checkpoint.
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "hyperspectral_vit_v3.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["healthy", "stressed"],
            "input_bands": 200,
            "patch_size": 5,
            "embed_dim": 64,
            "num_heads": 4,
            "depth": 2,
            "num_classes": 2,
            "epochs": epochs,
            "training_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "testing_samples": len(test_dataset),
            "training_data": "synthetic mock data",
            "architecture": "HyperspectralViT with training-only per-band standardization",
            "random_seed": 42,
            "band_mean": torch.tensor(band_mean),
            "band_std": torch.tensor(band_std),
        },
        model_path,
    )

    print("\nViT V3 model saved to:", model_path)
    print("Training completed successfully.")
    print("Validation and test data were not used for training.")


if __name__ == "__main__":
    main()
