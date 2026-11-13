
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.models.cnn_3d import Hyperspectral3DCNN
from src.models.hyperspectral_dataset import HyperspectralDataset


def main():
    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")
    print("Training device:", device)

    # Load synthetic hyperspectral dataset
    dataset = HyperspectralDataset(
        image_path="data/sample/mock_hyperspectral.tif",
        label_path="data/sample/mock_labels.tif",
        patch_size=5,
    )

    labels = np.array([
        int(dataset.labels[row, col])
        for row, col in dataset.coordinates
    ])

    indices = np.arange(len(dataset))

    # First split: 60% training, 40% temporary
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.4,
        random_state=42,
        stratify=labels,
    )

    # Second split: 20% validation, 20% testing
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.5,
        random_state=42,
        stratify=labels[temporary_indices],
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

    # Calculate class weights using training data only
    train_labels = labels[train_indices]
    class_counts = np.bincount(train_labels, minlength=2)

    class_weights = len(train_labels) / (
        2.0 * class_counts
    )

    weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    print("Training class distribution:", class_counts.tolist())
    print("Class weights:", weights.tolist())

    # Initialize CNN
    model = Hyperspectral3DCNN(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # Train using only the training partition
    epochs = 3

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

    # Save checkpoint without overwriting the earlier experiment
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "hyperspectral_3d_cnn_v2.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["healthy", "stressed"],
            "input_bands": 200,
            "patch_size": 5,
            "epochs": epochs,
            "training_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "testing_samples": len(test_dataset),
            "training_data": "synthetic mock data",
        },
        model_path,
    )

    print("\nNew model saved to:", model_path)
    print("Training completed successfully.")
    print("Validation and test data were not used for training.")


if __name__ == "__main__":
    main()
