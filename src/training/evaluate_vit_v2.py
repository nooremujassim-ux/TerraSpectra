
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from torch.utils.data import DataLoader, Subset

from src.models.hyperspectral_vit_v2 import HyperspectralViTV2
from src.models.hyperspectral_dataset import HyperspectralDataset


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")

    dataset = HyperspectralDataset(
        "data/sample/mock_hyperspectral.tif",
        "data/sample/mock_labels.tif",
        patch_size=5,
    )

    labels = np.array([
        int(dataset.labels[row, col])
        for row, col in dataset.coordinates
    ])

    indices = np.arange(len(dataset))

    # Reproduce the same stratified 60/20/20 split.
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

    print("Final Test Dataset")
    print("-" * 40)
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(validation_indices)}")
    print(f"Testing samples: {len(test_indices)}")

    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # Initialize V2 model.
    model = HyperspectralViTV2(
        num_bands=200,
        patch_size=5,
        embed_dim=64,
        num_heads=4,
        depth=2,
        num_classes=2,
    ).to(device)

    checkpoint = torch.load(
        "models/hyperspectral_vit_v2.pth",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    threshold = 0.50

    actual = []
    predicted = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)

            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)

            predictions = (
                probabilities[:, 1] >= threshold
            ).long()

            actual.extend(targets.tolist())
            predicted.extend(predictions.tolist())

    print("\nFinal ViT V2 Test Results")
    print("-" * 40)
    print(f"Classification threshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy_score(actual, predicted):.4f}")
    print(
        f"Precision: {precision_score(actual, predicted, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(actual, predicted, zero_division=0):.4f}"
    )
    print(f"F1-score:  {f1_score(actual, predicted, zero_division=0):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predicted))

    print("\nClass order: 0 = Healthy, 1 = Stressed")

    print("\nPrediction Distribution:")
    print(f"Actual samples: {len(actual)}")
    print(f"Predicted healthy: {predicted.count(0)}")
    print(f"Predicted stressed: {predicted.count(1)}")

    print("\nImportant:")
    print("Evaluation uses the reserved test split only.")
    print("Classification threshold is fixed at 0.50.")
    print("Dataset contains synthetic mock observations.")
    print("Results do not establish real-world crop disease performance.")
    print("Spatial independence is not guaranteed because patches may overlap.")


if __name__ == "__main__":
    main()
