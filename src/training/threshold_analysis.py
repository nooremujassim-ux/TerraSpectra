
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

from src.models.cnn_3d import Hyperspectral3DCNN
from src.models.hyperspectral_dataset import HyperspectralDataset


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Load synthetic hyperspectral dataset.
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

    # Reproduce the exact 60/20/20 split used during training.
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

    # Load only the reserved test samples.
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=32,
        shuffle=False,
    )

    # Load the newly trained CNN model.
    model = Hyperspectral3DCNN(num_classes=2)

    checkpoint = torch.load(
        "models/hyperspectral_3d_cnn_v2.pth",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Threshold selected using validation data.
    threshold = 0.50

    actual = []
    predicted = []

    # Generate predictions on the reserved test set.
    with torch.no_grad():
        for features, targets in test_loader:
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)

            predictions = (
                probabilities[:, 1] >= threshold
            ).long()

            actual.extend(targets.tolist())
            predicted.extend(predictions.tolist())

    # Calculate final test metrics.
    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual,
        predicted,
        zero_division=0,
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0,
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0,
    )

    print("\nFinal CNN Test Results")
    print("-" * 40)

    print(f"Classification threshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predicted))

    print("\nClass order: 0 = Healthy, 1 = Stressed")

    print("\nPrediction Distribution:")
    print(f"Actual samples: {len(actual)}")
    print(f"Predicted healthy: {predicted.count(0)}")
    print(f"Predicted stressed: {predicted.count(1)}")

    print("\nImportant:")
    print("Evaluation uses the reserved test split only.")
    print("Threshold was selected using validation data.")
    print("Dataset contains synthetic mock observations.")
    print(
        "Results do not establish real-world crop disease performance."
    )


if __name__ == "__main__":
    main()
