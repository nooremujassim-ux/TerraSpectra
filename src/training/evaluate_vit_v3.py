
import numpy as np
import torch

import rasterio

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
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
    torch.manual_seed(42)

    device = torch.device("cpu")
    model_path = "models/hyperspectral_vit_v3.pth"

    # Load the saved checkpoint.
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    # Recreate the original dataset split.
    with rasterio.open("data/sample/mock_labels.tif") as src:
        raw_labels = src.read(1)

    coordinates = np.argwhere(np.isin(raw_labels, [0, 1]))
    labels = raw_labels[
        coordinates[:, 0],
        coordinates[:, 1],
    ].astype(np.int64)

    indices = np.arange(len(coordinates))

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

    # Reuse training-only statistics saved in the checkpoint.
    band_mean = checkpoint["band_mean"].cpu().numpy()
    band_std = checkpoint["band_std"].cpu().numpy()

    dataset = StandardizedHyperspectralDataset(
        image_path="data/sample/mock_hyperspectral.tif",
        label_path="data/sample/mock_labels.tif",
        patch_size=5,
        band_mean=band_mean,
        band_std=band_std,
    )

    test_dataset = Subset(dataset, test_indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # Recreate and load the trained ViT.
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

    all_predictions = []
    all_targets = []

    # Evaluate only on the reserved test partition.
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)

            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())

    predictions = np.asarray(all_predictions)
    actual = np.asarray(all_targets)

    print("\nViT V3 Test Evaluation")
    print("=" * 45)
    print("Training samples:", len(train_indices))
    print("Validation samples:", len(validation_indices))
    print("Testing samples:", len(test_indices))

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
    print("Predicted class counts:", np.bincount(predictions, minlength=2).tolist())

    print("\nNote:")
    print("- Evaluation uses synthetic mock data.")
    print("- Standardization statistics came from training pixels only.")
    print("- Spatial independence is not guaranteed because patches overlap.")
    print("- These results do not establish real-world disease detection.")


if __name__ == "__main__":
    main()
