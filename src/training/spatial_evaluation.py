
from pathlib import Path

import numpy as np
import rasterio

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    project_root = Path(__file__).resolve().parents[2]

    image_path = project_root / "data" / "sample" / "mock_hyperspectral.tif"
    label_path = project_root / "data" / "sample" / "mock_labels.tif"

    # Load hyperspectral image and labels.
    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)

    with rasterio.open(label_path) as src:
        labels = src.read(1).astype(np.int64)

    bands, height, width = image.shape

    if labels.shape != (height, width):
        raise ValueError("Image and label dimensions do not match.")

    # Convert image from (bands, rows, columns)
    # into (pixels, spectral bands).
    features = np.moveaxis(image, 0, -1).reshape(-1, bands)
    targets = labels.reshape(-1)

    # Build spatial masks.
    rows, columns = np.indices((height, width))

    # Training region: outside the lower-right quadrant.
    train_mask = (rows < 28) | (columns < 28)

    # Testing region: lower-right quadrant.
    test_mask = (rows >= 32) & (columns >= 32)

    # Pixels between these regions are excluded.
    train_features = features[train_mask.reshape(-1)]
    train_labels = labels[train_mask]

    test_features = features[test_mask.reshape(-1)]
    test_labels = labels[test_mask]

    print("Spatial Evaluation")
    print("-" * 40)
    print(f"Image dimensions: {height} x {width}")
    print(f"Spectral bands: {bands}")
    print(f"Training samples: {len(train_labels)}")
    print(f"Testing samples: {len(test_labels)}")
    print(f"Excluded buffer samples: {height * width - len(train_labels) - len(test_labels)}")

    print("\nTraining class distribution:")
    print(f"Healthy: {np.sum(train_labels == 0)}")
    print(f"Stressed: {np.sum(train_labels == 1)}")

    print("\nTesting class distribution:")
    print(f"Healthy: {np.sum(test_labels == 0)}")
    print(f"Stressed: {np.sum(test_labels == 1)}")

    if len(np.unique(train_labels)) < 2:
        raise ValueError("Training data must contain both classes.")

    if len(np.unique(test_labels)) < 2:
        raise ValueError("Testing data must contain both classes.")

    # Train the baseline model.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    )

    print("\nTraining Logistic Regression...")
    model.fit(train_features, train_labels)

    # Evaluate on the spatially held-out region.
    predictions = model.predict(test_features)

    print("\nSpatial Evaluation Results")
    print("-" * 40)
    print(f"Accuracy:  {accuracy_score(test_labels, predictions):.4f}")
    print(
        f"Precision: {precision_score(test_labels, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(test_labels, predictions, zero_division=0):.4f}"
    )
    print(f"F1-score:  {f1_score(test_labels, predictions, zero_division=0):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, predictions))

    print("\nClass order: 0 = Healthy, 1 = Stressed")
    print("Note: Evaluation uses synthetic mock data.")
    print("This is a spatial holdout from one synthetic image,")
    print("not an independent real-world validation.")


if __name__ == "__main__":
    main()
