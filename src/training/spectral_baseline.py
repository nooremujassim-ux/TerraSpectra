
import rasterio
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def main():
    # Load hyperspectral image and pixel labels
    with rasterio.open("data/sample/mock_hyperspectral.tif") as src:
        image = src.read().astype(np.float32)

    with rasterio.open("data/sample/mock_labels.tif") as src:
        labels = src.read(1)

    # Convert (bands, height, width) into (pixels, bands)
    X = image.reshape(image.shape[0], -1).T
    y = labels.reshape(-1)

    # Keep only labelled pixels
    valid = np.isin(y, [0, 1])
    X = X[valid]
    y = y[valid]

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Spectral features:", X_train.shape[1])

    # Standardize features and balance class weights
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    )

    print("\nTraining Logistic Regression baseline...")
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)

    print("\nBaseline Evaluation")
    print("-------------------")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClass order: 0 = Healthy, 1 = Stressed")

    print("\nNote: Evaluation uses synthetic mock data.")


if __name__ == "__main__":
    main()
