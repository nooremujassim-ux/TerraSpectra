
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


CLASS_NAMES = {
    0: "Healthy",
    1: "Stressed",
}


def validate_predictions(actual, predicted, coordinates):
    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    coordinates = np.asarray(coordinates, dtype=np.int64)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Actual and predicted labels must be 1D arrays.")

    if len(actual) == 0:
        raise ValueError("Prediction arrays cannot be empty.")

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths must match.")

    if coordinates.shape != (len(actual), 2):
        raise ValueError("Coordinates must have shape (samples, 2).")

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Actual labels must contain only 0 and 1.")

    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("Predicted labels must contain only 0 and 1.")

    return actual, predicted, coordinates


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")

    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Labels must be non-empty and have equal lengths.")

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1_score": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).tolist(),
        "class_order": ["Healthy", "Stressed"],
        "total_samples": int(len(actual)),
        "actual_class_counts": {
            CLASS_NAMES[class_id]: int(np.sum(actual == class_id))
            for class_id in [0, 1]
        },
        "predicted_class_counts": {
            CLASS_NAMES[class_id]: int(np.sum(predicted == class_id))
            for class_id in [0, 1]
        },
    }


def create_prediction_dataframe(
    actual,
    predicted,
    coordinates,
    probabilities=None,
):
    actual, predicted, coordinates = validate_predictions(
        actual, predicted, coordinates
    )

    data = {
        "row": coordinates[:, 0],
        "col": coordinates[:, 1],
        "actual_class": actual,
        "actual_label": [
            CLASS_NAMES[int(value)] for value in actual
        ],
        "predicted_class": predicted,
        "predicted_label": [
            CLASS_NAMES[int(value)] for value in predicted
        ],
        "correct": actual == predicted,
    }

    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)

        if probabilities.shape != (len(actual), 2):
            raise ValueError(
                "Probabilities must have shape (samples, 2)."
            )

        if not np.isfinite(probabilities).all():
            raise ValueError("Probabilities contain invalid values.")

        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Probabilities must be between 0 and 1.")

        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Each probability row must sum to 1.")

        data["healthy_probability"] = probabilities[:, 0]
        data["stressed_probability"] = probabilities[:, 1]

    return pd.DataFrame(data)


def save_model_analysis(
    model_name,
    actual,
    predicted,
    coordinates,
    output_dir,
    probabilities=None,
    metadata=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = create_prediction_dataframe(
        actual,
        predicted,
        coordinates,
        probabilities,
    )

    metrics = calculate_metrics(actual, predicted)

    if metadata:
        metrics["evaluation_metadata"] = metadata

    csv_path = output_dir / "predictions.csv"
    json_path = output_dir / "metrics.json"
    matrix_path = output_dir / "confusion_matrix.png"

    dataframe.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    matrix = np.asarray(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(f"{model_name} — Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    ax.set_xticks([0, 1], ["Healthy", "Stressed"])
    ax.set_yticks([0, 1], ["Healthy", "Stressed"])

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                fontsize=12,
            )

    fig.colorbar(image, ax=ax, label="Number of samples")
    fig.tight_layout()
    fig.savefig(matrix_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return metrics, dataframe


def create_comparison_visualizations(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(results.keys())
    actual_counts = [
        results[name]["actual_class_counts"] for name in model_names
    ]
    predicted_counts = [
        results[name]["predicted_class_counts"] for name in model_names
    ]

    positions = np.arange(len(model_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))

    for index, (class_name, color) in enumerate(
        [("Healthy", "seagreen"), ("Stressed", "indianred")]
    ):
        actual_values = [
            counts[class_name] for counts in actual_counts
        ]
        predicted_values = [
            counts[class_name] for counts in predicted_counts
        ]

        ax.bar(
            positions + (index * 2 - 1.5) * width,
            actual_values,
            width,
            label=f"Actual {class_name}",
            color=color,
            alpha=0.45,
        )

        ax.bar(
            positions + (index * 2 - 0.5) * width,
            predicted_values,
            width,
            label=f"Predicted {class_name}",
            color=color,
        )

    ax.set_title("Actual vs Predicted Class Distribution")
    ax.set_ylabel("Number of test samples")
    ax.set_xticks(positions, model_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    fig.savefig(
        output_dir / "actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, metrics in results.items():
        ax.bar(
            name,
            metrics["predicted_class_counts"]["Healthy"],
            label="Healthy",
            color="seagreen",
        )
        ax.bar(
            name,
            metrics["predicted_class_counts"]["Stressed"],
            bottom=metrics["predicted_class_counts"]["Healthy"],
            label="Stressed",
            color="indianred",
        )

    ax.set_title("Predicted Class Distribution by Model")
    ax.set_ylabel("Number of test samples")
    ax.tick_params(axis="x", rotation=10)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
