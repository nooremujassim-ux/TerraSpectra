
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.models.cnn_3d import Hyperspectral3DCNN
from src.models.hyperspectral_dataset import HyperspectralDataset
from src.models.hyperspectral_dataset_v2 import (
    StandardizedHyperspectralDataset,
)
from src.models.hyperspectral_vit import HyperspectralViT
from src.models.hyperspectral_hybrid import HyperspectralHybridViT

from src.evaluation.prediction_analysis import (
    save_model_analysis,
    create_comparison_visualizations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "sample"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "prediction_analysis"

IMAGE_PATH = DATA_DIR / "mock_hyperspectral.tif"
LABEL_PATH = DATA_DIR / "mock_labels.tif"

CLASS_NAMES = ["Healthy", "Stressed"]
DEVICE = torch.device("cpu")


def load_raster_labels():
    with rasterio.open(LABEL_PATH) as src:
        return src.read(1).astype(np.int64)


def get_probabilities(model, loader, threshold=None):
    actual = []
    predicted = []
    probabilities_all = []

    model.eval()

    with torch.no_grad():
        for features, targets in loader:
            outputs = model(features.to(DEVICE))
            probabilities = torch.softmax(outputs, dim=1)

            if threshold is None:
                predictions = torch.argmax(probabilities, dim=1)
            else:
                predictions = (
                    probabilities[:, 1] >= threshold
                ).long()

            actual.extend(targets.cpu().numpy().tolist())
            predicted.extend(predictions.cpu().numpy().tolist())
            probabilities_all.extend(
                probabilities.cpu().numpy().tolist()
            )

    return (
        np.asarray(actual, dtype=np.int64),
        np.asarray(predicted, dtype=np.int64),
        np.asarray(probabilities_all, dtype=np.float64),
    )


def run_cnn():
    print("\n[1/3] Evaluating CNN V2...")

    torch.manual_seed(42)
    np.random.seed(42)

    dataset = HyperspectralDataset(
        str(IMAGE_PATH),
        str(LABEL_PATH),
        patch_size=5,
    )

    labels = np.asarray([
        int(dataset.labels[row, col])
        for row, col in dataset.coordinates
    ])

    indices = np.arange(len(dataset))

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

    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=32,
        shuffle=False,
    )

    checkpoint_path = MODEL_DIR / "hyperspectral_3d_cnn_v2.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model = Hyperspectral3DCNN(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])

    actual, predicted, probabilities = get_probabilities(
        model,
        test_loader,
        threshold=0.50,
    )

    coordinates = dataset.coordinates[test_indices]

    return {
        "name": "CNN V2",
        "folder": "cnn_v2",
        "actual": actual,
        "predicted": predicted,
        "probabilities": probabilities,
        "coordinates": coordinates,
        "metadata": {
            "evaluation_protocol": "Stratified random 60/20/20 split",
            "random_state": 42,
            "patch_size": 5,
            "batch_size": 32,
            "classification_rule": "Stressed probability >= 0.50",
            "classification_threshold": 0.50,
            "training_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "test_samples": len(test_indices),
            "data_type": "Synthetic mock data",
            "limitations": [
                "Random split may allow spatial overlap between patches.",
                "Results do not establish real-world crop disease performance.",
            ],
        },
    }


def run_spatial_vit():
    print("\n[2/3] Evaluating Spatial ViT...")

    checkpoint_path = MODEL_DIR / "hyperspectral_vit_spatial.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    labels = load_raster_labels()

    height, width = labels.shape
    rows, columns = np.indices((height, width))

    test_mask = (
        (rows >= 32)
        & (columns >= 32)
        & np.isin(labels, [0, 1])
    )

    coordinates = np.argwhere(test_mask)

    test_indices = np.ravel_multi_index(
        (coordinates[:, 0], coordinates[:, 1]),
        (height, width),
    )

    dataset = StandardizedHyperspectralDataset(
        image_path=str(IMAGE_PATH),
        label_path=str(LABEL_PATH),
        patch_size=checkpoint["patch_size"],
        band_mean=checkpoint["band_mean"].cpu().numpy(),
        band_std=checkpoint["band_std"].cpu().numpy(),
    )

    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = HyperspectralViT(
        num_bands=checkpoint["input_bands"],
        patch_size=checkpoint["patch_size"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        depth=checkpoint["depth"],
        num_classes=checkpoint["num_classes"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    actual, predicted, probabilities = get_probabilities(
        model,
        test_loader,
    )

    return {
        "name": "Spatial ViT",
        "folder": "spatial_vit",
        "actual": actual,
        "predicted": predicted,
        "probabilities": probabilities,
        "coordinates": coordinates,
        "metadata": {
            "evaluation_protocol": "Spatial holdout",
            "test_mask": "rows >= 32 and columns >= 32",
            "patch_size": checkpoint["patch_size"],
            "batch_size": 32,
            "classification_rule": "Argmax over two class probabilities",
            "training_samples": checkpoint["training_samples"],
            "test_samples": len(coordinates),
            "data_type": "Synthetic mock data",
            "limitations": [
                "5x5 patches near the training boundary may overlap the test region.",
                "Results are not independent real-world validation.",
            ],
        },
    }


def run_hybrid():
    print("\n[3/3] Evaluating Hybrid CNN-Transformer...")

    checkpoint_path = MODEL_DIR / "hyperspectral_hybrid_spatial.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    labels = load_raster_labels()

    height, width = labels.shape
    rows, columns = np.indices((height, width))

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

    test_indices = np.ravel_multi_index(
        (
            test_coordinates[:, 0],
            test_coordinates[:, 1],
        ),
        (height, width),
    )

    dataset = StandardizedHyperspectralDataset(
        image_path=str(IMAGE_PATH),
        label_path=str(LABEL_PATH),
        patch_size=checkpoint["patch_size"],
        band_mean=checkpoint["band_mean"].cpu().numpy(),
        band_std=checkpoint["band_std"].cpu().numpy(),
    )

    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = HyperspectralHybridViT(
        num_bands=checkpoint["num_bands"],
        patch_size=checkpoint["patch_size"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        depth=checkpoint["depth"],
        num_classes=checkpoint["num_classes"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    actual, predicted, probabilities = get_probabilities(
        model,
        test_loader,
    )

    return {
        "name": "Hybrid CNN-Transformer",
        "folder": "hybrid_spatial",
        "actual": actual,
        "predicted": predicted,
        "probabilities": probabilities,
        "coordinates": test_coordinates,
        "metadata": {
            "evaluation_protocol": "Spatial holdout",
            "training_mask": "rows < 28 or columns < 28",
            "test_mask": "rows >= 32 and columns >= 32",
            "patch_size": checkpoint["patch_size"],
            "batch_size": 32,
            "classification_rule": "Argmax over two class probabilities",
            "training_samples": len(train_coordinates),
            "test_samples": len(test_coordinates),
            "data_type": "Synthetic mock data",
            "limitations": [
                "5x5 patches near the training boundary may overlap the test region.",
                "Results are not independent real-world validation.",
            ],
        },
    }


def create_test_prediction_map(result, output_dir):
    height, width = load_raster_labels().shape

    actual_map = np.full((height, width), np.nan)
    predicted_map = np.full((height, width), np.nan)

    rows = result["coordinates"][:, 0]
    columns = result["coordinates"][:, 1]

    actual_map[rows, columns] = result["actual"]
    predicted_map[rows, columns] = result["predicted"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, image, title in zip(
        axes,
        [actual_map, predicted_map],
        ["Actual Test Labels", "Predicted Test Labels"],
    ):
        plotted = ax.imshow(
            image,
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    colorbar = fig.colorbar(
        plotted,
        ax=axes,
        ticks=[0, 1],
        fraction=0.035,
        pad=0.04,
    )
    colorbar.ax.set_yticklabels(CLASS_NAMES)

    fig.suptitle(
        f"{result['name']} — Reserved Test Pixels Only",
        fontsize=13,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "test_prediction_map.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_summary(results):
    summary = {}

    for result in results:
        folder = OUTPUT_DIR / result["folder"]
        folder.mkdir(parents=True, exist_ok=True)

        metrics, dataframe = save_model_analysis(
            model_name=result["name"],
            actual=result["actual"],
            predicted=result["predicted"],
            coordinates=result["coordinates"],
            output_dir=folder,
            probabilities=result["probabilities"],
            metadata=result["metadata"],
        )

        create_test_prediction_map(result, folder)

        summary[result["name"]] = metrics

        print(f"\n{result['name']}")
        print("-" * 40)
        print(f"Test samples: {len(dataframe)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        print(f"Output: {folder}")

    create_comparison_visualizations(
        summary,
        OUTPUT_DIR,
    )

    summary_path = OUTPUT_DIR / "prediction_analysis_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nPrediction analysis completed.")
    print(f"All outputs saved to: {OUTPUT_DIR}")


def main():
    print("=" * 60)
    print("TERRASPECTRA — MODEL PREDICTION ANALYSIS")
    print("=" * 60)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)

    if not LABEL_PATH.exists():
        raise FileNotFoundError(LABEL_PATH)

    results = [
        run_cnn(),
        run_spatial_vit(),
        run_hybrid(),
    ]

    create_summary(results)


if __name__ == "__main__":
    main()
