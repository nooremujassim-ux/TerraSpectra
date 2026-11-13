
"""Export trained CNN predictions for the TerraSpectra React dashboard."""

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.cnn_3d import Hyperspectral3DCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = PROJECT_ROOT / "data" / "sample" / "mock_hyperspectral.tif"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "hyperspectral_3d_cnn_v2.pth"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "public" / "cnn_predictions.json"

PATCH_SIZE = 5
BATCH_SIZE = 32
THRESHOLD = 0.50


class HyperspectralInferenceDataset(Dataset):
    """Extract image patches without loading reference labels."""

    def __init__(self, image_path, patch_size=5):
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("Patch size must be a positive odd number.")

        with rasterio.open(image_path) as src:
            self.image = src.read().astype(np.float32)
            self.height = src.height
            self.width = src.width
            self.bands = src.count

        if self.image.ndim != 3:
            raise ValueError("Expected a 3D hyperspectral image.")

        if not np.isfinite(self.image).all():
            raise ValueError("Image contains invalid values.")

        self.patch_size = patch_size
        self.padding = patch_size // 2

        self.padded_image = np.pad(
            self.image,
            (
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="reflect",
        )

        self.coordinates = [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
        ]

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        row, col = self.coordinates[index]

        patch = self.padded_image[
            :,
            row:row + self.patch_size,
            col:col + self.patch_size,
        ]

        patch = np.ascontiguousarray(patch[None, ...])

        return torch.from_numpy(patch), row, col


def export_predictions():
    """Run CNN inference and export pixel-level predictions."""

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Hyperspectral image not found: {IMAGE_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    torch.manual_seed(42)
    device = torch.device("cpu")

    dataset = HyperspectralInferenceDataset(
        IMAGE_PATH,
        patch_size=PATCH_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = Hyperspectral3DCNN(num_classes=2).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pixels = []

    with torch.no_grad():
        for features, rows, cols in loader:
            features = features.to(device)

            logits = model(features)
            probabilities = torch.softmax(logits, dim=1)

            stressed_probabilities = probabilities[:, 1].cpu().numpy()

            for row, col, probability in zip(
                rows.tolist(),
                cols.tolist(),
                stressed_probabilities,
            ):
                predicted_class = int(probability >= THRESHOLD)

                pixels.append({
                    "position": [col, row],
                    "row": row,
                    "col": col,
                    "predicted_class": predicted_class,
                    "predicted_label": (
                        "Stressed" if predicted_class == 1 else "Healthy"
                    ),
                    "stressed": bool(predicted_class == 1),
                    "stressed_probability": round(float(probability), 6),
                })

    predicted_stressed = sum(
        pixel["predicted_class"] == 1 for pixel in pixels
    )

    predicted_healthy = len(pixels) - predicted_stressed

    result = {
        "dataset_type": "SYNTHETIC_MOCK_DATA_MODEL_PREDICTIONS",
        "description": (
            "CNN predictions on artificially generated hyperspectral data. "
            "These predictions are not real crop disease observations."
        ),
        "model": "Hyperspectral3DCNN",
        "checkpoint": CHECKPOINT_PATH.name,
        "classification_threshold": THRESHOLD,
        "metadata": {
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.bands,
            "patch_size": PATCH_SIZE,
            "device": str(device),
        },
        "total_pixels": len(pixels),
        "predicted_healthy_pixels": predicted_healthy,
        "predicted_stressed_pixels": predicted_stressed,
        "pixels": pixels,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("CNN predictions exported successfully.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Total pixels: {len(pixels)}")
    print(f"Predicted healthy: {predicted_healthy}")
    print(f"Predicted stressed: {predicted_stressed}")
    print("Reference labels were not used during inference.")


if __name__ == "__main__":
    export_predictions()
