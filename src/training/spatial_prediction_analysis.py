
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def main():
    project_root = Path(__file__).resolve().parents[2]

    prediction_path = (
        project_root / "frontend" / "public" / "cnn_predictions.json"
    )

    output_path = (
        project_root / "reports" / "cnn_spatial_analysis.json"
    )

    with prediction_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    metadata = data["metadata"]
    height = metadata["height"]
    width = metadata["width"]

    # Reconstruct the CNN prediction grid.
    prediction_grid = np.zeros((height, width), dtype=np.uint8)
    probability_grid = np.zeros((height, width), dtype=np.float32)

    for pixel in data["pixels"]:
        row = pixel["row"]
        col = pixel["col"]

        prediction_grid[row, col] = pixel["predicted_class"]
        probability_grid[row, col] = pixel["stressed_probability"]

    # Identify connected stressed regions.
    stressed_mask = prediction_grid == 1

    structure = np.ones((3, 3), dtype=np.uint8)

    region_labels, region_count = ndimage.label(
        stressed_mask,
        structure=structure,
    )

    regions = []

    for region_id in range(1, region_count + 1):
        region_mask = region_labels == region_id
        rows, cols = np.where(region_mask)

        regions.append({
            "region_id": int(region_id),
            "pixel_count": int(np.sum(region_mask)),
            "average_stressed_probability": round(
                float(np.mean(probability_grid[region_mask])), 6
            ),
            "bounding_box": {
                "min_row": int(rows.min()),
                "max_row": int(rows.max()),
                "min_col": int(cols.min()),
                "max_col": int(cols.max()),
            },
        })

    regions.sort(
        key=lambda region: region["pixel_count"],
        reverse=True,
    )

    total_pixels = height * width
    stressed_pixels = int(np.sum(stressed_mask))
    healthy_pixels = total_pixels - stressed_pixels

    average_probability = float(np.mean(probability_grid))

    # Prepare the analysis report.
    report = {
        "dataset_type": data["dataset_type"],
        "analysis_type": "CNN prediction spatial analysis",
        "image_dimensions": {
            "height": height,
            "width": width,
        },
        "total_pixels": total_pixels,
        "predicted_healthy_pixels": healthy_pixels,
        "predicted_stressed_pixels": stressed_pixels,
        "predicted_stressed_percentage": round(
            stressed_pixels / total_pixels * 100, 2
        ),
        "average_stressed_probability": round(
            average_probability, 6
        ),
        "connected_region_count": int(region_count),
        "connectivity": "8-connected",
        "regions": regions,
        "limitations": [
            "Analysis uses CNN predictions from synthetic mock data.",
            "Connected regions are spatial clusters, not confirmed disease outbreaks.",
            "No real-world crop area or geographic acreage is inferred.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("CNN Spatial Prediction Analysis")
    print("-" * 45)
    print(f"Image dimensions: {height} x {width}")
    print(f"Total pixels: {total_pixels}")
    print(f"Predicted healthy pixels: {healthy_pixels}")
    print(f"Predicted stressed pixels: {stressed_pixels}")
    print(
        f"Predicted stressed percentage: "
        f"{report['predicted_stressed_percentage']}%"
    )
    print(f"Average stressed probability: {average_probability:.4f}")
    print(f"Connected stressed regions: {region_count}")

    print("\nLargest predicted regions:")

    for region in regions[:5]:
        print(
            f"Region {region['region_id']}: "
            f"{region['pixel_count']} pixels"
        )

    print(f"\nReport saved to: {output_path}")
    print("\nNote: Results are based on synthetic mock data.")


if __name__ == "__main__":
    main()
