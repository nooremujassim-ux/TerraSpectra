
import json

import numpy as np

from src.training.spatial_prediction_analysis import main


def test_spatial_analysis_generates_valid_report(tmp_path, monkeypatch):
    project_root = tmp_path

    prediction_dir = project_root / "frontend" / "public"
    prediction_dir.mkdir(parents=True)

    report_dir = project_root / "reports"

    pixels = []

    # Create a 4x4 grid with a 2x2 connected stressed region.
    for row in range(4):
        for col in range(4):
            stressed = row in (1, 2) and col in (1, 2)

            pixels.append({
                "row": row,
                "col": col,
                "predicted_class": int(stressed),
                "stressed_probability": 0.8 if stressed else 0.2,
            })

    prediction_data = {
        "dataset_type": "SYNTHETIC_MOCK_DATA_MODEL_PREDICTIONS",
        "metadata": {
            "height": 4,
            "width": 4,
            "bands": 10,
            "patch_size": 5,
            "device": "cpu",
        },
        "pixels": pixels,
    }

    prediction_path = prediction_dir / "cnn_predictions.json"

    prediction_path.write_text(
        json.dumps(prediction_data),
        encoding="utf-8",
    )

    # Redirect the analysis to the temporary test project.
    monkeypatch.setattr(
        "src.training.spatial_prediction_analysis.Path.resolve",
        lambda self: project_root / "src" / "training" / "spatial_prediction_analysis.py",
    )

    main()

    report_path = report_dir / "cnn_spatial_analysis.json"

    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["total_pixels"] == 16
    assert report["predicted_healthy_pixels"] == 12
    assert report["predicted_stressed_pixels"] == 4
    assert report["connected_region_count"] == 1
    assert report["regions"][0]["pixel_count"] == 4
    assert report["regions"][0]["bounding_box"] == {
        "min_row": 1,
        "max_row": 2,
        "min_col": 1,
        "max_col": 2,
    }

    assert report["dataset_type"] == (
        "SYNTHETIC_MOCK_DATA_MODEL_PREDICTIONS"
    )
