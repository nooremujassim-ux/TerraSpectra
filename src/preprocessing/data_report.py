
"""Generate a summary report for hyperspectral sample data."""

from pathlib import Path

import numpy as np
import rasterio

from src.preprocessing.pipeline import preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = PROJECT_ROOT / "data" / "sample" / "mock_hyperspectral.tif"
LABEL_PATH = PROJECT_ROOT / "data" / "sample" / "mock_labels.tif"
REPORT_DIR = PROJECT_ROOT / "reports"


def generate_data_report() -> Path:
    """Generate a text report describing the sample dataset."""

    with rasterio.open(IMAGE_PATH) as dataset:
        image = dataset.read()
        metadata = {
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": str(dataset.crs),
        }

    with rasterio.open(LABEL_PATH) as dataset:
        labels = dataset.read(1)

    processed = preprocess_image(
        IMAGE_PATH,
        apply_dimensionality_reduction=True,
        n_components=30,
    )

    healthy_pixels = int(np.count_nonzero(labels == 0))
    stressed_pixels = int(np.count_nonzero(labels == 1))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "week1_data_summary.txt"

    report = f"""
TERRASPECTRA - WEEK 1 DATA SUMMARY
=================================

DATA SOURCE
-----------
Synthetic mock hyperspectral dataset.
Not real satellite imagery or field observations.

IMAGE METADATA
--------------
Width: {metadata['width']}
Height: {metadata['height']}
Spectral bands: {metadata['bands']}
Data type: {metadata['dtype']}
CRS: {metadata['crs']}

LABEL SUMMARY
-------------
Healthy mock pixels: {healthy_pixels}
Stressed mock pixels: {stressed_pixels}
Total pixels: {labels.size}

PREPROCESSING
-------------
Normalization: Completed
PCA: Completed
Original spectral bands: {metadata['bands']}
Reduced components: {processed.shape[0]}
Output shape: {processed.shape}
Output data type: {processed.dtype}
All output values finite: {bool(np.isfinite(processed).all())}

LIMITATIONS
-----------
The dataset and labels are artificially generated.
They are intended only for pipeline development and testing.
They do not demonstrate real crop disease detection,
model accuracy, or disease forecasting.
"""

    report_path.write_text(report.strip() + "\n", encoding="utf-8")

    print("Week 1 data report generated successfully.")
    print(f"Report location: {report_path}")

    return report_path


if __name__ == "__main__":
    generate_data_report()

