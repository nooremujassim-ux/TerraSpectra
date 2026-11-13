# TerraSpectra: Hyperspectral Crop Disease Forecasting

## 1. Project Overview

TerraSpectra is a Data Science and Machine Learning project focused on hyperspectral crop imagery analysis for identifying crop stress patterns and supporting crop health assessment.

The project develops an end-to-end workflow involving hyperspectral data loading, preprocessing, spectral analysis, dimensionality reduction, deep learning, model evaluation, and geospatial visualization.

**Important:** The current experiments use clearly labelled synthetic hyperspectral data. Results demonstrate pipeline functionality and do not establish real-world crop disease detection performance.

## 2. Project Objectives

- Process hyperspectral imagery containing multiple spectral bands.
- Analyze spectral and spatial characteristics of crop data.
- Perform data validation, preprocessing, and exploratory analysis.
- Apply Principal Component Analysis (PCA) for dimensionality reduction.
- Develop a 3D Convolutional Neural Network (3D-CNN).
- Evaluate classification performance using appropriate metrics.
- Visualize crop stress patterns using a geospatial dashboard.
- Develop an API for future model inference.

## 3. Project Modules

1. Hyperspectral Data Loading
2. Data Validation and Preprocessing
3. Spectral Analysis and PCA
4. 3D-CNN Model Development
5. Model Training and Evaluation
6. Spatial Analysis
7. Geospatial Visualization Dashboard
8. Future FastAPI Inference Backend

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming | Python 3.11 |
| Data Processing | NumPy, Pandas, SciPy |
| Geospatial Processing | Rasterio |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| Visualization | Matplotlib, Seaborn, Plotly |
| Frontend | React |
| Geospatial Visualization | Deck.gl |
| Version Control | Git, GitHub |

## 5. Project Structure

```text
TerraSpectra/
├── backend/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── docs/
├── frontend/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── inference/
│   └── evaluation/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## 6. Development Methodology

The project follows these stages:

1. Requirement analysis and dataset identification.
2. Hyperspectral data loading and validation.
3. Data preprocessing and normalization.
4. Exploratory data analysis and spectral analysis.
5. PCA-based dimensionality reduction.
6. 3D-CNN architecture development.
7. Model training using training data.
8. Validation-based threshold selection.
9. Final evaluation using a reserved test split.
10. Dashboard development and documentation.

## 7. Current Implementation

### Hyperspectral Data

- Synthetic hyperspectral GeoTIFF sample.
- 200 spectral bands.
- Image dimensions: 64 × 64 pixels.
- Total pixels: 4,096.
- Healthy samples: 3,520.
- Stressed samples: 576.

### Dimensionality Reduction

PCA was applied to the synthetic hyperspectral dataset.

The first 30 principal components explain approximately 87.72% of the variance in the generated sample.

### Deep Learning Model

A 3D Convolutional Neural Network was implemented using PyTorch.

- Input: Hyperspectral spatial patches.
- Input dimensions per sample: 1 × 200 × 5 × 5.
- Classes: Healthy (0) and Stressed (1).
- Training device: CPU.
- Dataset split: 60% training, 20% validation, 20% testing.

### Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 98.90% |
| Precision | 94.17% |
| Recall | 98.26% |
| F1-score | 96.17% |

Confusion matrix:

```text
[[698   7]
 [  2 113]]
```

The classification threshold was selected using validation data and then applied to the reserved test split.

**Evaluation limitation:** These results are from a synthetic dataset and a random pixel split. Neighbouring spatial patches may overlap between partitions. Therefore, these metrics should not be interpreted as independent spatial validation or real-world crop disease performance.

## 8. Installation and Setup

### Clone the repository

```bash
git clone https://github.com/nooremujassim-ux/TerraSpectra.git
cd TerraSpectra
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CPU-only PyTorch installation, use the official CPU wheel index if needed:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 9. Running the Project

Run the preprocessing tests:

```powershell
python -m pytest -q
```

Train the 3D-CNN:

```powershell
python -m src.training.train_cnn
```

Analyze the validation threshold:

```powershell
python -m src.training.threshold_analysis
```

Evaluate the reserved test dataset:

```powershell
python -m src.training.evaluate_cnn
```

Run the frontend from the frontend directory:

```powershell
cd frontend
npm install
npm run dev
```

## 10. Evaluation Metrics

The project uses:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Additional spatial and spectral evaluation experiments are included in the training module.

## 11. Research and Validation Policy

Synthetic data is used for pipeline development and testing. It is not presented as genuine agricultural observation data.

Real-world crop stress or disease detection claims require suitable field-collected or independently sourced hyperspectral datasets, reliable ground-truth labels, and spatially independent validation.

## 12. Internship Information

**Organization:** Infotact Solutions  
**Project:** TerraSpectra: Hyperspectral Crop Disease Forecasting  
**Project Start Date:** 10 November 2026  
**Project Deadline:** 10 December 2026

## 13. Author

**Noor E Mujassim**  
Data Science and Machine Learning Intern