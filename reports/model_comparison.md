
# TerraSpectra — Model Comparison Report

## 1. Objective

This report compares the classification performance of the machine-learning models developed for the TerraSpectra hyperspectral image classification pipeline.

All experiments in this report use synthetic mock hyperspectral data. The results demonstrate pipeline and model behavior on the mock dataset, not real-world crop disease detection.

## 2. Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3D CNN | 0.9890 | 0.9417 | 0.9826 | 0.9617 |
| ViT V1 (CLS token) | 0.8598 | 0.0000 | 0.0000 | 0.0000 |
| ViT V2 (Mean pooling) | 0.1402 | 0.1402 | 1.0000 | 0.2460 |
| ViT V3 (Standardized input) | 0.9976 | 1.0000 | 0.9826 | 0.9912 |

**Important:** Logistic Regression used a separate 80/20 random split. Its results are not directly comparable with the other models, which used a stratified 60/20/20 split.

## 3. ViT V3 Confusion Matrix

Class order: 0 = Healthy, 1 = Stressed.

| Actual / Predicted | Healthy | Stressed |
|---|---:|---:|
| Healthy | 705 | 0 |
| Stressed | 2 | 113 |

ViT V3 correctly classified 818 of 820 test samples in this experiment.

## 4. Observations

- Logistic Regression achieved perfect metrics on its separate test split.
- The 3D CNN achieved strong classification results.
- ViT V1 predicted all test samples as healthy.
- ViT V2 predicted all test samples as stressed.
- ViT V3 achieved substantially better results after introducing training-set per-band standardization and training for 10 epochs.

The ViT V3 experiment changed both input standardization and training duration compared with V1 and V2. Therefore, the improvement cannot be attributed to standardization alone.

## 5. Experimental Limitations

1. All reported experiments use synthetic mock hyperspectral data.
2. The random pixel-level split does not guarantee spatial independence.
3. The 5×5 patches can overlap across training, validation, and test partitions.
4. The synthetic dataset does not establish real crop disease detection capability.
5. The Logistic Regression experiment used a different split.
6. ViT V3 used 10 training epochs, while ViT V1 and V2 used 3 epochs.

## 6. Conclusion

ViT V3 produced the strongest F1-score among the models evaluated on the shared 60/20/20 split.

However, these results should be treated as experimental performance on synthetic data. Spatially independent evaluation and validation on genuine hyperspectral agricultural data are necessary before making real-world performance claims.

## 7. Spatial Holdout Experiment

A separate spatial holdout experiment was conducted using the same training and testing regions for Logistic Regression and ViT V3.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Spatial ViT | 0.9932 | 0.9597 | 0.9931 | 0.9761 |

### Spatial ViT Confusion Matrix

| Actual / Predicted | Healthy | Stressed |
|---|---:|---:|
| Healthy | 874 | 6 |
| Stressed | 1 | 143 |

### Interpretation

The spatial ViT achieved an F1-score of 97.61%. Logistic Regression achieved 100% across the reported metrics.

Both experiments used the same synthetic image and spatial regions. These results are useful for comparing model behavior on this dataset, but they do not establish real-world agricultural performance.

Further evaluation on genuine hyperspectral imagery from independent locations is required.
## Hybrid 3D-CNN + Vision Transformer (Spatial Holdout)

### Architecture
- 3D CNN extracts local spectral-spatial features.
- Vision Transformer models relationships between spatial tokens.
- Hybrid classifier predicts Healthy and Stressed classes.
- Total parameters: 78,242.

### Training Configuration
- Training samples: 2,800
- Spatial test samples: 1,024
- Epochs: 10
- Batch size: 32
- Optimizer: Adam
- Learning rate: 0.001
- Device: CPU
- Normalization statistics calculated from training pixels only.

### Spatial Test Results

| Metric | Score |
|---|---:|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1-score | 1.0000 |

Confusion Matrix:

| | Predicted Healthy | Predicted Stressed |
|---|---:|---:|
| Actual Healthy | 880 | 0 |
| Actual Stressed | 0 | 144 |

### Limitations
- Evaluation used synthetic mock hyperspectral data.
- The test region was excluded from model training.
- Results represent one spatial holdout from one synthetic scene.
- These metrics do not establish real-world crop disease detection performance.