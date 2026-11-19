# TerraSpectra - Model Evaluation and Scientific Validation Report

## 1. Objective and Scope

This report documents the classification experiments conducted in TerraSpectra using a synthetic hyperspectral image.

The project evaluates spectral and spatial machine-learning approaches for distinguishing two synthetic classes:

- Class 0: Healthy
- Class 1: Stressed

The experiments demonstrate data-processing, model-training, classification, evaluation, and visualization workflows. They do not establish real-world crop disease detection capability.

## 2. Dataset Description

| Property | Value |
|---|---|
| Data source | Synthetic mock hyperspectral GeoTIFF |
| Image dimensions | 64 x 64 pixels |
| Spectral bands | 200 |
| Spatial pixels | 4,096 |
| Classification classes | Healthy and Stressed |
| Patch size | 5 x 5 pixels |

The dataset is a synthetic sample and should not be interpreted as independently collected agricultural field data.

## 3. Current Model Evaluation Results

The following results are taken from the current dashboard evaluation summary.

| Model | Evaluation protocol | Test samples | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|---:|
| CNN V2 | Stratified random 60/20/20 | 820 | 0.9890 | 0.9417 | 0.9826 | 0.9617 |
| Spatial ViT | Spatial holdout | 1,024 | 0.9932 | 0.9597 | 0.9931 | 0.9761 |
| Hybrid CNN-Transformer | Spatial holdout | 1,024 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Comparison caution:** CNN V2 uses a random pixel-level split, whereas Spatial ViT and Hybrid CNN-Transformer use spatial holdout regions. These results are not a controlled head-to-head benchmark and should not be used to claim that one architecture is universally superior.

## 4. CNN V2 Evaluation

### Evaluation configuration

| Setting | Value |
|---|---|
| Evaluation protocol | Stratified random 60/20/20 split |
| Random state | 42 |
| Training samples | 2,457 |
| Validation samples | 819 |
| Test samples | 820 |
| Patch size | 5 x 5 |
| Batch size | 32 |
| Classification threshold | 0.50 |
| Classification rule | Stressed probability >= 0.50 |

The evaluation script loads the trained CNN V2 checkpoint and evaluates the reserved test partition. The classification threshold is reported as selected using validation data.

### Confusion matrix

Class order: Healthy, Stressed.

| Actual / Predicted | Healthy | Stressed |
|---|---:|---:|
| Healthy | 698 | 7 |
| Stressed | 2 | 113 |

The model correctly classified 811 of 820 test samples.

### Interpretation

CNN V2 achieved an F1-score of 0.9617 on its random test partition. Its evaluation demonstrates strong classification performance on the synthetic dataset.

However, randomly assigning pixel-level samples does not guarantee spatial independence. Since neighboring samples are extracted as image patches, patches belonging to different partitions may share source pixels.

## 5. Spatial Vision Transformer Evaluation

### Training and evaluation configuration

| Setting | Value |
|---|---|
| Evaluation protocol | Spatial holdout |
| Training region | row < 28 OR column < 28 |
| Test region | row >= 32 AND column >= 32 |
| Training samples | 2,800 |
| Test samples | 1,024 |
| Patch size | 5 x 5 |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Transformer depth | 2 |
| Batch size | 32 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Random seed | 42 |
| Normalization | Training-region per-band statistics |
| Classification rule | Argmax over class probabilities |

The training script calculates per-band means and standard deviations using training-region spectra only. These statistics are saved in the model checkpoint and reused during evaluation.

### Confusion matrix

Class order: Healthy, Stressed.

| Actual / Predicted | Healthy | Stressed |
|---|---:|---:|
| Healthy | 874 | 6 |
| Stressed | 1 | 143 |

The model correctly classified 1,017 of 1,024 test samples.

### Interpretation

The Spatial ViT achieved an F1-score of 0.9761 on the spatial test region.

Its training and test regions are separated by an excluded coordinate buffer. With 5 x 5 patches, the nearest training and test patch centers are at least five pixels apart along one coordinate axis. Each patch extends two pixels from its center, so the selected training and test patches do not share source pixels under this mask configuration.

This reduces direct patch overlap between the selected training and test samples. It does not establish independent validation across different scenes, locations, or acquisition conditions.

## 6. Hybrid CNN-Transformer Evaluation

### Training and evaluation configuration

| Setting | Value |
|---|---|
| Evaluation protocol | Spatial holdout |
| Training region | row < 28 OR column < 28 |
| Test region | row >= 32 AND column >= 32 |
| Training samples | 2,800 |
| Test samples | 1,024 |
| Patch size | 5 x 5 |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Transformer depth | 2 |
| Batch size | 32 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Random seed | 42 |
| Normalization | Training-region per-band statistics |
| Classification rule | Argmax over class probabilities |

The hybrid architecture combines local spectral-spatial feature extraction using a 3D CNN with spatial token relationship modelling using a Vision Transformer.

Normalization statistics are calculated from training-region pixels only and stored in the checkpoint.

### Confusion matrix

Class order: Healthy, Stressed.

| Actual / Predicted | Healthy | Stressed |
|---|---:|---:|
| Healthy | 880 | 0 |
| Stressed | 0 | 144 |

All 1,024 samples in this synthetic spatial test region were classified correctly.

### Interpretation

The Hybrid CNN-Transformer achieved 1.0000 across the reported classification metrics.

Perfect performance on this experiment should be interpreted cautiously. It represents one spatial holdout from one synthetic scene and does not establish generalization to independent real-world agricultural imagery.

## 7. Spatial Evaluation Considerations

The Spatial ViT and Hybrid CNN-Transformer use the same spatial training and test coordinate definitions.

The training region is defined by:

- Row coordinate less than 28, or
- Column coordinate less than 28.

The test region is defined by:

- Row coordinate at least 32, and
- Column coordinate at least 32.

Pixels outside these selected regions, including the buffer and any excluded labels, are not used as training or test samples in these experiments.

Both models use 5 x 5 patches. Based on the region masks and patch radius of two pixels, selected training and test patches do not share source pixels.

Nevertheless, the experiments share the same synthetic scene. A spatial holdout within one image is not equivalent to validation on independent agricultural fields or separate hyperspectral acquisitions.

## 8. Historical Model Experiments

Earlier development experiments included Logistic Regression and three Vision Transformer configurations.

| Historical model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ViT V1 (CLS token) | 0.8598 | 0.0000 | 0.0000 | 0.0000 |
| ViT V2 (Mean pooling) | 0.1402 | 0.1402 | 1.0000 | 0.2460 |
| ViT V3 (Standardized input) | 0.9976 | 1.0000 | 0.9826 | 0.9912 |

These are historical experiments and are not the current dashboard's three-model evaluation set.

The Logistic Regression experiment used a separate 80/20 random split. The earlier ViT V1, V2, and V3 experiments used a stratified 60/20/20 split.

ViT V3 changed both input standardization and training duration compared with V1 and V2. Its performance improvement cannot be attributed to standardization alone.

A separate historical spatial experiment reported Logistic Regression and Spatial ViT results. Its metrics should also be interpreted within that experiment's own protocol.

## 9. Scientific Limitations

1. All current experiments use synthetic mock hyperspectral data.
2. The experiments do not establish real-world crop disease detection performance.
3. CNN V2 uses a random pixel-level split and does not guarantee spatial independence between partitions.
4. Spatial ViT and Hybrid CNN-Transformer use one fixed spatial holdout from one synthetic scene.
5. A spatial holdout within one image does not establish generalization across independent locations, seasons, sensors, or acquisition conditions.
6. Perfect metrics from the Hybrid experiment require further investigation using independent data.
7. The different evaluation protocols and test sample counts prevent a direct controlled ranking of the three current models.
8. Synthetic class patterns may be easier to distinguish than real agricultural stress conditions.
9. The historical experiments use different configurations and should not be combined into a single controlled benchmark.

## 10. Recommended Future Validation

Further work should prioritize:

- Evaluation on genuine hyperspectral agricultural imagery with reliable reference labels.
- Testing on independent scenes and geographically separate locations.
- Repeated spatial holdout experiments.
- Independent test data that is not used for model selection or threshold tuning.
- Evaluation across different crops, growth stages, sensors, and environmental conditions.
- Reporting per-class metrics and class distributions alongside aggregate scores.
- Investigating the Hybrid model's perfect synthetic test result on more challenging independent datasets.

## 11. Conclusion

TerraSpectra demonstrates an end-to-end synthetic hyperspectral classification workflow involving data preprocessing, model training, evaluation, spatial analysis, explainability, and dashboard visualization.

The current CNN V2, Spatial ViT, and Hybrid CNN-Transformer experiments produce strong classification metrics on the synthetic dataset. Their results must be interpreted according to their individual evaluation protocols.

The spatial experiments reduce direct patch overlap through separated training and test regions, while the CNN experiment uses a random pixel-level split.

These results demonstrate experimental pipeline functionality, not validated real-world crop disease forecasting. Independent real hyperspectral imagery and geographically separate validation are necessary before making agricultural deployment claims.
