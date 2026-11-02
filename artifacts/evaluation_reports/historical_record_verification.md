# Recorded evaluation results — verification

- Generated: 2026-11-02T16:11:50+00:00 by `python -m lungai.evaluation.historical` (lungai 1.0.0). Do not edit by hand.
- Evidence level: RECORDED: aggregate results from earlier runs. Threshold metrics are checked for internal consistency against the stored confusion matrices. They cannot be reproduced: no checkpoint, image or per-image prediction is in the repository. The internal test split has known patient leakage (see artifacts/data_validation).

## Internal test split (V5, N = 1,570) — recomputed from stored confusion matrices

| Run | Accuracy (95% CI) | Macro P | Macro R | Macro F1 | Weighted F1 | Macro ROC-AUC (recorded, unverifiable) | Consistency |
|---|---|---:|---:|---:|---:|---:|---|
| **Canonical (DenseNet-121 + Gaussian low-pass preprocessing)** | 82.93% (80.99%–84.71%) | 79.31% | 80.73% | 78.35% | 84.18% | 0.9755 | consistent (0 mismatches / 29 checks) |
| Historical: DenseNet-121 ERM baseline (no low-pass filter) | 76.18% (74.01%–78.22%) | 74.82% | 74.62% | 72.63% | 78.79% | 0.96 | consistent (0 mismatches / 29 checks) |
| Historical: DenseNet-121 + Deep CORAL | 80.89% (78.87%–82.76%) | 77.69% | 78.93% | 76.85% | 82.58% | 0.9671 | consistent (0 mismatches / 29 checks) |
| Historical: DenseNet-121 + DANN | 76.24% (74.07%–78.28%) | 75.78% | 75.30% | 72.99% | 79.12% | 0.9603 | consistent (0 mismatches / 29 checks) |
| Historical: DenseNet-121 + low-pass + CORAL | 78.22% (76.11%–80.19%) | 75.57% | 76.50% | 73.95% | 80.19% | 0.9609 | consistent (0 mismatches / 29 checks) |

## Canonical run — per class (internal test)

| Class | Support | Precision | Recall (95% CI) | Specificity | F1 |
|---|---:|---:|---|---:|---:|
| COVID-19 | 292 | 0.9964 | 0.9452 (0.913–0.966) | 0.9992 | 0.9701 |
| Normal | 394 | 0.9086 | 0.8832 (0.848–0.911) | 0.9702 | 0.8958 |
| Pleural Effusion | 152 | 0.6000 | 0.5132 (0.434–0.591) | 0.9633 | 0.5532 |
| Pneumonia | 423 | 0.9492 | 0.7943 (0.753–0.830) | 0.9843 | 0.8649 |
| Pulmonary Nodule / Mass | 108 | 0.3849 | 0.8519 (0.773–0.907) | 0.8995 | 0.5303 |
| Tuberculosis | 201 | 0.9198 | 0.8557 (0.800–0.898) | 0.9890 | 0.8866 |

Confusion matrix (rows = true, columns = predicted, canonical class order):

```
COVID-19                 [276, 0, 0, 10, 0, 6]
Normal                   [0, 348, 4, 6, 33, 3]
Pleural Effusion         [0, 0, 78, 0, 74, 0]
Pneumonia                [0, 29, 28, 336, 24, 6]
Pulmonary Nodule / Mass  [0, 1, 15, 0, 92, 0]
Tuberculosis             [1, 5, 5, 2, 16, 172]
```

## External cohort (Montgomery County, binary) — recomputed from stored prediction breakdowns

| Run | N (TB / normal) | Sensitivity, predicted ≠ Normal (95% CI) | Specificity (95% CI) | Exact TB recall | Interpretation |
|---|---|---|---|---|---|
| Canonical (DenseNet-121 + Gaussian low-pass preprocessing) | 138 (58 / 80) | 58/58 = 100.00% (93.79%–100.00%) | 0/80 = 0.00% (0.00%–4.58%) | 0/58 | Degenerate: every scan, including every normal scan, was predicted abnormal. Sensitivity 58/58 carries no screening value with specificity 0/80. |
| Historical: DenseNet-121 ERM baseline (no low-pass filter) | 138 (58 / 80) | 58/58 = 100.00% (93.79%–100.00%) | 0/80 = 0.00% (0.00%–4.58%) | 0/58 | Degenerate: every scan, including every normal scan, was predicted abnormal. Sensitivity 58/58 carries no screening value with specificity 0/80. |
| Historical: DenseNet-121 + Deep CORAL | 138 (58 / 80) | 58/58 = 100.00% (93.79%–100.00%) | 0/80 = 0.00% (0.00%–4.58%) | 1/58 | Degenerate: every scan, including every normal scan, was predicted abnormal. Sensitivity 58/58 carries no screening value with specificity 0/80. |
| Historical: DenseNet-121 + DANN | 138 (58 / 80) | 58/58 = 100.00% (93.79%–100.00%) | 0/80 = 0.00% (0.00%–4.58%) | 0/58 | Degenerate: every scan, including every normal scan, was predicted abnormal. Sensitivity 58/58 carries no screening value with specificity 0/80. |
| Historical: DenseNet-121 + low-pass + CORAL | 138 (58 / 80) | 56/58 = 96.55% (88.27%–99.05%) | 0/80 = 0.00% (0.00%–4.58%) | 0/58 | No normal scan was classified Normal (specificity 0/80); the model does not separate normal from TB scans. |

## Validation-set evidence available for model selection

| Run | Source | Evidence |
|---|---|---|
| Canonical (DenseNet-121 + Gaussian low-pass preprocessing) | `experiments/densenet_frequency_v5/frequency_validation_grid.json` | Filter chosen on validation: the ERM baseline checkpoint evaluated on val under 10 preprocessing filters; Gaussian sigma=1.0 had the highest val macro-F1 (0.8457 vs 0.7495 unfiltered). The trained model's own validation score was logged but not saved. |
| Historical: DenseNet-121 ERM baseline (no low-pass filter) | `experiments/densenet_v5/training_history.json` | Final-epoch val accuracy 0.7771 (epoch before: 0.8056). |
| Historical: DenseNet-121 + Deep CORAL | `experiments/densenet_coral_v5/coral_lambda_search.json` | Selected lambda 0.01: val accuracy 0.8151, val macro-F1 0.7771. |
| Historical: DenseNet-121 + DANN | `experiments/densenet_dann_v5/dann_lambda_search.json` | Selected lambda 0.1: val accuracy 0.7688, val macro-F1 0.7397. |
| Historical: DenseNet-121 + low-pass + CORAL | `experiments/densenet_hybrid_v5/hybrid_config.json` | No validation score recorded. |
