# Experiment results (recorded)

The values on this page are taken from `artifacts/evaluation_reports/historical_record_verification.json`, which `python -m lungai.evaluation.historical` regenerates. `tests/test_docs_metrics.py` checks that the figures here and in the README match that file.

**Evidence level for every number here:** RECORDED. Each is an aggregate from a single earlier run (seed 42, 4 epochs). Threshold metrics were recomputed from the stored confusion matrices and all match. ROC-AUC values are as recorded and unverifiable. All runs share the same patient-leaked V5 test split.

## Internal test split (V5, N = 1,570)

| Run | Role | Accuracy (95% CI) | Macro F1 | Weighted F1 | Macro ROC-AUC (recorded) |
|---|---|---|---:|---:|---:|
| DenseNet-121 + Gaussian low-pass preprocessing | **Canonical** | 82.93% (80.99–84.71) | 78.35% | 84.18% | 0.9755 |
| DenseNet-121 + Deep CORAL | Historical | 80.89% (78.87–82.76) | 76.85% | 82.58% | 0.9671 |
| DenseNet-121 + low-pass + CORAL | Historical | 78.22% (76.11–80.19) | 73.95% | 80.19% | 0.9609 |
| DenseNet-121 + DANN | Historical | 76.24% (74.07–78.28) | 72.99% | 79.12% | 0.9603 |
| DenseNet-121 ERM baseline | Historical | 76.18% (74.01–78.22) | 72.63% | 78.79% | 0.9600 |

The baseline's 76.18% accuracy and 0.9600 ROC-AUC are often quoted for "the model". They belong to the ERM baseline, not to the canonical model.

## External cohort (Montgomery County, binary)

| Run | Sensitivity (predicted ≠ Normal) | Specificity | Exact TB recall |
|---|---|---|---|
| Canonical | 58/58 | **0/80** | 0/58 |
| Deep CORAL | 58/58 | 0/80 | 1/58 |
| low-pass + CORAL | 56/58 | 0/80 | 0/58 |
| DANN | 58/58 | 0/80 | 0/58 |
| ERM baseline | 58/58 | 0/80 | 0/58 |

No run classified any normal external scan as Normal.

## Validation evidence (the only legitimate basis for selection)

| Run | What was recorded on validation |
|---|---|
| Canonical | Filter search: the baseline checkpoint with a σ = 1.0 Gaussian low-pass scored val macro-F1 0.8457, against 0.7495 unfiltered. The trained model's own validation score was not saved |
| ERM baseline | Final-epoch val accuracy 0.7771; the epoch before scored 0.8056 |
| Deep CORAL | λ = 0.01: val accuracy 0.8151, macro-F1 0.7771 |
| DANN | λ = 0.1: val accuracy 0.7688, macro-F1 0.7397 |
| low-pass + CORAL | None recorded |

## Corrections to earlier documents

- **Comparator tables:** the benchmark table in earlier drafts differed from the result files in 13 of 35 cells, and every difference made the comparison runs look worse. The tables above come from the files.
- **The "58/58 sensitivity" claim:** earlier drafts presented it as screening ability. It is a degenerate result, because specificity is 0/80.
- **Thesis figures:** Figures 4, 6 and 7 in the archived thesis drafts show the ERM baseline but are captioned as the canonical model. See [docs/archive/README.md](archive/README.md).
