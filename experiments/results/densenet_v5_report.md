# Controlled DenseNet-121 Baseline Evaluation Report (Dataset V5)

**Project**: LungAI Six-Class Disease Detector (M.Tech Thesis)  
**Experiment**: Phase 3B Component 2 — DenseNet-121 Baseline on Unified Dataset V5  
**Date**: October 2026  
**Artifact Metrics**: `experiments/results/densenet_v5_metrics.json`  
**Model Checkpoint**: `experiments/densenet_v5/densenet121_v5.h5`  

---

## 1. Overall Internal Test Performance ($N=1,570$)

| Metric | DenseNet-121 (V3 Full) | DenseNet-121 (V5 Full) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 78.53% | **76.18%** | **-2.35%** |
| **Macro Precision** | 74.26% | **74.82%** | **+0.56%** |
| **Macro Recall** | 75.27% | **74.62%** | **-0.65%** |
| **Macro F1-Score** | 73.28% | **72.63%** | **-0.65%** |
| **Weighted F1-Score** | 80.54% | **78.79%** | **-1.75%** |
| **Macro ROC-AUC** | 0.9677 | **0.9600** | **-0.0077** |
| **Macro PR-AUC** | 0.8015 | **0.7927** | **-0.0088** |

---

## 2. Per-Class Performance Breakdown (V5 Internal Test Split)

| Diagnostic Class | Support | Precision | Recall | Specificity | F1-Score | V3 F1 | Delta F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | 292 | 100.00% | 82.19% | 100.00% | **90.23%** | 91.85% | **-1.62%** |
| **Normal** | 394 | 89.22% | 84.01% | 96.60% | **86.54%** | 90.00% | **-3.46%** |
| **Pleural Effusion** | 152 | 47.74% | 62.50% | 92.67% | **54.13%** | 58.24% | **-4.11%** |
| **Pneumonia** | 423 | 92.16% | 69.50% | 97.82% | **79.25%** | 79.84% | **-0.59%** |
| **Pulmonary Nodule / Mass** | 108 | 28.30% | 69.44% | 87.00% | **40.21%** | 36.23% | **+3.98%** |
| **Tuberculosis** | 201 | 91.48% | 80.10% | 98.90% | **85.41%** | 83.55% | **+1.86%** |

---

## 3. Quarantined Montgomery External Evaluation ($N=138$)

* **Exact Tuberculosis Recall**: **0.00%** (0/58)
* **Exact Normal Specificity**: **0.00%** (0/80)
* **Binary Abnormal Sensitivity**: **100.00%** (58/58)

### Predictions on Active TB Cases ($N=58$):
* **Pulmonary Nodule / Mass**: 48 scans (82.8%)
* **Pleural Effusion**: 10 scans (17.2%)

### Predictions on Normal Controls ($N=80$):
* **Pulmonary Nodule / Mass**: 79 scans (98.8%)
* **Pleural Effusion**: 1 scans (1.2%)

---

## 4. Key Thesis Insights & Phase 3B Conclusion

1. **Multi-Source Diversity Benefit**:
   Adding the expanded NIH cohort in V5 provides multi-source grounding for both Pleural Effusion (VinDr + NIH) and Pulmonary Nodule / Mass (VinDr + JSRT + NIH).
2. **Persistent Montgomery Domain Shift**:
   Despite the increased training diversity in V5, the model still exhibits acquisition-dependence when tested on unaligned external cohorts (Montgomery County), confirming that dataset size expansion alone cannot solve domain shift.
3. **Firm Rationale for Domain Adaptation (Phase 4)**:
   This empirical evidence establishes the core thesis hypothesis: domain-invariant feature learning (such as Domain Adversarial Training / CORAL / MMD) is necessary to achieve true clinical generalization across institutions.
