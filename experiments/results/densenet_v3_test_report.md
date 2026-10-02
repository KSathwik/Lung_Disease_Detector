# DenseNet-121 V3 Comprehensive Test Report

**Model**: DenseNet-121  
**Dataset**: `experiments/data/unified_manifest_v3.csv`  
**Test Samples**: 1,514  
**Test Accuracy**: **80.25%**  
**Macro F1-Score**: **75.10%**  
**Weighted F1-Score**: **82.01%**  
**Macro ROC-AUC**: **97.13%**  
**Macro PR-AUC**: **81.99%**  

---

## Per-Class Results

| Class Name | Support | Precision | Recall | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | 292 | 100.00% | 84.93% | 100.00% | **91.85%** |
| **Normal** | 370 | 92.24% | 90.00% | 97.55% | **91.11%** |
| **Pleural Effusion** | 142 | 52.63% | 56.34% | 94.75% | **54.42%** |
| **Pneumonia** | 420 | 90.50% | 77.14% | 96.89% | **83.29%** |
| **Pulmonary Nodule / Mass** | 89 | 32.86% | 78.65% | 89.96% | **46.36%** |
| **Tuberculosis** | 201 | 87.91% | 79.60% | 98.32% | **83.55%** |

---

## Top Failure & Confusion Modes

| Rank | True Class $\rightarrow$ Predicted Class | Confusion Count |
| :---: | :--- | :---: |
| 1 | `Pleural Effusion -> Pulmonary Nodule / Mass` | **62** |
| 2 | `Pneumonia -> Pulmonary Nodule / Mass` | **35** |
| 3 | `Pneumonia -> Pleural Effusion` | **34** |
| 4 | `Normal -> Pulmonary Nodule / Mass` | **21** |
| 5 | `COVID-19 -> Pneumonia` | **20** |
| 6 | `Tuberculosis -> Pulmonary Nodule / Mass` | **20** |
| 7 | `Pneumonia -> Normal` | **19** |
| 8 | `Pulmonary Nodule / Mass -> Pleural Effusion` | **19** |

---

## Montgomery External Validation Summary
* **Exact TB Recall**: **0.00%**
* **Exact Normal Specificity**: **0.00%**
* **Binary Abnormal Sensitivity**: **100.00%**
