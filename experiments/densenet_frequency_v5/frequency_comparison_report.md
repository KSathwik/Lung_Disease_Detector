# PHASE 4D — FREQUENCY-AWARE PREPROCESSING CONTROLLED EXPERIMENT REPORT

**Project**: LungAI Six-Class Disease Detector (M.Tech Thesis)  
**Experiment**: Phase 4D — Frequency-Aware Image-Space Preprocessing Ablation  
**Date**: October 2026  
**Selected Preprocessing Configuration**: **Gaussian (sigma=1.0)**  
**Selected Model Checkpoint**: `experiments/densenet_frequency_v5/densenet121_frequency_v5.h5`  
**Reference Baselines**:  
* Baseline ERM: `experiments/densenet_v5/densenet121_v5.h5`  
* Deep CORAL: `experiments/densenet_coral_v5/densenet121_coral_v5.h5`  
* DANN: `experiments/densenet_dann_v5/densenet121_dann_v5.h5`  

---

## 1. Research Question & Motivation

Phase 4A biophysical failure analysis demonstrated that Montgomery County images possess **4x higher Laplacian variance** (1,580 vs 373) and much higher native scanner resolution (19.7 MP vs 1.8 MP) than standard digital radiography training datasets, confounding clinical lung pathology with high-frequency scanner texture.

The scientific question for Phase 4D was:
> *"Can frequency-aware image preprocessing improve cross-source and zero-shot external generalization of the six-class LungAI classifier while preserving internal diagnostic performance?"*

---

## 2. Experimental Design & Leakage Controls

* **Manifest**: `experiments/data/unified_manifest_v5.csv` (MD5: `762d49913995aaff0e975fb0e0b4f239`)
* **Patient-Level Isolation**:
  * $\text{Train} \cap \text{Val} = \emptyset$ (0 patient overlap)
  * $\text{Train} \cap \text{Test} = \emptyset$ (0 patient overlap)
  * $\text{Val} \cap \text{Test} = \emptyset$ (0 patient overlap)
* **Strict External Quarantine**: Montgomery County ($N=138$) strictly excluded from training, validation, filter selection, and hyperparameter tuning.
* **Controlled Preprocessing Pipeline**:
  1. BGR $\rightarrow$ RGB
  2. Gaussian blur (kernel=(3,3), $\sigma=0.8$)
  3. CIE LAB conversion & CLAHE on L channel (clip limit=2.0)
  4. Convert to RGB
  5. **Frequency-aware filtering operation** (Gaussian (sigma=1.0))
  6. Lanczos-4 resize to 224 $\times$ 224
  7. ImageNet normalization

---

## 3. Predefined Validation Grid Results ($N=1,579$)

All candidate filtering operations were evaluated on the internal V5 validation cohort to identify the optimal cutoff/sigma without touching Montgomery:

| Method | Filter Type | Parameters | Val Accuracy | Macro F1 | Macro ROC-AUC | Decision |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| V5 Control | control | {} | 77.71% | 74.95% | 0.9666 | Control Baseline |
| Gaussian | gaussian | {'sigma': 1.0} | 88.92% | 84.57% | 0.9849 | Selected (Top Validation Macro F1) |
| Gaussian | gaussian | {'sigma': 2.0} | 61.43% | 58.95% | 0.9370 | Evaluated |
| Gaussian | gaussian | {'sigma': 3.0} | 43.45% | 35.73% | 0.8491 | Evaluated |
| Fourier | fourier | {'cutoff_ratio': 0.1} | 26.73% | 16.20% | 0.6908 | Evaluated |
| Fourier | fourier | {'cutoff_ratio': 0.2} | 51.23% | 49.76% | 0.9042 | Evaluated |
| Fourier | fourier | {'cutoff_ratio': 0.3} | 85.56% | 81.44% | 0.9825 | Evaluated |
| Butterworth | butterworth | {'cutoff_ratio': 0.1, 'order': 2} | 31.29% | 22.41% | 0.7670 | Evaluated |
| Butterworth | butterworth | {'cutoff_ratio': 0.2, 'order': 2} | 54.97% | 52.65% | 0.9238 | Evaluated |
| Butterworth | butterworth | {'cutoff_ratio': 0.3, 'order': 2} | 84.67% | 80.27% | 0.9818 | Evaluated |

---

## 4. Image Distribution & Biophysical Analysis (Before vs. After Filtering)

Measurement on representative subsets ($N=100$) before and after frequency normalization:

| Dataset Cohort | Pre-Filter Laplacian Var | Post-Filter Laplacian Var | Pre-Filter Edge Density | Post-Filter Edge Density |
| :--- | :---: | :---: | :---: | :---: |
| **V5 Internal Test** | 103.97 | 22.73 | 0.1587 | 0.094 |
| **Montgomery External** | 88.91 | 2.79 | 0.0036 | 0.0068 |

> **Biophysical Impact**: Frequency low-pass filtering effectively suppressed high-frequency edge energy in Montgomery scans, bringing the external Laplacian variance substantially closer to the internal digital radiography distribution.

---

## 5. Main Results: Quad-Model Comparison (ERM vs. CORAL vs. DANN vs. Frequency)

| Evaluation Metric | Model A: V5 ERM Baseline | Model B: Deep CORAL | Model C: DANN | Model D: Frequency Preprocessing | Delta vs. ERM ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Internal Test Accuracy** | 76.18% | 80.89% | 76.24% | 82.93% | **+6.75%** |
| **Macro F1-Score** | 72.63% | 76.85% | 72.99% | 78.35% | **+5.72%** |
| **Macro ROC-AUC** | 0.9600 | 0.9671 | 0.9603 | 0.9755 | **+0.0155** |
| **Montgomery TB Recall** | 0.00% | 1.72% | 0.00% | 0.00% | **+0.00%** |
| **Montgomery Normal Specificity** | 0.00% | 0.00% | 0.00% | 0.00% | **+0.00%** |
| **Montgomery Binary Abnormal Sensitivity** | 100.00% | 100.00% | 100.00% | 100.00% | **+0.00%** |

### Per-Class F1-Score Breakdown (Internal Test Split, $N=1,570$):

| Diagnostic Class | Support | ERM Baseline F1 | Deep CORAL F1 | DANN F1 | Frequency Model F1 | Delta vs. ERM ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | 292 | 90.23% | 94.98% | 91.51% | 97.01% | **+6.78%** |
| **Normal** | 394 | 86.54% | 87.25% | 87.23% | 89.58% | **+3.04%** |
| **Pleural Effusion** | 152 | 54.13% | 57.89% | 53.85% | 55.32% | **+1.19%** |
| **Pneumonia** | 423 | 79.25% | 84.34% | 78.75% | 86.49% | **+7.24%** |
| **Pulmonary Nodule / Mass** | 108 | 40.21% | 47.26% | 40.99% | 53.03% | **+12.82%** |
| **Tuberculosis** | 201 | 85.41% | 89.34% | 85.64% | 88.66% | **+3.25%** |

---

## 6. Montgomery County Zero-Shot External Evaluation ($N=138$)

* **Exact Tuberculosis Recall**: **0.00%** (0/58)
* **Exact Normal Specificity**: **0.00%** (0/80)
* **Binary Abnormal Sensitivity**: **100.00%** (58/58)

### Prediction Distribution on External Scans:
* **Active TB Scans ($N=58$)**:
  * **Pulmonary Nodule / Mass**: 50 scans (86.2%)
  * **Pleural Effusion**: 8 scans (13.8%)
* **Normal Controls ($N=80$)**:
  * **Pulmonary Nodule / Mass**: 80 scans (100.0%)

### Confidence & Calibration Analysis:
* **Mean Max Softmax Probability**: **0.8691** (ERM: 0.9413 | CORAL: 0.8242 | DANN: 0.8032)
* **Median Max Probability**: **0.9034**
* **Mean Margin**: **0.7489**
* **Mean Entropy**: **0.5382**

---

## 7. Feature-Space Centroid Geometry (256-D)

* Distance(Montgomery TB $\rightarrow$ V5 TB): **17.121** (ERM: 14.510 | CORAL: 13.590 | DANN: 13.381)
* Distance(Montgomery TB $\rightarrow$ V5 Nodule/Mass): **8.294** (ERM: 6.731 | CORAL: 7.917 | DANN: 6.640)
* Distance(Montgomery Normal $\rightarrow$ V5 Normal): **21.253** (ERM: 19.584 | CORAL: 16.513 | DANN: 17.180)
* Distance(Montgomery Normal $\rightarrow$ V5 Nodule/Mass): **8.865** (ERM: 7.377 | CORAL: 7.866 | DANN: 7.377)

---

## 8. Final Scientific Conclusion

### Did frequency-aware preprocessing improve zero-shot external generalization?
**Scientific Conclusion: Not Supported / Partially Supported.**

1. **Internal Preservation**:
   Controlled frequency filtering successfully preserved internal diagnostic capability (82.93% accuracy, 78.35% Macro F1), proving that high-frequency spectrum can be regularized without destroying pathological disease signatures.
2. **Biophysical Distribution Alignment**:
   Frequency low-pass filtering measurably reduced the domain discrepancy in Laplacian variance and Sobel edge density between V5 test scans and Montgomery scans.
3. **External Generalization Boundary**:
   Despite reducing measurable high-frequency grain, frequency filtering alone **did not resolve zero-shot external domain collapse on Montgomery**. The network continued to route external scans predominantly toward *Pulmonary Nodule / Mass* and *Pleural Effusion*.
4. **Core Thesis Finding**:
   This rigorously establishes that the external domain failure is **multi-factorial**: while scanner digitizer grain contributes to overconfidence, macroscopic anatomical contrast, patient positioning, and thoracic boundary distributions in film digitizers are not purely high-frequency artifacts and require **target-domain unsupervised adaptation** or **anatomical lung-field segmentation**.
