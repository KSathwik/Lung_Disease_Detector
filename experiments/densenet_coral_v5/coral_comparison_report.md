# PHASE 4B — DEEP CORAL CONTROLLED DOMAIN-GENERALIZATION REPORT

**Project**: LungAI Six-Class Disease Detector (M.Tech Thesis)  
**Experiment**: Phase 4B — Deep CORAL Controlled Domain-Generalization Experiment  
**Date**: October 2026  
**Model Checkpoint**: `experiments/densenet_coral_v5/densenet121_coral_v5.h5`  
**Reference Baseline**: `experiments/densenet_v5/densenet121_v5.h5` (Frozen V5 ERM)  
**Selected Lambda (Validation-Tuned)**: **0.01**  

---

## 1. Experiment Objective

Phase 4A identified substantial source-dependent distribution differences across training cohorts, including differences in image resolution, intensity characteristics, and high-frequency image structure, that were associated with the observed external-domain failure on the Montgomery County cohort.

The objective of Phase 4B was to test the following research question experimentally:
> *"Can feature-distribution alignment across the available training domains using Deep CORAL improve cross-source generalization compared with the standard DenseNet-121 empirical-risk-minimization baseline?"*

---

## 2. Method & Architecture

* **Backbone**: DenseNet-121 initialized with ImageNet weights.
* **Feature Representation for Alignment**: The 256-dimensional dense embedding immediately preceding the 6-class softmax classification layer (`dense_features`, ReLU activated).
* **Deep CORAL Formulation**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{classification}} + \lambda \times \mathcal{L}_{\text{CORAL}}$$
  $$\mathcal{L}_{\text{CORAL}} = \frac{1}{4 d^2} \|C_s - C_t\|_F^2$$
  where $d=256$, and $C_s, C_t$ are empirical covariance matrices of feature vectors sampled from distinct source domains $D_s$ and $D_t$.
* **Domain-Pair Sampling Strategy**: In each training step, two distinct source domains are sampled with probability proportional to their representation in the training cohort. Mini-batches of 16 images per domain are processed simultaneously.
* **Controlled Training Schedule**:
  * Phase 1: 2 epochs (backbone frozen, Adam, lr = $10^{-3}$).
  * Phase 2: 2 epochs (top 30 DenseNet layers unfrozen, Adam, lr = $10^{-5}$).
  * Total duration: 4 epochs (strictly matching the baseline schedule).

---

## 3. Data & Leakage Controls

* **Manifest**: `experiments/data/unified_manifest_v5.csv` (MD5: `762d49913995aaff0e975fb0e0b4f239`)
* **Patient-Level Isolation**:
  * $\text{Train} \cap \text{Val} = \emptyset$ (0 patient overlap)
  * $\text{Train} \cap \text{Test} = \emptyset$ (0 patient overlap)
  * $\text{Val} \cap \text{Test} = \emptyset$ (0 patient overlap)
* **External Quarantine**: Montgomery County ($N=138$) was strictly excluded from training, validation, feature alignment, and hyperparameter selection.
* **Predefined Lambda Search**: Grid search evaluated $\lambda \in [0.01, 0.1, 1.0]$ exclusively on the internal validation set:

| Lambda ($\lambda$) | Val Accuracy | Val Macro F1 | Val Macro ROC-AUC | Decision |
| :---: | :---: | :---: | :---: | :---: |
| 0.01 | 81.51% | 77.71% | 0.9722 | (Selected) |
| 0.1 | 81.63% | 77.36% | 0.9679 |  |
| 1.0 | 81.32% | 76.96% | 0.9684 |  |

---

## 4. Main Results: V5 ERM Baseline vs. V5 + Deep CORAL

| Evaluation Metric | V5 ERM Baseline | V5 + Deep CORAL | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Internal Accuracy** | 76.18% | 80.89% | **+4.71%** |
| **Macro F1-Score** | 72.63% | 76.85% | **+4.22%** |
| **Macro ROC-AUC** | 0.9600 | 0.9671 | **+0.0071** |
| **Macro PR-AUC** | 0.7927 | 0.8224 | **+0.0297** |
| **Montgomery TB Recall** | 0.00% | 1.72% | **+1.72%** |
| **Montgomery Normal Specificity** | 0.00% | 0.00% | **+0.00%** |
| **Montgomery Binary Abnormal Sensitivity** | 100.00% | 100.00% | **+0.00%** |

### Per-Class F1-Score Breakdown (Internal Test Split, $N=1,570$):

| Diagnostic Class | Support | ERM Baseline F1 | Deep CORAL F1 | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **COVID-19** | 292 | 90.23% | 94.98% | **+4.75%** |
| **Normal** | 394 | 86.54% | 87.25% | **+0.71%** |
| **Pleural Effusion** | 152 | 54.13% | 57.89% | **+3.76%** |
| **Pneumonia** | 423 | 79.25% | 84.34% | **+5.09%** |
| **Pulmonary Nodule / Mass** | 108 | 40.21% | 47.26% | **+7.05%** |
| **Tuberculosis** | 201 | 85.41% | 89.34% | **+3.93%** |

---

## 5. Quarantined Montgomery External Evaluation ($N=138$)

* **Exact Tuberculosis Recall**: **1.72%** (1/58)
* **Exact Normal Specificity**: **0.00%** (0/80)
* **Binary Abnormal Sensitivity**: **100.00%** (58/58)

### Prediction Distribution on External Scans:
* **Active TB Scans ($N=58$)**:
  * **Pulmonary Nodule / Mass**: 42 scans (72.4%)
  * **Pleural Effusion**: 15 scans (25.9%)
  * **Tuberculosis**: 1 scans (1.7%)
* **Normal Controls ($N=80$)**:
  * **Pulmonary Nodule / Mass**: 72 scans (90.0%)
  * **Tuberculosis**: 6 scans (7.5%)
  * **Pleural Effusion**: 2 scans (2.5%)

---

## 6. Representation & Feature-Space Analysis

Embeddings extracted from the 256-D dense feature layer:

![Feature Space Projection](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/densenet_coral_v5/coral_feature_space.png)

### Centroid Distances in 256-D Space:
* Distance(Montgomery TB $\rightarrow$ V5 TB): **13.590** (Baseline: 14.510)
* Distance(Montgomery TB $\rightarrow$ V5 Nodule/Mass): **7.917** (Baseline: 6.731)
* Distance(Montgomery Normal $\rightarrow$ V5 Normal): **16.513** (Baseline: 19.584)
* Distance(Montgomery Normal $\rightarrow$ V5 Nodule/Mass): **7.866** (Baseline: 7.377)

---

## 7. Interpretation & Scientific Conclusion

1. **Internal Performance Maintained**:
   Deep CORAL preserves strong internal multi-class classification (80.89% accuracy, 0.9671 macro ROC-AUC).
2. **External Domain Shift Remains Unresolved**:
   Deep correlation alignment across the *training* source domains (VinDr, NIH, TBX11K, Guangzhou) was **insufficient to bridge the domain gap to the unseen Montgomery digitizer scanner**.
   Montgomery active TB cases continue to cluster adjacent to Pulmonary Nodule / Mass in feature space, resulting in 0.00% exact TB recall and 0.00% exact normal specificity, while maintaining 100.00% binary abnormal detection.
3. **Scientific Implication**:
   Second-order covariance alignment across heterogeneous source domains with non-overlapping label distributions cannot disentangle scanner-specific high-frequency contrast without explicit domain-adversarial invariance or cross-domain normalization.

---

## 8. Limitations

* **Label Sourcing Confounding**: Classes like COVID-19 and Tuberculosis are predominantly single-source in training, meaning covariance alignment can inadvertently align disease features with domain artifacts.
* **Missing Projection Information**: Montgomery projection metadata remains unrecorded.
* **Single Unseen Benchmark**: Montgomery represents one specific digitizer technology.

---

## 9. Recommendation for Phase 4C

Based on the empirical findings of Phase 4B:
* Deep CORAL (covariance alignment) does not alter the fundamental failure mode on digitized film radiographs.
* We recommend exploring **Domain-Adversarial Neural Networks (DANN)** or **Frequency-Aware Normalization (Fourier Domain Adaptation / High-Pass Attenuation)** in Phase 4C to explicitly suppress digitizer high-frequency features.
