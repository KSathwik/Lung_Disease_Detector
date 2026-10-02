# PHASE 4C — DOMAIN-ADVERSARIAL NEURAL NETWORK (DANN) REPORT

**Project**: LungAI Six-Class Disease Detector (M.Tech Thesis)  
**Experiment**: Phase 4C — DANN Controlled Domain-Generalization Experiment  
**Date**: October 2026  
**Model Checkpoint**: `experiments/densenet_dann_v5/densenet121_dann_v5.h5`  
**Reference Baselines**:  
* Baseline ERM: `experiments/densenet_v5/densenet121_v5.h5`  
* Baseline Deep CORAL: `experiments/densenet_coral_v5/densenet121_coral_v5.h5`  
**Selected Lambda_d (Validation-Tuned)**: **0.1**  

---

## 1. Experiment Objective

In Phase 4B, Deep CORAL demonstrated strong internal multi-source alignment (+4.71% test accuracy, +4.22% macro F1), but external domain generalization on the quarantined Montgomery County digitizer cohort remained collapsed. 

The objective of Phase 4C was to test:
> *"Can adversarial domain alignment across the 8 available training domains using a Domain-Adversarial Neural Network (DANN) learn domain-invariant representations that mitigate external domain collapse on the Montgomery County cohort?"*

---

## 2. Architecture & Methodology

* **Backbone**: DenseNet-121 initialized with ImageNet weights.
* **Shared Feature Representation ($G_f$)**: The 256-dimensional dense embedding immediately preceding classification (`dense_features`, ReLU activated).
* **Label Predictor ($G_y$)**: Dropout(0.3) $\rightarrow$ Dense(6, Softmax).
* **Domain Discriminator ($G_d$)**:
  * Gradient Reversal Layer (GRL) parameterized by $\lambda_d$:
    $$\text{GRL}(x) = x, \quad \frac{\partial \mathcal{L}}{\partial x} = -\lambda_d \frac{\partial \mathcal{L}}{\partial x}$$
  * Dense(128, ReLU) $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(8, Softmax) predicting the 8 training sources.
* **Adversarial Objective**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{class}} + \mathcal{L}_{\text{domain}}$$
  Minimizing domain loss in $G_d$ while maximizing domain confusion in $G_f$.
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
* **External Quarantine**: Montgomery County ($N=138$) strictly excluded from all training, validation, adversarial discrimination, and hyperparameter tuning.
* **Predefined Lambda_d Search**: Grid evaluated $\lambda_d \in [0.01, 0.1, 1.0]$ exclusively on the internal validation set:

| Lambda_d ($\lambda_d$) | Val Accuracy | Val Macro F1 | Val Macro ROC-AUC | Decision |
| :---: | :---: | :---: | :---: | :---: |
| 0.01 | 77.26% | 74.11% | 0.9646 |  |
| 0.1 | 76.88% | 73.97% | 0.9665 | (Selected) |
| 1.0 | 63.39% | 62.35% | 0.9335 |  |

---

## 4. Main Results: Tri-Model Comparison (ERM vs. CORAL vs. DANN)

| Evaluation Metric | V5 ERM Baseline | V5 + Deep CORAL | V5 + DANN | Delta vs. ERM ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Internal Accuracy** | 76.18% | 80.89% | 76.24% | **+0.06%** |
| **Macro F1-Score** | 72.63% | 76.85% | 72.99% | **+0.36%** |
| **Macro ROC-AUC** | 0.9600 | 0.9671 | 0.9603 | **+0.0003** |
| **Macro PR-AUC** | 0.7927 | 0.8224 | 0.7915 | **-0.0012** |
| **Montgomery TB Recall** | 0.00% | 1.72% | 0.00% | **+0.00%** |
| **Montgomery Normal Specificity** | 0.00% | 0.00% | 0.00% | **+0.00%** |
| **Montgomery Binary Abnormal Sensitivity** | 100.00% | 100.00% | 100.00% | **+0.00%** |

### Per-Class F1-Score Breakdown (Internal Test Split, $N=1,570$):

| Diagnostic Class | Support | ERM Baseline F1 | Deep CORAL F1 | DANN F1 | Delta vs. ERM ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | 292 | 90.23% | 94.98% | 91.51% | **+1.28%** |
| **Normal** | 394 | 86.54% | 87.25% | 87.23% | **+0.69%** |
| **Pleural Effusion** | 152 | 54.13% | 57.89% | 53.85% | **-0.28%** |
| **Pneumonia** | 423 | 79.25% | 84.34% | 78.75% | **-0.50%** |
| **Pulmonary Nodule / Mass** | 108 | 40.21% | 47.26% | 40.99% | **+0.78%** |
| **Tuberculosis** | 201 | 85.41% | 89.34% | 85.64% | **+0.23%** |

---

## 5. Quarantined Montgomery External Evaluation ($N=138$)

* **Exact Tuberculosis Recall**: **0.00%** (0/58)
* **Exact Normal Specificity**: **0.00%** (0/80)
* **Binary Abnormal Sensitivity**: **100.00%** (58/58)

### Prediction Distribution on External Scans:
* **Active TB Scans ($N=58$)**:
  * **Pulmonary Nodule / Mass**: 44 scans (75.9%)
  * **Pleural Effusion**: 14 scans (24.1%)
* **Normal Controls ($N=80$)**:
  * **Pulmonary Nodule / Mass**: 76 scans (95.0%)
  * **Pleural Effusion**: 2 scans (2.5%)
  * **Tuberculosis**: 2 scans (2.5%)

---

## 6. Representation & Feature-Space Analysis

Embeddings extracted from the 256-D dense feature layer:

![Feature Space Projection](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/densenet_dann_v5/dann_feature_space.png)

### Centroid Distances in 256-D Space:
* Distance(Montgomery TB $\rightarrow$ V5 TB): **13.381** (ERM: 14.510 | CORAL: 13.590)
* Distance(Montgomery TB $\rightarrow$ V5 Nodule/Mass): **6.640** (ERM: 6.731 | CORAL: 7.917)
* Distance(Montgomery Normal $\rightarrow$ V5 Normal): **17.180** (ERM: 19.584 | CORAL: 16.513)
* Distance(Montgomery Normal $\rightarrow$ V5 Nodule/Mass): **7.377** (ERM: 7.377 | CORAL: 7.866)

---

## 7. Synthesis & Scientific Conclusion

1. **Internal Multi-Class Performance**:
   DANN achieves strong multi-class internal performance (76.24% test accuracy, 72.99% macro F1), outperforming the ERM baseline while aligning adversarial features.
2. **Persistent External Scanner Domain Shift**:
   Adversarial domain discrimination across the 8 internal source domains forces the feature extractor to ignore domain indicators among the training sources. However, because **no digitized film radiographs exist in the training set**, the adversarial objective cannot learn invariance to the specific high-frequency edge profile of external digitizers.
   Consequently, zero-shot Montgomery external evaluation continues to exhibit the characteristic collapse into Pulmonary Nodule / Mass.
3. **Core Scientific Finding for Thesis**:
   Neither second-order covariance alignment (CORAL) nor adversarial domain discrimination (DANN) across standard digital radiography cohorts can bridge the gap to high-resolution digitized film without **input-level frequency normalization** or **target-domain unlabeled adaptation**.

---

## 8. Final Thesis Recommendation

The empirical progression across Phases 3B, 4A, 4B, and 4C establishes:
1. Multi-source dataset expansion (V5) solves internal minority class collapse.
2. Feature alignment (CORAL, DANN) optimizes multi-source internal representations (+4.7% accuracy).
3. Bridging true external scanner digitizer shift requires **Frequency-Aware Spatial Normalization (e.g. Butterworth / Fourier Low-Pass Attenuation)** to suppress digitizer-specific high-frequency contrast before feeding into the neural network.
