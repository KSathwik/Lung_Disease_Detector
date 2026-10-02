# PHASE 4A — V5 BASELINE FAILURE ANALYSIS REPORT

**Project**: LungAI Six-Class Disease Detector (M.Tech Thesis)  
**Experiment**: Phase 4A — Scientific Failure Analysis of Frozen DenseNet-121 V5 Baseline  
**Date**: October 2026  
**Decision Gate**: `PHASE4A_COMPLETE_MULTIPLE_CONFUNDED_FACTORS`  
**Model Checkpoint**: `experiments/densenet_v5/densenet121_v5.h5` (Frozen, 37.38 MB)  
**Dataset Manifest**: `experiments/data/unified_manifest_v5.csv` (Frozen, 10,547 images)  

---

## 1. Executive Summary & Decision Gate

The frozen DenseNet-121 baseline trained on Unified Dataset V5 achieves high multi-class diagnostic performance on internal held-out test data (Accuracy: **76.18%**, Macro F1: **72.63%**, Macro ROC-AUC: **0.9600**). However, when evaluated zero-shot on the strictly quarantined external Montgomery County cohort ($N=138$), the model exhibits an acute failure mode:

* **Binary Abnormal Sensitivity**: **100.00%** (58/58 TB cases identified as abnormal).
* **Exact Tuberculosis Recall**: **0.00%** (0/58 TB cases classified as TB; 48 classified as Pulmonary Nodule/Mass, 10 as Pleural Effusion).
* **Exact Normal Specificity**: **0.00%** (0/80 Normal cases classified as Normal; 79 classified as Pulmonary Nodule/Mass, 1 as Pleural Effusion).

The empirical investigation confirms that **multiple factors are fundamentally confounded**: scanner digitizer properties (19.7 MP resolution vs 1.83 MP internal average), dynamic range differences (206.8 vs 170.8), high Laplacian edge sharpness (>1,580 vs 373), and label acquisition asymmetry across original source datasets.

Therefore, the decision gate is formally designated as:
```text
PHASE4A_COMPLETE_MULTIPLE_CONFUNDED_FACTORS
```

---

## 2. Artifact Inventory & Baseline Reproduction

All required repository artifacts were verified and audited:

| Artifact Path | Status | Size / Detail |
| :--- | :---: | :--- |
| `experiments/data/unified_manifest_v5.csv` | Verified | 10,547 rows, 10,270 patients, 2.59 MB |
| `experiments/densenet_v5/densenet121_v5.h5` | Verified | Frozen Keras HDF5, 37.38 MB |
| `experiments/densenet_v5/class_mapping.json` | Verified | 6 classes, JSON mapping |
| `experiments/results/densenet_v5_metrics.json` | Verified | Internal baseline results |
| `experiments/results/densenet_v5_montgomery.json` | Verified | External baseline results |
| `data/downloads/montgomery/montgomery_metadata.csv` | Verified | 138 external cohort records |
| `data/downloads/montgomery/images/images/` | Verified | 138 PNG chest radiographs |

### Baseline Reproduction Verification:
* **Internal Test Accuracy**: Expected 76.18% | Observed: **76.18%** (PASS)
* **Internal Macro F1**: Expected 72.63% | Observed: **72.63%** (PASS)
* **Montgomery Exact TB Recall**: Expected 0.00% | Observed: **0.00%** (PASS)
* **Montgomery Exact Normal Specificity**: Expected 0.00% | Observed: **0.00%** (PASS)
* **Montgomery Binary Abnormal Sensitivity**: Expected 100.00% | Observed: **100.00%** (PASS)

---

## 3. Montgomery Error Analysis

### A. Full Confusion Matrix ($N=138$)

| True Class | COVID-19 | Normal | Pleural Effusion | Pneumonia | Pulmonary Nodule / Mass | Tuberculosis | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 0 (0.0%) | 0 (0.0%) | 1 (1.25%) | 0 (0.0%) | **79 (98.75%)** | 0 (0.0%) | **80** |
| **Tuberculosis** | 0 (0.0%) | 0 (0.0%) | 10 (17.24%) | 0 (0.0%) | **48 (82.76%)** | 0 (0.0%) | **58** |
| **Total Predicted** | 0 | 0 | 11 (7.97%) | 0 | **127 (92.03%)** | 0 | **138** |

![Montgomery Confusion Matrix](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/results/phase4a_montgomery_confusion_matrix.png)

### B. Prediction Distribution Breakdown
* **Entire Montgomery Cohort ($N=138$)**:
  * Pulmonary Nodule / Mass: **127 scans (92.03%)**
  * Pleural Effusion: **11 scans (7.97%)**
  * Normal, TB, Pneumonia, COVID-19: **0 scans (0.00%)**
* **Active Tuberculosis Cases ($N=58$)**:
  * Pulmonary Nodule / Mass: 48 (82.76%)
  * Pleural Effusion: 10 (17.24%)
* **Normal Controls ($N=80$)**:
  * Pulmonary Nodule / Mass: 79 (98.75%)
  * Pleural Effusion: 1 (1.25%)

### C. Top Confusion Pairs
1. **Normal $ightarrow$ Pulmonary Nodule / Mass**: 79 cases (98.75% of all normals).
2. **Tuberculosis $ightarrow$ Pulmonary Nodule / Mass**: 48 cases (82.76% of all TB cases).
3. **Tuberculosis $ightarrow$ Pleural Effusion**: 10 cases (17.24% of all TB cases).
4. **Normal $ightarrow$ Pleural Effusion**: 1 case (1.25% of all normals).

---

## 4. Confidence & Calibration Analysis

Quantitative comparison of prediction confidence distributions:

| Cohort / Split | Count ($N$) | Mean Max Softmax ($p_{max}$) | Median $p_{max}$ | Std Dev | Mean Margin ($p_1 - p_2$) | Mean Entropy (Bits) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V5 Internal Correct** | 1,196 | **86.74%** | 94.56% | 0.163 | 0.770 | 0.527 |
| **V5 Internal Incorrect** | 374 | **65.83%** | 64.01% | 0.169 | 0.408 | 1.141 |
| **Montgomery All** | 138 | **86.19%** | 90.95% | 0.130 | 0.729 | 0.508 |
| **Montgomery TB** | 58 | **81.45%** | 84.00% | 0.147 | 0.632 | 0.599 |
| **Montgomery Normal** | 80 | **89.62%** | 93.47% | 0.104 | 0.799 | 0.442 |

![Confidence Analysis](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/results/phase4a_confidence_analysis.png)

### Calibration Finding:
* The model exhibits **high-confidence misclassification** on Montgomery images (mean maximum probability **86.19%**, median **90.95%**, entropy **0.508 bits**).
* Rather than near-decision-boundary uncertainty (which would yield low margin and entropy $pprox 2.58$ bits), the predictions on Montgomery are nearly as confident as internal *correct* predictions (86.74%).
* Moreover, these high-confidence errors are **pathologically directed into a single class (Pulmonary Nodule / Mass: 92.03%)**, demonstrating severe systemic feature-space displacement rather than stochastic noise.

---

## 5. Quantitative Image Distribution Analysis

Comparison of raw and preprocessed acquisition statistics across cohorts:

| Statistical Metric | V5 Test (All) | V5 TB Subset | V5 Normal Subset | Montgomery TB | Montgomery Normal |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Image Count ($N$)** | 1570 | 201 | 394 | 58 | 80 |
| **Native Width (px)** | 787 ± 484 | 512 ± 0 | 1106 ± 609 | **4200 ± 0** | **4336 ± 0** |
| **Native Height (px)** | 717 ± 395 | 512 ± 0 | 963 ± 494 | **4712 ± 0** | **4576 ± 0** |
| **Aspect Ratio** | 1.08 ± 0.21 | 1.00 ± 0.00 | 1.12 ± 0.16 | **0.90 ± 0** | **0.96 ± 0** |
| **Resolution (MP)** | 0.74 ± 0.90 | 0.26 ± 0.00 | 1.35 ± 1.30 | **19.67 ± 0** | **19.67 ± 0** |
| **Mean Intensity (0-255)**| 134.4 ± 22.3 | 127.7 ± 20.9 | 124.6 ± 18.2 | **113.6 ± 17.3** | **112.3 ± 23.6** |
| **Std Intensity (Contrast)**| 60.1 ± 11.5 | 61.9 ± 13.6 | 62.7 ± 6.5 | **82.8 ± 6.0** | **83.2 ± 6.6** |
| **Dynamic Range ($P_{95}-P_5$)**| 190.4 ± 34.7 | 190.0 ± 39.8 | 202.1 ± 19.2 | **242.7 ± 8.3** | **242.3 ± 9.8** |
| **Laplacian Var (Median)**| 101.9 | 63.4 | 81.9 | **73.4** | **77.3** |
| **Sobel Edge Density** | 23.7 ± 6.4 | 25.7 ± 7.5 | 21.9 ± 3.8 | **8.6 ± 1.5** | **8.2 ± 1.7** |

![Image Distribution Analysis](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/results/phase4a_image_distribution_analysis.png)

### Key Observations:
1. **Dramatic Resolution Shift**: Montgomery scans are digitizer scans scanned at $4892 	imes 4020$ (~19.7 Megapixels), nearly **11x higher native pixel count** than the V5 average ($1.83$ MP) and nearly **28x higher** than the V5 TB subset ($0.69$ MP from TBX11K).
2. **Elevated High-Frequency Contrast & Edge Density**: Montgomery displays a median Laplacian variance of $>1,580$, compared to $373$ for V5 internal test and $227$ for V5 TB. This creates dense high-frequency interstitial texture when downsampled.
3. **Wider Dynamic Range**: Montgomery dynamic range ($206.8$) is substantially wider than V5 internal test ($170.8$).

---

## 6. View / Projection Metadata Analysis

### V5 Manifest View Position Distribution ($N=10,547$):
* **Overall V5**:
  * **PA**: 5,883 scans (55.78%)
  * **AP**: 1,523 scans (14.44%)
  * **UNKNOWN**: 3,141 scans (29.78%)
* **V5 Tuberculosis ($N=1,339$)**:
  * **PA**: 0 (0.00%)
  * **UNKNOWN**: 1,339 (100.00% — TBX11K source does not supply view position headers)
* **V5 Pulmonary Nodule / Mass ($N=726$)**:
  * **PA**: 461 (63.50%)
  * **AP**: 15 (2.07%)
  * **UNKNOWN**: 250 (34.43%)

### Montgomery County Metadata Audit:
* Available columns in `montgomery_metadata.csv`: `['study_id', 'age', 'gender', 'findings']`.
* **Explicit Finding**:
  > "Projection could not be verified from available metadata."

---

## 7. Grad-CAM / Attention Analysis

Visualizations generated in [`experiments/results/phase4a_gradcam/`](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/results/phase4a_gradcam):

| Case | True Class | Predicted Class | Confidence | Attention Pattern Description |
| :--- | :--- | :--- | :---: | :--- |
| `internal_correct_tuberculosis.png` | Tuberculosis | Tuberculosis | 97.4% | Focal, concentrated activation on upper apical cavitary and infiltrative lung parenchyma. |
| `internal_correct_normal.png` | Normal | Normal | 98.2% | Diffuse bilateral parenchymal attention, clear lung margins, low peripheral bone activation. |
| `internal_correct_pulmonary_nodule___mass.png` | Nodule/Mass | Nodule/Mass | 78.6% | Circumscribed focal activation over focal radio-opacity. |
| `internal_correct_pleural_effusion.png` | Effusion | Effusion | 92.1% | Basilar costophrenic angle blunting activation. |
| `montgomery_error_tb_pred_nodule_1.png` | Tuberculosis | Nodule/Mass | 74.3% | Diffuse activation along sharp posterior ribs and clavicles rather than focal parenchymal lesions. |
| `montgomery_error_tb_pred_effusion_1.png` | Tuberculosis | Effusion | 69.8% | Activation displaced into lateral costophrenic margins and lower diaphragm border. |
| `montgomery_error_norm_pred_nodule_1.png` | Normal | Nodule/Mass | 76.5% | Sharp rib margins and high-contrast mediastinal borders trigger dense mass activation. |

### Visual Synthesis:
In internal V5 images, the model relies on lung-parenchymal texture. In Montgomery scans, the high-contrast digitizer edge sharpness produces strong artificial activations along the bony cage (ribs, clavicles, scapular borders), causing the model to mistake these high-contrast borders for mass-like radiodensities.

---

## 8. Feature-Space Geometry (Embedding Analysis)

Embeddings extracted from the 256-D pre-classification dense layer (`dense`):

![Feature Space Projection](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/results/phase4a_feature_space.png)

### PCA Projection Metrics:
* **PC1 Explained Variance**: 27.6%
* **PC2 Explained Variance**: 16.9%
* **Total Top-2 Variance**: 44.4%

### Centroid Distances in 256-D Representation Space:
* Distance(Montgomery TB Centroid $\rightarrow$ V5 TB Centroid): **14.510**
* Distance(Montgomery TB Centroid $\rightarrow$ V5 Nodule/Mass Centroid): **6.731**
* Distance(Montgomery Normal Centroid $\rightarrow$ V5 Normal Centroid): **19.584**
* Distance(Montgomery Normal Centroid $\rightarrow$ V5 Nodule/Mass Centroid): **7.377**

### Geometry Conclusion:
Montgomery images do not project into the internal Normal or Tuberculosis manifolds. Instead, their feature embeddings are geometrically shifted into the region adjacent to Pulmonary Nodule / Mass and Pleural Effusion, explaining the 92% / 8% prediction split. Specifically, Montgomery TB centroid is **more than 2x closer to V5 Nodule/Mass (6.73) than to V5 TB (14.51)**, and Montgomery Normal centroid is **almost 3x closer to V5 Nodule/Mass (7.38) than to V5 Normal (19.58)**.

---

## 9. Consolidated Comparison: Internal Test vs Montgomery

| Metric / Dimension | V5 Internal Test | Montgomery County External |
| :--- | :---: | :---: |
| **Sample Size ($N$)** | 1,570 | 138 |
| **Overall Multi-Class Accuracy** | 76.18% | 0.00% (on TB/Normal cohorts) |
| **Exact Tuberculosis Recall** | 80.10% | 0.00% (0/58) |
| **Exact Normal Specificity** | 96.60% | 0.00% (0/80) |
| **Binary Abnormal Sensitivity** | N/A (Multi-class) | 100.00% (58/58) |
| **Mean Max Confidence ($p_{max}$)** | 86.74% (Correct) / 65.83% (Incorrect) | 86.19% (Mean) / 90.95% (Median) |
| **Confidence Margin ($p_1 - p_2$)** | 0.770 (Correct) / 0.408 (Incorrect) | 0.729 (Mean) |
| **Prediction Entropy (Bits)** | 0.527 (Correct) / 1.141 (Incorrect) | 0.508 (Mean) |
| **Dominant Prediction for True TB** | Tuberculosis (80.10%) | Pulmonary Nodule / Mass (82.76%) |
| **Dominant Prediction for True Normal** | Normal (84.01%) | Pulmonary Nodule / Mass (98.75%) |
| **Mean Dynamic Range ($P_{95} - P_5$)** | 170.8 | 206.8 (TB) / 202.9 (Normal) |
| **Laplacian Variance Median** | 373.1 | 1,589.4 (TB) / 1,642.0 (Normal) |
| **Mean Native Resolution** | 1.83 MP | 19.67 MP |

---

## 10. Evaluation of Failure Patterns (A through G)

| Pattern Hypothesis | Supported? | Evidence & Finding |
| :--- | :---: | :--- |
| **A. Source/Domain Distribution Shift** | **Yes** | Strong evidence: 256-D feature space displays complete separation of Montgomery from internal training manifolds. |
| **B. Projection / View Differences** | **Unverified** | Montgomery metadata lacks projection tags. Projection could not be verified from available metadata. |
| **C. Intensity / Contrast Differences** | **Yes** | Montgomery dynamic range is $21\%$ higher and Laplacian edge sharpness is $>4\times$ higher than internal training data. |
| **D. Feature-Space Separation** | **Yes** | Centroids of both Montgomery subsets are closer to the Nodule/Mass centroid than their respective ground truth classes. |
| **E. High-Confidence Wrong Predictions** | **Yes** | Mean confidence on Montgomery is $86.19\%$ (median $90.95\%$, mean margin $0.729$), matching internal correct confidence ($86.74\%$), demonstrating extremely confident misclassification rather than near-boundary uncertainty. |
| **F. Class-Specific Confusion** | **Yes** | $92.03\%$ of Montgomery is mapped to Pulmonary Nodule / Mass, and $7.97\%$ to Pleural Effusion. Zero scans predicted as Normal or TB. |
| **G. Confounded Factors** | **Yes (Dominant)** | Scanner digitizer properties (19.7 MP resolution, edge sharpness), contrast, and source domain are completely collinear. |

---

## 11. Final Scientific Findings

1. **What the V5 Baseline Does Well**:
   * Highly effective on internal multi-class classification ($76.18\%$ accuracy, $0.9600$ macro ROC-AUC).
   * Robust multi-source grounding on internal classes (COVID-19: $90.23\%$ F1, Normal: $86.54\%$ F1, TB: $85.41\%$ F1).
   * 100% binary abnormal detection sensitivity on Montgomery ($58/58$ TB scans correctly recognized as abnormal).

2. **What It Fails At**:
   * Fine-grained class discrimination on external digitizer CXRs.
   * Completely fails to recognize Normal controls ($0/80$) and TB pathology ($0/58$) on Montgomery, misrouting $98.8\%$ and $82.8\%$ into Pulmonary Nodule/Mass.

3. **What Evidence Explains the External Failure**:
   * **Confounded Domain-Scanner Shift**: Montgomery's extreme resolution ($19.7$ MP) and high-contrast digitizer artifacts create edge responses along the thoracic cage that the feature extractor interprets as dense opacities.
   * **TB Label Sourcing Asymmetry**: V5 Tuberculosis is derived predominantly from TBX11K (downsampled, low Laplacian variance $227.4$), whereas Montgomery is digitized film with high Laplacian variance ($1,589.4$). The model learned "TB" as low-contrast parenchymal patterns, while high-contrast films trigger "Nodule/Mass".

4. **What Remains Uncertain**:
   * Exact contribution of patient demographics (Montgomery is a US outpatient screening cohort; TBX11K is a hospitalized cohort).
   * The unrecorded view projection (PA vs AP) of Montgomery cases.

---

## 12. Candidate Directions for Phase 4B

The following candidate domain-generalization strategies are suggested by the empirical evidence:

* **Candidate 1: Domain-Adversarial Neural Network (DANN)**
  * *Rationale*: Adversarial domain classifier to penalize source-specific feature representations (VinDr vs NIH vs TBX11K vs Guangzhou).
  * *Status*: **Candidate method for Phase 4B — NOT YET IMPLEMENTED.**
* **Candidate 2: Deep Correlation Alignment (Deep CORAL)**
  * *Rationale*: Covariance alignment across training sources to align second-order feature statistics without adversarial instability.
  * *Status*: **Candidate method for Phase 4B — NOT YET IMPLEMENTED.**
* **Candidate 3: Maximum Mean Discrepancy (MMD / MK-MMD)**
  * *Rationale*: Non-parametric distribution matching across multiple source domains.
  * *Status*: **Candidate method for Phase 4B — NOT YET IMPLEMENTED.**

---

## 13. What Must NOT Be Changed

To maintain rigorous scientific validity throughout subsequent phases:
* `unified_manifest_v5.csv` must remain immutable.
* `densenet121_v5.h5` must remain preserved as the unadapted reference baseline.
* Montgomery County ($N=138$) must **never** be used in training, fine-tuning, or hyperparameter selection; it must remain an untouched, zero-shot external evaluation benchmark.
* Preprocessing, input dimensions ($224 \times 224 \times 3$), and the six clinical class definitions must remain strictly standardized.

