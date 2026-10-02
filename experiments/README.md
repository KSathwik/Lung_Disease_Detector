# LUNGAI — MASTER THESIS EXPERIMENTAL INDEX

This directory houses the complete scientific experimental series developed for the M.Tech thesis on multi-source domain generalization in chest radiography.

All experiments use the identical leak-free unified cohort ($N=10,547$ scans, 10,270 patients, 0% patient leakage), identical DenseNet-121 backbone, 256-D latent embedding space, inverse-frequency class weighting, and 4-epoch schedule (2 epochs frozen base $\text{lr}=10^{-3}$, 2 epochs fine-tuning top 30 layers $\text{lr}=10^{-5}$, seed 42).

---

## 1. Master 5-Paradigm Experimental Comparison

| Phase | Scientific Paradigm | Model Architecture | Key Internal Metric | Montgomery External TB Recall | Primary Artifact Directory | Paradigm Role |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **3A** | **Unified V5 Reconstruction** | Manifest Construction | 10,547 scans / 10,270 patients | — | [`experiments/data/`](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/data/) | Leak-free data protocol |
| **3B** | **ERM Baseline** | DenseNet-121 | 76.18% Acc / 72.63% F1 | 0.00% (0/58) | [`experiments/densenet_v5/`](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/densenet_v5/) | Scientific baseline |
| **4A** | **External Failure Analysis** | Biophysical Analysis | 4× Laplacian edge discrepancy | — | [`experiments/results/`](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/results/) | Sensor shift root cause |
| **4B** | **Deep CORAL** | DenseNet-121 + CORAL | 80.89% Acc / 76.85% F1 | 1.72% (1/58) | [`experiments/densenet_coral_v5/`](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/densenet_coral_v5/) | Latent covariance alignment |
| **4C** | **DANN Adversarial** | DenseNet-121 + DANN | 76.24% Acc / 72.99% F1 | 0.00% (0/58) | [`experiments/densenet_dann_v5/`](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/densenet_dann_v5/) | Adversarial alignment study |
| **4D** | **Frequency LP ($\sigma=1.0$)** | **DenseNet-121 + Gaussian LP** | **82.93% Acc / 78.35% F1** | **0.00% (0/58)** | [`experiments/densenet_frequency_v5/`](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/densenet_frequency_v5/) | **FINAL PRODUCTION MODEL (MODEL D)** |
| **4E** | **Hybrid Synthesis** | DenseNet-121 + LP + CORAL | 78.22% Acc / 73.95% F1 | 0.00% (0/58) | [`experiments/densenet_hybrid_v5/`](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/densenet_hybrid_v5/) | Synthesis study; peak COVID-19 F1 (98.62%) |

---

## 2. Final Production Model (Model D)

The designated final inference model for the LungAI application is **Model D**:
* **Checkpoint**: `experiments/densenet_frequency_v5/densenet121_frequency_v5.h5`
* **Pre-processing**: Gaussian Low-Pass filter ($\sigma=1.0$) on standardized CIE LAB CLAHE enhanced scans
* **Internal Test Performance**:
  * Accuracy: **82.93%**
  * Macro F1-Score: **78.35%** (+5.72% gain over ERM baseline)
  * Macro ROC-AUC: **0.9755**
  * Macro PR-AUC: **0.8391**
* **Per-Class Metrics**:
  * COVID-19: **97.01% F1**
  * Normal: **89.58% F1**
  * Pneumonia: **86.49% F1**
  * Tuberculosis: **88.66% F1**
  * Pulmonary Nodule / Mass: **53.03% F1**
  * Pleural Effusion: **55.32% F1**

---

## 3. Quarantined External Benchmark (Montgomery County)

Across all five model paradigms, zero-shot evaluation on the untouched external Montgomery cohort ($N=138$) consistently collapsed toward *Pulmonary Nodule / Mass*:
* Film-digitizer high-frequency noise and extreme dynamic range mismatch (analog film digitizer vs. digital detector) create an insurmountable sensor domain shift for blind models.
* While Model D successfully eliminates internal sensor grain overfitting (lifting internal accuracy from 76.18% to 82.93%), zero-shot external transfer proves that sensor shift is multi-factorial, requiring target-domain calibration or lung-field anatomical segmentation.

---

## 4. Preservation Directive

All artifacts in this directory (metrics, reports, confusion matrices, ROC/PR curves, t-SNE feature plots, and Grad-CAM overlays) constitute empirical thesis evidence. They must remain preserved and reproducible.
