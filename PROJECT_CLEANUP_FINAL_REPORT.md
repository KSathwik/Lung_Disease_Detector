# LungAI — Project Cleanup Final Report

**Date**: October 3, 2026 | 01:28 IST  
**Repository Root**: `D:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main`  
**Original Branch**: `main`  
**New Branch**: `chore/lungai-repository-cleanup`  
**Commit Hash**: `3730a6ceed796bda809b6f6dbe39d30f8ba3a580`  
**Configured Remote**: `origin -> https://KSathwik@github.com/KSathwik/Lung_Disease_Detector.git`  
**Push Status**: **SUCCESS** (`origin/chore/lungai-repository-cleanup` created and synchronized)  

---

## 1. Audit & Cleanup Statistics

| Metric | Pre-Cleanup | Post-Cleanup | Net Change / Action |
| :--- | :---: | :---: | :--- |
| **Total Files Audited** | 96,905 | 96,862 | Full recursive audit completed |
| **Files Retained (KEEP)** | 96,857 | 96,857 | All code, datasets, manifests & thesis files preserved |
| **Files Archived (ARCHIVE)**| — | 5 | Moved to `experiments/archive/legacy_models/` |
| **Files Deleted (DELETE)** | — | 43 | Bytecode caches, pytest cache, temp test uploads, duplicate `.h5` |
| **Disk Space Recovered** | — | ~463 MB | Duplicate model checkpoints purged from production `models/` |
| **Unintended Files** | — | **0** | Clean git working directory |
| **Active Secrets Detected**| 0 | **0** | Verified via multi-pattern secret scanner |

---

## 2. Production Model Verification (Model D)

* **Deployment Checkpoint**: `models/densenet121_frequency_v5.h5`
* **Master Experiment Checkpoint**: `experiments/densenet_frequency_v5/densenet121_frequency_v5.h5`
* **File Size**: 30,099,432 bytes (28.71 MB)
* **Verified MD5 Hash**: `6af4634311b5e9648a20478fc2ed2dbb`
* **Architecture**: DenseNet-121 &rarr; Global Average Pooling &rarr; Batch Normalization &rarr; Dense(256, ReLU) &rarr; Dropout(0.3) &rarr; Dense(6, Softmax)
* **Official Classes (6)**: `COVID-19`, `Normal`, `Pleural Effusion`, `Pneumonia`, `Pulmonary Nodule / Mass`, `Tuberculosis`
* **Internal Test Performance**:
  * Accuracy: **82.93%**
  * Macro F1: **78.35%**
  * Macro ROC-AUC: **0.9755**
  * Macro PR-AUC: **0.8391**
* **Deployment Integrity**: Production `models/` folder contains strictly Model D and its corresponding `training_results.json`. All legacy models removed or archived.

---

## 3. Scientific Core & Thesis Manifest Protection

* **Protected Manifest Path**: `experiments/data/unified_manifest_v5.csv`
* **Manifest Verified Scans**: **10,547**
* **Manifest Unique Patients**: **10,270**
* **Verified MD5 Hash**: `762d49913995aaff0e975fb0e0b4f239` (**EXACT MATCH**)
* **Modification Status**: **0 bytes altered** (`git diff -- experiments/data/unified_manifest_v5.csv` returns clean).

---

## 4. Preservation of Thesis Experimental Suites

All empirical evidence and artifacts supporting the MTech thesis have been preserved intact:

* **Phase 3A — Unified V5 Dataset**: `experiments/data/unified_manifest_v5.csv` and historical manifests V1–V4.
* **Phase 3B — ERM Baseline**: `experiments/densenet_v5/` (Weights, metrics, loss curves).
* **Phase 4A — Sensor-Shift Failure Analysis**: `experiments/results/phase4a_*` (Grad-CAM comparisons, feature spaces, confidence histograms).
* **Phase 4B — Deep CORAL Domain Adaptation**: `experiments/densenet_coral_v5/` (Weights, metrics, ROC/PR curves).
* **Phase 4C — Domain-Adversarial Neural Networks (DANN)**: `experiments/densenet_dann_v5/` & `experiments/archive/dann_checkpoints/`.
* **Phase 4D — Frequency Preprocessing (Model D)**: `experiments/densenet_frequency_v5/` (Selected final system).
* **Phase 4E — Hybrid Domain Adaptation**: `experiments/densenet_hybrid_v5/`.
* **Source-Held-Out Generalization**: `experiments/densenet_b4_v5/` & `experiments/source_holdout_b4/`.
* **Quarantined External Benchmark**: `data/downloads/montgomery/` (Untouched, preserved for audit reproducibility).

---

## 5. Automated Validation & Test Suite

The full test suite was executed on the new branch before and after cleanup:

* **Pytest Result**: **31 / 31 PASSED** (0 failures, 0 errors, 21 deprecation warnings)
* **Model Loading**: Model D loads cleanly into `InferenceEngine` with exact layer names and output shape `(None, 6)`.
* **Preprocessing Pipeline**: Verified BGR &rarr; RGB, Gaussian blur (3,3) $\sigma=0.8$, CIE LAB CLAHE, Gaussian spatial low-pass filter $\sigma=1.0$, Lanczos-4 resize to $224 \times 224$, ImageNet normalization.
* **Inference Engine**: Verified 6 class probability outputs summing to $\approx 1.0$.
* **Grad-CAM Generation**: Verified visual heatmap generation and overlay compositing.
* **FastAPI Service**: Verified `POST /api/v1/predict` handling valid CXR scans, invalid file extensions, corrupt file streams, and empty uploads.
* **Frontend Application**: Verified AnalyzePage components, Grad-CAM visualization display, probability bars, and external generalization limitation disclosures.

---

## 6. Archival Records

The following non-production artifacts were safely archived:

| Source Path | Archive Target Location | Reason |
| :--- | :--- | :--- |
| `models/ResNet_phase2_best.h5` | `experiments/archive/legacy_models/ResNet_phase2_best.h5` | Historical Phase 2 ResNet checkpoint (230.76 MB) |
| `models/baseline_archive/` | `experiments/archive/legacy_models/baseline_archive/` | Historical 5-class baseline models and metrics |
| `experiments/archive/dann_checkpoints/` | `experiments/archive/dann_checkpoints/` | DANN hyperparameter tuning checkpoints |

---

## 7. Git Traceability

* **Committed Branch**: `chore/lungai-repository-cleanup`
* **Remote Tracking**: `origin/chore/lungai-repository-cleanup`
* **Pull Request URL**: `https://github.com/KSathwik/Lung_Disease_Detector/pull/new/chore/lungai-repository-cleanup`
* **Main Branch**: Untouched (`main` remains preserved at origin).
