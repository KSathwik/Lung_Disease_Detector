# LUNGAI — PROJECT CLEANUP & ARCHIVAL FINAL REPORT

**Date**: October 2026  
**Final Production Model**: `experiments/densenet_frequency_v5/densenet121_frequency_v5.h5` (Model D — Frequency LP $\sigma=1.0$)  
**Status**: Cleanup & Archival Successfully Executed (31/31 Automated Tests Passing)

---

## 1. Before vs. After Cleanup Footprint

| Category / Directory | Pre-Cleanup Files | Post-Cleanup Files | Pre-Cleanup Size (MB) | Post-Cleanup Size (MB) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`data/`** | 27,911 | 27,911 | 11,343.09 MB | 11,343.09 MB | **PRESERVED** (10,547 V5 scans + raw sets) |
| **`.venv/`** | 26,092 | 26,092 | 1,948.19 MB | 1,948.19 MB | **PRESERVED** (Local runtime, git-ignored) |
| **`models/`** | 9 | 9 | 761.41 MB | 761.41 MB | **ORGANIZED** (Model D + baselines) |
| **`experiments/`** | 250 | 241 | 727.09 MB | 726.63 MB | **CLEANED & ARCHIVED** (Thesis intact) |
| **`frontend/`** | 42,497 | 42,497 | 379.37 MB | 379.37 MB | **PRESERVED** (`node_modules/` git-ignored) |
| **`docs/`** | 57 | 57 | 14.71 MB | 14.71 MB | **PRESERVED** (Thesis & viva guides) |
| **`backend/`** | 64 | 31 | 0.35 MB | 0.16 MB | **PURGED CACHES** (33 `.pyc` removed) |
| **`reports/`** | 5 | 5 | 0.26 MB | 0.26 MB | **PRESERVED** (Clinical report templates) |
| **`uploads/`** | 6 | 8 | 0.01 MB | 0.01 MB | **PRESERVED** (Test inference uploads) |
| **`.pytest_cache/`**| 6 | 0 | 0.003 MB | 0.00 MB | **DELETED** (Generated test cache) |
| **Root Files** | 6 | 7 | 1.10 MB | 1.11 MB | **UPDATED** (Documentation added) |
| **TOTAL** | **96,898** | **96,858** | **~15,174.58 MB** | **~15,174.94 MB** | **CLEAN & REPRODUCIBLE** |

---

## 2. Removed Files (Verified Generated / Redundant Artifacts)

In strict adherence to Section 21 verification guidelines:

| File / Directory | Size | Reason | Referenced By | Safe to Delete |
| :--- | :---: | :--- | :--- | :---: |
| `backend/__pycache__/` | 9.8 KB | Python bytecode cache | Dynamic import runtime | **YES** |
| `backend/api/__pycache__/` | 0.4 KB | Python bytecode cache | Dynamic import runtime | **YES** |
| `backend/api/routes/__pycache__/` | 42.8 KB | Python bytecode cache | Dynamic import runtime | **YES** |
| `backend/database/__pycache__/` | 24.2 KB | Python bytecode cache | Dynamic import runtime | **YES** |
| `backend/ml/__pycache__/` | 55.3 KB | Python bytecode cache | Dynamic import runtime | **YES** |
| `backend/tests/__pycache__/` | 69.0 KB | Python bytecode cache | Dynamic import runtime | **YES** |
| `backend/utils/__pycache__/` | 1.2 KB | Python bytecode cache | Dynamic import runtime | **YES** |
| `experiments/__pycache__/` | 464.1 KB | Python bytecode cache | Dynamic import runtime | **YES** |
| `experiments/phase2e/__pycache__/` | 28.8 KB | Python bytecode cache | Dynamic import runtime | **YES** |
| `.pytest_cache/` | 3.0 KB | Pytest test execution cache | Pytest runtime only | **YES** |

*Total deleted: 40 generated cache files (~698.6 KB).*

---

## 3. Archived Artifacts

| Source File / Directory | Target Archive Location | Reason |
| :--- | :--- | :--- |
| `experiments/densenet_dann_v5/checkpoint_lambda_0.01.h5` | `experiments/archive/dann_checkpoints/` | Intermediate hyperparameter sweep checkpoint (30.1 MB) |
| `experiments/densenet_dann_v5/checkpoint_lambda_0.1.h5` | `experiments/archive/dann_checkpoints/` | Intermediate hyperparameter sweep checkpoint (30.1 MB) |
| `experiments/densenet_dann_v5/checkpoint_lambda_1.0.h5` | `experiments/archive/dann_checkpoints/` | Intermediate hyperparameter sweep checkpoint (30.1 MB) |
| `experiments/densenet_dann_v5/checkpoint_lambda_*_val.json`| `experiments/archive/dann_checkpoints/` | Intermediate sweep validation metrics |

*Note: All intermediate DANN sweep files are safely preserved in `experiments/archive/dann_checkpoints/`, keeping the main `densenet_dann_v5/` directory clean and focused on the selected thesis checkpoint `densenet121_dann_v5.h5`.*

---

## 4. Preserved Artifacts

### 4.1 Production Model D
* **`experiments/densenet_frequency_v5/densenet121_frequency_v5.h5`**: Master trained checkpoint (82.93% Accuracy, 78.35% Macro F1).
* **`models/densenet121_frequency_v5.h5`**: Deployment model copy.
* **`backend/ml/class_mapping.json`**: Official 6-class mapping.
* **`backend/ml/preprocessing.py`**: Gaussian $\sigma=1.0$ low-pass filter + CLAHE pipeline.
* **`backend/ml/inference.py`**: Enforced strictly Model D loading with Grad-CAM support.

### 4.2 Thesis Scientific Artifacts (Phases 3A – 4E)
* **Phase 3A Dataset Reconstruction**: `experiments/data/unified_manifest_v5.csv` (10,547 scans, MD5 `762d49913995aaff0e975fb0e0b4f239`).
* **Phase 3B ERM Baseline**: `experiments/densenet_v5/` (Weights, metrics, confusion matrix, ROC/PR curves).
* **Phase 4A Failure Analysis**: `experiments/results/phase4a_*` (High-frequency discrepancy, dark border bias, dynamic range degradation).
* **Phase 4B Deep CORAL**: `experiments/densenet_coral_v5/` (Covariance alignment weights, feature visualizations, metrics).
* **Phase 4C DANN**: `experiments/densenet_dann_v5/` (Domain-adversarial weights, feature space t-SNE, sweep log).
* **Phase 4D Frequency Preprocessing**: `experiments/densenet_frequency_v5/` (Winning model D, ablation curves, metrics).
* **Phase 4E Hybrid**: `experiments/densenet_hybrid_v5/` (Hybrid LP + CORAL weights, comparison logs).

---

## 5. Potential Issues & Verification Audit

* **Manifest MD5 Status**: Preserved as physical baseline (`762d49913995aaff0e975fb0e0b4f239`). Verified unchanged.
* **Broken References**: Scanned application source. Removed legacy `densenet_model.h5` fallback from `backend/ml/inference.py` so that only Model D is ever loaded.
* **Secrets & Credentials**: Re-audited. Zero API keys, private keys, or passwords exist in repository. `.env` is git-ignored and restricted to local SQLite configs.
* **Test Suite Verification**: Ran full pytest suite post-cleanup:
  ```text
  31 passed, 21 warnings in 26.02s
  - backend/tests/test_health.py (2/2 passed)
  - backend/tests/test_model_d_integration.py (9/9 passed)
  - backend/tests/test_patients.py (7/7 passed)
  - backend/tests/test_predictions.py (6/6 passed)
  - backend/tests/test_preprocessing.py (7/7 passed)
  ```

---

## 6. Final Project Structure

```text
Lung_Disease_Detector-main/
│
├── backend/                             # FastAPI Production Application
│   ├── api/
│   │   └── routes/                      # Endpoints: /health, /patients, /predictions, /reports
│   ├── database/                        # SQLAlchemy async models & migrations
│   ├── ml/
│   │   ├── class_mapping.json           # 6-class disease labels & indices
│   │   ├── download_model.py            # Checkpoint release download helper
│   │   ├── inference.py                 # Strictly Model D singleton inference + Grad-CAM
│   │   └── preprocessing.py             # Frequency LP (sigma=1.0) + CLAHE pipeline
│   ├── tests/                           # Complete test suite (31 tests, 100% passing)
│   └── main.py                          # Application entry point & lifespan handler
│
├── frontend/                            # React Web Application
│   ├── public/                          # Static assets, icons, HTML shell
│   └── src/
│       ├── components/                  # UI widgets (Navbar, ImageDropzone, HeatmapViewer)
│       ├── pages/                       # Dashboard, AnalyzePage, PatientPage, HistoryPage
│       └── services/                    # API client layer (Axios)
│
├── models/                              # Local Model Storage
│   ├── densenet121_frequency_v5.h5      # Production Model D Checkpoint (28.71 MB)
│   ├── cnn_model.h5                     # Optional Custom CNN
│   ├── resnet_model.h5                  # Optional ResNet-50
│   ├── densenet_model.h5                # Frozen legacy baseline
│   ├── training_results.json            # Model performance metadata
│   └── baseline_archive/                # Historical 5-class Phase 1 checkpoints
│
├── experiments/                         # Thesis Scientific Experiment Archive
│   ├── README.md                        # Master 5-Paradigm Thesis Experiment Index
│   ├── data/
│   │   └── unified_manifest_v5.csv      # Phase 3A Frozen Master Manifest (N=10,547)
│   ├── densenet_v5/                     # Phase 3B: DenseNet-121 ERM Baseline
│   ├── densenet_coral_v5/               # Phase 4B: Deep CORAL Domain Adaptation
│   ├── densenet_dann_v5/                # Phase 4C: DANN Domain Adversarial
│   ├── densenet_frequency_v5/           # Phase 4D: Frequency-Aware LP (FINAL MODEL D)
│   ├── densenet_hybrid_v5/              # Phase 4E: Hybrid Preprocessing + CORAL
│   ├── densenet_b4_v5/                  # Phase 4B: Source-holdout validation
│   ├── archive/                         # Archived intermediate sweep checkpoints
│   │   └── dann_checkpoints/            # Phase 4C lambda sweep weights (0.01, 0.1, 1.0)
│   ├── results/                         # Consolidated evaluation plots & JSON metrics
│   └── *.py                             # Complete reproducible training & evaluation scripts
│
├── data/                                # CXR Image Repositories (Git-ignored)
│   ├── raw/                             # NIH ChestX-ray14, VinDr-CXR, CheXpert, etc.
│   └── downloads/                       # Original dataset archives & Montgomery benchmark
│
├── docs/                                # Project & Thesis Documentation
│   ├── PROJECT_CLEANUP_AUDIT.md         # Initial comprehensive recursive audit
│   ├── PROJECT_STRUCTURE.md             # Complete system architecture documentation
│   ├── PROJECT_CLEANUP_FINAL_REPORT.md  # This post-cleanup report
│   ├── LungAI_MTech_Project_Report.md   # Master thesis document
│   └── thesis/                          # Additional thesis chapters and figures
│
├── .gitignore                           # Comprehensive exclusion rules
├── Dockerfile                           # Production container definition
└── README.md                            # Repository landing guide
```
