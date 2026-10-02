# LUNGAI — PROJECT DIRECTORY CLEANUP AUDIT & ARCHIVAL PLAN

**Audit Timestamp**: October 2026  
**Repository**: `Lung_Disease_Detector-main`  
**Final Production Model**: `experiments/densenet_frequency_v5/densenet121_frequency_v5.h5`  
**Status**: Pre-cleanup Audit Complete — Awaiting Approval (Destructive Deletions Frozen)

---

## 1. Executive Summary of Project Footprint

A comprehensive recursive audit of the repository identified the following top-level directory composition:

| Top-Level Directory | File Count | Disk Footprint (MB) | Purpose / Category | Status |
| :--- | :---: | :---: | :--- | :--- |
| **`data/`** | 27,911 | 11,343.09 MB (~11.08 GB) | Multi-source CXR scans (`data/raw/` and `data/downloads/`) | **PRESERVE** (Referenced by V5 manifest) |
| **`.venv/`** | 26,092 | 1,948.19 MB (~1.90 GB) | Local Python 3.12 Virtual Environment | **LOCAL DEV** (Git-ignored) |
| **`models/`** | 9 | 761.41 MB | Production checkpoints and historical weights | **PRESERVE / AUDIT** |
| **`experiments/`** | 250 | 727.09 MB | Thesis experiments (Phases 3A–4E), manifests, plots | **PRESERVE / ARCHIVE** |
| **`frontend/`** | 42,497 | 379.37 MB | React application & `node_modules/` | **PRESERVE CODE** (Dependencies git-ignored) |
| **`docs/`** | 57 | 14.71 MB | Thesis documentation, reports, presentation slides | **PRESERVE** |
| **`backend/`** | 64 | 0.35 MB | FastAPI application, database, inference engine | **PRESERVE** |
| **`reports/`** | 5 | 0.26 MB | Generated clinical PDF/text reports | **PRESERVE** |
| **`uploads/`** | 6 | 0.01 MB | Local temporary test uploads | **PRESERVE** |
| **`.pytest_cache/`**| 6 | 0.00 MB | Pytest execution cache | **CANDIDATE FOR CLEANUP** |
| **Root Files** | 6 | 1.10 MB | `README.md`, `Dockerfile`, `.gitignore`, slides | **PRESERVE** |
| **TOTAL** | **96,893** | **~15,174.58 MB (~14.82 GB)** | Complete Repository | — |

---

## 2. Final Model Artifact Verification (Model D)

The designated final inference model is **Model D — Frequency Preprocessing (Gaussian $\sigma=1.0$)**:

* **Experiment Checkpoint**: `experiments/densenet_frequency_v5/densenet121_frequency_v5.h5` (28.71 MB, MD5 verified)
* **Application Checkpoint**: `models/densenet121_frequency_v5.h5` (28.71 MB, synchronized copy)
* **Official Class Mapping**: `backend/ml/class_mapping.json` & `experiments/densenet_frequency_v5/class_mapping.json`
* **Preprocessing Pipeline**: `backend/ml/preprocessing.py` (Gaussian $\sigma=1.0$ Low-Pass, CIE LAB CLAHE, Lanczos-4 resize, ImageNet normalization)
* **Inference Service**: `backend/ml/inference.py` (Singleton service, direct softmax output, Grad-CAM visualization)
* **Automated Tests**: `backend/tests/test_model_d_integration.py` (9/9 automated tests passing)

**Directive**: Model D and its reproduction pipeline are strictly **FROZEN & IMMUTABLE**.

---

## 3. Critical V5 Manifest Audit & Discrepancy Notice

* **File**: `experiments/data/unified_manifest_v5.csv`
* **Size**: 2,592,544 bytes ($N=10,547$ scans, 10,270 unique patients)
* **Calculated MD5 Hash**: `762d49913995aaff0e975fb0e0b4f239`
* **Reference MD5 Stated in Directive**: `42f494954a1a44e59ef26bf6fb966f91`

> [!WARNING]
> ### Manifest MD5 Discrepancy Report (Per Section 12)
> The physical file `experiments/data/unified_manifest_v5.csv` on disk has MD5 hash `762d49913995aaff0e975fb0e0b4f239`.
> This hash matches the exact MD5 logged across all Phase 3B (`train_densenet_v5.py`), Phase 4A (`phase4a_failure_analysis.py`), Phase 4B (`train_densenet_coral_v5.py`), Phase 4C (`train_densenet_dann_v5.py`), Phase 4D (`train_densenet_frequency_v5.py`), and Phase 4E (`train_densenet_hybrid_v5.py`) experimental executions.
> In accordance with Section 12 ("*If the hash has changed: STOP. Do not replace or regenerate it. Report the discrepancy*"), the file has **NOT** been replaced, touched, or regenerated. It is preserved as the authentic experimental baseline.

---

## 4. Comprehensive Model Checkpoint Audit

The project contains 23 model weight files (`*.h5`):

| File Path | Size (MB) | Paradigm / Architecture | Experiment / Role | Recommendation |
| :--- | :---: | :--- | :--- | :--- |
| `experiments/densenet_frequency_v5/densenet121_frequency_v5.h5` | 28.71 | DenseNet-121 + Gaussian LP ($\sigma=1.0$) | **Phase 4D — Final Production Model D** | **KEEP (PRIMARY)** |
| `models/densenet121_frequency_v5.h5` | 28.71 | DenseNet-121 + Gaussian LP ($\sigma=1.0$) | **Production Deployment Checkpoint** | **KEEP (APP)** |
| `experiments/densenet_v5/densenet121_v5.h5` | 35.66 | DenseNet-121 ERM Baseline | **Phase 3B — Thesis Scientific Baseline** | **KEEP (THESIS)** |
| `experiments/densenet_coral_v5/densenet121_coral_v5.h5` | 28.70 | DenseNet-121 + Deep CORAL ($\lambda=0.01$) | **Phase 4B — Thesis Domain Alignment** | **KEEP (THESIS)** |
| `experiments/densenet_dann_v5/densenet121_dann_v5.h5` | 28.71 | DenseNet-121 + DANN Adversarial | **Phase 4C — Thesis Domain Adversarial** | **KEEP (THESIS)** |
| `experiments/densenet_dann_v5/checkpoint_lambda_0.01.h5` | 28.71 | DenseNet-121 DANN Sweep ($\lambda=0.01$) | Phase 4C Grid Search Intermediate | **ARCHIVE** |
| `experiments/densenet_dann_v5/checkpoint_lambda_0.1.h5` | 28.71 | DenseNet-121 DANN Sweep ($\lambda=0.10$) | Phase 4C Grid Search Intermediate | **ARCHIVE** |
| `experiments/densenet_dann_v5/checkpoint_lambda_1.0.h5` | 28.71 | DenseNet-121 DANN Sweep ($\lambda=1.00$) | Phase 4C Grid Search Intermediate | **ARCHIVE** |
| `experiments/densenet_hybrid_v5/densenet121_hybrid_v5.h5` | 28.70 | DenseNet-121 + Hybrid LP + CORAL | **Phase 4E — Thesis Synthesis Model** | **KEEP (THESIS)** |
| `experiments/densenet_b4_v5/densenet121_b4a_v5_vindr_to_nih.h5` | 30.73 | DenseNet-121 Source Holdout | Phase 4B Source Holdout (VinDr $\rightarrow$ NIH) | **KEEP (THESIS)** |
| `experiments/densenet_b4_v5/densenet121_b4b_v5_nih_to_vindr.h5` | 30.73 | DenseNet-121 Source Holdout | Phase 4B Source Holdout (NIH $\rightarrow$ VinDr) | **KEEP (THESIS)** |
| `experiments/densenet_v3/densenet121_v3.h5` | 40.17 | DenseNet-121 (Pre-V5) | Phase 2 Historical Baseline | **ARCHIVE** |
| `experiments/densenet_v3_full/densenet121_v3_full.h5` | 40.17 | DenseNet-121 (Pre-V5 Full) | Phase 2 Historical Baseline | **ARCHIVE** |
| `experiments/models/densenet121_unified.h5` | 40.17 | DenseNet-121 (Pre-V5 Unified) | Phase 2 Historical Baseline | **ARCHIVE** |
| `experiments/models/resnet50_unified.h5` | 214.25 | ResNet-50 (Pre-V5 Unified) | Phase 2 Historical Baseline | **ARCHIVE** |
| `experiments/source_holdout_b4/densenet121_b4a_vindr_to_nih.h5`| 30.73 | DenseNet-121 Source Holdout | Phase 2 Pre-V5 Holdout Run | **ARCHIVE** |
| `experiments/source_holdout_b4/densenet121_b4b_nih_to_vindr.h5`| 30.73 | DenseNet-121 Source Holdout | Phase 2 Pre-V5 Holdout Run | **ARCHIVE** |
| `models/cnn_model.h5` | 8.38 | Custom 4-block CNN | Legacy App Algorithm 1 | **KEEP (OPTIONAL APP)** |
| `models/densenet_model.h5` | 40.17 | DenseNet-121 (Pre-V5) | Legacy App Checkpoint | **ARCHIVE** |
| `models/resnet_model.h5` | 214.25 | ResNet-50 | Legacy App Algorithm 2 | **KEEP (OPTIONAL APP)** |
| `models/ResNet_phase2_best.h5` | 230.76 | ResNet-50 Phase 2 Fine-Tuned | Phase 2 Fine-Tuned Checkpoint | **ARCHIVE** |
| `models/baseline_archive/cnn_model_old_5class.h5` | 8.38 | Custom CNN 5-class | Historical Phase 1 Archive | **ARCHIVED** |
| `models/baseline_archive/resnet_model_old_5class.h5` | 230.76 | ResNet-50 5-class | Historical Phase 1 Archive | **ARCHIVED** |

*Note: In accordance with the prompt's instruction ("DO NOT delete any model merely because it is not the final model"), zero models will be deleted. Intermediates and pre-V5 models are recommended for archival rather than deletion.*

---

## 5. Dataset & Manifest Inventory

* **Total Scans in `data/`**: 27,911 files across `data/raw/` and `data/downloads/`.
* **Total Dataset Size**: 11,343.09 MB (~11.08 GB).
* **Reference Verification**:
  * Scans referenced by `unified_manifest_v5.csv`: 10,547 images across COVID-19, Normal, Pneumonia, Tuberculosis, Pleural Effusion, and Pulmonary Nodule / Mass cohorts.
  * Zero-shot external cohort: Montgomery County (`data/downloads/montgomery/images/images/`, $N=138$ scans).
  * Original ZIP archives in `data/downloads/`:
    * `chest-xray-pneumonia.zip` (2.29 GB)
    * `covid19-radiography-database.zip` (778.2 MB)
    * `tuberculosis-tb-chest-xray-dataset.zip` (663.4 MB)
    * `chest-ctscan-images.zip` (118.6 MB)
* **Manifests in `experiments/data/`**:
  * `unified_manifest_v5.csv` (2.47 MB, Active Master V5)
  * `unified_manifest_v4.csv` (2.37 MB, Historical V4)
  * `unified_manifest_v3.csv` (2.57 MB, Historical V3)
  * `unified_manifest_v2.csv` (2.19 MB, Historical V2)
  * `unified_manifest.csv` (2.28 MB, Historical V1)

---

## 6. Temporary Files, Caches, and Log Audit

* **`__pycache__/` directories in project source**:
  * `backend/__pycache__/` (9.5 KB)
  * `backend/api/__pycache__/` (0.4 KB)
  * `backend/api/routes/__pycache__/` (41.8 KB)
  * `backend/database/__pycache__/` (23.7 KB)
  * `backend/ml/__pycache__/` (54.0 KB)
  * `backend/tests/__pycache__/` (67.3 KB)
  * `backend/utils/__pycache__/` (1.1 KB)
  * `experiments/__pycache__/` (453.2 KB)
  * `experiments/phase2e/__pycache__/` (28.2 KB)
  * *Total Source Cache*: 9 directories, 679.2 KB. **SAFE TO PURGE**.
* **Pytest Cache**: `.pytest_cache/` (6 files, 0.0 MB). **SAFE TO PURGE**.
* **Temporary Editor / Swap Files**: 0 files (`*.tmp`, `*.bak`, `*.swp`).
* **Active Log Files**: 0 loose `.log` files in source (1 internal in `node_modules`).

---

## 7. Secrets and Credentials Audit

* **`.env` files checked**:
  * `backend/.env`: Local development configuration only (`DATABASE_URL=sqlite+aiosqlite:///./lung_disease.db`, `API_PORT=8000`, `MODELS_DIR=models/`). No cloud tokens, private keys, or passwords.
  * `backend/.env.example`: Safe template file.
* **Private keys / Certificates (`*.pem`, `*.key`)**: None found.
* **Credentials JSON**: None found.
* **Git Status**: `backend/.env` is already listed in `.gitignore`.

---

## 8. Git Status & `.gitignore` Assessment

Current `.gitignore` covers:
* `__pycache__/`, `*.py[cod]`, `venv/`, `.venv/`, `.pytest_cache/`, `.env`, `backend/.env`, `models/*.h5`, `data/raw/`, `data/downloads/`, `frontend/node_modules/`, `frontend/build/`, `*.log`.

Recommended additions to `.gitignore`:
```text
.next/
dist/
build/
coverage/
.cache/
.ipynb_checkpoints/
*.tmp
*.temp
*.bak
*.swp
```

---

## 9. Comprehensive Archival Plan

To organize without breaking any script imports or references:

### A. Completed Scientific Paradigms to KEEP Exactly in Place:
* `experiments/data/unified_manifest_v5.csv` (Frozen Phase 3A dataset)
* `experiments/densenet_v5/` (Phase 3B Baseline ERM)
* `experiments/results/phase4a_*` (Phase 4A Failure Analysis reports)
* `experiments/densenet_coral_v5/` (Phase 4B Deep CORAL)
* `experiments/densenet_dann_v5/` (Phase 4C DANN)
* `experiments/densenet_frequency_v5/` (Phase 4D Frequency Model — Final Model D)
* `experiments/densenet_hybrid_v5/` (Phase 4E Hybrid Model)

### B. Pre-V5 Historical Experiments to Archive in `experiments/archive/`:
* `experiments/densenet_v3/`
* `experiments/densenet_v3_full/`
* `experiments/source_holdout_b4/`
* `experiments/phase2d/`, `phase2e/`, `phase2f/`, `phase2g/`
* Historical manifests `unified_manifest.csv`, `_v2.csv`, `_v3.csv`, `_v4.csv`

### C. Generated Caches to DELETE:
* `backend/**/__pycache__/`
* `experiments/**/__pycache__/`
* `.pytest_cache/`
