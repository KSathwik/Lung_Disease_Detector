# LungAI — Full Repository Forensic Cleanup Report

**Date & Time**: October 3, 2026 | 01:15 IST  
**Repository Root**: `D:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main`  
**Current Branch**: `main`  
**Remote**: `origin` (`https://github.com/KSathwik/Lung_Disease_Detector.git`)  
**Audit Status**: Complete Forensic Audit Executed — Awaiting User Approval (Critical Safety Stop)  

---

## 1. Repository Identification & Summary

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Git Root** | `D:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main` | Verified via `git rev-parse --show-toplevel` |
| **Current Branch** | `main` | Clean baseline before branching |
| **Configured Remote** | `origin -> https://github.com/KSathwik/Lung_Disease_Detector.git` | Fetch & Push endpoints confirmed |
| **Total Files (Disk)** | **96,905** | Includes `.venv` (26,092), `node_modules` (42,486), `data` (27,911) |
| **Total Size (Disk)** | **14.82 GB (15,175 MB)** | CXR dataset `data/` accounts for 11.08 GB |
| **Project Source Files** | **416** | Codebase excluding runtime environments & raw data |
| **Git Tracked Files** | **63** | In git index (`git ls-files`) |
| **Git Untracked Files**| **353** | New experiment results, reports, figures, thesis assets |
| **Baseline Test Status**| **31 / 31 PASSED** | Executed via pytest with Model D integration |

---

## 2. Scientific Core & Integrity Verification

### 2.1 Unified V5 Dataset Manifest
* **File Path**: `experiments/data/unified_manifest_v5.csv`
* **File Size**: 2,592,544 bytes
* **Total CXR Scans**: **10,547**
* **Unique Patients**: **10,270**
* **Calculated MD5 Hash**: `762d49913995aaff0e975fb0e0b4f239`
* **Integrity Status**: **PERFECT MATCH** (Identical to expected benchmark hash; preserved untouched).

### 2.2 Final Production Model D Checkpoint
* **Production Path**: `models/densenet121_frequency_v5.h5`
* **Experiment Master**: `experiments/densenet_frequency_v5/densenet121_frequency_v5.h5`
* **File Size**: 30,099,432 bytes (28.71 MB)
* **Calculated MD5 Hash**: `6af4634311b5e9648a20478fc2ed2dbb` (Both copies identical)
* **Architecture**: DenseNet-121 &rarr; GAP &rarr; Batch Normalization &rarr; Dense(256, ReLU) &rarr; Dropout(0.3) &rarr; Dense(6, Softmax)
* **Internal Test Metrics**: Accuracy: 82.93%, Macro F1: 78.35%, Macro ROC-AUC: 0.9755, Macro PR-AUC: 0.8391
* **Preprocessing Pipeline**: BGR &rarr; RGB, Gaussian blur (3,3) $\sigma=0.8$, CIE LAB CLAHE, Gaussian spatial low-pass $\sigma=1.0$, Lanczos-4 resize to $224 \times 224$, ImageNet normalization.

### 2.3 Scientific Experiment Suites (Phases 3A – 4E)
* **Phase 3A — Unified V5 Dataset**: `experiments/data/unified_manifest_v5.csv` & manifests V1–V4 (**PRESERVED**)
* **Phase 3B — ERM Baseline**: `experiments/densenet_v5/` (**PRESERVED**)
* **Phase 4A — Sensor Shift Failure Analysis**: `experiments/results/phase4a_*` (**PRESERVED**)
* **Phase 4B — Deep CORAL Domain Adaptation**: `experiments/densenet_coral_v5/` (**PRESERVED**)
* **Phase 4C — Domain-Adversarial Neural Networks (DANN)**: `experiments/densenet_dann_v5/` (**PRESERVED**)
* **Phase 4D — Frequency Spatial Filtering (Model D)**: `experiments/densenet_frequency_v5/` (**PRESERVED**)
* **Phase 4E — Hybrid Adaptation**: `experiments/densenet_hybrid_v5/` (**PRESERVED**)
* **Source-Held-Out Generalization Benchmarks**: `experiments/densenet_b4_v5/` & `experiments/source_holdout_b4/` (**PRESERVED**)
* **Quarantined Montgomery Benchmark**: `data/downloads/montgomery/` (**PRESERVED**)

---

## 3. Production Dependency Analysis

* **Inference Engine (`backend/ml/inference.py`)**:
  * Loads exclusively Model D (`models/densenet121_frequency_v5.h5` or `experiments/densenet_frequency_v5/densenet121_frequency_v5.h5`).
  * Enforces the 6 Model D clinical classes: `COVID-19`, `Normal`, `Pleural Effusion`, `Pneumonia`, `Pulmonary Nodule / Mass`, `Tuberculosis`.
  * Generates Grad-CAM visual attention overlays without clinical diagnosis claims.
  * Discloses Montgomery external scanner-shift generalization limitations in response payload.
  * Contains zero legacy fallback to obsolete CNN or ResNet models.
* **API Endpoints (`backend/api/routes/predictions.py`)**:
  * Strictly wired to Model D inference service.
  * Robust input validation for image payloads.
* **Frontend Application (`frontend/src/pages/AnalyzePage.js`)**:
  * Updated to display Model D predictions, Grad-CAM overlays, six-class probability distributions, and domain shift limitations.
* **Test Suite (`backend/tests/`)**:
  * 31 automated tests passing covering inference, API validation, preprocessing, and error handling.

---

## 4. Proposed Cleanup Classification

### 4.1 DELETE (Safely Removable Temporary & Generated Material)
These files are generated cache files and temporary upload artifacts created during test runs. None contain source code or scientific results.

1. **Python Bytecode Caches**:
   * `backend/__pycache__/` (*.pyc)
   * `backend/api/__pycache__/` (*.pyc)
   * `backend/api/routes/__pycache__/` (*.pyc)
   * `backend/database/__pycache__/` (*.pyc)
   * `backend/ml/__pycache__/` (*.pyc)
   * `backend/tests/__pycache__/` (*.pyc)
   * `backend/utils/__pycache__/` (*.pyc)
   * `experiments/__pycache__/` (*.pyc)
   * `experiments/phase2e/__pycache__/` (*.pyc)
   * `experiments/phase2f/__pycache__/` (*.pyc)
2. **Pytest Execution Cache**:
   * `.pytest_cache/`
3. **Temporary Test Upload Artifacts**:
   * `uploads/0aaae2e1-ef26-4d5f-980e-a94a25b192a5_scan.jpg`
   * `uploads/3488478c-d13b-4b91-9547-900424609b83_scan.jpg`
   * `uploads/5b68a09e-c093-4435-b2b1-a3ba2afa8c93_scan.jpg`
   * `uploads/7c99fc72-c317-4e44-823d-950371b64e0e_valid_cxr.jpg`
   * `uploads/7fdfab3e-1fd6-434e-8f06-88f8b3762bc6_valid_cxr.jpg`
   * `uploads/b4e02ed0-7110-4bce-9008-3a3d2c969fe7_valid_cxr.jpg`
   * `uploads/d306e101-76da-44f1-93de-1d1e3456183f_scan.jpg`
   * `uploads/dc6dbc84-cc03-42b6-90e8-8878956b36e9_scan.jpg`
   * `uploads/e0994163-e540-4a18-92ef-62ffb37f8208_scan.jpg`
   * `uploads/fafe4723-dcc1-481e-96d2-266358285850_valid_cxr.jpg`
   *(A `.gitkeep` file will be created in `uploads/` to maintain the directory structure for runtime).*
4. **Duplicate Exact Model Checkpoints in `models/`**:
   * `models/cnn_model.h5` (8.38 MB, identical duplicate of `models/baseline_archive/cnn_model_old_5class.h5`)
   * `models/densenet_model.h5` (40.17 MB, identical duplicate of `experiments/models/densenet121_unified.h5`)
   * `models/resnet_model.h5` (214.25 MB, identical duplicate of `experiments/models/resnet50_unified.h5`)

### 4.2 ARCHIVE (Historical Artifacts Moved to `experiments/archive/`)
Historical checkpoints that should NOT reside in the production `models/` directory, but must be retained for scientific lineage and reproducibility:

1. **`models/ResNet_phase2_best.h5`** (230.76 MB) &rarr; `experiments/archive/legacy_models/ResNet_phase2_best.h5`
   * *Reason*: Historical Phase 2 ResNet checkpoint. Must not be deleted, but should not sit in active production `models/`.
2. **`models/baseline_archive/`** &rarr; `experiments/archive/legacy_models/baseline_archive/`
   * `cnn_model_old_5class.h5` (8.38 MB)
   * `resnet_model_old_5class.h5` (230.76 MB)
   * `training_results_old_5class.json`
   * *Reason*: Preserves the early 5-class baselines in the experiment archive rather than the production deployment folder.
3. **`experiments/archive/dann_checkpoints/`** (Already safely placed in archive)
   * `checkpoint_lambda_0.01.h5`
   * `checkpoint_lambda_0.1.h5`
   * `checkpoint_lambda_1.0.h5`
   * Intermediate validation JSONs
   * *Reason*: DANN hyperparameter tuning lineage.

### 4.3 KEEP (Active Production & Scientific Thesis Assets)
All of the following are strictly preserved:

1. **Production Assets**:
   * `models/densenet121_frequency_v5.h5` (Model D weights)
   * `models/training_results.json` (Model D metrics)
   * `backend/` (All source files: `main.py`, `ml/inference.py`, `ml/preprocessing.py`, `ml/class_mapping.json`, routes, database, utils, etc.)
   * `backend/tests/` (All 31 unit, API, preprocessing, and Model D integration tests)
   * `frontend/` (All React source files, public assets, configurations)
   * `pytest.ini` (Configures test execution to target `backend/tests` cleanly)
   * `Dockerfile`, `.dockerignore`
   * `README.md`
2. **Scientific Datasets & Manifests**:
   * `experiments/data/unified_manifest_v5.csv` (Protected V5 dataset manifest)
   * `experiments/data/unified_manifest*.csv` (V1–V4 manifests documenting dataset lineage)
   * `data/raw/` (Original multi-source scans)
   * `data/downloads/` (External datasets including Montgomery benchmark)
3. **Thesis Experiments & Evidentiary Artifacts**:
   * `experiments/densenet_frequency_v5/` (Master Model D training outputs, Grad-CAM, ROC/PR curves)
   * `experiments/densenet_v5/` (Phase 3B ERM outputs)
   * `experiments/densenet_coral_v5/` (Phase 4B Deep CORAL outputs)
   * `experiments/densenet_dann_v5/` (Phase 4C DANN outputs)
   * `experiments/densenet_hybrid_v5/` (Phase 4E Hybrid outputs)
   * `experiments/densenet_b4_v5/` & `experiments/source_holdout_b4/` (Source-held-out experiments)
   * `experiments/results/` (All 103 JSON results, markdown reports, confusion matrices, Grad-CAM samples)
   * `experiments/*.py` (All experimental training, evaluation, reconstruction, and failure analysis scripts)
   * `experiments/models/` (Unified baseline reference checkpoints `densenet121_unified.h5`, `resnet50_unified.h5`)
4. **Academic Documentation**:
   * `docs/thesis/THESIS.md` (Master thesis manuscript)
   * `docs/LungAI_MTech_Project_Report.md` & DOCX / PDF builds
   * `docs/LungAI_Viva_Preparation_Guide.docx`
   * `docs/LungAI_Presentation.pptx` & `LungAI_Presentation.pptx`
   * `docs/extracted_figures/` & `docs/generated_figures/` & `docs/images/`
   * `docs/generate_ppt.py` (Restored generator script)
   * `reports/` (Clinical report templates and baseline summaries)

### 4.4 REVIEW (Intentionally Preserved Under Observation)
1. **`LungAI_Presentation.pptx` (in repo root)**:
   * 1.13 MB presentation deck located at the repository root. A separate 53 KB presentation deck exists at `docs/LungAI_Presentation.pptx`. Both are preserved because the root presentation contains rich embedded slides for thesis defense.
2. **`PROJECT_STRUCTURE.md` & `PROJECT_CLEANUP_AUDIT.md`**:
   * Repository documentation created during earlier auditing. Preserved in root.

---

## 5. Security & Secret Audit Findings

* **Scanned Files**: 416 source/text files across backend, frontend, experiments, and docs.
* **Scan Rules**: Private keys, AWS keys, GitHub tokens, generic API keys, hardcoded passwords, database credentials.
* **Findings**:
  * `backend/.env.example:6` — Commented-out PostgreSQL connection example (`# DATABASE_URL=postgresql+asyncpg://...`). Non-sensitive template.
  * `backend/database/connection.py:15` — Commented-out PostgreSQL connection example. Non-sensitive template.
* **Active Secrets Detected**: **0** (Zero active secrets or credentials detected).
* **Gitignore Status**: `backend/.env`, `*.db`, `*.sqlite3` are strictly ignored.

---

## 6. Gitignore Optimization

To prevent accidental staging of large binary model files (exceeding GitHub's 100 MB limit) while ensuring all thesis manifests, code, and metrics are preserved, `.gitignore` is verified with:
```gitignore
# Python bytecode & environments
__pycache__/
*.py[cod]
*.pyo
venv/
.venv/
env/

# Testing & Coverage
.pytest_cache/
.coverage
coverage/
.ipynb_checkpoints/

# Node
frontend/node_modules/
node_modules/
frontend/build/
.next/
dist/
build/
.cache/

# Local databases & environment
.env
backend/.env
frontend/.env.local
*.db
*.sqlite3

# Uploads
uploads/*
!uploads/.gitkeep
backend/uploads/*
!backend/uploads/.gitkeep

# Logs & temp
*.log
logs/
*.tmp
*.temp
*.bak
*.swp
*~
.DS_Store
Thumbs.db
.vscode/
.idea/

# Large model binaries (>100MB cannot be pushed to standard GitHub)
*.h5
*.keras

# Large dataset raw images
data/raw/
data/processed/
data/downloads/
```

---

## 7. Next Actions (Post-Approval)

Upon receiving user approval, execution will proceed through:
1. **Phase 13**: Execute approved file deletions (`__pycache__`, `.pytest_cache`, temporary test `uploads/*.jpg`, redundant duplicate `.h5` files in `models/`).
2. **Phase 13 (cont.)**: Move legacy models (`ResNet_phase2_best.h5` and `models/baseline_archive/`) to `experiments/archive/legacy_models/`.
3. **Phase 14**: Run full test validation (`pytest -q` &rarr; 31 passed).
4. **Phase 15**: Verify `git diff` and ensure `experiments/data/unified_manifest_v5.csv` and Model D checkpoints are completely untouched.
5. **Phase 16**: Create branch `chore/lungai-repository-cleanup`.
6. **Phase 17**: Run final test verification on new branch.
7. **Phase 18**: Stage approved files and commit.
8. **Phase 19**: Push branch `chore/lungai-repository-cleanup` to `origin`.
9. **Phase 20**: Generate `PROJECT_CLEANUP_FINAL_REPORT.md` and display final summary.
