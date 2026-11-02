# LungAI Safe Disk-Space Cleanup Preview Report

**Document Date:** October 3, 2026  
**Auditor / Agent:** DeepMind Antigravity Safe Maintenance Agent  
**Branch:** `develop`  
**Purpose:** Dry-run preview identifying genuinely redundant temporary, cache, and obsolete backup artifacts while preserving 100% of the active LungAI working software, research protocols, test suites, and final deliverables.

---

## 1. Safe to Delete

The following files and directories are verified as genuinely redundant temporary files, build/bytecode caches, superseded presentation drafts, deprecated baseline plots from early un-curated prototypes, and historical dev cleanup reports. They are completely safe to remove without affecting project reproducibility, testing, or final thesis deliverables.

| Path | Type | Size | Reason |
| :--- | :--- | ---: | :--- |
| `scratch/__pycache__/` | Directory (Bytecode Cache) | 227.88 KB | Temporary Python compilation cache generated during audit script runs. |
| `LungAI_Presentation.pptx` | File (Root Duplicate) | 1,103.56 KB | Obsolete, un-synchronized 22-slide presentation draft superseded by `LungAI_Updated_Professional_Presentation.pptx`. |
| `docs/LungAI_Presentation.pptx` | File (Docs Duplicate) | 52.35 KB | Obsolete duplicate presentation deck superseded by `LungAI_Updated_Professional_Presentation.pptx`. |
| `docs/confusion_matrix_CNN.png` | File (Obsolete Plot) | 63.54 KB | Deprecated baseline confusion matrix from early uncurated V1 prototype. |
| `docs/confusion_matrix_ResNet.png` | File (Obsolete Plot) | 65.56 KB | Deprecated baseline confusion matrix from early uncurated V1 prototype. |
| `docs/training_curves.png` | File (Obsolete Plot) | 194.55 KB | Deprecated loss/accuracy curves from uncurated 5-class baseline. |
| `docs/archive/old_screenshots/01_landing_analyze_page.png` | File (Superseded Screenshot) | 348.48 KB | Early low-res UI mock superseded by `docs/screenshots/application/02_analyze.png`. |
| `docs/archive/old_screenshots/02_patient_management.png` | File (Superseded Screenshot) | 202.80 KB | Early low-res UI mock superseded by `docs/screenshots/application/07_patients.png`. |
| `docs/archive/old_screenshots/03_model_metrics.png` | File (Superseded Screenshot) | 158.83 KB | Early low-res UI mock superseded by `docs/screenshots/application/05_metrics.png`. |
| `docs/archive/old_screenshots/04_history_records.png` | File (Superseded Screenshot) | 164.77 KB | Early low-res UI mock superseded by `docs/screenshots/application/06_history.png`. |
| `PROJECT_CLEANUP_AUDIT.md` | File (Temporary Dev Report) | 11.42 KB | Intermediate development session audit notes from prior refactoring phases. |
| `PROJECT_CLEANUP_FINAL_REPORT.md` | File (Temporary Dev Report) | 5.69 KB | Intermediate development session report from prior refactoring phases. |
| `PROJECT_FINAL_ASSET_AND_CLEANUP_REPORT.md` | File (Temporary Dev Report) | 9.30 KB | Intermediate development session report from prior refactoring phases. |
| `PROJECT_FORENSIC_CLEANUP_REPORT.md` | File (Temporary Dev Report) | 12.97 KB | Intermediate development session report from prior refactoring phases. |
| `PROJECT_STRUCTURE.md` | File (Temporary Dev Report) | 7.04 KB | Outdated directory layout snapshot superseded by active repository state. |

**Total Safe to Delete:** **2.57 MB** (2,691,837 bytes across 15 items)

---

## 2. Requires Confirmation

The following files represent internal developer utilities, historical iteration manifests, and research benchmark archives. Because they may serve as historical audit trails or generation tools, they are held for user confirmation and will **NOT** be deleted without explicit instruction.

| Path | Type | Size | Why |
| :--- | :--- | ---: | :--- |
| `scratch/*.py` (31 scripts) | Directory (Scripts) | 338.45 KB | Contains one-off compilation scripts (`compile_markdown.py`, `build_docx_thesis.py`, chapter modules, and synchronization scripts). Kept because they document the exact build pipeline for the thesis. |
| `experiments/data/unified_manifest.csv` | File (Dataset Manifest V1) | 2.28 MB | Manifest for initial un-curated V1 dataset. Documents the forensic transition from V1 to V5. |
| `experiments/data/unified_manifest_v2.csv` | File (Dataset Manifest V2) | 2.20 MB | Manifest for V2 deduplicated iteration. Documents data forensics evolution. |
| `experiments/data/unified_manifest_v3.csv` | File (Dataset Manifest V3) | 2.57 MB | Manifest for V3 5-class cleaned cohort. Documents data forensics evolution. |
| `experiments/data/unified_manifest_v4.csv` | File (Dataset Manifest V4) | 2.37 MB | Manifest for V4 cohort prior to CT slice elimination. Documents data forensics evolution. |
| `experiments/archive/` | Directory (Research Archive) | 1.82 MB | Contains historical DANN lambda tuning checkpoints and 5-class training logs. |
| `docs/generate_ppt.py` | File (Script) | 24.32 KB | Early python-pptx presentation generation script. |

**Total Requiring Confirmation:** **11.58 MB** (12,143,211 bytes)

---

## 3. Must Preserve

The following files, directories, models, test suites, and documentation form the core foundation of the LungAI research platform and university submission. They are strictly protected:

| Path | Reason |
| :--- | :--- |
| `LungAI_MTech_Project_Report_Final_Corrected.docx` | Primary university submission thesis document (120 pages, bordered tables, embedded figures). |
| `LungAI_MTech_Project_Report_Final_Corrected.pdf` | Formatted, verified university thesis PDF (120 pages, dynamic footer page numbering). |
| `LungAI_Updated_Professional_Presentation.pptx` | Canonical 23-slide defense presentation deck synchronized with thesis narrative and metrics. |
| `README.md` | Primary repository documentation summarizing architecture, V5 cohort, Model D metrics, and gate parameters. |
| `backend/` | Complete FastAPI asynchronous backend application, API routes, models, services, and configuration. |
| `backend/ml/` | Machine learning inference engine, preprocessing pipeline, Grad-CAM generator, and CXR gate module. |
| `backend/tests/` | Complete 44-test automated verification suite (`test_cxr_gate.py`, `test_predictions.py`, `conftest.py`). |
| `frontend/` | Complete React 18 single-page application, interactive DICOM viewport, analysis, and metrics UI. |
| `experiments/` | Research experiment code, protocols, evaluation scripts, and cross-source generalization benchmarks. |
| `experiments/data/unified_manifest_v5.csv` | Canonical V5 dataset manifest ($N=10,547$ images, 10,270 patients, 6 classes, 0% leakage). |
| `experiments/cxr_gate/` | CXR gate manifest, evaluation dataset ($N=125$), and test verification metrics. |
| `experiments/densenet_frequency_v5/` | Model D frequency preprocessing evaluation artifacts, metrics JSON, and Montgomery evaluation. |
| `experiments/densenet_coral_v5/` | Deep CORAL latent covariance alignment experiment metrics and feature analyses. |
| `experiments/densenet_dann_v5/` | DANN adversarial domain adaptation benchmark metrics and negative result documentation. |
| `experiments/densenet_hybrid_v5/` | Hybrid frequency-domain + CORAL domain generalization benchmark metrics. |
| `models/` | Production model configurations, class mappings, and training histories. |
| `reports/` | Generated PDF clinical report templates and evaluation summaries. |
| `scripts/` | Diagram generation automation scripts (`generate_diagrams.py`). |
| `uploads/` | Local directory for temporary user uploads (`.gitkeep` preserved). |
| `docs/LungAI_MTech_Project_Report.md` & `docs/thesis/THESIS.md` | Master Markdown thesis sources synchronized with TOC, LOF, LOT, and 16 chapters. |
| `docs/FINAL_SUBMISSION_CONSISTENCY_AUDIT.md` | Final 20-point comprehensive submission consistency audit report. |
| `docs/FINAL_THESIS_DOCUMENTATION_AUDIT.md` | Forensic documentation audit report across IEEE/PubMed literature. |
| `docs/CXR_GATE_FINAL_FORENSIC_AUDIT.md` | Forensic audit of the two-stage CXR validation gate and cat defect resolution. |
| `docs/CXR_INPUT_VALIDATION_AUDIT.md` | Detailed architectural safety report for the semantic input validation gate. |
| `docs/diagrams/` | 12 high-resolution rendered architecture diagrams, UML diagrams, and Mermaid source files. |
| `docs/figures/` | 24 analytical performance figures, ROC/PR curves, t-SNE projections, and Montgomery failure panels. |
| `docs/generated_figures/` | 16 workflow diagrams and preprocessing progression panels. |
| `docs/screenshots/application/` | 16 high-resolution production application screenshots (Home, Analyze, Grad-CAM, History, Patients). |
| `docs/thesis/LITERATURE_MATRIX.md` | Evidence-based literature comparative matrix analyzing 24 peer-reviewed studies. |

---

## 4. Estimated Disk Recovery

- **Immediate Safe Disk Recovery:** **2.57 MB** (2,691,837 bytes)
- **Potential Additional Recovery (Pending User Review):** **11.58 MB** (12,143,211 bytes)
- **Current Total Working Project Size:** **~76.5 MB** (Clean, lean working repository)

---

## 5. Verification Assessment

All files classified under **Safe to Delete** are genuinely redundant duplicates, caches, or deprecated baseline plots. No production source code, test suites, research protocols, manifests, models, or documentation will be impacted.
