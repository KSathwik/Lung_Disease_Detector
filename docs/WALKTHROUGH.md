# Lung Disease Detector — Execution & Model Evaluation Walkthrough

## Executive Summary
This document provides the complete end-to-end execution, evaluation, medical safety audit, and system validation summary for the **Lung Disease Detection System**. Both the **Custom 4-Block CNN** and the **ResNet50 Transfer Learning** architectures were trained, tuned, and evaluated on a held-out test set of 1,630 real clinical X-ray images.

- **Selected Best Model**: **ResNet50 (Fine-Tuned Transfer Learning)**
- **Held-Out Test Set Accuracy**: **95.21%**
- **Held-Out Test Set AUC-ROC**: **99.39%**
- **Test Set F1-Score**: **95.17%**
- **Backend Test Suite Status**: **22 / 22 Tests Passed (100%)**

---

## 1. Dataset Structure & Stratified Audit

A comprehensive pre-training audit was conducted across the raw dataset directory (`data/raw`). 10,864 total chest radiograph images were cataloged across 5 active disease categories:

| Disease Class | Image Count | Dataset Share (%) | Status |
| :--- | :--- | :--- | :--- |
| **Pneumonia** | 4,273 | 39.33% | Active |
| **COVID-19** | 3,616 | 33.28% | Active |
| **Normal** | 1,583 | 14.57% | Active |
| **Tuberculosis** | 700 | 6.44% | Active |
| **Lung Cancer** | 692 | 6.37% | Active |
| **COPD** | 0 | 0.00% | Empty Directory |
| **Pleural Effusion** | 0 | 0.00% | Empty Directory |
| **TOTAL** | **10,864** | **100.00%** | **5 Active Classes** |

### Preprocessing & Normalization Pipeline:
1. **Denoising & Contrast Enhancement**: Applied Gaussian blur (`(3,3)`) and **CLAHE** (Contrast Limited Adaptive Histogram Equalization with `clipLimit=2.0`, `tileGridSize=(8,8)`) on single-channel uint8 contiguous memory arrays to enhance X-ray bone structure and soft-tissue boundaries.
2. **Standardization**: Resized images to `(224, 224, 3)` using bilinear interpolation and applied **ImageNet channel normalization** ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
3. **Data Splitting**: Strict stratified division: **70% Training (7,604 images)**, **15% Validation (1,630 images)**, **15% Held-Out Test Set (1,630 images)** using random seed `42`.

---

## 2. Model Training & Evaluation Metrics

Both model architectures were trained and evaluated against identical validation and held-out test sets:

| Performance Metric | Custom CNN Model | ResNet50 Model (Selected) | Delta ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Training Accuracy** | 90.82% | 97.68% | +6.86% |
| **Validation Accuracy** | 78.04% | **95.28%** | **+17.24%** |
| **Held-Out Test Accuracy** | 78.22% | **95.21%** | **+16.99%** |
| **Test Precision** | 78.50% | **95.18%** | **+16.68%** |
| **Test Recall** | 78.22% | **95.21%** | **+16.99%** |
| **Test F1-Score** | 72.16% | **95.17%** | **+23.01%** |
| **Test AUC-ROC** | 88.50% | **99.39%** | **+10.89%** |
| **Composite Score** | 0.7919 | **0.9603** | **+0.1684** |

---

## 3. ResNet50 Per-Class Held-Out Test Performance

The fine-tuned ResNet50 model demonstrated high diagnostic sensitivity across all 5 active classes on the held-out test set (1,630 unseen images):

| Target Condition | Precision | Recall (Sensitivity) | F1-Score | Test Support |
| :--- | :--- | :--- | :--- | :--- |
| **COVID-19** | **0.96** | **0.98** | **0.97** | 543 |
| **Lung Cancer** | **1.00** | **1.00** | **1.00** | 104 |
| **Pneumonia** | **0.96** | **0.97** | **0.96** | 641 |
| **Normal** | **0.91** | **0.90** | **0.91** | 237 |
| **Tuberculosis** | **0.92** | **0.81** | **0.86** | 105 |
| **Macro Average** | **0.95** | **0.93** | **0.94** | 1,630 |
| **Weighted Average** | **0.95** | **0.95** | **0.95** | 1,630 |

---

## 4. Medical Error & Safety Analysis

> [!WARNING]
> **Clinical Safety & Regulatory Scope**: This system is designed as an educational and research prototype. It is NOT FDA/CE certified as a primary diagnostic device. All predictions MUST be reviewed by a certified radiologist or qualified clinical practitioner.

### Critical Safety Findings:
1. **Zero False Negative Rate for Lung Cancer**: ResNet50 achieved 100% sensitivity and 100% specificity for `Lung Cancer` on test set images, ensuring critical oncological findings are not missed.
2. **High COVID-19 & Pneumonia Sensitivity**: High recall rates (**98% COVID-19**, **97% Pneumonia**) minimize false negative triage risks in acute respiratory care.
3. **Tuberculosis Moderate Recall (81%)**: Lower sample representation for Tuberculosis (6.44% of total dataset) resulted in an 81% recall rate. In clinical deployment, low-confidence predictions ($\le 0.70$) trigger automatic triage flags recommending sputum culture / GeneXpert confirmation.

---

## 5. Artifacts Generated

All models, metrics, and visualization artifacts have been built and saved to the project directory:

- **Model Files**:
  - `models/resnet_model.h5` — Production inference engine (ResNet50 best weights)
  - `models/cnn_model.h5` — Custom CNN baseline model
  - `models/training_results.json` — Comprehensive metrics payload
- **Visualizations**:
  - `docs/confusion_matrix_ResNet.png` — ResNet50 test set confusion matrix
  - `docs/confusion_matrix_CNN.png` — Custom CNN baseline confusion matrix
  - `docs/training_curves.png` — Loss & accuracy convergence plots
- **Database Storage**:
  - `lung_disease.db` — SQLite database populated with `ModelMetrics` records via SQLAlchemy ORM.

---

## 6. System & API Verification

The backend test suite was executed via `pytest`:
```bash
.venv\Scripts\python.exe -m pytest backend/tests/
```
**Results**: `22 passed, 0 failed in 3.26s`.

### Validated Backend Functionality:
- `GET /` — API root status check
- `GET /api/v1/health` — System health monitoring
- `POST /api/v1/patients/` — Patient registration & management
- `POST /api/v1/predict` — Image upload, CLAHE preprocessing, model inference, and report creation
- SQLite database schema & async session management

---

## 7. Production Readiness Checklist

- [x] Data cleaning and single-channel contiguous CLAHE enhancement implemented.
- [x] Multi-threaded CPU parallel data loading implemented.
- [x] Models trained & evaluated on completely held-out test split (15%).
- [x] Metric persistence in DB schema implemented.
- [x] Backend test suite 100% passing.
- [x] Medical safety disclaimer included across UI and API payloads.
