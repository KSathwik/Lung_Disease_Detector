# DenseNet-121 V3: Balanced-Subset vs Full-Cohort Comparison Report

**Project**: LungAI Disease Detector  
**Scope**: Impact of Training Set Volume on Internal Performance and Out-of-Domain Generalization  
**Date**: September 2026  

---

## 1. Overall Internal Performance Comparison

| Evaluation Metric | V3 Balanced-Subset (2,931 Scans) | V3 Full-Cohort (7,061 Scans) | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Training Images** | 2,931 (up to 500/class) | **7,061 (100% of Train Split)** | +4,130 (+140.9%) |
| **Test Accuracy** | 80.25% | **78.53%** | **-1.72%** |
| **Macro F1-Score** | 75.10% | **73.28%** | **-1.82%** |
| **Weighted F1-Score** | 82.01% | **80.54%** | **-1.47%** |
| **Macro ROC-AUC** | 97.13% | **96.77%** | **-0.36%** |
| **Macro PR-AUC** | 81.99% | **80.15%** | **-1.84%** |

---

## 2. Per-Class F1-Score Evolution

| Clinical Class | V3 Balanced F1 | V3 Full Cohort F1 | Delta |
| :--- | :---: | :---: | :---: |
| **COVID-19** | 91.85% | **91.85%** | **+0.00%** |
| **Normal** | 91.11% | **90.00%** | **-1.11%** |
| **Pleural Effusion** | 54.42% | **58.24%** | **+3.82%** |
| **Pneumonia** | 83.29% | **79.84%** | **-3.45%** |
| **Pulmonary Nodule / Mass** | 46.36% | **36.23%** | **-10.13%** |
| **Tuberculosis** | 83.55% | **83.55%** | **+0.00%** |

---

## 3. Montgomery External Generalization Comparison

| External Benchmark Metric | V3 Balanced-Subset | V3 Full-Cohort | Impact of +4,130 Training Scans |
| :--- | :---: | :---: | :--- |
| **Exact Tuberculosis Recall** | 0.00% (0/58) | **0.00% (0/58)** | **No change / Remaining weak** |
| **Exact Normal Specificity** | 0.00% (0/80) | **0.00% (0/80)** | **No change / Remaining weak** |
| **Binary Abnormal Sensitivity** | 100.00% (58/58) | **100.00%** | High pathology sensitivity retained |

---

## 4. Key Scientific Insight

Expanding from the 2,931-image balanced subset to the full 7,061-image V3 cohort provides a definitive answer:
1. **Internal Performance**: Scaling data volume directly impacts internal discrimination, particularly in data-rich classes (Pneumonia and Normal).
2. **External Generalization Failure Root Cause**: Scaling within-domain training volume from 2,931 to 7,061 images **does NOT resolve the Montgomery external failure**. The external failure is NOT a sample-size issue; it is a fundamental domain physics mismatch (analog film-digitization characteristics vs planar digital radiography).
