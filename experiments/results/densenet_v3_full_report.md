# DenseNet-121 Full V3 Training & Evaluation Report

**Model**: DenseNet-121  
**Training Cohort**: Full V3 Manifest Training Split ($N=7,061$ images; NO subsampling)  
**Validation Cohort**: Full V3 Validation Split ($N=1,515$ images)  
**Internal Test Cohort**: Full V3 Test Split ($N=1,514$ images)  
**External Benchmark**: Quarantined Montgomery County Dataset ($N=138$ images)  

---

## 1. Overall Internal Test Performance

* **Internal Test Accuracy**: **78.53%**
* **Macro Precision**: **74.26%**
* **Weighted Precision**: **84.43%**
* **Macro Recall**: **75.27%**
* **Weighted Recall**: **78.53%**
* **Macro F1-Score**: **73.28%**
* **Weighted F1-Score**: **80.54%**
* **Macro ROC-AUC**: **96.77%**
* **Macro PR-AUC**: **80.15%**

---

## 2. Per-Class Metrics Breakdown

| Class Name | Support | Precision | Recall | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | 292 | 100.00% | 84.93% | 100.00% | **91.85%** |
| **Normal** | 370 | 90.00% | 90.00% | 96.77% | **90.00%** |
| **Pleural Effusion** | 142 | 50.00% | 69.72% | 92.78% | **58.24%** |
| **Pneumonia** | 420 | 90.88% | 71.19% | 97.26% | **79.84%** |
| **Pulmonary Nodule / Mass** | 89 | 26.74% | 56.18% | 90.39% | **36.23%** |
| **Tuberculosis** | 201 | 87.91% | 79.60% | 98.32% | **83.55%** |

---

## 3. Montgomery External Validation Results ($N=138$)

* **Exact Tuberculosis Recall**: **0.00%** (0/58)
* **Exact Normal Specificity**: **0.00%** (0/80)
* **Binary Abnormal Sensitivity**: **100.00%** (58/58)

### Active Tuberculosis Predictions ($N=58$):
* **Pulmonary Nodule / Mass**: 51 scans (87.9%)
* **Pleural Effusion**: 7 scans (12.1%)

### Normal Control Predictions ($N=80$):
* **Pulmonary Nodule / Mass**: 79 scans (98.8%)
* **Tuberculosis**: 1 scans (1.2%)
