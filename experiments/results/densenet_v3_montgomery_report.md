# DenseNet-121 V3 Montgomery External Validation Report

**Cohort**: Montgomery County Chest X-ray Dataset (NLM/NIH)  
**Total External Scans**: 138 (58 Active TB, 80 Normal Controls)  
**Quarantine Integrity**: 100% Strictly Untouched During V3 Model Training and Selection.  

---

## 1. Exact Six-Class Classification Performance

* **Exact Tuberculosis Recall**: **0.00%** (0/58)
* **Exact Normal Specificity**: **0.00%** (0/80)

---

## 2. Binary Pathology Detection Performance (Abnormal vs Normal)

* **Binary Abnormal Sensitivity**: **100.00%** (58/58)
* **Binary Normal Specificity**: **0.00%**

*Critical Scientific Distinction: Binary abnormal detection flags whether the model recognizes pathology on film-digitized scans, whereas Exact Recall measures correct multiclass assignment to Tuberculosis.*

---

## 3. Predicted Class Distributions on External Cohort

### Active Tuberculosis Cases ($N=58$):
* **Pulmonary Nodule / Mass**: 56 scans (96.6%)
* **Pleural Effusion**: 2 scans (3.4%)

### Normal Control Cases ($N=80$):
* **Pulmonary Nodule / Mass**: 77 scans (96.2%)
* **Tuberculosis**: 3 scans (3.8%)
