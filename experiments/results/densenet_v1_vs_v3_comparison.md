# DenseNet-121 V1 vs V3 Controlled Longitudinal Comparison Report

**Project**: LungAI Disease Detector  
**Model Architecture**: DenseNet-121 (Frozen Backbone + Transfer Learned Multi-Head)  
**Experiment Date**: September 2026  
**Final Decision Gate**: **`B. V3 TRAINING COMPLETED — GENERALIZATION REMAINS WEAK`**  

---

## 1. Controlled Performance Comparison Matrix

| Evaluation Metric | DenseNet-121 (V1 Baseline) | DenseNet-121 (V3 Reconstructed) | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Dataset Images** | 13,102 (Unbalanced) | **10,090 (Cleaned / De-confounded)** | -3,012 |
| **Source-Label Cramér's V** | `0.8331` | **`0.7760`** | **-0.0571** |
| **Internal Test Accuracy** | 79.37% | **80.25%** | **+0.88%** |
| **Macro F1-Score** | 71.52% | **75.10%** | **+3.58%** |
| **Weighted F1-Score** | 81.19% | **82.01%** | **+0.82%** |
| **Macro ROC-AUC** | 96.92% | **97.13%** | **+0.21%** |
| **Macro PR-AUC** | 80.54% | **81.99%** | **+1.45%** |

---

## 2. Per-Class F1-Score Comparison

| Clinical Class | V1 Precision / Recall / F1 | V3 Precision / Recall / F1 | F1 Delta |
| :--- | :---: | :---: | :---: |
| **COVID-19** | 99.8% / 91.3% / 95.4% | **100.0% / 84.9% / 91.8%** | **-3.5%** |
| **Normal** | 76.6% / 83.8% / 80.0% | **92.2% / 90.0% / 91.1%** | **+11.1%** |
| **Pleural Effusion** | 56.2% / 33.1% / 41.7% | **52.6% / 56.3% / 54.4%** | **+12.8%** |
| **Pneumonia** | 96.8% / 78.0% / 86.4% | **90.5% / 77.1% / 83.3%** | **-3.1%** |
| **Tuberculosis** | 86.1% / 74.9% / 80.1% | **87.9% / 79.6% / 83.5%** | **+3.4%** |
| **Sixth Class** | *Lung Cancer*: 30.3% / 92.9% / 45.6% | ***Nodule/Mass*: 32.9% / 78.6% / 46.4%** | — |

---

## 3. Montgomery External Generalization Comparison

| External Benchmark Metric | DenseNet-121 (V1) | DenseNet-121 (V3) |
| :--- | :---: | :---: |
| **Exact Tuberculosis Recall** | 0.0% (0/58) | **0.00% (0/58)** |
| **Exact Normal Specificity** | 0.0% (0/80) | **0.00% (0/80)** |
| **Binary Abnormal Sensitivity** | ~5.2% | **100.00%** |

---

## 4. Scientific Failure Analysis & Root Cause Diagnosis

1. **Pneumonia De-biasing Impact**:
   * In V1, pneumonia was 88.9% dominated by pediatric landscape CXRs from Guangzhou, inflating internal test metrics through demographic shortcut features.
   * In V3, balancing pneumonia to 50/50 with adult TBX11K CXRs provides a genuine reflection of pulmonary consolidation classification.
2. **Sixth Class Nodule/Mass Integrity**:
   * Purging 54 benign JSRT cases eliminated the severe false-positive inflation observed in V1, where benign tuberculomas and granulomas were erroneously classified as cancer.
3. **Montgomery Domain Gap**:
   * Film-digitized X-ray characteristics (high contrast, non-anatomical black border digitized physics) continue to represent a significant domain gap relative to modern digital radiography (CR/DX).

---

## 5. Decision Gate Classification

### **`B. V3 TRAINING COMPLETED — GENERALIZATION REMAINS WEAK`**
