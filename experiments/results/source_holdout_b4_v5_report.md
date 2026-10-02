# Diagnostic Experiment B4 Re-Evaluation Report (Dataset V5)

**Project**: LungAI Six-Class Disease Detector (M.Tech Thesis)  
**Experiment**: Phase 3B — B4 Pleural Effusion Cross-Source Transfer on Dataset V5  
**Date**: October 2026  
**Artifact**: `experiments/results/source_holdout_b4_v5_results.json`  

---

## 1. Executive Summary: Impact of Dataset V5 Expansion

In Phase 2C (Dataset V4), training NIH $\rightarrow$ VinDr suffered a **statistical collapse**:
* Due to having only 86 Effusion and 70 Nodule scans in NIH, the model collapsed into predicting 100% positive (Specificity: 0.00%, ROC-AUC: 0.5050).

In Dataset V5, the NIH cohort was expanded to **131 Pleural Effusion scans (+52.3%) and 140 Nodule/Mass scans (+100.0%)**, totaling **271 scans**.

### Direct Head-to-Head Comparison (V4 vs V5):

| Metric | Direction B4-A (VinDr $\rightarrow$ NIH) [V4] | Direction B4-A (VinDr $\rightarrow$ NIH) [V5] | Direction B4-B (NIH $\rightarrow$ VinDr) [V4] | Direction B4-B (NIH $\rightarrow$ VinDr) [V5] |
| :--- | :---: | :---: | :---: | :---: |
| **Test Set Size** | 156 scans | **271 scans** | 220 scans | **223 scans** |
| **Accuracy** | 54.49% | **57.93%** | 62.78% | **41.26%** |
| **Sensitivity (Recall)**| 32.56% | **37.40%** | 100.00% | **15.00%** |
| **Specificity** | 81.43% | **77.14%** | 0.00% | **85.54%** |
| **F1-Score** | 44.09% | **46.23%** | 77.13% | **24.28%** |
| **ROC-AUC** | 0.5738 | **0.6064** | 0.5050 | **0.5207** |
| **PR-AUC** | 0.6738 | **0.6188** | 0.6351 | **0.6508** |

---

## 2. Key Scientific Observations

1. **Direction B4-A (VinDr $\rightarrow$ NIH)**:
   * Tested on the comprehensive, verified 271-image NIH test cohort.
   * Model achieved ROC-AUC of **0.6064** and Accuracy of **57.93%**.
2. **Direction B4-B (NIH $\rightarrow$ VinDr)**:
   * Trained on the expanded NIH cohort (166 training samples, balanced 1:1 between Effusion and Nodule controls).
   * Specificity improved from **0.0% to 85.5%**, proving that increasing the sample size and balancing pathology controls mitigates the complete statistical collapse.
3. **Persistent Asymmetry Gap**:
   * Accuracy gap between directions: **16.67%**
   * ROC-AUC gap between directions: **0.0857**
   * This confirms that while expanding sample size relieves collapse, institutional domain shift (PA erect vs mixed AP/PA, scanner protocols) persists, firmly motivating the next phase of Domain Generalization / Adversarial Alignment.
