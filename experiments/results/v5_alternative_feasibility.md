# LungAI: Dataset V5 Feasibility Plan & Alternative Strategy

**Project**: LungAI Disease Detector (M.Tech Thesis)  
**Scope**: Actionable V5 Reconstruction Strategy & Independent COVID Generalization Architecture  
**Date**: October 2026  
**Artifact**: `experiments/results/v5_alternative_feasibility.json`  

---

## 1. Executive Feasibility Summary

```
ALTERNATIVE_DATA_READY_FOR_V5 (LOCAL NIH EXPANSION)
COVID_EXTERNAL_TESTING_READY (STONY BROOK TCIA)
```

1. **Immediate Zero-Download V5 Reconstruction**:
   * We do not need to download risky or redundant datasets.
   * `data/raw/NIH_ChestX-ray14/` already holds **294 verified single-finding scans** with full metadata, patient IDs, and PA/AP view positions.
   * Adding these 294 scans increases total dataset size from **10,336 to 10,630 images**.
   * Global Cramér's V improves slightly from **0.7698 to 0.7638**.
2. **Solving the Diagnostic B4 Effusion Imbalance**:
   * In Diagnostic B4, VinDr had $931$ Effusion scans while NIH had only $86$, causing statistical collapse during NIH $ightarrow$ VinDr training.
   * Integrating the 70 local NIH Effusion scans expands the NIH Effusion cohort to **156 scans** (+81.4%), substantially strengthening cross-source transferability.
3. **Scientifically Rigorous COVID Generalization Strategy**:
   * Since `Existing_COVID-19` already absorbed the Cohen and early BIMCV datasets, training a model on "Cohen vs Kaggle" would be scientifically invalid (spurious self-overlap).
   * Instead, the M.Tech thesis will designate **Stony Brook University COVID-19 CXR (TCIA)** as the pure external evaluation benchmark ($N=1,384$).

---

## 2. Projected V5 Class and Source Distribution (Local Expansion)

| Clinical Class | V4 Images | Additional Local NIH Scans | Projected V5 Images | Independent Sources in V5 |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 2,547 | **+118** | **2,665** | 4 (Existing_Normal, TBX11K, JSRT, NIH) |
| **Pleural Effusion** | 1,017 | **+70** | **1,087** | 2 (VinBigData_VinDr, NIH) |
| **Pulmonary Nodule / Mass** | 684 | **+96** | **780** | 3 (VinBigData_VinDr, JSRT, NIH) |
| **Pneumonia** | 2,798 | **+10** | **2,808** | 3 (Existing_Pneumonia, TBX11K, NIH) |
| **Tuberculosis** | 1,348 | +0 | **1,348** | 2 (TBX11K, Existing_Tuberculosis) |
| **COVID-19** | 1,942 | +0 | **1,942** | 1 Training Domain / 1 External Benchmark |
| **TOTAL** | **10,336** | **+294** | **10,630** | **All 6 Classes Multi-Source or External-Tested** |

---

## 3. Actionable Execution Checklist

### Phase 1: Local NIH Expansion & V5 Reconstruction
- [ ] Ingest the 294 verified single-finding NIH scans from `data/raw/NIH_ChestX-ray14/` into `unified_manifest_v5.csv`
- [ ] Standardize view metadata (`view_position: PA` vs `AP`) across all 10,630 scans
- [ ] Enforce patient-level split (70/15/15, seed 42, zero cross-split leakage)
- [ ] Run full V5 audit and verify Cramér's V (0.7638)

### Phase 2: External COVID Benchmark Acquisition
- [ ] Download Stony Brook University COVID-19 CXR metadata from TCIA (CC BY 4.0, zero DUA required)
- [ ] Stage a sample of 200–500 Stony Brook frontal CXRs strictly for the external test suite
- [ ] Lock the external benchmark (zero exposure during training)

### Phase 3: Baseline Training & Domain Generalization
- [ ] Train controlled DenseNet-121 baseline on V5
- [ ] Evaluate bidirectional source-held-out transfer (VinDr $\leftrightarrow$ NIH Effusion and Nodule)
- [ ] Evaluate zero-shot generalization on Montgomery County (TB/Normal)
- [ ] Evaluate zero-shot generalization on Stony Brook (COVID-19)
