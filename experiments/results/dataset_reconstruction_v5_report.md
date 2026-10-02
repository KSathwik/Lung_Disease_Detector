# LungAI: Dataset V5 Reconstruction & Audit Report

**Project**: LungAI Six-Class Disease Detector (M.Tech Thesis)  
**Scope**: Phase 3A — Unified Dataset V5 Construction via Local NIH ChestX-ray14 Expansion  
**Date**: October 2026  
**Artifact Manifest**: `experiments/data/unified_manifest_v5.csv`  
**Prior Baseline**: `experiments/data/unified_manifest_v4.csv`  

---

## 1. Executive Summary & Verification State

```
V5_RECONSTRUCTION_COMPLETE
ZERO_PATIENT_LEAKAGE: VERIFIED (0 cross-split patients)
MONTGOMERY_QUARANTINE: VERIFIED (0 scans)
DEDUPLICATION_VERIFIED: 83 perceptual near-duplicates excluded (Hamming distance <= 3)
```

| Metric | Dataset V4 | Dataset V5 | Delta |
| :--- | :---: | :---: | :---: |
| **Total Images** | **10,336** | **10,547** | **+211** (+2.04%) |
| **Total Unique Patients** | **10,140** | **10,270** | **+130** |
| **Train Set Images (Patients)** | 7,242 (7,097) | **7,398** (**7,189**) | +156 images |
| **Val Set Images (Patients)** | 1,550 (1,520) | **1,579** (**1,540**) | +29 images |
| **Test Set Images (Patients)** | 1,544 (1,523) | **1,570** (**1,541**) | +26 images |
| **Cramér's V (Confounding)** | 0.7698 | **0.7654** | **-0.0044** (Improved) |
| **Patient Leakage Across Splits** | 0 | **0** | **0** (Strict isolation) |
| **Montgomery External Cohort** | Quarantined (138) | **Quarantined (138)** | **Untouched** |

---

## 2. Resolving the Diagnostic B4 Sample-Size Imbalance

In Diagnostic Experiment B4 (Pleural Effusion cross-source transfer), model generalization suffered from an acute sample size disparity:
* VinDr provided **931 Effusion scans** (100% PA erect).
* NIH provided only **86 Effusion scans** in V4 (PA and AP mixture).

### Resolution in V5:
By ingesting local, verified, single-finding Effusion scans from `data/raw/NIH_ChestX-ray14/`:
* **NIH Pleural Effusion expanded from 86 to 131 scans (+52.33% increase)**.
* Total Pleural Effusion cohort in V5 reached **1,062 scans** across Vietnam (VinDr) and USA (NIH).
* Similarly, **NIH Pulmonary Nodule / Mass expanded from 70 to 140 scans (+100.00% increase)**.

| Pathology | VinDr Cohort | NIH Cohort (V4) | NIH Cohort (V5) | NIH Growth Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Pleural Effusion** | 931 | 86 | **131** | **+52.33%** |
| **Pulmonary Nodule / Mass** | 536 | 70 | **140** | **+100.00%** |

---

## 3. Class-by-Source Distribution Matrix (Unified V5)

```
source_dataset           Existing_COVID-19  Existing_Normal  Existing_Pneumonia  Existing_Tuberculosis  JSRT  NIH_ChestX-ray14  TBX11K  VinBigData_VinDr_CXR
clinical_label                                                                                                                                              
COVID-19                              1942                0                   0                      0     0                 0       0                     0
Normal                                   0             1199                   0                      0    67               171    1199                     0
Pleural Effusion                         0                0                   0                      0     0               131       0                   931
Pneumonia                                0                0                1395                      0     0                15    1395                     0
Pulmonary Nodule / Mass                  0                0                   0                      0    78               140       0                   536
Tuberculosis                             0                0                   0                    665     0                 0     683                     0
```

### Class Entropy and Dominance Breakdown:

* **COVID-19**: 1942 scans (+0 vs V4)
  * Independent Sources: 1 (Existing_COVID-19: 1942)
  * Dominant Source: `Existing_COVID-19` (100.0%)
  * Normalized Entropy: -0.0000

* **Normal**: 2636 scans (+89 vs V4)
  * Independent Sources: 4 (Existing_Normal: 1199, TBX11K: 1199, NIH_ChestX-ray14: 171, JSRT: 67)
  * Dominant Source: `Existing_Normal` (45.49%)
  * Normalized Entropy: 0.4749

* **Pleural Effusion**: 1062 scans (+45 vs V4)
  * Independent Sources: 2 (VinBigData_VinDr_CXR: 931, NIH_ChestX-ray14: 131)
  * Dominant Source: `VinBigData_VinDr_CXR` (87.66%)
  * Normalized Entropy: 0.1796

* **Pneumonia**: 2805 scans (+7 vs V4)
  * Independent Sources: 3 (Existing_Pneumonia: 1395, TBX11K: 1395, NIH_ChestX-ray14: 15)
  * Dominant Source: `Existing_Pneumonia` (49.73%)
  * Normalized Entropy: 0.3476

* **Pulmonary Nodule / Mass**: 754 scans (+70 vs V4)
  * Independent Sources: 3 (VinBigData_VinDr_CXR: 536, NIH_ChestX-ray14: 140, JSRT: 78)
  * Dominant Source: `VinBigData_VinDr_CXR` (71.09%)
  * Normalized Entropy: 0.3799

* **Tuberculosis**: 1348 scans (+0 vs V4)
  * Independent Sources: 2 (TBX11K: 683, Existing_Tuberculosis: 665)
  * Dominant Source: `TBX11K` (50.67%)
  * Normalized Entropy: 0.3333

---

## 4. Standardized View Position Distribution

All 10,547 scans now feature normalized `view_position` metadata:
* **PA (Posteroanterior)**: **5,883 scans** (55.8%)
* **AP (Anteroposterior)**: **1,523 scans** (14.4%)
* **UNKNOWN (Retrospective mixture)**: **3,141 scans** (29.8%)

### View Distribution Across Diseases:
| Clinical Label | PA | AP | UNKNOWN | Total |
| :--- | :---: | :---: | :---: | :---: |
| **COVID-19** | 0 | 0 | 1,942 | 1,942 |
| **Normal** | 1,415 | 22 | 1,199 | 2,636 |
| **Pleural Effusion** | 998 | 64 | 0 | 1,062 |
| **Pneumonia** | 1,402 | 1,403 | 0 | 2,805 |
| **Pulmonary Nodule / Mass** | 720 | 34 | 0 | 754 |
| **Tuberculosis** | 1,348 | 0 | 0 | 1,348 |

---

## 5. Methodological & Governance Confirmations

1. **V1–V4 Manifest Immutability**: All prior manifest files (`unified_manifest.csv`, `v2`, `v3`, `v4`) remain completely untouched.
2. **Deterministic Partition Splitting**: 48 patients who had previous scans in V4 were assigned strictly to their prior partition (zero cross-split movement). The 176 newly introduced patients were partitioned with seed 42 into 70% Train, 15% Val, and 15% Test.
3. **External Benchmarks Preserved**: Montgomery County ($N=138$) remains in strict isolation. Stony Brook University COVID-19 ($N=1,384$) is reserved exclusively as an external evaluation cohort.
4. **Baseline DenseNet-121 Preserved**: No models were retrained during this dataset construction phase.
