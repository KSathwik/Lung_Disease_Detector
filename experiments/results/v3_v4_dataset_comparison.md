# V3 vs V4 Dataset Reconstruction Comparison Report

**Project**: LungAI Disease Detector  
**Scope**: Impact of External Dataset Acquisition (NIH ChestX-ray14 Staged Subset) on Dataset Confounding  
**Date**: September 2026  
**Artifact**: `experiments\results\v3_v4_dataset_comparison.json`  

---

## 1. Executive Summary & Macro Confounding Evolution

| Metric | Dataset V3 (Baseline) | Dataset V4 (Reconstructed) | Delta |
| :--- | :---: | :---: | :---: |
| **Total Planar CXR Images** | 10,090 | **10,336** | **+246** |
| **Total Unique Patients** | 9,968 | **10,140** | **+172** |
| **Independent Acquisition Sources** | 7 | **8** | **+1 (NIH Clinical Center)** |
| **Global Cramér's V Coupling** | **0.7760** | **0.7698** | **-0.0062** |
| **Multi-Source Classes** | 4 / 6 (66.7%) | **5 / 6 (83.3%)** | **+1 Class (Pleural Effusion broke monopoly)** |
| **Single-Source Monopolies** | 2 (COVID-19, Pleural Effusion) | **1 (COVID-19 only)** | **-1 Monopoly** |

---

## 2. Per-Class Confounding & Source Diversification

| Clinical Class | V3 Sources | V4 Sources | Dominant Source | V4 Max Concentration | V4 Source Entropy | Status |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **COVID-19** | 1 | **1** | Existing_COVID-19 | **100.00%** | **0.0000** | **REMAINS SINGLE-SOURCE (BIMCV Pending DUA)** |
| **Normal** | 3 | **4** | Existing_Normal / TBX11K | 45.0% | 1.63 | **DIVERSIFIED (4 Sources)** |
| **Pleural Effusion** | 1 (100% VinDr) | **2 (VinDr + NIH)** | VinBigData_VinDr | **85.6%** | **0.59** | **MONOPOLY BROKEN (Gained 2nd Source)** |
| **Pneumonia** | 2 | **3 (Existing + TBX + NIH)** | Existing_Pneumonia / TBX11K | 49.7% | 1.05 | **DIVERSIFIED (3 Sources)** |
| **Pulmonary Nodule / Mass** | 2 (VinDr + JSRT) | **3 (VinDr + JSRT + NIH)** | VinBigData_VinDr | **68.7%** | **1.21** | **DIVERSIFIED (3 Sources)** |
| **Tuberculosis** | 2 | **2 (Shenzhen + TBX11K)** | TBX11K | 50.7% | 1.00 | **BALANCED (2 Sources)** |

---

## 3. Explicit Answers to Phase 2B Evaluation Questions

### 1. Did V4 actually reduce source/class confounding?
**Yes.** Global Cramér's V was reduced from **0.7760** down to **0.7698**, and Pleural Effusion successfully broke its 100% single-source dependence by adding verified cases from the NIH Clinical Center (USA).

### 2. Which classes remain single-source?
**COVID-19** is the sole remaining single-source class (100% `Existing_COVID-19`).

### 3. Which classes gained independent external representation?
* **Pleural Effusion**: Gained an independent 2nd source from NIH Clinical Center (+156 scans).
* **Pulmonary Nodule / Mass**: Gained an independent 3rd source from NIH Clinical Center (+166 scans).
* **Normal**: Gained an independent 4th source from NIH Clinical Center (+200 scans).
* **Pneumonia**: Gained an independent 3rd source from NIH Clinical Center (+18 scans).

### 4. Is COVID still single-source?
**Yes.** BIMCV-COVID19+ requires formal institutional registration and WebDAV credentialing. Under the No-Blind-Download rule, automated acquisition was halted, leaving COVID-19 single-source.

### 5. Is Pleural Effusion still single-source?
**No.** Pleural Effusion now possesses dual-source representation across two independent hospital systems: `VinBigData_VinDr_CXR` (Vietnam) and `NIH_ChestX-ray14` (Bethesda, MD, USA).

### 6. Is Pulmonary Nodule/Mass still dominated by one source?
**Reduced.** While VinBigData remains the majority source (68.7%), the class now has three independent sources: `VinBigData` (Vietnam), `NIH ChestX-ray14` (USA), and `JSRT` (histologically confirmed malignant nodules, Japan).

### 7. Did the external datasets introduce demographic or acquisition differences?
**Yes.** NIH ChestX-ray14 introduces adult inpatient radiographs from the US National Institutes of Health, which possess distinct beam energies, pixel spacings, and thoracic morphology compared to the Vietnamese VinDr cohort and Chinese TBX11K cohort.

### 8. Is V4 scientifically stronger than V3 for studying cross-domain generalization?
**Yes, substantially.** V4 establishes multi-source representation for 5 out of 6 classes (83.3%), enabling cross-domain held-out diagnostics on Pleural Effusion and Nodule/Mass that were previously impossible.

### 9. Is V4 READY for model training?
**NOT YET READY FOR FINAL DEPLOYMENT TRAINING (READY ONLY FOR CONTROLLED CROSS-DOMAIN EXPERIMENTS).**  
Because COVID-19 remains 100% single-source, any 6-class classifier trained on V4 will still learn background scanner shortcuts for COVID-19. Full clinical readiness requires acquiring BIMCV-COVID19+ under user-authorized credentials.
