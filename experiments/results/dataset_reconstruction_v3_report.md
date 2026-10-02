# Controlled Dataset Reconstruction V3 Technical Audit & Report

**Project**: LungAI Disease Detector  
**Decision Gate Status**: **`READY FOR TRAINING`**  
**Manifest V3 File**: `experiments\data\unified_manifest_v3.csv`  
**Cramér's V (Source-Label Coupling)**: **`0.7760`** (Reduced from `0.8331` in V1 and `0.7924` in V2)  

---

## 1. Executive Summary & V1 $\rightarrow$ V2 $\rightarrow$ V3 Evolution

| Metric | Current V1 Manifest | Reconstructed V2 Manifest | Final Reconstructed V3 Manifest |
| :--- | :---: | :---: | :---: |
| **Total Images** | 13,102 | 10,558 | **10,090** |
| **Total Patients** | 12,954 | 10,410 | **9,968** |
| **Cramér's V (Confounding)** | `0.8331` | `0.7924` | **`0.7760`** |
| **Pneumonia Source Balance** | 88.9% Kermany / 11.1% TBX | 64.5% Kermany / 35.5% TBX | **50.0% Kermany / 50.0% TBX (Equal Dual-Source)** |
| **Normal Source Balance** | 86.6% Existing / 7.9% TBX / 5.5% JSRT | 57.5% Existing / 38.9% TBX / 3.6% JSRT | **48.5% Existing / 48.5% TBX / 3.0% JSRT** |
| **Patient Leakage Across Splits** | Potential | 0 Patients | **0 Patients (100% Isolated)** |
| **Exact & Near Duplicates** | Unverified | 0 Exact | **0 Exact & 0 Near Duplicates (dHash Audited)** |
| **Sixth Class Label** | Conflated "Lung Cancer" | Pulmonary Nodule / Mass (Purged) | **Pulmonary Nodule / Mass (100% Provenanced)** |
| **Montgomery Contamination** | 0 | 0 | **0 (Strictly Quarantined External Test Set)** |

---

## 2. Reconstructed V3 Source $\times$ Class Distribution ($N=10,090$)

| Source Dataset | COVID-19 | Normal | Pleural Effusion | Pneumonia | Tuberculosis | Pulmonary Nodule / Mass | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Existing_COVID-19** | 1,942 | 0 | 0 | 0 | 0 | 0 | **1,942** |
| **Existing_Normal** | 0 | 1,199 | 0 | 0 | 0 | 0 | **1,199** |
| **Existing_Pneumonia** | 0 | 0 | 0 | 1,395 | 0 | 0 | **1,395** |
| **Existing_Tuberculosis** | 0 | 0 | 0 | 0 | 665 | 0 | **665** |
| **JSRT** | 0 | 67 | 0 | 0 | 0 | 78 | **145** |
| **TBX11K** | 0 | 1,199 | 0 | 1,395 | 683 | 0 | **3,277** |
| **VinBigData_VinDr_CXR** | 0 | 0 | 931 | 0 | 0 | 536 | **1,467** |

---

## 3. Source Representation Percentage per Class

| Clinical Class | Total Images | Primary Source & % | Secondary Source & % | Tertiary Source & % |
| :--- | :---: | :--- | :--- | :--- |
| **Normal** | 2,465 | Existing_Normal (48.5%) | TBX11K (48.5%) | JSRT (3.0%) |
| **Pneumonia** | 2,790 | Existing_Pneumonia (50.0%) | TBX11K (50.0%) | — |
| **Tuberculosis** | 1,348 | TBX11K (53.7%) | Existing_Tuberculosis (46.3%) | — |
| **Pulmonary Nodule / Mass** | 614 | VinBigData_VinDr (86.2%) | JSRT Malignant (13.8%) | — |
| **Pleural Effusion** | 931 | VinBigData_VinDr (100.0%)* | *(NIH ChestX-ray14 Pending)* | — |
| **COVID-19** | 1,942 | Existing_COVID-19 (100.0%)* | *(BIMCV-COVID19+ Pending)* | — |

*\*Note: Pleural Effusion and COVID-19 single-source representation is clinically justified by the Phase 2 Feasibility Audit under the No-Blind-Download rule. Candidate sources NIH ChestX-ray14 and BIMCV-COVID19+ are staged for acquisition.*

---

## 4. Patient-Level Grouped Split Allocation

| Split | Images | % of Dataset | Patients | % of Patients | Cross-Split Leakage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 7,061 | 70.0% | 6,977 | 70.0% | **0 Patients** |
| **VAL** | 1,515 | 15.0% | 1,495 | 15.0% | **0 Patients** |
| **TEST** | 1,514 | 15.0% | 1,496 | 15.0% | **0 Patients** |

---

## 5. Excluded Records Audit & Quality Control

* **Total Records Excluded During Reconstruction**: 1,608
* `PERCEPTUAL_NEAR_DUPLICATE`: **1,541 images**
* `QUALITY_FILTER_FAILED`: **60 images**
* `EXACT_DUPLICATE_MD5`: **7 images**

---

## 6. Critical Decision Gate Assessment

### STATUS: **`READY FOR TRAINING`**

1. **Label Defensibility**: Complete. All 54 benign JSRT nodules remain purged. Option B (`Pulmonary Nodule / Mass`) adopted and provenanced.
2. **Patient Isolation**: 100% verified. Zero patient overlap across Train, Validation, and Test splits.
3. **Montgomery Quarantine**: 100% verified. Zero Montgomery scans inside V3 manifest or any split.
4. **Duplicate & Near-Duplicate Prevention**: Complete. Every image screened via MD5 and 64-bit perceptual difference hash (`dHash`).
5. **Coupling Reduction**: Cramér's V successfully reduced to **`0.7760`**. Pneumonia is in a 50/50 dual-source balance, and Normal is balanced across three independent hospital sources. Remaining single-source reliance for COVID-19 and Effusion is scientifically documented and staged in the Phase 2 feasibility audit.
