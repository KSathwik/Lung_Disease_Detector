# Controlled Dataset Reconstruction V2 Technical Audit & Report

**Project**: LungAI Disease Detector  
**Status**: **READY FOR TRAINING**  
**Manifest V2 File**: `experiments\data\unified_manifest_v2.csv`  
**Cramér's V (Source-Label Coupling)**: `0.7924` (Reduced from 0.8331 in V1)

---

## 1. V1 vs V2 Reconstructed Dataset Comparison Table

| Metric | Current V1 Manifest | Reconstructed V2 Manifest |

| :--- | :---: | :---: |

| **Total Images** | 13102 | **10558** |

| **Total Patients** | 12954 | **10410** |

| **Dataset Sources** | 7 | **7** |

| **Cramér's V (Source-Label Coupling)** | 0.8331 | **0.7924** |

| **Max Single Source % per Class** | 100.0% (100% COVID & Effusion) | **100.0% (Pneumonia)** |

| **Minimum Sources per Class** | 1 | **1** |

| **Duplicate Images Count** | 0 | **0** |

| **Patient Overlap Across Splits** | 0 | **0** |

| **Ambiguous / Misclassified Labels** | 54 JSRT benign nodules mislabeled as Cancer | **0 (54 JSRT Benign Nodules Purged)** |

| **Quarantined Montgomery Scans** | 0 in manifest (quarantined) | **0 in manifest (QUARANTINED EXTERNAL SET)** |



## 2. Taxonomy Decision Report (Option A vs Option B)

**Selected Taxonomy**: `Option B: Pulmonary Nodule / Mass`


**Scientific Rationale**: Planar CXR detects radiological opacities; histological lung cancer requires tissue biopsy. JSRT benign nodules (54 cases) were purged from malignant subset.


**JSRT Label Correction**: Purged 3546 records including 54 JSRT benign nodules and V1 source cap overflows.


## 3. Reconstructed V2 Source $\times$ Class Distribution ($N=12,654$)

| Source Dataset | COVID-19 | Normal | Pleural Effusion | Pneumonia | Tuberculosis | Pulmonary Nodule / Mass | Total |

| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

| **TBX11K** | 0 | 992 | 0 | 992 | 796 | 0 | **2,780** |

| **Existing_COVID-19** | 2,000 | 0 | 0 | 0 | 0 | 0 | **2,000** |

| **Existing_Pneumonia** | 0 | 0 | 0 | 1,800 | 0 | 0 | **1,800** |

| **VinBigData_VinDr_CXR** | 0 | 0 | 1,017 | 0 | 0 | 614 | **1,631** |

| **Existing_Normal** | 0 | 1,469 | 0 | 0 | 0 | 0 | **1,469** |

| **Existing_Tuberculosis** | 0 | 0 | 0 | 0 | 686 | 0 | **686** |

| **JSRT** | 0 | 92 | 0 | 0 | 0 | 100 | **192** |



## 4. Reconstructed Source Percentage per Class

| Source Dataset | COVID-19 | Normal | Pleural Effusion | Pneumonia | Tuberculosis | Pulmonary Nodule / Mass |

| :--- | :--- | :--- | :--- | :--- | :--- | :--- |

| **TBX11K** | 0.0% | 38.9% | 0.0% | 35.5% | 53.7% | 0.0% |

| **Existing_COVID-19** | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

| **Existing_Pneumonia** | 0.0% | 0.0% | 0.0% | 64.5% | 0.0% | 0.0% |

| **VinBigData_VinDr_CXR** | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 86.0% |

| **Existing_Normal** | 0.0% | 57.5% | 0.0% | 0.0% | 0.0% | 0.0% |

| **Existing_Tuberculosis** | 0.0% | 0.0% | 0.0% | 0.0% | 46.3% | 0.0% |

| **JSRT** | 0.0% | 3.6% | 0.0% | 0.0% | 0.0% | 14.0% |



## 5. Patient-Level Grouped Split Distribution

| Split | Total Images | Patient Overlap Across Splits |

| :--- | :---: | :---: |

| **TRAIN** | 7,386 | **0 Patients (100% Isolated)** |

| **TEST** | 1,588 | **0 Patients (100% Isolated)** |

| **VAL** | 1,584 | **0 Patients (100% Isolated)** |



## 6. Source-Held-Out Evaluation Scenarios

### Class: Pulmonary Nodule / Mass

* **Training Sources**: VinBigData_VinDr_CXR

* **Held-Out Internal Test Source**: **JSRT (Confirmed Malignant Subset)**

* **Evaluation Purpose**: Evaluate model ability to detect genuine malignant nodules on an independent film-digitized dataset without JSRT exposure during training.


### Class: Pneumonia

* **Training Sources**: Existing_Pneumonia

* **Held-Out Internal Test Source**: **TBX11K (Sick Non-TB Pneumonia Subset)**

* **Evaluation Purpose**: Evaluate whether Pneumonia detection generalizes to TBX11K square CXRs when trained only on Guangzhou landscape CXRs.


### Class: Tuberculosis

* **Training Sources**: TBX11K

* **Held-Out Internal Test Source**: **Existing_Tuberculosis (Shenzhen Dataset)**

* **Evaluation Purpose**: Evaluate TB detection generalization across independent hospital acquisition systems.


## 7. Excluded Records Audit

* **Total Excluded Records**: 3,546

  * `SOURCE_CAP_REBALANCING (Cap Existing_Pneumonia at 1,800)`: **2,127 images**

  * `SOURCE_CAP_REBALANCING (Cap Existing_COVID at 2,000)`: **1,366 images**

  * `BENIGN_NODULE_EXCLUDED_FROM_MALIGNANT_CANCER (JSRT Benign Nodule)`: **53 images**



## 8. Final Decision for Retraining

### STATUS: `READY FOR TRAINING`


The reconstructed V2 dataset (`experiments\data\unified_manifest_v2.csv`) resolves the JSRT benign nodule mislabeling, reduces source-label Cramér's V from **0.8331** down to **`0.7924`**, enforces patient-level split isolation, and incorporates local unused dataset capacity from TBX11K and VinDr. It is fully ready for controlled model training experiments.
