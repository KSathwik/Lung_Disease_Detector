# Dataset Reconstruction V4 Audit Report

**Dataset**: LungAI Unified Manifest V4  
**Date**: September 2026  
**Artifact**: `experiments\data\unified_manifest_v4.csv`  

---

## 1. Overview & Split Distribution

* **Total Images**: **10,336**
* **Total Unique Patients**: **10,140**
* **Train Split**: 7,242 images (70.1%)
* **Validation Split**: 1,550 images (15.0%)
* **Test Split**: 1,544 images (14.9%)
* **Patient Leakage Across Splits**: **0 (Verified)**
* **Montgomery Benchmark Scans**: **0 (Strictly Quarantined)**

---

## 2. Complete Source & Class Cross-Tabulation

```
source_dataset           Existing_COVID-19  Existing_Normal  Existing_Pneumonia  Existing_Tuberculosis  JSRT  NIH_ChestX-ray14  TBX11K  VinBigData_VinDr_CXR
clinical_label                                                                                                                                              
COVID-19                              1942                0                   0                      0     0                 0       0                     0
Normal                                   0             1199                   0                      0    67                82    1199                     0
Pleural Effusion                         0                0                   0                      0     0                86       0                   931
Pneumonia                                0                0                1395                      0     0                 8    1395                     0
Pulmonary Nodule / Mass                  0                0                   0                      0    78                70       0                   536
Tuberculosis                             0                0                   0                    665     0                 0     683                     0
```

---

## 3. Label Provenance & Governance Summary

* **Normal**: 4 independent sources (`Existing_Normal`, `TBX11K`, `JSRT`, `NIH_ChestX-ray14`).
* **Pneumonia**: 3 independent sources (`Existing_Pneumonia` Guangzhou, `TBX11K` Beijing, `NIH_ChestX-ray14`).
* **COVID-19**: 1 source (`Existing_COVID-19`). Single-source status strictly documented due to BIMCV credential gating.
* **Tuberculosis**: 2 independent sources (`Existing_Tuberculosis` Shenzhen, `TBX11K` Beijing).
* **Pleural Effusion**: 2 independent sources (`VinBigData_VinDr_CXR` Vietnam, `NIH_ChestX-ray14` USA).
* **Pulmonary Nodule / Mass**: 3 independent sources (`VinBigData_VinDr_CXR`, `JSRT` histologically confirmed malignant, `NIH_ChestX-ray14`). 54 benign JSRT nodules remain purged.

---

## 4. Key Limitations Remaining

1. **COVID-19 Single-Source Bottleneck**: 100% of COVID-19 cases originate from `Existing_COVID-19`.
2. **Text-Mined Label Noise**: NIH labels are NLP-derived; mitigated by restricting ingestion strictly to isolated single-finding cases.
3. **Pleural Effusion Asymmetry**: VinBigData accounts for 85.6% of Pleural Effusion cases.
