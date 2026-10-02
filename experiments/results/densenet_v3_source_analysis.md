# DenseNet-121 V3 Source-Aware Diagnostic Analysis

**Experiment**: Controlled DenseNet-121 V3 Baseline  
**Evaluation Set**: Internal Test Split ($N=1,514$)  
**Purpose**: Diagnostic evaluation of per-source performance to inspect remaining shortcut vulnerabilities.  

---

## Source Performance Breakdown

| Source Dataset | Test Samples | Test Accuracy | Dominant Classes |
| :--- | :---: | :---: | :--- |
| **Existing_COVID-19** | 292 | **84.93%** | COVID-19 (292) |
| **Existing_Normal** | 180 | **93.33%** | Normal (180) |
| **Existing_Pneumonia** | 210 | **90.00%** | Pneumonia (210) |
| **Existing_Tuberculosis** | 99 | **91.92%** | Tuberculosis (99) |
| **TBX11K** | 492 | **75.00%** | Pneumonia (210), Normal (180), Tuberculosis (102) |
| **JSRT** | 21 | **52.38%** | Pulmonary Nodule / Mass (11), Normal (10) |
| **VinBigData_VinDr_CXR** | 220 | **63.18%** | Pleural Effusion (142), Pulmonary Nodule / Mass (78) |

---

## Key Diagnostic Observations:
1. **TBX11K vs Existing_Pneumonia**: Diagnostic assessment of whether Pneumonia accuracy remains consistent across Guangzhou pediatric landscape CXRs and TBX11K square CXRs.
2. **VinDr Effusion & Nodule Consistency**: Evaluation of high-resolution radiologist consensus records in the absence of external candidate datasets.
3. **Domain Coupling Footprint**: Remaining source-class coupling (`Cramér's V = 0.7760`) means that single-source classes (COVID-19, Effusion) retain characteristic source-specific acquisition signatures.
