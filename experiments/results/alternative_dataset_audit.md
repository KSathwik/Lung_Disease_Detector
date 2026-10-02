# LungAI: Alternative Dataset Audit Report

**Project**: LungAI Disease Detector (M.Tech Thesis)  
**Scope**: Systematic Evaluation of Alternative Public CXR Collections & Provenance Audit  
**Date**: October 2026  
**Artifact**: `experiments/results/alternative_dataset_audit.json`  

---

## 1. Executive Summary & Forensic Discovery

```
PARTIAL_DATA_READY
```

Our empirical forensic audit uncovered critical dataset realities regarding the COVID-19 cohort:

1. **Candidate A (`COVID-19 Radiography Database` / Kaggle)**: **100% REDUNDANT**. It is the exact parent archive from which `Existing_COVID-19` in V1–V4 was extracted.
2. **Candidate B (`ieee8023/covid-chestxray-dataset` / Cohen et al.)**: **CONTAMINATED**. The Kaggle collection scraped Cohen's GitHub repo. Exactly **143 images** from Cohen's dataset are ALREADY embedded inside V4's `Existing_COVID-19` (e.g. `COVID-961` to `COVID-1043`). Ingesting Candidate B would cause **uncontrolled patient leakage**.
3. **Internal Provenance of V4 COVID-19**:
   * **BIMCV (Spain)**: **1,622 images (83.5%)**
   * **Cohen / GitHub**: **282 images (14.5%)**
   * **SIRM (Italy)**: **38 images (2.0%)**
4. **Candidate C (`NIH ChestX-ray14`)**: **READY LOCALLY**. Exactly **294 verified single-finding scans** (118 Normal, 70 Effusion, 96 Nodule/Mass, 10 Pneumonia) are ALREADY extracted on local disk in `data/raw/NIH_ChestX-ray14/` with ZERO V4 overlap.
5. **Candidate D (`Stony Brook University COVID-19` / TCIA)**: **GENUINE INDEPENDENT COVID COHORT**. 1,384 hospitalized COVID CXRs under CC BY 4.0 license, free from Kaggle/Cohen overlap.

---

## 2. Alternative Candidate Decision Matrix

| Candidate Dataset | Origin / Provider | Available Scans | Relevant Disease Labels | V4 Overlap Status | Decision | Primary Role |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **A. COVID-19 Radiography Database** | Tawsifur Rahman (Kaggle) | 3,616 COVID | COVID-19, Normal, Pneumonia | **100% Self-Overlap** (Parent of Existing_COVID) | **REJECTED** | Redundant baseline |
| **B. COVID-19 Image Data Collection** | Cohen et al. (GitHub `ieee8023`) | ~930 CXR/CT | COVID-19, ARDS, Pneumonia | **143 scans already in V4** | **REJECTED** | Patient leakage risk |
| **C. Additional NIH ChestX-ray14** | NIH Clinical Center (Bethesda, MD) | 294 Local / 112K Archive | Effusion, Nodule, Mass, Normal, Pneumonia | **0 Overlap** (Disjoint patients) | **ELIGIBLE_FOR_TRAINING** | Balance Effusion/Nodules in V5 |
| **D. Stony Brook COVID-19** | Stony Brook Univ. Hospital (TCIA) | 1,384 Frontal CXR | COVID-19 (Hospitalized Inpatient) | **0 Overlap** (Independent US hospital) | **ELIGIBLE_FOR_EXTERNAL_TESTING** | External COVID Generalization |
| **E. MIDRC-RICORD-1a** | RSNA / TCIA / AAPM | 120 Frontal CXR | COVID-19 (International Multi-Site) | **0 Overlap** (Independent cohort) | **ELIGIBLE_FOR_EXTERNAL_TESTING** | Multi-national benchmark |

---

## 3. Local NIH Scans Ready for Immediate V5 Integration

Without running any downloads, local inspection confirms 294 single-finding scans in `data/raw/NIH_ChestX-ray14/`:
* **Pleural Effusion**: **+70 scans** (Expands NIH Effusion in V5 from 86 to 156 scans).
* **Pulmonary Nodule / Mass**: **+96 scans** (+68 Nodule, +28 Mass, expanding NIH cohort from 70 to 166 scans).
* **Normal**: **+118 scans** (Expands NIH Normal from 82 to 200 scans).
* **Pneumonia**: **+10 scans** (Expands NIH Pneumonia from 8 to 18 scans).

These scans have verified Patient IDs, frontal view positions (PA and AP), and zero patient leakage into V4 partitions.
