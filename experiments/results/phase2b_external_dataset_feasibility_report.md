# Phase 2B External Dataset Feasibility & Compatibility Report

**Project**: LungAI Disease Detector  
**Scope**: External Dataset Acquisition Feasibility & Compatibility Audit  
**Date**: September 2026  
**Artifact**: `experiments\results\phase2b_external_dataset_feasibility.json`  

---

## 1. Candidate External Dataset Feasibility Matrix

| Dataset | Candidate classes | Patients | Images | Label provenance | Patient IDs | View info | Format | Access/license | Source independence | Mapping confidence | Acquisition status | Decision |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- | :---: | :--- | :--- | :---: | :--- | :---: |
| **NIH ChestX-ray14** | Pleural Effusion, Nodule/Mass, Normal, Pneumonia | 30,805 | 112,120 | NegBio NLP text-mined (10-18% noise) | Yes | Frontal only (67k PA, 44k AP) | PNG (1024x1024) | CC0 / Public Domain | High (NIH Clinical Center, USA) | High (isolated) | Staged subset verified & acquired | **PARTIALLY READY** |
| **BIMCV-COVID19+** | COVID-19 | 1,311 | 2,460 | RT-PCR molecular confirmation linked to EMR | Yes | Mixed (PA, AP, Lateral, CT) | DICOM | Open Research (DUA / Registration required) | Critical (Valencia Region, Spain) | High (RT-PCR) | Blocked (Requires credentialed registration) | **NEEDS VERIFICATION** |
| **PadChest** | Normal, Pneumonia, Pleural Effusion, Nodule/Mass | 67,000+ | 160,000+ | UMLS concepts (27% manual, 73% NLP) | Yes | 6 positions (PA, AP, Lateral, etc.) | DICOM | Research Agreement (Formal application) | High (Hospital San Juan, Spain) | Moderate | Blocked (Requires formal research approval) | **NEEDS VERIFICATION** |
| **Montgomery County** | Tuberculosis, Normal | 138 | 138 | Clinical culture / clinic records | Yes | 100% PA frontal | PNG | Public Research | High (Montgomery County HHS, USA) | High | Pristine external benchmark | **REJECTED (QUARANTINED)** |

---

## 2. In-Depth Candidate Analysis & Governance

### A. NIH ChestX-ray14 (NIH Clinical Center, Bethesda, MD, USA)
* **Access & Storage**: 112,120 images (~42 GB across 12 tar.gz archives). Downloaded official metadata `Data_Entry_2017_v2020.csv` (9 MB) and official download script `batch_download_zips.py`.
* **Label Integrity**: Labels mined via NegBio NLP from diagnostic reports. Documented 10–18% label noise. To guarantee high precision, strict filtering is enforced: only **single-finding isolated cases** are eligible for ingestion. Multi-label combinations (e.g. `Effusion|Infiltration`) are rejected.
* **Taxonomy Alignment**:
  * `Effusion` matches `Pleural Effusion` (156 single-finding scans in batch 001).
  * `Nodule` (106 scans) and `Mass` (60 scans) map to `Pulmonary Nodule / Mass` (NOT cancer).
  * `No Finding` maps to `Normal` (200 single-view control scans sampled).
  * `Pneumonia` (18 scans).
* **Sample Quality Audit**: 10 representative scans streamed and inspected: 100% valid 1024x1024 planar CXRs, mean intensity 94.9–171.3, std 42.8–78.1, zero corrupted files, zero duplicates with V3.
* **Decision**: **PARTIALLY READY**. Staged single-finding cohort extracted and incorporated into V4.

### B. BIMCV-COVID19+ (Valencian Region Medical Image Bank, Spain)
* **Status**: Critical scientific value (only independent European COVID-19 source).
* **Technical Constraints**: Contains axial CT volumes and lateral CXRs requiring multi-stage DICOM filtering (`Modality in ['CR', 'DX']` and `PatientPosition in ['PA', 'AP']`).
* **Governance Blocker**: Access requires academic user registration, signed institutional Data Use Agreement (DUA), and authenticated WebDAV credentials.
* **Decision**: **NEEDS VERIFICATION / PENDING CREDENTIALED REGISTRATION**. In accordance with the No-Blind-Download rule, automated acquisition cannot proceed without user credentials. COVID-19 remains single-source until credentials are provided.

### C. PadChest (Hospital San Juan de Alicante, Spain)
* **Status**: High potential for Normal, Pneumonia, Effusion, and Nodule/Mass.
* **Governance Blocker**: Access is gated behind a formal Research Use Agreement. Redistribution is strictly prohibited.
* **Decision**: **NEEDS VERIFICATION / PENDING RESEARCH ACCESS APPROVAL**. Acquisition halted under Rule 4.

### D. Montgomery County CXR (Department of Health and Human Services, MD, USA)
* **Status**: Quarantined external validation benchmark ($N=138$).
* **Governance Guarantee**: Strictly excluded from all internal training, validation, test, and reconstruction manifests. Contamination count: **0**.
