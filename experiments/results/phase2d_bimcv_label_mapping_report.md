# Phase 2D: BIMCV-COVID19+ Access Verification & Label Mapping Specification

**Project**: LungAI Disease Detector  
**Scope**: Clinical Ground Truth & Mapping Rules for BIMCV-COVID19+  
**Date**: October 2026  
**Artifact**: `experiments/results/phase2d_bimcv_label_mapping.json`  

---

## 1. Governance & Access Status

* **Access Status**: `ACCESS_PENDING`
* **Repository**: Valencian Region Medical Image Bank (BIMCV-COVID19+), Regional Ministry of Health, Valencia, Spain.
* **Licensing**: BIMCV Open Research License (academic & non-commercial research).
* **Access Prerequisites**:
  1. User account registration via the official BIMCV portal.
  2. Executed institutional Data Use Agreement (DUA).
  3. Authenticated WebDAV / SFTP endpoint credentials.
* **Compliance Enforcement**: In strict adherence to project governance and the **No-Blind-Download** rule, automated unauthenticated crawling is blocked. Acquisition remains suspended until credentials are authorized.

---

## 2. Clinical Ground Truth & Label Mapping Rules

To eliminate the risk of shortcut learning and label ambiguity, the following mapping specifications are established for prospective BIMCV ingestion:

| Criterion | Specification | Clinical / Scientific Rationale |
| :--- | :--- | :--- |
| **Target Class** | `COVID-19` | Sole target class; non-COVID cases in BIMCV will not be ingested to prevent control cohort contamination. |
| **Diagnostic Ground Truth** | **RT-PCR Confirmed Positive** | Imaging signs alone (e.g. peripheral ground-glass) are non-specific; molecular confirmation via RT-PCR within $\pm 7$ days is mandatory. |
| **Imaging Modality** | **Planar CXR (CR/DX)** | CT slices, 3D reconstructions, and ultrasound scans must be strictly excluded. |
| **Projections Allowed** | **PA and AP Frontal Views** | Lateral radiographs (LL/RL) must be strictly filtered out to maintain planar frontal consistency. |
| **Age Criteria** | **Adult Cohort ($\ge 15$ years)** | Pediatric anatomy introduces substantial structural confounders. |
| **View Tracking** | **Mandatory View Tagging** | Each ingested image must record `view_position: PA` or `view_position: AP` to support View-Controlled Evaluation B. |

---

## 3. Exclusion Matrix for Ingestion

* **CT Volumes / Slices**: Exclude automatically via DICOM modality tag (`Modality != 'CT'`).
* **Lateral CXRs**: Exclude automatically via DICOM `ViewPosition` (`ViewPosition not in ['PA', 'AP']`).
* **Unconfirmed Cases**: Exclude cases labeled as "COVID-19 suspicious" lacking laboratory RT-PCR molecular verification.
* **Duplicate Acquisitions**: Exclude identical series or repeat exposures taken within the same imaging minute.
