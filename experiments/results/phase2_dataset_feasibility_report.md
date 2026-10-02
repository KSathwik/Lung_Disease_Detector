# Phase 2 Dataset Feasibility & Compatibility Audit Report

**Project**: LungAI Disease Detector  
**Scope**: Dataset Engineering & Candidate Evaluation  
**Status**: Completed  
**Artifact**: `experiments\results\phase2_dataset_feasibility.json`  

---

## 1. Executive Summary & Candidate Evaluation Matrix

| Candidate Dataset | Source & Provenance | License | Images / Patients | Taxonomy Match | Confounding Impact | Decision Under No-Blind-Download Rule |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **TBX11K (Simplified)** | Liu et al., ACM MM 2020 | Academic Open | 8,811 / 8,811 | Normal, Pneumonia, TB | **Very High** (breaks Pneumonia monopoly) | **ACCEPTED (Locally Verified & Audited)** |
| **VinDr-CXR (VinBigData)** | 17 Radiologist Consensus | PhysioNet | 4,394 / 4,394 | Pleural Effusion, Nodule/Mass | **High** (primary source for effusion & nodule) | **ACCEPTED (Locally Verified & Audited)** |
| **JSRT Pathology Set** | Histology Biopsy / Shiraishi | JSRT Open | 247 / 247 | Nodule/Mass (100), Normal (92) | **Moderate** (54 benign nodules purged) | **ACCEPTED (Pathology Audited)** |
| **NIH ChestX-ray14** | NIH Clinical Center (Wang 2017) | CC0 / Public | 112,120 / 30,805 | Effusion, Pneumonia, Nodule/Mass | **High** (2nd source for effusion/nodule) | **PENDING MANUAL VERIFICATION & STAGED SUBSET ACQUISITION** |
| **BIMCV-COVID19+** | Valencian Region (Spain) | Open Research | 2,460 / 1,311 | COVID-19 (RT-PCR confirmed) | **Critical** (breaks COVID-19 single-source) | **PENDING MANUAL VERIFICATION & REGISTRATION** |
| **Montgomery County** | Montgomery Health Dept / NLM | Open Research | 138 / 138 | Normal (80), TB (58) | **N/A** (Held-out external test) | **STRICTLY QUARANTINED (0 in Internal Manifests)** |

---

## 2. In-Depth Candidate Analysis

### A. NIH ChestX-ray14 (NIH Clinical Center, USA)
* **Access & Storage**: ~45 GB uncompressed across 112,120 images. Stored in NIH Box repository.
* **Label Integrity**: Mined automatically from diagnostic radiology reports via NegBio NLP. Oakden-Rayner (2020) demonstrated a 10%–18% label noise rate in comparison to consensus re-reading.
* **Taxonomy Alignment**:
  * `Effusion` matches `Pleural Effusion` (radiological finding).
  * `Nodule` and `Mass` map defensibly to `Pulmonary Nodule / Mass` (Option B).
  * `Pneumonia` has high label noise; requires exclusion of co-occurring edema/consolidation.
  * `No Finding` represents a healthy control source after filtering follow-ups.
* **Decision**: **PENDING MANUAL VERIFICATION & STAGED SUBSET ACQUISITION**. In accordance with the No-Blind-Download rule, downloading 45 GB of unverified data is deferred. A targeted manifest of single-finding scans is ready for download in Phase 2B.

### B. BIMCV-COVID19+ (Valencian Region Medical Image Bank, Spain)
* **Access & Storage**: Multi-gigabyte archive hosted by Generalitat Valenciana under Open Research terms.
* **Label Integrity**: Laboratory RT-PCR confirmed COVID-19 diagnoses with clinical follow-up. This represents the gold standard for clinical ground truth.
* **Taxonomy Alignment**: Directly provides verified `COVID-19` cases to break the current 100% single-source dominance.
* **Technical Constraints**: Contains axial CT slices and lateral projections. Requires automated DICOM tag filtering (`Modality == 'DX'` or `'CR'` and `PatientPosition in ['PA', 'AP']`).
* **Decision**: **PENDING MANUAL VERIFICATION & REGISTRATION**. Awaiting registration credentialing and automated DICOM filter validation.

### C. TBX11K (Verified Local Archive)
* **Availability**: 8,811 planar chest radiographs already present on local disk (`data/downloads/tbx11k-simplified/`).
* **Taxonomy Alignment**:
  * 3,800 `healthy` -> `Normal`
  * 3,800 `sick_but_no_tb` -> `Pneumonia` (non-TB pulmonary infections)
  * 1,211 `tb` -> `Tuberculosis`
* **Impact**: Ingesting verified non-overlapping records immediately establishes a 50/50 balance in Pneumonia (1,800 Existing_Pneumonia vs 1,800 TBX11K) and provides substantial Normal and TB multi-source diversity without download risk.
* **Decision**: **ACCEPTED FOR V3 INGESTION**.

---

## 3. Strict Quarantine Governance (Montgomery County)
* **Montgomery Scans in V3 Manifest**: Exactly **0** (Verified).
* **Isolation Guarantee**: Zero Montgomery images are used for training, validation, internal test, threshold tuning, or hyperparameter selection.
