# Phase 2E: BIMCV-COVID19+ Ingestion Specification & Quality Standard

**Project**: LungAI Disease Detector  
**Scope**: Ingestion Protocol & Schema Requirements for Prospective Dataset V5 Release  
**Governance Status**: `BIMCV_STATUS = ACCESS_PENDING`  
**Artifact**: `experiments/results/phase2e_bimcv_ingestion_spec.json`  

---

## 1. Required Manifest Schema

Every candidate BIMCV scan evaluated for admission into Unified Dataset V5 must populate the following verified fields:

| Field Name | Type | Description / Standard |
| :--- | :--- | :--- |
| `patient_id` | `string` | De-identified patient code prefixed with `bimcv_pt_` |
| `study_id` | `string` | Encounter / examination accession identifier |
| `image_id` | `string` | Unique image filename identifier prefixed with `bimcv_` |
| `source_dataset` | `string` | Literal identifier `BIMCV_COVID19+` |
| `original_label` | `string` | Raw clinical diagnostic text from Valencian Health Authority EMR |
| `clinical_label` | `string` | Standardized target class `COVID-19` |
| `label_provenance` | `string` | Categorical: `MOLECULAR_CONFIRMED` or `DOCUMENTED_BIMCV_COVID` |
| `rt_pcr_status` | `string` | Laboratory confirmation status (`POSITIVE`) |
| `imaging_date` | `string` | Date of chest radiograph acquisition |
| `confirmation_date` | `string` | Date of RT-PCR specimen collection |
| `date_difference_days` | `integer` | Absolute days between radiograph and PCR confirmation ($|\Delta| \le 7$) |
| `view` | `string` | Standardized projection (`PA`, `AP`, `LATERAL`, `OTHER`, `UNKNOWN`) |
| `projection` | `string` | Acquisition posture (`ERECT`, `SUPINE`, `SEMI-ERECT`) |
| `image_modality` | `string` | Planar radiography modality (`DX` or `CR`) |
| `image_format` | `string` | Storage format (lossless 8-bit/16-bit PNG) |
| `image_dimensions` | `tuple` | Width and height in pixels |
| `acquisition_metadata` | `object` | Hospital center code and scanner manufacturer |
| `exclusion_reason` | `string / null` | Reason code if rejected (null if accepted) |

---

## 2. Ingestion Acceptance Rules

1. **Modality Verification**: Planar CXR only. Axial CT slices and volumetric reconstructions are automatically excluded (`CT_MODALITY_EXCLUDED`).
2. **Molecular Confirmation**: Must possess positive RT-PCR confirmation within $\pm 7$ days of imaging (`TEMPORAL_WINDOW_EXCEEDED`).
3. **Projection Normalization**: Lateral projections (`LL`, `RL`) are quarantined (`LATERAL_PROJECTION_EXCLUDED`). Ambiguous views are tagged `VIEW_UNCERTAIN`.
4. **Duplicate Protection**: Exact MD5 matches and perceptual dHash near-duplicates (Hamming distance $\le 3$) against existing COVID-19 archives are purged.
5. **Cross-Split Patient Isolation**: Partitioning enforces zero patient overlap across Train, Validation, and Test splits.
