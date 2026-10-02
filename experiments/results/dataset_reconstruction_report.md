# Dataset Acquisition Feasibility & Label-Integrity Verification Report

**Project**: LungAI Disease Detector  
**Status**: Executable Acquisition Plan (Zero Model Retraining / Zero Downloads Performed)  
**Current Manifest Coupling**: Cramér's V = `0.8331`

---

## 1. Local Archive Inventory & Reconciled Image Counts

| Dataset Source | Manifest Count | Local Archive Location | Total Local Available | Unused Capacity |

| :--- | :--- | :--- | :--- | :--- |

| **Existing_COVID-19** | 3,366 | `covid19-radiography-database.zip` | 3,616 | **+250** |

| **Existing_Normal** | 1,469 | `chest-xray-pneumonia.zip` | 1,583 | **+114** |

| **Existing_Pneumonia** | 3,927 | `chest-xray-pneumonia.zip` | 4,273 | **+346** |

| **Existing_Tuberculosis** | 686 | `tuberculosis-tb-chest-xray-dataset.zip` | 700 | **+14** |

| **JSRT** | 245 | `data/downloads/jsrt` | 247 | **+2** |

| **TBX11K** | 1,780 | `data/downloads/tbx11k-simplified` | 11,700 | **+9,920** |

| **VinBigData_VinDr_CXR** | 1,629 | `data/downloads/vinbigdata` | 4,394 | **+2,765** |

| **Montgomery (Quarantined)** | 0 | `data/downloads/montgomery` | 138 | **+138** |



## 2. JSRT Pathology Metadata Audit & Scientific Evaluation

* **Total JSRT Cases**: 247

* **Pathology Breakdown**: {'malignant': 100, 'non-nodule': 93, 'benign': 54}

* **Key Audit Finding**: Out of 247 JSRT cases, exactly 100 are confirmed malignant lung cancer, 54 are benign nodules (tuberculomas, granulomas, hematomas), and 93 are healthy normal controls.


### Scientific Evaluation of 'Lung Cancer / Nodule' Class

**Is Combined Class Valid?**: `False`


**Rationale**: Equating radiological 'Nodule/Mass' opacities with confirmed 'Lung Cancer' is medically invalid. In general screening populations, up to 90-95% of radiologically detected pulmonary nodules are benign (e.g., calcified granulomas from prior TB/fungal infection, hamartomas, focal scarring). In the JSRT dataset, 35.1% of nodule images (54/154) represent benign lesions (tuberculomas, granulomas, hematomas). Mislabeling benign granulomas as 'Lung Cancer' introduces severe false-positive label corruption during training.


**Corrective Policy**:
1. Exclude the 54 benign JSRT nodule images from the Lung Cancer class.
2. For the project class 'Lung Cancer', restrict inclusion strictly to histologically confirmed malignant cases (e.g., JSRT 100 malignant nodule cases, biopsy-verified cancer cases).
3. If radiologic nodules/masses from VinDr-CXR, NIH, or PadChest are incorporated, the class MUST be formally renamed to 'Pulmonary Nodule / Mass' (a radiological finding class) or partitioned into two separate categories: 'Confirmed Lung Cancer' (histological diagnosis) vs 'Pulmonary Nodule/Mass' (radiological finding).


## 3. Candidate Dataset Feasibility & Licensing Verification

### NIH ChestX-ray14 (NIH Clinical Center (USA))

* **Licensing**: Public Domain (CC0 / Open Access)

* **Acquisition Type**: Hospital PACS Digital CR/DR (PA/AP)

* **Patient IDs**: Yes (patient_id 1 to 30805 available in Data_Entry_2017.csv)

* **Label Provenance**: NLP text mining from radiology reports (Estimated 90%+ precision)

* **Missing Classes Covered**: Pleural Effusion, Pulmonary Nodule/Mass, Pneumonia, Normal

* **Feasibility**: FEASIBLE & HIGH PRIORITY. Required to break the 100% VinDr lock on Pleural Effusion.


### BIMCV-COVID19+ (Valencian Region Medical Image Bank (Spain))

* **Licensing**: BIMCV Open Research License

* **Acquisition Type**: Multi-Hospital Digital CR/DR

* **Patient IDs**: Yes (sub-session identifiers)

* **Label Provenance**: RT-PCR laboratory confirmation + clinical diagnosis

* **Missing Classes Covered**: COVID-19, Normal

* **Feasibility**: FEASIBLE & CRITICAL. Required to break 100% single-source lockdown of COVID-19.


### CheXpert (Stanford Health Care (USA))

* **Licensing**: Stanford Open Research License

* **Acquisition Type**: Hospital Digital CR/DR (PA/AP/Lateral)

* **Patient IDs**: Yes (patientXXXXX)

* **Label Provenance**: CheXpert NLP report extractor (Uncertainty labels present)

* **Missing Classes Covered**: Pleural Effusion, Pneumonia, Normal

* **Feasibility**: FEASIBLE. Secondary source for Effusion and Pneumonia.


## 4. Exact Executable Acquisition Manifest

| Clinical Class | Target $N$ | Sources & Breakdown | Provenance & Status | Exclusion Rules |

| :--- | :--- | :--- | :--- | :--- |

| **COVID-19** | **3,000** | • Existing_COVID-19 (Local Archive): 2,000<br>• BIMCV-COVID19+ (External Candidate): 1,000 | • Scraped Public CXRs / COVID-19 Radiography DB (Locally Available)<br>• RT-PCR Lab Confirmed Multi-Hospital CXRs (Requires External Download) | Exclude duplicates between web-scraped COVID sets; exclude lateral projections and pediatric cases (<15 yrs). |

| **Pleural Effusion** | **2,000** | • VinBigData_VinDr_CXR (Local Archive): 1,000<br>• NIH ChestX-ray14 (External Candidate): 1,000 | • Consensus Radiologist Bounding Box Annotations (Locally Available)<br>• NLP Report Mining from NIH Clinical Center (Requires External Download) | Exclude complex multi-label co-occurrences containing Pneumothorax or Pneumonia to ensure single-disease focus. |

| **Lung Cancer / Nodule** | **1,200** | • JSRT (Local Archive - Confirmed Malignant): 100<br>• VinBigData_VinDr_CXR (Local Archive): 600<br>• NIH ChestX-ray14 (External Candidate): 500 | • Histologically Confirmed Malignant Lung Cancer (Locally Available)<br>• Radiologist Nodule/Mass Consensus Annotations (Locally Available)<br>• NLP Report Mapped Nodule/Mass Findings (Requires External Download) | STRICT EXCLUSION: Exclude 54 benign JSRT nodule cases (tuberculomas, granulomas). Exclude unverified opacities < 5mm. |

| **Pneumonia** | **3,000** | • Existing_Pneumonia (Local Archive): 1,500<br>• TBX11K (Local Archive): 500<br>• NIH ChestX-ray14 (External Candidate): 1,000 | • Guangzhou Women & Children's Medical Center (Locally Available)<br>• TBX11K Non-TB Sick Pneumonia Split (Locally Available)<br>• NIH Report Mapped Pneumonia Cases (Requires External Download) | Exclude duplicate pediatric images from Kaggle Pneumonia set; exclude viral vs bacterial ambiguous cases. |

| **Tuberculosis** | **2,000** | • Existing_Tuberculosis (Local Archive - Shenzhen): 700<br>• TBX11K (Local Archive): 800<br>• Belarus / NIAID TB Portal (External Candidate): 500 | • Shenzhen No. 3 People's Hospital TB Split (Locally Available)<br>• TBX11K Active Tuberculosis Split (Locally Available)<br>• National TB Portal Confirmed Active TB (Requires External Download) | STRICT EXCLUSION: Montgomery (138 scans) is STRICTLY QUARANTINED for external testing. Exclude inactive calcified scars. |

| **Normal** | **3,000** | • Existing_Normal (Local Archive): 1,200<br>• TBX11K (Local Archive): 600<br>• JSRT (Local Archive): 93<br>• NIH ChestX-ray14 (External Candidate): 1,107 | • Normal CXR Subset (Locally Available)<br>• TBX11K Healthy Control Split (Locally Available)<br>• JSRT Non-Nodule Healthy Controls (Locally Available)<br>• NIH No Finding Controls (Requires External Download) | Exclude any control scan showing subtle apical scarring, rib fractures, or spinal hardware. |



## 5. Patient-Level Internal & Out-of-Domain Evaluation Strategy

* **Internal Ratios**: 70% Train / 15% Validation / 15% Test

* **Patient Grouping**: STRICT PATIENT-LEVEL SEPARATION. All images for patient_id X are assigned exclusively to ONE split. Zero cross-split patient overlap.

* **Source Stratification**: Joint stratification by (clinical_label, source_dataset) to ensure equal source representation across Train, Val, and Test.

* **Out-of-Domain Validation**: For multi-source classes, reserve one entire source domain during validation (e.g. hold out BIMCV for COVID-19, or hold out JSRT for Lung Cancer) to verify domain generalization BEFORE external testing.

* **External Montgomery Quarantine**: Montgomery County CXR Dataset (N=138: 58 TB, 80 Normal) — 100% STRICTLY QUARANTINED. Zero exposure during dataset building or training.


## 6. Unresolved Blockers & Mitigation Plan

### ⚠️ NIH ChestX-ray14 & BIMCV-COVID19+ External Downloads

* **Impact**: Required to break 100% single-source locks on Pleural Effusion and COVID-19.

* **Mitigation**: Can proceed with Phase 1 local rebalancing (using unused VinDr + TBX11K local images) immediately while external downloads are queued.


### ⚠️ Lung Cancer Nodule vs Malignancy Taxonomy Alignment

* **Impact**: Including radiologic nodules without biopsy confirmation creates label noise.

* **Mitigation**: Enforce strict exclusion of 54 benign JSRT nodules; explicitly rename combined class to 'Pulmonary Nodule / Mass' if non-biopsied radiologic nodules are included.

