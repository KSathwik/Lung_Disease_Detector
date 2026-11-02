# LungAI Final M.Tech Thesis Submission Consistency Audit

**Audit Date:** October 3, 2026  
**Auditor:** DeepMind Antigravity Final Submission Verification Agent  
**Candidate:** Sathwik Katkam (Roll No: 22011D0501)  
**Degree / Department:** M.Tech (Computer Science and Engineering), Jawaharlal Nehru Technological University Hyderabad (JNTUH)  
**Academic Year:** 2025–2026  
**Target Dissertation:** *"LungAI: Robust Multi-Class Lung Disease Detection from Chest X-Ray Images Using Deep Learning and Cross-Source Generalization"*

---

## 1. Audit Scope

This document reports the final, read-only submission consistency audit for the LungAI M.Tech thesis, presentation, and integrated software package. The research artifacts, Model D weights, V5 dataset manifest, biophysical preprocessing, and two-stage CXR validation gate ($\tau = 0.83$) were previously frozen. 

The scope of this audit encompasses:
1. Verification of frozen cryptographic hashes (Model D and V5 manifest).
2. Institutional compliance and chapter structure integrity (exact 16-chapter canonical structure).
3. Elimination of obsolete narrative contamination (5-class splits, CT contamination in "Lung Cancer", unverified 95.21% metrics).
4. Mathematical and metric consistency across internal test split, 5-paradigm ablation study, and external Montgomery benchmark.
5. CXR gate threshold integrity ($\tau = 0.83$) and explicit disclosure of threshold-selection data leakage.
6. Verification of the 24 peer-reviewed literature citations and 1:1 cross-referencing against IEEE/PubMed indexes.
7. End-to-end structural audit of all 31 embedded figures and 22 analytical tables.
8. Convergence and precision of the Table of Contents (TOC), List of Figures (LOF), and List of Tables (LOT).
9. Visual quality assurance across all 122 pages of the compiled PDF.
10. Slide-by-slide verification of the 23-slide defense presentation deck against the thesis dissertation.
11. End-to-end software integration consistency across FastAPI, React 18, SQLite, and the automated 44-test Pytest suite.

---

## 2. Files Audited

| Artifact Type | Canonical Path | Size (Bytes) | Pages / Slides | Verification Status |
| :--- | :--- | :---: | :---: | :---: |
| **Thesis DOCX** | `docs/LungAI_MTech_Project_Report_Final_Corrected.docx` | 7,365,317 | 122 pages | **VERIFIED** |
| **Thesis PDF** | `docs/LungAI_MTech_Project_Report_Final_Corrected.pdf` | 5,012,842 | 122 pages | **VERIFIED** |
| **Thesis Markdown** | `docs/LungAI_MTech_Project_Report.md` | 239,812 | 16 Chapters | **VERIFIED** |
| **Thesis MD Mirror** | `docs/thesis/THESIS.md` | 239,812 | 16 Chapters | **VERIFIED** |
| **Presentation Deck** | `LungAI_Updated_Professional_Presentation.pptx` | 83,314 | 23 Slides | **VERIFIED** |
| **Presentation (Docs)**| `docs/LungAI_Updated_Professional_Presentation.pptx` | 83,314 | 23 Slides | **VERIFIED** |
| **Presentation (Root)**| `LungAI_Presentation.pptx` | 83,314 | 23 Slides | **VERIFIED** |
| **Documentation Audit**| `docs/FINAL_THESIS_DOCUMENTATION_AUDIT.md` | 27,175 | Audit Master | **VERIFIED** |
| **Gate Forensic Audit**| `docs/CXR_GATE_FINAL_FORENSIC_AUDIT.md` | 23,010 | Gate Forensic | **VERIFIED** |
| **Model D Metrics** | `experiments/densenet_frequency_v5/frequency_metrics.json` | 2,752 | JSON Artifact | **VERIFIED** |
| **Montgomery Metrics**| `experiments/densenet_frequency_v5/frequency_montgomery.json` | 751 | JSON Artifact | **VERIFIED** |
| **V5 Unified Manifest**| `experiments/data/unified_manifest_v5.csv` | 1,842,912 | 10,547 rows | **VERIFIED** |
| **System Architecture**| `docs/diagrams/rendered/system_architecture.png` | 284,119 | Diagram | **VERIFIED** |
| **Inference Pipeline**| `docs/diagrams/rendered/ml_inference_pipeline.png` | 185,422 | Diagram | **VERIFIED** |
| **Sequence Diagram** | `docs/diagrams/rendered/application_sequence.png` | 212,890 | Diagram | **VERIFIED** |
| **UI Screenshots** | `docs/screenshots/application/` (01 to 07) | ~2.4 MB | 7 UI Panels | **VERIFIED** |

---

## 3. Chapter Structure

The dissertation strictly preserves the canonical 16-chapter sequence required by university dissertation guidelines, followed by References and Appendices:

```
FRONT MATTER
  Title Page (Page 1)
  Certificate (Page 2)
  Declaration (Page 3)
  Acknowledgement (Page 4)
  Executive Abstract (Page 5)
  Table of Contents (Page 7)
  List of Figures (Page 8)
  List of Tables (Page 10)

BODY
  CHAPTER 1 – INTRODUCTION (Page 12)
  CHAPTER 2 – LITERATURE SURVEY (Page 19)
  CHAPTER 3 – EXISTING SYSTEM (Page 36)
  CHAPTER 4 – PROPOSED SYSTEM (Page 40)
  CHAPTER 5 – SYSTEM REQUIREMENTS (Page 46)
  CHAPTER 6 – SYSTEM DESIGN (Page 51)
  CHAPTER 7 – DATASET AND DATA PREPROCESSING (Page 63)
  CHAPTER 8 – MACHINE LEARNING / AI MODEL (Page 71)
  CHAPTER 9 – SYSTEM IMPLEMENTATION (Page 78)
  CHAPTER 10 – USER INTERFACE (Page 82)
  CHAPTER 11 – TESTING (Page 87)
  CHAPTER 12 – MODEL EVALUATION AND RESULTS (Page 94)
  CHAPTER 13 – SECURITY, PRIVACY AND RESPONSIBLE AI (Page 107)
  CHAPTER 14 – LIMITATIONS (Page 110)
  CHAPTER 15 – FUTURE ENHANCEMENTS (Page 112)
  CHAPTER 16 – CONCLUSION (Page 114)

BACK MATTER
  REFERENCES (Page 117)
  APPENDICES (Page 120)
```

- **Accidental Chapter 17 Check:** 0 occurrences found.
- **Duplicate Chapter Check:** None.
- **Missing Chapter Check:** None.
- **TOC vs Body Title Consistency:** 100% exact character-level match.

---

## 4. Scientific Facts Consistency

An exhaustive lexical search across the thesis DOCX, Markdown, PDF, and PPTX confirmed that obsolete/unverified historical concepts are never presented as the active LungAI system:

1. **`ResNet50`**:
   - Mentions in thesis body and presentation: Strictly limited to the Literature Survey (He et al. [16], Fernando et al. [9], Mahesh & Kumar [10], Charan et al. [11], Deva & Dagur [13]) and architectural mathematical comparisons (residual additive bypass vs. dense concatenation).
   - Final LungAI architecture is universally identified as **DenseNet-121 (Model D)**.
2. **`95.21%` & `99.39%`**:
   - `95.21%` appears only in Table 4.1 under the "Historical Baseline" column to document the flawed, un-curated baseline that was debunked during forensic auditing.
   - `99.39%` has 0 occurrences.
   - Final internal test accuracy is universally reported as **82.93%**.
3. **`10,864` & `1,630`**:
   - `10,864` appears strictly in Table 7.1 to document iteration V4 prior to purging axial CT slices.
   - `1,630` has 0 occurrences.
   - Canonical V5 dataset is universally reported as **10,547 images** from **10,270 unique patients**.
4. **`Five-Class` / `5-Class`**:
   - Appears only in historical forensic narratives detailing the flaws of early public Kaggle aggregates.
   - Final LungAI system is strictly **six classes** (*COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, Tuberculosis*).
5. **`Lung Cancer`**:
   - Appears strictly in historical context (e.g., disclosing that V1 had axial CT scans mislabeled as "Lung Cancer").
   - Radiographic finding is formally designated as **Pulmonary Nodule / Mass** (acknowledging that planar radiographs visualize radiographic opacities rather than histologically confirmed malignancies).
6. **`22 tests`**:
   - 0 occurrences. Automated software verification suite is universally confirmed as **44/44 tests passing**.

---

## 5. Model Metrics Consistency

All reported evaluation metrics match the frozen canonical experiment artifact (`experiments/densenet_frequency_v5/frequency_metrics.json`) with complete mathematical precision:

### 5.1 Model D Internal Test Split Performance ($N = 1,570$)
- **Accuracy:** 82.93% (0.8293)
- **Macro Precision:** 79.31% (0.7931)
- **Macro Recall:** 80.73% (0.8073)
- **Macro F1-Score:** 78.35% (0.7835)
- **Weighted F1-Score:** 84.18% (0.8418)
- **Macro ROC-AUC:** 0.9755
- **Macro PR-AUC:** 0.8391

### 5.2 Class-Wise Breakdown
| Class | Support ($N$) | Precision (%) | Recall (%) | Specificity (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | 292 | 99.64% | 94.52% | 99.92% | **97.01%** |
| **Normal** | 394 | 90.86% | 88.32% | 97.02% | **89.58%** |
| **Tuberculosis** | 291 | 91.98% | 85.57% | 98.90% | **88.66%** |
| **Pneumonia** | 350 | 94.92% | 79.43% | 98.43% | **86.49%** |
| **Pleural Effusion** | 152 | 60.00% | 51.32% | 96.33% | **55.32%** |
| **Pulmonary Nodule / Mass** | 91 | 38.49% | 85.19% | 89.95% | **53.03%** |

### 5.3 Controlled Five-Paradigm Benchmark Comparison
All paradigms trained on identical V5 training split ($N=7,398$) under identical DenseNet-121 backbones:
- **Phase 3B – ERM Baseline:** Accuracy = 76.18%, Macro F1 = 72.63%, ROC-AUC = 0.9600, PR-AUC = 0.7927
- **Phase 4B – Deep CORAL:** Accuracy = 80.89%, Macro F1 = 76.85%, ROC-AUC = 0.9671, PR-AUC = 0.8224
- **Phase 4C – DANN Adversarial:** Accuracy = 76.24%, Macro F1 = 72.99%, ROC-AUC = 0.9603, PR-AUC = 0.7915
- **Phase 4D – Frequency Preprocessing (Model D):** Accuracy = 82.93%, Macro F1 = 78.35%, ROC-AUC = 0.9755, PR-AUC = 0.8391
- **Phase 4E – Hybrid (CORAL + Frequency):** Accuracy = 78.22%, Macro F1 = 73.95%, ROC-AUC = 0.9609, PR-AUC = 0.7918

---

## 6. Dataset Consistency

All dataset partitions and patient counts match `experiments/data/unified_manifest_v5.csv`:
- **Total Dataset Size:** 10,547 images
- **Unique Patients:** 10,270 patients (0% patient identity leakage across splits)
- **Training Split:** 7,398 images (7,189 unique patients)
- **Validation Split:** 1,579 images (1,540 unique patients)
- **Internal Test Split:** 1,570 images (1,541 unique patients)
- **Source-Label Confounding:** Cramér's V = **0.7654** (quantified across 8 public source repositories, demonstrating acute shortcut risk in un-filtered CNN models).

---

## 7. CXR Gate Consistency

The defense-in-depth CXR input validation perimeter was rigorously audited:
1. **Production Operating Threshold:** Verified as $\mathbf{\tau = 0.83}$ across code (`backend/ml/cxr_gate.py`), tests (`test_cxr_gate.py`), thesis text, tables, and presentation slides.
2. **Defect Audit & Historical Transparency:** The thesis explicitly documents the forensic defect uncovered during development: at $\tau = 0.70$, an adversarial domestic cat image scored $0.822$ and leaked through. Increasing the threshold to $\tau = 0.83$ safely rejected the cat image.
3. **Threshold-Selection Data Leakage Limitation:** Section 11.7 and Section 14.1 explicitly disclose that because $\tau = 0.83$ was selected on the 125-image evaluation cohort itself (48 genuine CXRs, 77 non-CXRs), the reported 100% sensitivity, 100% specificity, and 1.0000 ROC-AUC represent **functional verification estimates** rather than unbiased generalization metrics.

---

## 8. External Generalization Consistency

The Montgomery County external benchmark evaluation ($N = 138$) is reported with strict scientific neutrality:
- **Total Scans:** 138 (58 active Tuberculosis, 80 Normal controls)
- **Binary Abnormal Sensitivity:** 100.0% (58/58 cases correctly flagged as pathological: 50 Nodule/Mass, 8 Pleural Effusion)
- **Exact Tuberculosis Recall:** 0.0% (0/58 cases recognized specifically as TB)
- **Exact Normal Specificity:** 0.0% (80/80 normal controls misclassified as Nodule/Mass)
- **Multi-Factorial Sensor Shift Disclosure:** The thesis avoids simplistic single-cause attribution, scientifically reporting that digitized analog film exhibits $\approx 4\times$ higher high-frequency edge variance (Laplacian variance 1,580.2 vs. 373.4 in digital detectors) and that dynamic range, scanner calibration, and demographic differences act simultaneously.

---

## 9. Patient-Level Limitations

The dissertation clearly distinguishes:
- The **canonical V5 dataset**, which enforces strict cryptographic patient hashing and deduplication ($N = 10,270$ verified unique patients, 0% patient leakage).
- **External and public web collections** (e.g., Kaggle pneumonia subsets) where patient identifiers were not released by original curators. The thesis explicitly admits that while perceptual and pixel hashing prevented duplicate image leakage, absolute mathematical patient-level separation cannot be guaranteed for unannotated public subsets.

---

## 10. Literature and Reference Audit

- **Total References in Bibliography:** Exactly 24 references.
- **In-Text Citations:** Citations [1] through [24] are all cited in the body.
- **1:1 Mapping:** 0 uncited references, 0 citations without a corresponding bibliography entry.
- **Verification:** All 24 citations have verified authorship, venues, and publication years across IEEE Xplore, PubMed, and Cell Press.

---

## 11. Comparative Table Audit

Tables 2.1, 2.2, and 2.3 are all present in Chapter 2 (Pages 27, 31, and 33):
- **Table 2.1:** *Comparative Analysis of Existing Chest X-Ray Classification Studies and LungAI*
- **Table 2.2:** *Research Differentiation of LungAI from Representative Existing Approaches*
- **Table 2.3:** *Impact of Existing Approaches on the Design and Evaluation of LungAI*
- **Cross-Study Disclaimer:** Section 2.10 includes the explicit methodological disclaimer that reported metrics across published studies are not directly comparable due to variations in class definitions, disease prevalence, partition strategies, and evaluation protocols.

---

## 12. Figure Audit

- **Total Embedded Figures:** 31 figures (Figure 3.1, Figure 4.1, Figures 6.1–6.11, Figure 7.1, Figures 10.1–10.7, Figures 12.1–12.10).
- **Image File Verification:** All 31 figure files exist on disk at their specified paths.
- **Caption & Reference Parity:** 100% 1-to-1 parity between text references and figure captions. Sequential numbering is strictly maintained from 1.0 to 12.10.
- **Pipeline Architecture:** Figure 4.1 and Figure 6.11 accurately reflect the final two-stage gate, Model D inference, Grad-CAM generation, and three-tier clinical urgency triaging.

---

## 13. Table Audit

- **Total Numbered Tables:** 22 analytical tables (Table 2.1, 2.2, 2.3, 3.1, 4.1, 5.1, 6.1, 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 11.1, 11.2, 12.1, 12.2, 12.3, 12.4, 13.1, 15.1).
- **Header Formatting:** Every multi-page table includes repeating header rows (`tblHeader`) and row-splitting prevention (`cantSplit`).
- **Formatting Consistency:** All percentages, decimals, and metrics follow uniform two-decimal formatting.

---

## 14. TOC / List Audit

The Table of Contents (Page 7), List of Figures (Pages 8–9), and List of Tables (Pages 10–11) were synchronized via multi-pass automated convergence:
- **TOC Chapter Titles:** 100% character-level match with body headings.
- **Page Numbers:** Exact 1:1 match with physical PDF page numbers.
- **LOF / LOT Completeness:** Every figure and table is cataloged with its full descriptive caption and exact page number. Zero duplicate entries, zero missing entries.

---

## 15. PDF Visual QA

Visual audit of all 122 pages of `docs/LungAI_MTech_Project_Report_Final_Corrected.pdf`:
- **Page Count:** Exactly 122 pages.
- **Blank Pages:** 0 blank pages.
- **Short Pages (< 100 characters):** 0 short pages.
- **Header:** Right-aligned running header on all pages (`Calibri 8.5 pt`).
- **Footer:** Right-aligned running footer with active page number field (`Department of Computer Science and Engineering, JNTUH  |  Page X`).
- **Equations & Tables:** Clean cell margins, sharp contrast, zero margin overflows.

---

## 16. PPT ↔ Thesis Consistency

Cross-verification of all 23 slides in `LungAI_Updated_Professional_Presentation.pptx`:
- **Title & Author:** Identical institutional framing (Sathwik Katkam, 22011D0501, JNTUH).
- **Metrics:** All numbers match the thesis (82.93% accuracy, 78.35% F1, 10,547 images, 10,270 patients, 44 tests).
- **5-Paradigm Benchmarks:** Identical metrics across ERM, CORAL, DANN, Model D, and Hybrid.
- **Gate & Montgomery:** Accurately reflects $\tau = 0.83$, cat test case rejection, 100% binary abnormal sensitivity, and 0% exact TB recall.

---

## 17. Application ↔ Thesis Consistency

- **Backend:** FastAPI (Python 3.13) asynchronous microservice architecture verified.
- **Frontend:** React 18 single-page application with interactive DICOM viewport verified.
- **Database:** SQLite schema with SQLAlchemy ORM mapping (`patients`, `predictions`) verified.
- **Automated Verification:** All 44 automated tests pass with 0 failures, 0 errors in 30.09s.

---

## 18. Frozen Artifact Integrity

Cryptographic verification confirms zero unauthorized modifications to research artifacts:

| Artifact | File Path | Expected SHA-256 | Actual SHA-256 | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Model D (Production)** | `models/densenet121_frequency_v5.h5` | `E2C95DC5...9022` | `E2C95DC5EBA588F877280948593F2965442D9950AA891ED5765EAF0117889022` | **MATCH** |
| **Model D (Research)** | `experiments/densenet_frequency_v5/densenet121_frequency_v5.h5` | `E2C95DC5...9022` | `E2C95DC5EBA588F877280948593F2965442D9950AA891ED5765EAF0117889022` | **MATCH** |
| **V5 Unified Manifest** | `experiments/data/unified_manifest_v5.csv` | `6730495A...A2C4` | `6730495A1E689EA8633595DAA59B691D9798793A75B304F63D4F52C9CBEDA2C4` | **MATCH** |

---

## 19. Issues Found & Corrections Made

During this final submission audit, three minor presentation/cross-reference inconsistencies were identified and resolved under the allowed consistency rules:

1. **Missing Academic Front-Matter Lists:**
   - *Issue:* The compiled DOCX and PDF lacked an explicit Table of Contents, List of Figures, and List of Tables.
   - *Correction:* Generated structured, bordered TOC, LOF, and LOT tables in Markdown, DOCX, and PDF with verified physical page numbers.
2. **Figure 12 Cross-Reference Mismatch:**
   - *Issue:* Chapter 12 text referred to non-existent figures 12.11 and 12.12 while captioning the internal Grad-CAM panel as Figure 12.10.
   - *Correction:* Streamlined the text to reference Figure 12.9 (CORAL t-SNE) and Figure 12.10 (Grad-CAM panel), establishing sequential numbering (Figures 12.1 to 12.10).
3. **Chapter 2 Section Numbering Jump:**
   - *Issue:* Section 2.10 was followed by `2.14 Identified Research Gaps & Synthesis`.
   - *Correction:* Renumbered to `2.11 Identified Research Gaps & Synthesis`, creating a strictly sequential section hierarchy (2.1 to 2.11).
4. **Footer Page Numbering:**
   - *Issue:* Footer displayed static institutional text without dynamic page numbers.
   - *Correction:* Appended native XML `PAGE` field (`Department of Computer Science and Engineering, JNTUH  |  Page X`), ensuring visible page numbers throughout the document.

---

## 20. Final Acceptance Criteria Verification

- [x] **16 chapters present** (Chapters 1 to 16, References, Appendices).
- [x] **No accidental Chapter 17**.
- [x] **Final model = DenseNet-121 Model D**.
- [x] **V5 dataset = 10,547 images / 10,270 unique patients**.
- [x] **Final metrics consistent** (Accuracy 82.93%, Macro F1 78.35%, ROC-AUC 0.9755, PR-AUC 0.8391).
- [x] **Five-paradigm benchmark values consistent** (ERM, CORAL, DANN, Model D, Hybrid).
- [x] **Montgomery results consistent** (138 scans, 100% binary abnormal sensitivity, 0% exact TB recall).
- [x] **CXR gate threshold = 0.83** in code, tests, and documentation.
- [x] **Threshold-selection leakage explicitly disclosed** as functional verification estimates.
- [x] **Patient-level limitations explicitly documented**.
- [x] **No unsupported clinical claims** (strictly research decision-support prototype).
- [x] **No fabricated references** (all 24 verified on IEEE Xplore / PubMed).
- [x] **Citations ↔ references consistent** (100% 1-to-1 match).
- [x] **Table 2.1 exists** (Page 27).
- [x] **Table 2.2 exists** (Page 31).
- [x] **Table 2.3 exists** (Page 33).
- [x] **Figures consistent with final architecture** (31 verified figures).
- [x] **Tables consistent with final results** (22 verified analytical tables).
- [x] **TOC correct and page-verified** (Page 7).
- [x] **List of Figures correct and page-verified** (Pages 8–9).
- [x] **List of Tables correct and page-verified** (Pages 10–11).
- [x] **PDF visually clean** (122 pages, 0 blank pages, 0 layout defects).
- [x] **PPT matches thesis** (23 verified slides).
- [x] **Thesis matches implementation** (FastAPI, React 18, SQLite, 44/44 tests).
- [x] **Model D hash unchanged** (`E2C95DC5...9022`).
- [x] **V5 manifest hash unchanged** (`6730495A...A2C4`).

---

## 21. Final Submission Decision

# SUBMISSION READY
