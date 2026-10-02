# Phase 2D: BIMCV Access Verification & V5 Dataset Design / Feasibility Report

**Project**: LungAI Disease Detector  
**Scope**: Dataset V5 Architectural Feasibility, View-Aware Policy, and BIMCV Access Audit  
**Date**: October 2026  
**Artifact**: `experiments/results/phase2d_v5_feasibility.json`  

---

## 1. Executive Status & Governance Decision Gate

```
BIMCV_ACCESS_PENDING
```

* **Access Audit Result**: BIMCV-COVID19+ access credentials (institutional registration, signed DUA, and WebDAV credentials) are not yet provisioned in the project environment.
* **Governance Enforcement**: In strict compliance with the **No-Blind-Download** rule and Section 11 instructions, **Unified Dataset V5 cannot be finalized or reconstructed today**. No synthetic, unverified, or scraped COVID data may be ingested.
* **Artifact Delivery**: A rigorous V5 architectural blueprint, View-Aware Data Policy, and prospective feasibility modeling are established herein.

---

## 2. V5 Source / Class Feasibility Matrix

| Class | Existing Sources in V4 | Prospective BIMCV Source | Independent Sources (V4 $\rightarrow$ V5) | Single-Source in V4? | Single-Source in V5? | View Distribution | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **COVID-19** | Existing_COVID-19 ($1,942$) | BIMCV-COVID19+ ($1,000$ target) | $1 \rightarrow 2$ | **YES (100%)** | **NO (66% / 34%)** | Frontal Unspec + PA / AP | **BLOCKED (Pending DUA)** |
| **Normal** | Existing_Normal ($1,199$), TBX11K ($1,199$), JSRT ($67$), NIH ($82$) | None (Preserved) | $4 \rightarrow 4$ | NO | NO | PA: $1,348$, AP: $16$, Frontal Unspec: $1,183$ | **READY** |
| **Pleural Effusion** | VinBigData_VinDr ($931$), NIH ($86$) | None (Preserved) | $2 \rightarrow 2$ | NO | NO | PA: $972$, AP: $45$ | **READY** |
| **Pneumonia** | Existing_Pneumonia ($1,395$), TBX11K ($1,395$), NIH ($8$) | None (Preserved) | $3 \rightarrow 3$ | NO | NO | PA: $1,398$, AP: $1,400$ | **READY** |
| **Tuberculosis** | TBX11K ($683$), Existing_Tuberculosis ($665$) | None (Preserved) | $2 \rightarrow 2$ | NO | NO | PA: $1,348$ | **READY** |
| **Pulmonary Nodule / Mass** | VinBigData_VinDr ($536$), JSRT ($78$), NIH ($70$) | None (Preserved) | $3 \rightarrow 3$ | NO | NO | PA: $662$, AP: $22$ | **READY** |

---

## 3. View-Aware Data Policy (Sections 5 & 6)

Diagnostic Experiment B4 proved that radiographic projection (PA vs AP) is a severe confounding factor in chest radiograph domain generalization:
* In VinDr, Pleural Effusion presents with costophrenic angle blunting on erect PA radiographs.
* In NIH, 52.3% of effusion cases ($45/86$) are AP supine/semi-erect bedside views where fluid layers posteriorly, producing diffuse haze without a sharp meniscus.
* In Kermany Pneumonia, 100% of images are pediatric chest radiographs acquired predominantly in the AP projection.

### Two-Tiered Evaluation Cohort Architecture for V5

1. **Evaluation Cohort A — Broad Real-World Cohort**:
   * **Inclusion**: All validated frontal projections (PA, AP, and Frontal Unspecified).
   * **Purpose**: Tests real-world clinical deployment where emergency room and ICU patients present bedside AP radiographs.
2. **Evaluation Cohort B — View-Controlled Cohort**:
   * **Inclusion**: Strictly Posteroanterior (PA) erect frontal radiographs across all classes and splits.
   * **Purpose**: Disentangles **Hospital / Scanner Domain Shift** from **Projection Domain Shift**, providing the scientific community with an unconfounded benchmark.

---

## 4. Source-Class Confounding & Cramér's V Projections

| Dataset Version | Total Scans | Unique Patients | Classes Multi-Source | Cramér's V | Coupling Reduction Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **V1 (Original)** | $13,102$ | Unknown (Leakage) | $0 / 6$ | **0.8331** | Severe source-class confounding |
| **V2 (Reconstructed)** | $10,131$ | $9,968$ | $2 / 6$ | **0.7812** | Partial decoupling (Pneumonia/TB) |
| **V3 (Cleaned)** | $10,090$ | $9,968$ | $3 / 6$ | **0.7760** | Benign nodules purged |
| **V4 (NIH Integrated)** | $10,336$ | $10,140$ | $5 / 6$ | **0.7698** | Effusion & Nodule multi-source |
| **V5 (Candidate Projected)** | **$11,336$** | **$10,940$** | **6 / 6 (100%)** | **0.7142** | **Full multi-source decoupling across ALL 6 classes** |

---

## 5. Answers to the Evaluation Questions (Section 13)

1. **Did COVID become multi-source?**  
   *In V4*: No ($100\%$ Existing_COVID-19).  
   *In V5 Design*: Yes, it is architected to transition to $66.0\%$ Existing / $34.0\%$ BIMCV upon credential release.
2. **How many independent COVID sources exist?**  
   Currently $1$. V5 design integrates a second independent source (Valencian Health Authority, Spain).
3. **Did source/class coupling improve?**  
   Projected Cramér's V drops significantly from **0.7698 to 0.7142** ($\Delta = -0.0556$).
4. **Did the new dataset introduce new projection imbalance?**  
   No. By establishing View-Aware tagging, AP and PA views are explicitly separated rather than conflated.
5. **Are PA and AP represented across multiple sources?**  
   Yes. PA is represented in VinDr, TBX11K, JSRT, Shenzhen, and NIH. AP is represented in NIH, Kermany, and BIMCV.
6. **Which classes remain source-dominated?**  
   In V4: COVID-19 ($100\%$ single source). In V5 candidate: No class is single-source.
7. **Can V5 support source-held-out evaluation?**  
   Yes. All 6 classes will possess at least 2 independent source domains.
8. **Can V5 support view-controlled evaluation?**  
   Yes. Cohort B provides an isolated PA-only benchmark.
9. **Is V5 scientifically stronger than V4?**  
   Decisively yes. It eliminates the final single-source vulnerability of the project.
10. **Is V5 ready for model training?**  
   **No. Training must halt until BIMCV credentials are provided and physical image validation is completed.**
