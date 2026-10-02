# Phase 2G: Unified Dataset V5 & Domain Generalization Research Protocol

**Project**: LungAI Disease Detector  
**Scope**: Experimental Methodology, Dataset Architecture, and Governance Freeze  
**Status**: `V5_PROTOCOL_FROZEN_WAITING_FOR_BIMCV`  
**Artifact**: `experiments/results/phase2g_v5_protocol.json`  

---

## 1. Central Research Question

> **"Can a multi-source six-class chest X-ray classifier learn disease-relevant representations that generalize across independent acquisition sources, and can a domain-generalization strategy improve cross-source performance compared with a standard DenseNet-121 baseline?"**

This scientific inquiry is locked. Methodological goals cannot be altered based on observed test performance.

---

## 2. Six-Class Source Strategy Matrix

| Diagnostic Class | Current Verified Sources (V4) | Prospective BIMCV Role | Minimum Label Requirements | Cross-Source Holdout Pairs |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | Existing_Normal, TBX11K, JSRT, NIH | None (Preserved) | Clear lung fields, verified non-pathological CXR | `Existing+JSRT` $\leftrightarrow$ `TBX11K` |
| **Pneumonia** | Existing_Pneumonia, TBX11K, NIH | None (Preserved) | Alveolar consolidation, infiltrates | `Guangzhou` $\leftrightarrow$ `TBX11K` |
| **COVID-19** | Existing_COVID-19 ($1,942$) | Target $1,000$ molecular confirmed CXRs | **Mandatory RT-PCR confirmation within $\pm 7$ days** | `Existing_COVID` $\leftrightarrow$ `BIMCV` |
| **Tuberculosis** | TBX11K, Existing_Tuberculosis | None (Preserved) | Active pulmonary cavitary/infiltrative disease | `Shenzhen` $\leftrightarrow$ `TBX11K` |
| **Pleural Effusion** | VinBigData_VinDr, NIH | None (Preserved) | Costophrenic blunting / fluid layering | `VinDr` $\leftrightarrow$ `NIH` |
| **Pulmonary Nodule / Mass** | VinBigData_VinDr, JSRT, NIH | None (Preserved) | Verified solitary/multiple pulmonary lesions | `VinDr` $\leftrightarrow$ `NIH+JSRT` |

---

## 3. Data Split & View Policies

* **Patient-Level Isolation**: Partitioning strictly enforces:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
  Random seed is locked to `42`. Duplicate detection (MD5 bitwise match and dHash Hamming distance $\le 3$) strictly precedes splitting.
* **Cohort A (Broad Frontal CXR)**: Admits all verified frontal projections (PA and AP) to evaluate realistic emergency/inpatient clinical conditions.
* **Cohort B (View-Controlled)**: Restricts comparisons strictly to Posteroanterior (PA) erect frontal radiographs to decouple **Hospital / Scanner Domain Shift** from **Projection Domain Shift**.

---

## 4. Controlled DenseNet-121 Baseline Specification

* **Backbone**: DenseNet-121 pretrained on ImageNet.
* **Standard Preprocessing**: BGR $\rightarrow$ RGB, Gaussian Blur ($3\times3, \sigma=0.8$), CIE LAB L-channel CLAHE (clip limit $2.0$), Lanczos-4 resize to $224\times224$, ImageNet normalization.
* **Augmentation**: Random horizontal flip ($p=0.5$), random affine rotation ($\pm 10^\circ$).
* **Optimization**: Adam optimizer, initial $\text{lr}=10^{-4}$ with `ReduceLROnPlateau(patience=2, factor=0.5)`, batch size $32$, $20$ epochs, `EarlyStopping(patience=5)`.
* **Class Weighting**: Balanced inverse frequency weighting across all 6 classes.

---

## 5. Fair Comparison & Success Criteria for Domain Generalization

* **Control**: Standard Empirical Risk Minimization DenseNet-121 baseline.
* **Proposed Intervention**: Domain Generalization Architecture (DANN, CORAL, or MMD) selected based on the baseline failure analysis.
* **Invariants**: Identical train/val/test splits, identical preprocessing, identical metrics, and strictly zero peeking at external test cohorts during tuning.
* **Primary Success Criteria**:
  1. Statistically significant improvement in **Source-Held-Out Macro F1**.
  2. Substantial reduction in **Directional Cross-Source Asymmetry Gap**.
  3. Elevated worst-case single-hospital recall/specificity.
  4. Preserved or improved view-controlled generalization.
  *(Internal test accuracy is formally designated as a secondary diagnostic indicator).*

---

## 6. Avoidable Failure Modes & Governance Invariants

* **No Montgomery Contamination**: Montgomery County remains strictly external (evaluation-only benchmark).
* **No Post-Hoc Tuning**: Hyperparameters and architecture cannot be retrofitted to improve test figures.
* **Zero Premature Manifest Creation**: `unified_manifest_v5.csv` will not be generated until authentic BIMCV DICOMs are delivered.
