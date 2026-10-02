"""
Phase 2G: V5 Research Protocol Freeze
LungAI Disease Detector Project

Scope:
- Freezes the complete scientific methodology for prospective Dataset V5
- Formalizes the 6-class source strategy, data splitting policy, and view policies
- Specifies the baseline DenseNet-121 architecture, training protocol, and evaluation suites
- Formulates the fair comparison framework for prospective domain generalization (DANN/CORAL/MMD)
- Establishes the experiment registry and execution checklist
- Outputs phase2g_v5_protocol.json/.md, phase2g_experiment_registry.json, and phase2g_execution_checklist.md
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase2g_protocol_freeze")

WORKSPACE_ROOT = Path("d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main")
EXPERIMENTS_DIR = WORKSPACE_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
PHASE2G_DIR = EXPERIMENTS_DIR / "phase2g"

PHASE2G_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_protocol_freeze():
    logger.info("Executing Phase 2G: Freezing V5 Research Protocol...")

    # 1. PROTOCOL SPECIFICATION (Sections 1 to 12)
    protocol_data = {
        "protocol_title": "LungAI Dataset V5 & Domain Generalization Research Protocol",
        "governance_status": "FROZEN_WAITING_FOR_BIMCV",
        "primary_research_question": (
            "Can a multi-source six-class chest X-ray classifier learn disease-relevant "
            "representations that generalize across independent acquisition sources, and can a "
            "domain-generalization strategy improve cross-source performance compared with a standard "
            "DenseNet-121 baseline?"
        ),
        "target_classes": [
            "Normal",
            "Pneumonia",
            "COVID-19",
            "Tuberculosis",
            "Pleural Effusion",
            "Pulmonary Nodule / Mass"
        ],
        "class_source_strategy": {
            "Normal": {
                "verified_current_sources": ["Existing_Normal (1,199)", "TBX11K (1,199)", "JSRT (67)", "NIH (82)"],
                "prospective_bimcv_role": "None (Preserve current multi-source cohort)",
                "label_requirements": "Clear lung fields, absence of acute/chronic cardiopulmonary abnormalities",
                "source_holdout_pairs": [
                    ("Existing_Normal + JSRT", "TBX11K"),
                    ("TBX11K", "Existing_Normal + JSRT")
                ],
                "view_controlled_capability": "PA vs PA available across TBX11K, JSRT, and NIH",
                "exclusion_rules": "Exclude cases with subtle apical scarring or non-pulmonary thoracic artifacts"
            },
            "Pneumonia": {
                "verified_current_sources": ["Existing_Pneumonia (1,395)", "TBX11K (1,395)", "NIH (8)"],
                "prospective_bimcv_role": "None (Preserve current multi-source cohort)",
                "label_requirements": "Consolidation, bronchopneumonic opacities, or alveolar infiltrates",
                "source_holdout_pairs": [
                    ("Existing_Pneumonia (Guangzhou)", "TBX11K (Beijing)"),
                    ("TBX11K (Beijing)", "Existing_Pneumonia (Guangzhou)")
                ],
                "view_controlled_capability": "PA vs PA available in TBX11K; AP available in Kermany pediatric cohort",
                "exclusion_rules": "Exclude non-infectious alveolar hemorrhage or pulmonary edema"
            },
            "COVID-19": {
                "verified_current_sources": ["Existing_COVID-19 (1,942)"],
                "prospective_bimcv_role": "Target 1,000 RT-PCR molecular confirmed scans from Valencian Health Authority (BIMCV)",
                "label_requirements": "Mandatory positive RT-PCR SARS-CoV-2 confirmation within +/- 7 days of imaging encounter",
                "source_holdout_pairs": [
                    ("Existing_COVID-19", "BIMCV_COVID19+"),
                    ("BIMCV_COVID19+", "Existing_COVID-19")
                ],
                "view_controlled_capability": "PA vs PA and AP vs AP enabled by prospective BIMCV DICOM view tagging",
                "exclusion_rules": "Exclude cases lacking molecular RT-PCR confirmation, CT slices, and lateral projections"
            },
            "Tuberculosis": {
                "verified_current_sources": ["TBX11K (683)", "Existing_Tuberculosis (665)"],
                "prospective_bimcv_role": "None (Preserve current multi-source cohort)",
                "label_requirements": "Active pulmonary tuberculosis with cavitary, infiltrative, or nodular lesions",
                "source_holdout_pairs": [
                    ("Existing_Tuberculosis (Shenzhen)", "TBX11K (Beijing)"),
                    ("TBX11K (Beijing)", "Existing_Tuberculosis (Shenzhen)")
                ],
                "view_controlled_capability": "100% PA erect across both Shenzhen and Beijing cohorts",
                "exclusion_rules": "Exclude isolated calcified granulomas without active infiltrates"
            },
            "Pleural Effusion": {
                "verified_current_sources": ["VinBigData_VinDr_CXR (931)", "NIH_ChestX-ray14 (86)"],
                "prospective_bimcv_role": "None (Preserve current multi-source cohort)",
                "label_requirements": "Costophrenic angle blunting or fluid layering in the pleural space",
                "source_holdout_pairs": [
                    ("VinBigData_VinDr_CXR", "NIH_ChestX-ray14"),
                    ("NIH_ChestX-ray14", "VinBigData_VinDr_CXR")
                ],
                "view_controlled_capability": "PA vs PA strictly isolated by filtering NIH subset (41 PA vs 45 AP)",
                "exclusion_rules": "Exclude subpulmonic effusion without visible costophrenic blunting on PA"
            },
            "Pulmonary Nodule / Mass": {
                "verified_current_sources": ["VinBigData_VinDr_CXR (536)", "JSRT (78)", "NIH_ChestX-ray14 (70)"],
                "prospective_bimcv_role": "None (Preserve current multi-source cohort)",
                "label_requirements": "Discrete opacity <=3cm (nodule) or >3cm (mass) pathologically or radiologically verified",
                "source_holdout_pairs": [
                    ("VinBigData_VinDr_CXR", "NIH_ChestX-ray14 + JSRT"),
                    ("NIH_ChestX-ray14 + JSRT", "VinBigData_VinDr_CXR")
                ],
                "view_controlled_capability": "PA vs PA verified across VinDr, JSRT, and NIH",
                "exclusion_rules": "Exclude benign non-neoplastic granulomas (purged in Dataset V3)"
            }
        },
        "data_splitting_policy": {
            "isolation_principle": "Patient-level partitioning with complete decoupling across splits",
            "proportions": {"train": 0.70, "val": 0.15, "test": 0.15},
            "random_seed": 42,
            "leakage_guarantees": [
                "Train intersect Val = 0 patients",
                "Train intersect Test = 0 patients",
                "Val intersect Test = 0 patients",
                "Duplicate purging (MD5 bitwise and dHash <= 3) strictly precedes splitting"
            ],
            "freeze_condition": "Split partition is locked before any model training; post-hoc split adjustment is prohibited"
        },
        "view_policy": {
            "evaluation_cohort_a": {
                "name": "Broad Frontal CXR Cohort",
                "projections_included": ["PA", "AP", "Frontal Unspecified"],
                "clinical_purpose": "Evaluates operational generalization across real-world hospital emergency/bedside admission conditions"
            },
            "evaluation_cohort_b": {
                "name": "View-Controlled Cohort",
                "projections_included": ["Strictly Posteroanterior (PA) erect frontal"],
                "clinical_purpose": "Controls projection physics as a confounder to directly isolate institutional/scanner domain shift"
            }
        },
        "densenet_baseline_specification": {
            "architecture": "DenseNet-121 (pretrained on ImageNet, standard classification head)",
            "preprocessing": [
                "BGR to RGB color conversion",
                "Gaussian blur (kernel 3x3, sigma 0.8)",
                "CIE LAB L-channel CLAHE (clip limit 2.0, tileGridSize 8x8)",
                "Lanczos-4 interpolation resize to 224x224 pixels",
                "ImageNet channel-wise standardization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
            ],
            "augmentation": [
                "Random horizontal flip (p=0.5)",
                "Random affine rotation (+/- 10 degrees, border reflect)"
            ],
            "training_hyperparameters": {
                "optimizer": "Adam",
                "initial_learning_rate": 1e-4,
                "learning_rate_schedule": "ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)",
                "batch_size": 32,
                "epochs": 20,
                "early_stopping": "EarlyStopping(patience=5, restore_best_weights=True)",
                "class_weighting": "Inverse frequency weighting to balance six diagnostic classes",
                "random_seed": 42
            },
            "freeze_rule": "Hyperparameters are locked; post-hoc hyperparameter tuning on test splits is strictly forbidden"
        },
        "evaluation_protocol": {
            "internal_test_suite": [
                "Accuracy", "Macro Precision", "Macro Recall", "Macro F1",
                "Weighted F1", "Macro ROC-AUC", "Macro PR-AUC", "Confusion Matrix"
            ],
            "source_held_out_suite": [
                "Directional Accuracy", "Sensitivity / Recall", "Specificity",
                "Precision", "F1-Score", "ROC-AUC", "PR-AUC", "Directional Asymmetry Gap"
            ],
            "external_validation_suite": [
                "Montgomery County TB exact sensitivity (quarantined benchmark)",
                "Montgomery County Normal specificity"
            ]
        },
        "baseline_failure_analysis_framework": [
            "Source-dependent performance disparity calculation",
            "Directional asymmetry quantification (|Sens(A->B) - Sens(B->A)|)",
            "Projection sensitivity analysis (Performance gap between Cohort A and Cohort B)",
            "Confidence collapse evaluation (Confidence distribution on incorrect vs correct predictions)",
            "Cross-source confusion pattern shifts (Off-diagonal misclassification drift)"
        ],
        "domain_generalization_comparison_framework": {
            "control": "Standard Empirical Risk Minimization DenseNet-121",
            "proposed_candidates": [
                "Domain-Adversarial Neural Networks (DANN) with gradient reversal",
                "Correlation Alignment (CORAL) feature covariance regularization",
                "Maximum Mean Discrepancy (MMD) distribution alignment"
            ],
            "selection_gate": "Selected method must directly target the failure modes revealed by the V5 baseline analysis",
            "fair_comparison_invariants": [
                "Identical V5 train, validation, and internal test partitions",
                "Identical source-held-out evaluation cohorts",
                "Identical view-controlled cohorts",
                "Identical preprocessing and evaluation metrics",
                "Zero exposure to external test sets during training"
            ],
            "primary_success_criteria": [
                "Improvement in source-held-out macro F1",
                "Improvement in balanced cross-source sensitivity and specificity",
                "Reduction in directional cross-source asymmetry gap",
                "Improvement in worst-performing hospital domain",
                "Consistency of classification performance across independent sources"
            ]
        },
        "avoidable_failure_modes_safeguards": [
            "Source Leakage: Strict patient and institutional separation across comparative partitions",
            "Test-Set Tuning: Zero hyperparameter adjustments after observing holdout results",
            "Projection Confounding: Dual evaluation via Cohort A (real-world) and Cohort B (view-controlled)",
            "Montgomery Contamination: Permanent quarantine (zero images admitted into training or validation)",
            "Hypothesis Drift: Research question and success criteria frozen prior to data release"
        ]
    }

    with open(RESULTS_DIR / "phase2g_v5_protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol_data, f, indent=2)

    # 2. MARKDOWN PROTOCOL DOCUMENT
    protocol_md = f"""# Phase 2G: Unified Dataset V5 & Domain Generalization Research Protocol

**Project**: LungAI Disease Detector  
**Scope**: Experimental Methodology, Dataset Architecture, and Governance Freeze  
**Status**: `V5_PROTOCOL_FROZEN_WAITING_FOR_BIMCV`  
**Artifact**: `experiments/results/phase2g_v5_protocol.json`  

---

## 1. Central Research Question

> **"{protocol_data['primary_research_question']}"**

This scientific inquiry is locked. Methodological goals cannot be altered based on observed test performance.

---

## 2. Six-Class Source Strategy Matrix

| Diagnostic Class | Current Verified Sources (V4) | Prospective BIMCV Role | Minimum Label Requirements | Cross-Source Holdout Pairs |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | Existing_Normal, TBX11K, JSRT, NIH | None (Preserved) | Clear lung fields, verified non-pathological CXR | `Existing+JSRT` $\\leftrightarrow$ `TBX11K` |
| **Pneumonia** | Existing_Pneumonia, TBX11K, NIH | None (Preserved) | Alveolar consolidation, infiltrates | `Guangzhou` $\\leftrightarrow$ `TBX11K` |
| **COVID-19** | Existing_COVID-19 ($1,942$) | Target $1,000$ molecular confirmed CXRs | **Mandatory RT-PCR confirmation within $\\pm 7$ days** | `Existing_COVID` $\\leftrightarrow$ `BIMCV` |
| **Tuberculosis** | TBX11K, Existing_Tuberculosis | None (Preserved) | Active pulmonary cavitary/infiltrative disease | `Shenzhen` $\\leftrightarrow$ `TBX11K` |
| **Pleural Effusion** | VinBigData_VinDr, NIH | None (Preserved) | Costophrenic blunting / fluid layering | `VinDr` $\\leftrightarrow$ `NIH` |
| **Pulmonary Nodule / Mass** | VinBigData_VinDr, JSRT, NIH | None (Preserved) | Verified solitary/multiple pulmonary lesions | `VinDr` $\\leftrightarrow$ `NIH+JSRT` |

---

## 3. Data Split & View Policies

* **Patient-Level Isolation**: Partitioning strictly enforces:
  $$\\text{{Train}} \\cap \\text{{Val}} = \\emptyset, \\quad \\text{{Train}} \\cap \\text{{Test}} = \\emptyset, \\quad \\text{{Val}} \\cap \\text{{Test}} = \\emptyset$$
  Random seed is locked to `42`. Duplicate detection (MD5 bitwise match and dHash Hamming distance $\\le 3$) strictly precedes splitting.
* **Cohort A (Broad Frontal CXR)**: Admits all verified frontal projections (PA and AP) to evaluate realistic emergency/inpatient clinical conditions.
* **Cohort B (View-Controlled)**: Restricts comparisons strictly to Posteroanterior (PA) erect frontal radiographs to decouple **Hospital / Scanner Domain Shift** from **Projection Domain Shift**.

---

## 4. Controlled DenseNet-121 Baseline Specification

* **Backbone**: DenseNet-121 pretrained on ImageNet.
* **Standard Preprocessing**: BGR $\\rightarrow$ RGB, Gaussian Blur ($3\\times3, \\sigma=0.8$), CIE LAB L-channel CLAHE (clip limit $2.0$), Lanczos-4 resize to $224\\times224$, ImageNet normalization.
* **Augmentation**: Random horizontal flip ($p=0.5$), random affine rotation ($\pm 10^\\circ$).
* **Optimization**: Adam optimizer, initial $\\text{{lr}}=10^{{-4}}$ with `ReduceLROnPlateau(patience=2, factor=0.5)`, batch size $32$, $20$ epochs, `EarlyStopping(patience=5)`.
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
"""
    with open(RESULTS_DIR / "phase2g_v5_protocol.md", "w", encoding="utf-8") as f:
        f.write(protocol_md)

    # 3. EXPERIMENT REGISTRY (Section 13)
    registry_data = {
        "experiment_registry_version": "1.0",
        "governance": "LOCKED",
        "experiments": {
            "V5_BASELINE": {
                "identifier": "V5_BASELINE",
                "objective": "Establish controlled six-class DenseNet-121 ERM baseline on Unified Dataset V5",
                "dataset": "unified_manifest_v5.csv (Pending BIMCV Acquisition)",
                "train_cohort": "V5 Train Partition (70% patient-isolated, all 6 classes multi-source)",
                "val_cohort": "V5 Validation Partition (15% patient-isolated)",
                "test_cohort": "V5 Internal Test Partition (15% patient-isolated)",
                "model": "DenseNet-121 (Frozen backbone + Dense head, fine-tuned)",
                "metrics": ["Accuracy", "Macro F1", "Weighted F1", "Macro ROC-AUC", "Per-class Recall/Precision"],
                "leakage_controls": "Patient-level splitting (seed 42), MD5/dHash duplicate exclusion",
                "status": "BLOCKED_BIMCV_ACCESS"
            },
            "V5_SOURCE_HOLDOUT": {
                "identifier": "V5_SOURCE_HOLDOUT",
                "objective": "Evaluate bidirectional cross-hospital generalization across all multi-source classes",
                "dataset": "unified_manifest_v5.csv (Pending BIMCV Acquisition)",
                "train_cohort": "Source A independent domain",
                "test_cohort": "Source B held-out independent hospital domain",
                "model": "DenseNet-121 baseline",
                "metrics": ["Sensitivity", "Specificity", "F1-Score", "ROC-AUC", "Directional Asymmetry Gap"],
                "leakage_controls": "Strict source isolation; zero cross-source patient overlap",
                "status": "BLOCKED_BIMCV_ACCESS"
            },
            "V5_VIEW_CONTROLLED": {
                "identifier": "V5_VIEW_CONTROLLED",
                "objective": "Isolate institutional/scanner domain shift by strictly filtering cross-source evaluation to PA erect radiographs",
                "dataset": "unified_manifest_v5.csv (Pending BIMCV Acquisition)",
                "train_cohort": "Source A (PA strictly)",
                "test_cohort": "Source B (PA strictly)",
                "model": "DenseNet-121 baseline",
                "metrics": ["PA-Controlled Sensitivity", "Specificity", "F1-Score", "ROC-AUC"],
                "leakage_controls": "Exclusion of all AP bedside and non-PA radiographs",
                "status": "BLOCKED_BIMCV_ACCESS"
            },
            "V5_MONTGOMERY_EXTERNAL": {
                "identifier": "V5_MONTGOMERY_EXTERNAL",
                "objective": "External zero-shot clinical validation on quarantined Montgomery County archive (N=138)",
                "dataset": "data/downloads/montgomery (Strictly external)",
                "train_cohort": "None (Evaluation only)",
                "test_cohort": "Montgomery County (58 TB cases, 80 Normal controls)",
                "model": "DenseNet-121 baseline & subsequent DG models",
                "metrics": ["Montgomery TB Exact Recall", "Montgomery Normal Specificity", "ROC-AUC"],
                "leakage_controls": "Zero Montgomery images admitted into training, validation, or threshold tuning",
                "status": "BLOCKED_BIMCV_ACCESS"
            },
            "V5_DOMAIN_GENERALIZATION_01": {
                "identifier": "V5_DOMAIN_GENERALIZATION_01",
                "objective": "Implement and evaluate proposed domain-generalization method (DANN/CORAL/MMD) against frozen baseline",
                "dataset": "unified_manifest_v5.csv (Pending BIMCV Acquisition)",
                "train_cohort": "V5 Train Partition with domain labels",
                "test_cohort": "Identical V5 Internal Test, Source-Holdout, View-Controlled, and Montgomery cohorts",
                "model": "Domain-Invariant Architecture (Backbone + Task Classifier + Domain Regularizer)",
                "metrics": ["Source-Held-Out Macro F1", "Asymmetry Reduction", "Worst-Source Recall", "External Generalization"],
                "leakage_controls": "Identical frozen partitions; zero test-set peeking",
                "status": "NOT_STARTED"
            }
        }
    }

    with open(RESULTS_DIR / "phase2g_experiment_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)

    # 4. EXECUTION CHECKLIST (Section 14)
    checklist_md = """# Phase 2G: Unified Dataset V5 Execution Checklist

**Project**: LungAI Disease Detector  
**Scope**: Step-by-Step Operational Runbook Following Legitimate Credential Authorization  
**Current State**: Standing by for authenticated BIMCV-COVID19+ access credentials  

---

### Step 1: Credential Verification & Staged Ingestion
- [ ] Authorized BIMCV credentials obtained via official institutional portal
- [ ] BIMCV staged acquisition completed (target 1,000 planar CXRs)
- [ ] BIMCV labels verified against RT-PCR molecular confirmation (within +/- 7 days)
- [ ] BIMCV view metadata verified (PA vs AP categorized; lateral excluded)
- [ ] BIMCV duplicate checks completed (MD5 exact match and dHash <= 3)
- [ ] BIMCV patient overlap checks completed against existing COVID-19 and V4

### Step 2: V5 Reconstruction & Audit
- [ ] Unified Dataset V5 manifest created (`unified_manifest_v5.csv`)
- [ ] V5 audit completed and published (`dataset_reconstruction_v5_audit.json`)
- [ ] V5 Cramér's V calculated dynamically from manifest
- [ ] V5 source/class contingency matrix generated
- [ ] V5 patient-level split frozen (seed 42, zero leakage verified)

### Step 3: Baseline Training & Multi-Cohort Evaluation
- [ ] DenseNet-121 V5 baseline trained according to frozen hyperparameters
- [ ] Internal test suite completed (Cohort A: Broad Frontal)
- [ ] Source-held-out evaluation completed across all 6 classes
- [ ] View-controlled evaluation completed (Cohort B: PA-only)
- [ ] Montgomery County external zero-shot evaluation completed
- [ ] Baseline failure analysis completed (asymmetry, projection shift, calibration)

### Step 4: Domain Generalization Intervention
- [ ] Domain-generalization method selected based on baseline failure analysis
- [ ] Domain-generalization model trained on identical V5 partitions
- [ ] Same external and held-out cohorts evaluated
- [ ] Statistical comparison completed against frozen DenseNet-121 baseline
"""
    with open(RESULTS_DIR / "phase2g_execution_checklist.md", "w", encoding="utf-8") as f:
        f.write(checklist_md)

    logger.info("Phase 2G protocol freeze completed successfully.")


if __name__ == "__main__":
    build_protocol_freeze()
