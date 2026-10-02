# Phase 2G: Unified Dataset V5 Execution Checklist

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
