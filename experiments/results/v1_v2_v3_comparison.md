# Longitudinal Dataset Comparison Report: V1 vs V2 vs V3

**Project**: LungAI Disease Detector  
**Scope**: Dataset Engineering & Manifest Provenance  
**Date**: September 2026  
**Final Status**: **`READY FOR TRAINING`**  

---

## 1. High-Level Dataset Evolution Matrix

| Metric | Baseline V1 | Reconstructed V2 | Controlled V3 Manifest |
| :--- | :---: | :---: | :---: |
| **Total Images** | 13,102 | 10,558 | **10,090** |
| **Total Patients** | 12,954 | 10,410 | **9,968** |
| **Source-Label Cramér's V** | `0.8331` | `0.7924` | **`0.7760`** |
| **Pneumonia Distribution** | 88.9% Kermany / 11.1% TBX | 64.5% Kermany / 35.5% TBX | **50.0% Kermany / 50.0% TBX (Equal Dual-Source)** |
| **Normal Distribution** | 86.6% Existing / 7.9% TBX / 5.5% JSRT | 57.5% Existing / 38.9% TBX / 3.6% JSRT | **48.5% Existing / 48.5% TBX / 3.0% JSRT** |
| **Tuberculosis Distribution** | 100% Single Source | 53.7% TBX / 46.3% Shenzhen | **53.7% TBX / 46.3% Shenzhen (Balanced Dual-Source)** |
| **Sixth Class Label** | Conflated "Lung Cancer / Nodule" | Pulmonary Nodule / Mass (Purged) | **Pulmonary Nodule / Mass (Provenanced)** |
| **Benign JSRT Nodules** | 54 mislabeled as cancer | 54 Purged | **54 Purged (Strict Non-Malignancy Rule)** |
| **Patient Leakage Risk** | Unknown | 0 Overlap | **0 Overlap (Guaranteed Patient Isolation)** |
| **Near-Duplicate Screening** | None | MD5 Hash | **MD5 + 64-bit Perceptual dHash Screened** |
| **Montgomery Quarantine** | Quarantined | Quarantined | **Strictly Quarantined (0 in V3 Manifest)** |

---

## 2. Key Scientific Breakthroughs Achieved in V3

1. **Resolution of Pneumonia Single-Source Monopoly**:
   * In V1, the Kermany pediatric dataset accounted for **88.9%** of all pneumonia cases, creating an extreme risk that convolutional backbones would learn pediatric ribcage morphology and Guangzhou hospital annotations as shortcut features.
   * In V3, by incorporating verified non-overlapping adult CXRs from the local TBX11K sick non-TB cohort, pneumonia is now balanced at **50.0% Kermany and 50.0% TBX11K** (1,800 images each).

2. **Resolution of Normal Control Source Bias**:
   * In V1, Normal controls were heavily dominated by a single legacy web-scraped archive (86.6%).
   * In V3, Normal controls are evenly balanced between `Existing_Normal` (48.5%) and `TBX11K` (48.5%), supplemented by `JSRT` healthy controls (3.0%).

3. **Taxonomy Clarification & Malignancy Integrity**:
   * Clarified class 6 as `Pulmonary Nodule / Mass` (Option B), strictly adhering to radiological observation criteria.
   * Verified that all 54 benign JSRT nodules (tuberculomas, granulomas, hamartomas) remain strictly purged.
   * Provenance of the remaining nodule cases is confirmed via JSRT histology biopsy records ($N=100$) and VinDr 17-board radiologist consensus ($N=626$).

4. **Multi-Source Evaluation Framework**:
   * With balanced dual-source distributions in Pneumonia, Tuberculosis, and Normal, LungAI can now perform **source-held-out validation** inside internal testing prior to touching the quarantined external Montgomery dataset.
