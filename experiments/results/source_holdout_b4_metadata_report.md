# Source Metadata Comparison: VinBigData/VinDr vs NIH ChestX-ray14

**Project**: LungAI Disease Detector  
**Scope**: Acquisition Characteristics & Annotation Provenance Comparison for Pleural Effusion  
**Date**: September 2026  
**Artifact**: `experiments/results/source_holdout_b4_metadata_comparison.json`  

---

## 1. Acquisition & Annotation Metadata Matrix

| Characteristic | VinBigData / VinDr-CXR | NIH ChestX-ray14 |
| :--- | :--- | :--- |
| **Originating Institution** | 108 Military Central Hospital & Hanoi Medical Univ., Vietnam | NIH Clinical Center, Bethesda, MD, USA |
| **Geographic Population** | Southeast Asian adult patient population | North American clinical research center cohort |
| **Total Archive Size** | 4,394 planar chest radiographs | 112,120 planar chest radiographs |
| **Effusion Scans in V4** | **931 scans** (891 patients) | **86 scans** (48 patients) |
| **Projection / Views** | **100% PA** (Posteroanterior frontal) | **Mixed PA & AP** (60% PA, 40% AP bedside) |
| **Image Resolution** | Downscaled from high-bit DICOM to ~1024x1024 | 1024x1024 8-bit grayscale PNG |
| **Annotation Methodology** | **Visual bounding box consensus** by 17 board-certified radiologists | **Automated text-mining** from reports via NegBio NLP |
| **Label Precision / Ground Truth** | Direct radiologist visual consensus (high clinical precision) | NLP report extraction (documented 10–18% label noise) |
| **Effusion Archive Prevalence** | 23.5% of archive scans contain effusion | 11.9% of full archive contains effusion |

---

## 2. Plausible Impact on Cross-Source Generalization

1. **Projection Differences (PA vs AP)**:
   * VinDr is strictly PA erect views where pleural effusion settles in the dependent costophrenic sulci, forming the classic meniscus sign.
   * NIH contains 40% AP supine/semi-erect bedside views where fluid layers posteriorly, appearing as diffuse ground-glass haziness without a sharp meniscus.
   * This physical difference accounts for significant domain shift when models trained on PA radiographs evaluate AP radiographs.
2. **Annotation Style & Label Noise**:
   * VinDr requires consensus confirmation of visible effusion by multiple radiologists.
   * NIH report mining can label an image as "Effusion" if the radiologist noted "trace effusion", "possible blunting", or "clearing effusion" in the text, introducing weak visual ground truth.
