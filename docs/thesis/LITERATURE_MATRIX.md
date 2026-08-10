# Literature Survey Matrix

**Project:** Deep Learning Based Computer-Aided Diagnosis for Multi-Class Lung Disease Classification from Chest Radiographs
**Purpose:** Structured, evidence-based comparison of prior work, mapped to where each is used in this thesis.

> All entries are real, peer-reviewed or archival sources. DOIs for the three dataset papers ([1]–[3]) were verified against publisher records during preparation; remaining DOIs follow standard catalog records and should be re-confirmed at final submission.

## Master Comparison Table

| # | Paper Title | Authors (Year) | Venue | Methodology | Dataset | Key Result | Limitations | Relevance / Where Used |
|---|---|---|---|---|---|---|---|---|
| 1 | Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning | Kermany et al. (2018) | Cell 172(5) | Transfer learning (Inception-v3) | OCT + Pediatric CXR (5,856) | ~92.8% acc (pneumonia, binary) | Pediatric, binary, single-source | **Dataset source** (Normal, Pneumonia); baseline (§2.6, §6.1, §6.5) |
| 2 | Can AI Help in Screening Viral and COVID-19 Pneumonia? | Chowdhury et al. (2020) | IEEE Access 8 | Pre-trained CNNs + augmentation | COVID-19 Radiography DB | up to 99.7% (3-class) | Small COVID set, single-source | **Dataset source** (COVID-19); baseline (§2.5, §6.1, §6.5) |
| 3 | Reliable Tuberculosis Detection Using Chest X-Ray... | Rahman et al. (2020) | IEEE Access 8 | 9 CNNs + U-Net segmentation | TB DB (7,000) | 98.6% (DenseNet201, segmented) | Binary TB/Normal | **Dataset source** (TB); motivates segmentation future work (§2.5, §2.7, §6.1, §7.3) |
| 4 | Deep Residual Learning for Image Recognition | He et al. (2016) | IEEE CVPR | Residual skip connections | ImageNet | 3.57% top-5 err | General domain | **Backbone** (ResNet50) (§2.2, §3.4) |
| 5 | CheXNet: Radiologist-Level Pneumonia Detection | Rajpurkar et al. (2017) | arXiv | DenseNet-121 | ChestX-ray14 | Radiologist-level (pneumonia) | Weak/NLP labels | Methodological benchmark (§2.5) |
| 6 | ChestX-ray8: Hospital-scale CXR DB & Benchmarks | Wang et al. (2017) | IEEE CVPR | Multi-label CNN + localization | ChestX-ray14 (112k) | Per-class AUC varies | NLP-mined labels | Future multi-label / COPD & Effusion data (§2.5, §7.3) |
| 7 | CheXpert: Large CXR Dataset with Uncertainty Labels | Irvin et al. (2019) | AAAI | CNN + uncertainty handling | CheXpert (224k) | AUC up to 0.93 | Label uncertainty | Motivates uncertainty modelling (§2.5, §7.3) |
| 8 | Grad-CAM: Visual Explanations... | Selvaraju et al. (2017) | IEEE ICCV | Gradient-based class activation | — | Qualitative localization | Not quantitative | **Proposed XAI** improvement (§2.7, §7.3) |
| 9 | On Calibration of Modern Neural Networks | Guo et al. (2017) | ICML | Temperature scaling, ECE | CIFAR/ImageNet | NNs miscalibrated; T-scaling fixes | Vision-general | **Calibration** roadmap (§2.7, §3.10, §7.3) |
| 10 | A Survey on Deep Learning in Medical Image Analysis | Litjens et al. (2017) | Medical Image Analysis 42 | Survey | — | Taxonomy of DL in med-imaging | Survey (pre-2017) | Field overview (§2.1) |
| 11 | Dermatologist-level Classification of Skin Cancer | Esteva et al. (2017) | Nature 542 | CNN (Inception-v3) | 129k skin images | Dermatologist-level | Single domain | Clinical-grade CAD motivation (§2.1) |
| 12 | AI for Radiographic COVID-19 Detection Selects Shortcuts | DeGrave et al. (2021) | Nature Machine Intelligence 3 | Saliency/shortcut analysis | COVID CXR | Models use confounders | — | **Cross-source/shortcut** motivation (§2.7, §6.7, §7.3) |
| 13 | Very Deep Convolutional Networks (VGG) | Simonyan & Zisserman (2015) | ICLR | Deep small-filter CNN | ImageNet | Depth improves accuracy | Heavy params | Architecture context (§2.2) |
| 14 | Contrast Limited Adaptive Histogram Equalization | Zuiderveld (1994) | Graphics Gems IV | Local contrast enhancement | — | CLAHE algorithm | Classical | **Pre-processing** justification (§2.4, §3.4) |
| 15 | A Survey on Image Data Augmentation for Deep Learning | Shorten & Khoshgoftaar (2019) | Journal of Big Data 6 | Survey | — | Augmentation taxonomy | Survey | **Augmentation** justification (§2.4, §3.4) |
| 16 | ImageNet: A Large-Scale Hierarchical Image Database | Deng et al. (2009) | IEEE CVPR | Large-scale labeled corpus | ImageNet (14M) | Enabled pre-training | General | Transfer-learning basis (§2.3) |

## Thematic Grouping

| Theme | References | Used in |
|---|---|---|
| Datasets used | [1], [2], [3] | §6.1, §2.6 |
| Architectures/backbones | [4], [13], [16] | §3.4 |
| CXR disease detection | [1], [2], [3], [5], [6], [7] | §2.5 |
| Pre-processing & augmentation | [14], [15] | §3.4 |
| Explainability | [8] | §7.3 |
| Calibration & uncertainty | [7], [9] | §3.10, §7.3 |
| Generalization / shortcut learning | [12] | §6.7, §7.3 |
| Field surveys / motivation | [10], [11] | §2.1 |

## Identified Research Gaps (synthesised)
1. **Explainability** — predictions lack visual evidence localization ([8] proposed).
2. **Calibration** — confidence used for urgency is uncalibrated ([9]).
3. **Generalization** — multi-source/modality data risks shortcut learning ([12]).
4. **Workflow integration** — classifiers rarely embedded in auditable clinical workflows (this project's engineering contribution).
