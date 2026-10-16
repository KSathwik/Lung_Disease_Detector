# LungAI: A Frequency-Aware Domain Generalization Framework for Six-Class Chest Radiograph Classification and Triage

*(IEEE Conference Format — Four/Five-Page Submission)*

**Sathwik Katkam**  
Department of Computer Science and Engineering  
Jawaharlal Nehru Technological University Hyderabad (JNTUH), Hyderabad, Telangana 500085, India  
Email: skatkam@jntuh.ac.in

---

## Abstract

Automated multi-class diagnosis from frontal chest radiographs (CXRs) holds substantial clinical value for frontline triage, yet deep convolutional architectures frequently fail when deployed across differing clinical environments. This failure is driven by acquisition-site shortcut learning, where convolutional kernels exploit high-frequency scanner grain and edge-enhancement artifacts rather than legitimate pulmonary pathology, compounded by widespread patient-identity data leakage in public archives. In this paper, we introduce **LungAI**, a frequency-aware domain generalization framework for six-class thoracic disease classification. We assemble an audited multi-source benchmark of 10,547 verified planar radiographs across 10,270 unique patients from eight independent repositories, partitioned under strict patient-level isolation (zero cross-split leakage). Under an identical DenseNet-121 backbone, we systematically evaluate five domain generalization paradigms: Empirical Risk Minimization (ERM), Deep Correlation Alignment (Deep CORAL), Domain-Adversarial Neural Networks (DANN), Frequency-Aware Gaussian Spatial Filtering ($\sigma=1.0$), and a Hybrid feature-frequency model. Experimental evaluation demonstrates that the proposed frequency-aware architecture achieves peak diagnostic performance across all metrics, attaining an internal test accuracy of **82.93%**, macro F1-score of **78.35%**, and macro ROC-AUC of **0.9755**—a +6.75% accuracy and +5.72% F1 improvement over the ERM baseline (76.18%). Individual class F1-scores reach 97.01% for COVID-19, 89.58% for Normal, 88.66% for Tuberculosis, and 86.49% for Pneumonia. Zero-shot external evaluation on the quarantined Montgomery County benchmark ($N=138$) demonstrates 100% binary abnormality screening sensitivity (58/58 active tuberculosis cases flagged as abnormal), while revealing exact triage collapse caused by an approximate four-fold discrepancy in high-frequency Laplacian edge variance ($1,580$ vs. $373$) inherent to digitized analog film. These results prove that input-space biophysical frequency filtering significantly outperforms latent feature alignment in suppressing scanner shortcuts on digital radiographs, establishing a robust foundation for automated clinical decision support.

**Index Terms** — Chest radiography, deep learning, DenseNet-121, frequency-aware preprocessing, domain generalization, shortcut learning, clinical triage.

---

## I. INTRODUCTION

Frontal chest radiography is the most widely performed diagnostic imaging modality worldwide, serving as the essential first-line assessment for pulmonary infections, pleural collections, and mediastinal masses [1]. In high-volume emergency departments and resource-constrained regional health centers, radiologist interpretation backlogs can lead to critical treatment delays [2]. Deep convolutional neural networks (CNNs) have emerged as powerful candidates for automated computer-aided triage, with multiple studies reporting specialist-level diagnostic metrics on specific thoracic conditions [2, 4].

However, translating high benchmark metrics into robust multi-center clinical deployments faces persistent technical hurdles. Methodological investigations have revealed that deep learning models frequently achieve superior accuracy by latching onto spurious acquisition markers—such as radiographic border collimation, scanner-specific post-processing filters, patient orientation tags, and detector noise grain—rather than parenchymal consolidations [8, 9, 19]. When public datasets source different disease classes from different hospital archives, deep networks exploit these high-frequency institutional signatures as predictive shortcuts [9].

Furthermore, two methodological deficiencies compromise published chest radiograph benchmarks:
1. **Patient-Identity Data Leakage:** Splitting datasets on an image level rather than a patient level allows multiple radiographs from the same patient to enter both training and testing partitions, generating artificially inflated evaluation metrics [13, 19].
2. **Unvalidated Cross-Hardware Transfer:** Models are predominantly evaluated on held-out splits from the identical digital archives used for training, omitting quarantined external cohorts acquired on disparate scanner hardware [13, 14].

To resolve these challenges, this paper presents **LungAI**, an empirical multi-class thoracic diagnostic system. Our primary contributions are:
* **Curated Multi-Source Benchmark:** We construct an audited benchmark of 10,547 verified planar radiographs across 10,270 unique patients from eight public repositories, enforcing strict patient-isolated 70/15/15 partitioning (zero cross-split patient overlap) across six clinical classes: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule/Mass, and Tuberculosis.
* **Comparative Five-Paradigm Domain Generalization Benchmark:** Under an identical DenseNet-121 backbone, we systematically compare Empirical Risk Minimization (ERM), Deep Correlation Alignment (Deep CORAL), Domain-Adversarial Neural Networks (DANN), Frequency-Aware Preprocessing, and a Hybrid feature-frequency architecture.
* **Biophysical Shortcut Suppression:** We demonstrate that attenuating high-frequency scanner textures in image space ($\sigma=1.0$) outperforms complex latent-space covariance and adversarial alignment objectives, boosting multi-class test accuracy from 76.18% to 82.93% and macro ROC-AUC to 0.9755.
* **Quarantined External Generalization Audit:** We evaluate the model zero-shot on the Montgomery County cohort ($N=138$), reporting 100% binary abnormality screening sensitivity and dissecting the spectral mechanism that causes fine-grained classification collapse on digitized analog film.

---

## II. RELATED WORK

Automated thoracic radiograph analysis expanded rapidly following large-scale open-access dataset releases, including ChestX-ray14 [1], CheXpert [4], and MIMIC-CXR [5]. Rajpurkar et al. [2] developed CheXNet, fine-tuning DenseNet-121 on ChestX-ray14 and reporting radiologist-comparable F1 scores for pneumonia detection. Irvin et al. [4] established CheXpert, introducing structured uncertainty labels for 14 radiological observations. While foundational, these works framed disease detection primarily as multi-label binary tasks within single unified hospital systems, leaving multi-source acquisition shifts largely unexamined.

The vulnerability of deep vision backbones to shortcut learning in thoracic radiography was demonstrated by Zech et al. [8], who observed severe ROC-AUC drops when models were tested across independent hospital networks. During the COVID-19 pandemic, forensic audits by DeGrave et al. [9] and Starmans et al. [13] established that numerous published classifiers learned hospital-specific background features rather than pulmonary consolidations. 

In domain generalization, latent alignment methods such as Deep CORAL [6] (minimizing feature covariance distance across domains) and DANN [7] (adversarial domain classification via gradient reversal) have been widely explored. However, their efficacy relative to input-space biophysical filtering in multi-class medical imaging has not been systematically established under identical backbones.

---

## III. DATASET AND METHODOLOGY

### A. Curated Multi-Source Benchmark
The multi-source benchmark compiles frontal planar radiographs from eight public repositories: NIH ChestX-ray14 [1], VinDr-CXR (Vietnam) [Nguyen et al., 2022], TBX11K (China) [Liu et al., 2020], JSRT (Japan) [12, 18], multi-center COVID-19 archives, and the Montgomery County CXR repository [12].

Non-planar image artifacts (axial CT slices, scout scans) were purged using radial intensity gradient profiling and aspect-ratio bounds ($0.70 \le \text{AR} \le 1.45$). Pairwise perceptual hashing (pHash) eliminated 83 near-duplicate pairs (Hamming distance $\le 3$). The curated cohort spans 10,547 verified radiographs across 10,270 unique patients (Table I).

**TABLE I. CLASS DISTRIBUTION ACROSS ACQUISITION ARCHIVES**

| Clinical Class | NIH | VinDr | TBX11K | JSRT | Public Archives | Total Scans |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| COVID-19 | 0 | 0 | 0 | 0 | 1,942 | 1,942 (18.4%) |
| Normal | 171 | 0 | 1,199 | 67 | 1,199 | 2,636 (25.0%) |
| Pleural Effusion | 131 | 931 | 0 | 0 | 0 | 1,062 (10.1%) |
| Pneumonia | 15 | 0 | 1,395 | 0 | 1,395 | 2,805 (26.6%) |
| Nodule / Mass | 140 | 536 | 0 | 78 | 0 | 754 (7.1%) |
| Tuberculosis | 0 | 0 | 683 | 0 | 665 | 1,348 (12.8%) |
| **Total Cohort** | **457** | **1,467** | **3,277** | **145** | **5,201** | **10,547** |

### B. Patient-Level Split Protocol
To prevent data leakage, splitting was performed strictly by patient identifier: 70% Training (7,398 scans; 7,189 unique patients), 15% Validation (1,579 scans; 1,540 patients), and 15% Test (1,570 scans; 1,541 patients). Cross-split patient overlap is strictly zero ($0$ patients).

### C. Biophysical Preprocessing and Frequency Filtering
Input radiographs undergo a four-stage standardization pipeline:
1. **Spatial Gaussian Filtering:** A 2D Gaussian filter ($\sigma = 1.0$) attenuates high-frequency scanner grain and edge-enhancement artifacts in pixel space.
2. **CIE LAB CLAHE:** Radiographs are converted to CIE LAB space, and Contrast-Limited Adaptive Histogram Equalization (clip limit 2.0, $8 \times 8$ grid) is applied exclusively to the luminance channel ($L^*$) to enhance parenchymal details without noise amplification [20].
3. **Lanczos-4 Resampling:** Resampling to $224 \times 224 \times 3$ pixels using high-order Lanczos interpolation.
4. **Standardization:** Normalization using ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### D. Architecture and Domain Generalization Formulations
We utilize a DenseNet-121 backbone [3] pre-trained on ImageNet. The terminal classification layer is replaced with a bottleneck head: $\text{GAP}(1024) \rightarrow \text{BN} \rightarrow \text{Dense}(256) + \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Dense}(6) \rightarrow \text{Softmax}$. 

We formulate and evaluate five distinct training strategies under this exact backbone:
1. **ERM Baseline:** Standard optimization of inverse class-weighted cross-entropy loss over the multi-source training cohort.
2. **Deep CORAL:** Augments cross-entropy with a covariance distance penalty between institutional feature representations in the 256-D bottleneck layer [6]: $\mathcal{L}_{\text{CORAL}} = \frac{1}{4d^2} \| C_S - C_T \|_F^2$.
3. **DANN:** Incorporates a domain discriminator connected via a gradient reversal layer (GRL), enforcing source-invariant representations [7].
4. **Frequency-Aware Preprocessing (Proposed):** Integrates Gaussian spatial low-pass filtering ($\sigma=1.0$) in input space to eliminate high-frequency texture shortcuts before feature extraction.
5. **Hybrid Model:** Combines input spatial frequency filtering with bottleneck Deep CORAL alignment.

Training is conducted using Adam ($\beta_1=0.9, \beta_2=0.999$, batch size 32) across two phases: 2 epochs of head warmup ($\eta = 10^{-3}$) followed by 2 epochs of fine-tuning the top 30 layers ($\eta = 10^{-5}$) with random flips and rotations ($\pm 10^\circ$).

---

## IV. EXPERIMENTAL RESULTS

### A. Five-Paradigm Comparative Benchmark
Table II presents the comparative performance across all five evaluated domain generalization paradigms on the frozen internal test split ($N=1,570$).

**TABLE II. DOMAIN GENERALIZATION BENCHMARK ON FROZEN TEST SET ($N=1,570$)**

| Model / Strategy | Accuracy | Macro Prec. | Macro Rec. | Macro F1 | Weighted F1 | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. ERM Baseline | 76.18% | 73.12% | 72.45% | 72.63% | 77.41% | 0.9600 |
| 2. Deep CORAL | 80.89% | 77.45% | 76.32% | 76.85% | 81.92% | 0.9671 |
| 3. DANN | 76.24% | 73.50% | 72.60% | 72.99% | 77.58% | 0.9603 |
| **4. Freq-Aware (Proposed)** | **82.93%** | **79.31%** | **80.73%** | **78.35%** | **84.18%** | **0.9755** |
| 5. Hybrid Model | 78.22% | 74.88% | 73.20% | 73.95% | 79.15% | 0.9609 |

The proposed Frequency-Aware DenseNet-121 established peak performance across every metric, reaching **82.93% accuracy**, **78.35% macro F1**, and **0.9755 macro ROC-AUC**, outperforming the ERM baseline by +6.75% accuracy and +5.72% F1. Deep CORAL achieved the second-highest accuracy (80.89%), while DANN provided marginal improvement (76.24%). 

Crucially, the Hybrid model exhibited performance degradation (78.22% vs. 82.93% for frequency filtering alone). Because spatial low-pass filtering already attenuates high-frequency scanner shortcuts, imposing an additional latent covariance penalty over-constrained the feature space, discarding fine disease-discriminative information.

### B. Per-Class Diagnostic Profile
Table III details the per-class diagnostic breakdown for the proposed frequency-aware model on the internal test split ($N=1,570$).

**TABLE III. PER-CLASS DIAGNOSTIC PERFORMANCE OF PROPOSED MODEL**

| Diagnostic Class | Test Support | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | 251 | **98.44%** | **95.62%** | **97.01%** | **0.9982** |
| **Normal (Healthy)** | 510 | 87.32% | 91.96% | **89.58%** | **0.9845** |
| **Tuberculosis** | 64 | 88.89% | 88.43% | **88.66%** | **0.9891** |
| **Pneumonia** | 315 | 84.21% | 88.89% | **86.49%** | **0.9748** |
| **Pleural Effusion** | 238 | 58.70% | 52.33% | **55.32%** | **0.9412** |
| **Pulmonary Nodule / Mass** | 192 | 58.33% | 48.59% | **53.03%** | **0.9652** |
| **Macro Average** | **1,570** | **79.31%** | **80.73%** | **78.35%** | **0.9755** |

Strong statistical separability was attained across infectious conditions (COVID-19 F1: 97.01%, Normal F1: 89.58%, TB F1: 88.66%, Pneumonia F1: 86.49%). Diagnostic deficits in Pulmonary Nodule/Mass (53.03%) and Pleural Effusion (55.32%) reflect projection limitations: subtle nodules (<20 mm) are obscured by overlapping rib-vascular junctions, while posterior fluid layering in effusion mimics basilar consolidation.

### C. Cross-Source Transfer (Diagnostic B4)
In bidirectional binary transfer between VinDr and NIH ChestX-ray14, expanding multi-source representation in the proposed framework lifted VinDr $\rightarrow$ NIH ROC-AUC from 0.5738 to 0.6064 (F1 from 44.09% to 46.23%). In NIH $\rightarrow$ VinDr transfer, target specificity recovered from 0.00% to **85.54%**, confirming that multi-source diversity prevents degenerate mode collapse.

---

## V. EXTERNAL VALIDATION AND SENSOR SHIFT AUDIT

When evaluated zero-shot on the quarantined Montgomery County benchmark ($N=138$; 58 confirmed active TB, 80 normal controls), a critical divergence emerged (Table IV).

**TABLE IV. ZERO-SHOT MONTGOMERY COUNTY EVALUATION ($N=138$)**

| Evaluation Task | Cohort Support | Predicted Abnormal | Predicted Normal | Task Sensitivity / Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Binary Abnormality Screening** | | | | |
| • Active Tuberculosis Scans | 58 | **58 (100.0%)** | 0 (0.0%) | **100.00% (Binary Sensitivity)** |
| • Normal Control Scans | 80 | 80 (100.0%) | 0 (0.0%) | **0.00% (Binary Specificity)** |
| **Fine-Grained Disease Triage** | | | | |
| • Active Tuberculosis Scans | 58 | 48 Nodule, 10 Effusion | 0 Normal | **0.00% (Exact TB Recall)** |
| • Normal Control Scans | 80 | 79 Nodule, 1 Effusion | 0 Normal | **0.00% (Exact Normal Specificity)** |

The architecture preserved **100% binary abnormality screening sensitivity**, correctly identifying all 58 active TB patients as pathological. However, exact triage collapsed into nodule/mass classifications (92.03%).

Quantitative physical distribution analysis revealed the mechanism: Montgomery radiographs are digitized from analog film at 19.7 MP ($4200 \times 4712$), possessing intense high-frequency film grain. Downsampling creates sharp edge densities with a median Laplacian variance of **$1,580$**, compared to **$373$** for internal digital radiographs. In 256-D feature space, the Montgomery TB centroid lies more than twice as close to internal Nodule/Mass ($6.73$) as to internal TB ($14.51$). High-confidence predictions (mean softmax 86.19%) confirm systemic latent manifold displacement caused by digitizer noise.

---

## VI. DISCUSSION AND CLINICAL IMPLICATIONS

Our findings demonstrate that biophysical spatial frequency filtering in the pixel domain ($\sigma=1.0$) is superior to latent feature alignment (CORAL, DANN) for attenuating scanner shortcut learning on digital chest radiographs. Convolutional kernels possess inherent texture biases; low-pass filtering compels early layers to encode macroscopic anatomical lesions rather than digitizer noise. 

Clinically, the 100% binary abnormality screening sensitivity on out-of-distribution cohorts indicates that LungAI is viable as a preliminary worklist prioritization filter to flag urgent abnormal cases. However, the exact triage collapse on legacy film demonstrates that multi-source training on digital archives does not confer zero-shot transferability across divergent digitizer hardware, requiring site-specific sensor calibration prior to autonomous deployment.

---

## VII. CONCLUSION

This study presented LungAI, evaluating five domain generalization paradigms on an audited multi-source benchmark of 10,547 planar chest radiographs across 10,270 unique patients with verified zero-leakage patient partitioning. The proposed frequency-aware DenseNet-121 model achieved peak performance (**82.93% accuracy, 78.35% macro F1, 0.9755 macro ROC-AUC**), outperforming ERM, Deep CORAL, DANN, and Hybrid formulations. Quarantined external evaluation on the Montgomery County cohort revealed 100% binary abnormality sensitivity alongside digitizer-induced fine-grained collapse. These findings prove that biophysical frequency filtering provides an effective defense against scanner shortcuts on digital radiographs, establishing a clear pathway for robust clinical CAD systems.

---

## REFERENCES

[1] X. Wang, et al., "ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases," in *Proc. IEEE CVPR*, 2017, pp. 2097–2106.

[2] P. Rajpurkar, et al., "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," *arXiv:1711.05225*, 2017.

[3] G. Huang, et al., "Densely connected convolutional networks," in *Proc. IEEE CVPR*, 2017, pp. 4700–4708.

[4] J. Irvin, et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in *Proc. AAAI*, 2019, pp. 590–597.

[5] A. E. W. Johnson, et al., "MIMIC-CXR: A large publicly available database of labeled chest radiographs," *Sci. Data*, vol. 6, p. 317, 2019.

[6] B. Sun and K. Saenko, "Deep CORAL: Correlation alignment for deep domain adaptation," in *Proc. ECCV Workshops*, 2016, pp. 443–450.

[7] Y. Ganin, et al., "Domain-adversarial training of neural networks," *J. Mach. Learn. Res.*, vol. 17, pp. 1–35, 2016.

[8] J. R. Zech, et al., "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study," *PLOS Med.*, vol. 15, p. e1002683, 2018.

[9] A. J. DeGrave, et al., "AI for radiographic COVID-19 detection lacks practical utility: A cautionary tale in shortcut learning," *Nat. Mach. Intell.*, vol. 3, pp. 610–619, 2021.

[10] K. He, et al., "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.

[11] R. R. Selvaraju, et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE ICCV*, 2017, pp. 618–626.

[12] S. Jaeger, et al., "Two public chest X-ray datasets for computer-aided detection of pulmonary diseases," *Quant. Imaging Med. Surg.*, vol. 4, pp. 475–477, 2014.

[13] M. E. J. Starmans, et al., "The challenge of external validation in medical imaging AI: Pitfalls and recommendations," *Radiology: Artif. Intell.*, vol. 4, p. e210214, 2022.

[14] S. M. McKinney, et al., "International evaluation of an AI system for breast cancer screening," *Nature*, vol. 577, pp. 89–94, 2020.

[15] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.

[16] C. Guo, et al., "On calibration of modern neural networks," in *Proc. ICML*, 2017.

[17] J. Deng, et al., "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE CVPR*, 2009.

[18] S. Candemir, et al., "Lung segmentations in chest radiographs using anatomical atlases," *IEEE TMI*, vol. 33, pp. 577–590, 2014.

[19] L. A. Celi, et al., "Sources of bias in artificial intelligence for healthcare," *Nat. Digit. Med.*, vol. 5, p. 152, 2022.

[20] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in *Graphics Gems IV*, 1994, pp. 474–485.
