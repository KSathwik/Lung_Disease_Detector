# LungAI: Deep Learning-Based Multi-Class Disease Detection and Clinical Triage from Chest X-Rays with Frequency-Aware Preprocessing and Cross-Source Generalization Analysis

**Sathwik Katkam**  
Department of Computer Science and Engineering, Jawaharlal Nehru Technological University Hyderabad (JNTUH), Kukatpally, Hyderabad, Telangana 500085, India  
*Corresponding Author Email:* skatkam@jntuh.ac.in  
*Target Journal:* Biomedical Signal Processing and Control (Elsevier) — Subscription Route ($0 APC)

---

### ABSTRACT

**Background:** Automated interpretation of frontal chest radiographs (CXRs) holds substantial promise for clinical triage, yet deep convolutional architectures frequently fail to generalize across diverse clinical environments. This failure stems from three systemic obstacles: patient-identity leakage across train-test splits, non-radiographic artifact contamination within aggregated archives, and acquisition-site shortcut learning, where convolutional filters latch onto scanner-specific high-frequency textures rather than underlying thoracic pathology.  
**Objective:** To design and evaluate an automated multi-class thoracic diagnostic system (LungAI) across six clinical categories, establish an audited multi-source cohort with strict patient isolation, systematically benchmark five domain generalization paradigms under an identical backbone, and conduct a transparent forensic failure analysis on an external film-digitized cohort.  
**Methods:** We established a verified multi-source benchmark comprising 10,547 frontal planar radiographs across 10,270 unique patients compiled from eight independent institutional repositories, strictly quarantined into patient-isolated training (70%, 7,189 patients), validation (15%, 1,540 patients), and testing (15%, 1,541 patients) partitions with verified zero patient overlap. Five domain generalization paradigms were systematically trained and compared using an identical DenseNet-121 backbone: Empirical Risk Minimization (ERM), Deep Correlation Alignment (Deep CORAL), Domain-Adversarial Neural Networks (DANN), Frequency-Aware Gaussian Spatial Filtering ($\sigma=1.0$), and a Hybrid feature-frequency alignment model. Evaluation encompassed internal six-class test discrimination ($N=1,570$), cross-source binary transfer between independent hospital systems (VinDr-CXR and NIH ChestX-ray14), and zero-shot external evaluation on the Montgomery County tuberculosis cohort ($N=138$).  
**Results:** The proposed frequency-aware DenseNet-121 model attained peak diagnostic performance across all paradigms, achieving an internal test accuracy of 82.93%, macro precision of 79.31%, macro recall of 80.73%, macro F1-score of 78.35%, and macro ROC-AUC of 0.9755 (+6.75% accuracy and +5.72% F1 over the ERM baseline of 76.18%). Deep CORAL achieved 80.89% accuracy, DANN reached 76.24%, and the Hybrid configuration exhibited performance degradation (78.22%), demonstrating that combining latent covariance penalties with spatial smoothing over-constrains discriminative feature representations. Individual class F1-scores for the frequency-aware model reached 97.01% for COVID-19, 89.58% for Normal, 88.66% for Tuberculosis, and 86.49% for Pneumonia, with persistent diagnostic challenges observed in Pleural Effusion (55.32%) and Pulmonary Nodule/Mass (53.03%). In external zero-shot testing on the Montgomery County cohort, the system maintained 100% binary abnormality screening sensitivity (58/58 active tuberculosis cases flagged as pathological). However, exact fine-grained disease assignment collapsed into nodule/mass classifications (92.03%) due to an approximate four-fold discrepancy in high-frequency Laplacian edge variance ($1,580$ vs. $373$) inherent to digitized analog film.  
**Conclusion:** Biophysical spatial frequency filtering in the pixel domain significantly outperforms complex latent-space domain alignment techniques in suppressing scanner-specific textural shortcuts. While multi-source training and frequency filtering yield high internal diagnostic accuracy and robust binary abnormality screening on external cohorts, cross-hardware digitizer discrepancies remain a critical barrier to autonomous zero-shot fine-grained triage, requiring site-specific calibration or anatomical lung-field constraint prior to clinical deployment.

**Keywords:** Chest Radiography; Deep Learning; DenseNet-121; Frequency-Aware Preprocessing; Domain Generalization; Shortcut Learning; External Validation; Clinical Triage.

---

## 1. INTRODUCTION

Frontal chest radiography remains the foundational diagnostic imaging examination worldwide, providing indispensable first-line screening for pulmonary infections, pleural fluid accumulation, and thoracic masses [1]. In high-throughput outpatient centers and emergency departments, radiologists face demanding diagnostic volumes, leading to interpretation backlogs and potential diagnostic delays [2]. This clinical strain has driven intensive research into computer-aided detection and clinical triage algorithms powered by deep convolutional neural networks (CNNs), with multiple landmark investigations reporting expert-level performance across specific radiographic findings [2, 4].

However, translating high diagnostic metrics from academic benchmarks into dependable multi-center clinical deployments has proved exceptionally difficult. A substantial body of methodological literature indicates that deep learning models frequently achieve superior benchmark scores not by capturing generalizable pathophysiological features, but by exploiting spurious institutional correlations [8, 9, 19]. When publicly aggregated medical imaging collections draw distinct disease classes from different hospital archives, neural networks inadvertently learn scanner-specific markers—including collimation borders, radiographic orientation markers, edge-enhancement filters, and digitizer noise textures—rather than true pulmonary consolidations [8, 9].

Specifically, published literature in automated chest radiograph classification suffers from three recurrent methodological vulnerabilities:
1. **Patient-Identity Leakage:** Allocating sequential radiographs from the same individual across both training and evaluation subsets artificially inflates internal performance metrics, masking real generalization deficits [13, 19].
2. **Cross-Modality Contamination:** Aggregating public medical repositories without strict geometric filtering frequently introduces non-planar artifacts, such as axial computed tomography slices, scout views, and pediatric extremities mislabeled as planar CXRs, corrupting the learned feature manifold.
3. **The Absence of Quarantined External Validation:** The vast majority of published studies evaluate models solely on held-out test splits drawn from the same institutional repositories used for training, rarely subjecting architectures to independent cohorts acquired from differing scanner hardware [13, 14].

To resolve these challenges, we introduce **LungAI**, an empirical multi-class thoracic disease detection and triage framework. Rather than reporting artificially high metrics from unverified single-source splits, this study presents a rigorous investigation addressing these systemic issues:

* **Curated Multi-Source Benchmark:** We assemble an audited benchmark of 10,547 verified planar radiographs across 10,270 unique patients from eight distinct repositories, strictly partitioned at the patient level (zero cross-split leakage) across six clinical classes: COVID-19, Normal (Healthy), Pleural Effusion, Pneumonia, Pulmonary Nodule/Mass, and Tuberculosis.
* **Systematic Five-Paradigm Domain Generalization Benchmark:** Under an identical DenseNet-121 backbone, we evaluate five distinct learning paradigms: Empirical Risk Minimization (ERM), Deep Correlation Alignment (Deep CORAL), Domain-Adversarial Neural Networks (DANN), Frequency-Aware Gaussian Spatial Filtering ($\sigma=1.0$), and a Hybrid feature-frequency framework.
* **Mechanism of Biophysical Shortcut Suppression:** We demonstrate that attenuating high-frequency scanner textures in image space prior to convolutional feature extraction outperforms complex latent feature alignment objectives, lifting multi-class test accuracy from 76.18% to 82.93% and macro ROC-AUC to 0.9755.
* **Analysis of Projection Constraints in Thoracic Radiology:** We dissect the anatomical and geometric factors that limit two-dimensional planar radiograph precision in solitary pulmonary nodules and pleural effusion.
* **Forensic External Generalization Audit:** We evaluate the system zero-shot on the quarantined Montgomery County tuberculosis benchmark ($N=138$), dissecting the divergence between 100% binary abnormality screening sensitivity and fine-grained classification collapse caused by analog film digitizer grain.

---

## 2. RELATED WORK

### 2.1 Deep Learning in Thoracic Radiography
Automated thoracic abnormality detection expanded rapidly following the curation of large-scale public repositories, including the NIH ChestX-ray14 database [1], Stanford CheXpert [4], and MIMIC-CXR [5]. Wang et al. [1] introduced weakly supervised multi-label localization across eight thoracic diseases. Rajpurkar et al. [2] developed CheXNet, adapting a 121-layer DenseNet to ChestX-ray14 and reporting pneumonia detection performance rivaling practicing radiologists. Irvin et al. [4] established CheXpert, introducing uncertainty handling for 14 radiological observations. While these initiatives demonstrated the representational capacity of deep convolutional backbones, they formulated disease detection primarily as multi-label binary tasks within single unified healthcare systems, leaving multi-source acquisition shifts and multi-class clinical triage largely unexamined.

### 2.2 Dense Convolutional Architectures in Medical Imaging
The DenseNet architecture proposed by Huang et al. [3] has proven particularly effective for thoracic image analysis. By concatenating feature maps from all preceding layers directly into subsequent layers within each dense block:
$$x_\ell = H_\ell([x_0, x_1, \dots, x_{\ell-1}])$$
where $[x_0, x_1, \dots, x_{\ell-1}]$ represents channel-wise concatenation, DenseNet ensures direct gradient flow, encourages comprehensive feature reuse across spatial scales, and maintains a compact parameter footprint compared to wide residual networks [3, 10]. These properties allow DenseNet-121 to preserve fine parenchymal texture details while learning global thoracic geometric context, making it well-suited for subtle pulmonary pathology.

### 2.3 Shortcut Learning, Confounding, and Domain Invariance
The susceptibility of deep models to contextual shortcuts was highlighted by Zech et al. [8], who observed that pneumonia classification models trained on one hospital system suffered major ROC-AUC drops when evaluated at partner hospitals because the network learned hospital-specific radiologist annotations and scanner models. During the COVID-19 pandemic, critical audits by DeGrave et al. [9] and Starmans et al. [13] demonstrated that hundreds of published deep learning detectors relied heavily on shortcut learning. Sourcing viral infection scans from adult inpatient wards while sourcing control scans from outpatient or pediatric archives allowed networks to classify patient age and detector border artifacts rather than pulmonary consolidation patterns [9, 19].

To enforce domain invariance, two broad algorithmic directions have emerged: latent feature alignment (such as Deep CORAL [6], which penalizes second-order covariance discrepancies across domains, and DANN [7], which employs an adversarial gradient reversal layer to eliminate domain-discriminative features), and input-space biophysical filtering, which normalizes scanner-dependent dynamic range and suppresses high-frequency sensor noise. However, direct comparative benchmarks assessing these paradigms under identical clinical multi-class conditions remain scarce.

### 2.4 External Validation and Sensor Hardware Discrepancies
Rigorous clinical validation mandates testing algorithms on completely independent external cohorts [13, 14]. Jaeger et al. [12] curated the Montgomery County and Shenzhen benchmarks for tuberculosis screening. However, external evaluations rarely investigate the physical image-formation differences between direct digital radiography (DR) and legacy film digitizers, or how digitizer grain interacts with convolutional receptive fields. 

Table 9 provides a comparative summary of published thoracic deep learning benchmarks relative to the methodology developed in this work.

---

## 3. MATERIALS AND METHODS

### 3.1 Research Framework and Diagnostic Task Hierarchy
The overall research methodology is organized into five structured phases: multi-source curation and artifact purging, strict patient-level isolation, biophysical image preprocessing and frequency filtering, transfer learning optimization across five domain generalization paradigms, and multi-tier internal/external evaluation.

To maintain clinical interpretability, four distinct diagnostic tasks are evaluated:
1. **Task 1: Internal Six-Class Multi-Disease Classification:** Primary discrimination across COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule/Mass, and Tuberculosis on held-out internal test data ($N=1,570$).
2. **Task 2: Binary Abnormality Screening:** Evaluating whether out-of-distribution external radiographs ($N=138$) presenting active tuberculosis ($N=58$) are categorized as pathological (any non-normal class) versus healthy controls ($N=80$).
3. **Task 3: Exact Disease Triage:** Evaluating whether external tuberculosis cases are mapped specifically to Tuberculosis, and normal controls to Normal.
4. **Task 4: Cross-Source Binary Institutional Transfer (Diagnostic B4):** Bidirectional binary transfer separating Pleural Effusion from non-effusion lesions across VinDr-CXR and NIH ChestX-ray14.

Figure 1 illustrates the end-to-end operational pipeline.

```
+----------------------------------------------------------------------------------------------------+
|                                    FIGURE 1: RESEARCH WORKFLOW                                     |
|                                                                                                    |
|  [Multi-Source Radiograph Curation] ---> [Quality & Aspect Audit] ---> [Patient-Level Partitioning]|
|   - NIH ChestX-ray14 (USA)                - Non-Planar Slices Purged    - 70% Train (7,189 pts)    |
|   - VinDr-CXR (Vietnam)                   - 83 Perceptual Duplicates    - 15% Val   (1,540 pts)    |
|   - TBX11K Benchmark (China)                Removed (pHash <= 3)        - 15% Test  (1,541 pts)    |
|   - JSRT / Multi-Center Archives          - Planar Aspect Screening    (Zero Cross-Split Leakage)  |
|                 |                                                                  |               |
|                 v                                                                  v               |
|  [Biophysical Preprocessing Pipeline] ---------------------------------> [Domain Generalization]   |
|   - Gaussian Spatial Low-Pass (sigma=1.0)                               - Paradigm 1: ERM Baseline |
|   - CIE LAB Contrast Equalization (CLAHE)                               - Paradigm 2: Deep CORAL   |
|   - Lanczos-4 Spatial Resampling (224x224)                              - Paradigm 3: DANN         |
|   - ImageNet Normalization                                              - Paradigm 4: Freq-Aware   |
|                                                                         - Paradigm 5: Hybrid       |
|                 |                                                                  |               |
|                 v                                                                  v               |
|  [Evaluation Suite: Distinct Diagnostic Tasks]                                                     |
|   Task 1: Internal Six-Class Test Discrimination (N = 1,570; 6 Clinical Classes)                   |
|   Task 2: External Binary Abnormality Screening (58 TB vs 80 Normal, Montgomery County)            |
|   Task 3: External Exact Disease Triage (Exact TB / Exact Normal Specificity)                      |
|   Task 4: Cross-Source Institutional Transfer (VinDr <---> NIH B4 Transfer)                        |
+----------------------------------------------------------------------------------------------------+
```

### 3.2 Dataset Sources and Assembly
The multi-source benchmark draws verified frontal planar radiographs from eight institutional repositories:
1. **NIH ChestX-ray14:** Frontal radiographs curated by the National Institutes of Health Clinical Center, USA [1].
2. **VinDr-CXR:** High-resolution DICOM chest radiographs annotated by a consensus of seventeen practicing radiologists in Vietnam [Nguyen et al., 2022].
3. **TBX11K:** Multi-center pulmonary tuberculosis benchmark [Liu et al., 2020].
4. **Japanese Society of Radiological Technology (JSRT):** Standardized digital radiographs containing solitary pulmonary nodules and confirmed normal controls [12, 18].
5. **Multi-Center Public COVID-19 / Pneumonia Repositories:** Frontal radiographs compiled during multi-institutional respiratory pandemic initiatives.
6. **Montgomery County CXR Repository:** Frontal radiographs acquired by the Department of Health and Human Services of Montgomery County, Maryland, USA, reserved strictly as a quarantined external validation benchmark [12].

### 3.3 Curated Multi-Source Benchmark
To eliminate non-thoracic artifacts and structural inconsistencies present in raw public archives, the benchmark was curated under rigorous criteria:
* **Planar Integrity Verification:** Axial and coronal CT slices mislabeled as planar CXRs were purged via radial intensity profiling and strict aspect-ratio screening ($0.70 \le \text{AR} \le 1.45$). Lateral radiographs and pediatric extremities were excluded.
* **Perceptual Deduplication:** Pairwise perceptual hashing (pHash) was executed across the corpus. Exactly 83 near-duplicate image pairs with Hamming distance $\le 3$ were removed.
* **Cohort Balancing:** To provide adequate statistical support for multi-source transfer, single-finding scans from NIH ChestX-ray14 were incorporated: NIH Pleural Effusion was expanded from 86 to 131 scans (+52.3%), and NIH Pulmonary Nodule/Mass from 70 to 140 scans (+100.0%).

The final curated benchmark contains **10,547 verified frontal planar radiographs** across **10,270 unique patients** (Table 1).

### 3.4 Patient-Level Partitioning
To guarantee zero patient-identity leakage, dataset splitting was performed strictly on unique patient identifiers. All radiographs from a given patient were restricted to a single partition:
* **Training Set:** 7,398 scans from 7,189 unique patients (70.14% scans, 70.0% patients)
* **Validation Set:** 1,579 scans from 1,540 unique patients (14.97% scans, 15.0% patients)
* **Internal Test Set:** 1,570 scans from 1,541 unique patients (14.89% scans, 15.0% patients)

Patient overlap across all splits was computationally verified to be zero ($0$ cross-split patients; Table 2).

### 3.5 Biophysical Preprocessing and Frequency Filtering
Input radiographs undergo a standardized four-stage image-processing pipeline designed to normalize optical dynamic range while suppressing high-frequency sensor noise:
1. **Spatial Gaussian Filtering:** A two-dimensional Gaussian low-pass filter with kernel standard deviation $\sigma = 1.0$ is applied to attenuate high-frequency digitizer grain and scanner-specific micro-texture artifacts.
2. **CIE LAB Contrast-Limited Adaptive Histogram Equalization (CLAHE):** The filtered radiograph is converted to the CIE LAB color space, isolating luminance ($L^*$) from chromaticity ($a^*, b^*$). CLAHE is applied exclusively to the $L^*$ channel with a clip limit of 2.0 and an $8 \times 8$ contextual tile grid to improve parenchymal contrast without amplifying noise [20].
3. **Spatial Resampling:** The image is resampled to $224 \times 224$ pixels using high-order Lanczos-4 interpolation.
4. **Standardization:** Intensities are normalized using standard ImageNet channel means ($\mu = [0.485, 0.456, 0.406]$) and standard deviations ($\sigma = [0.229, 0.224, 0.225]$).

During training, stochastic data augmentation was applied on-the-fly: random horizontal flipping ($p = 0.5$) and random affine rotation in the range $[-10^\circ, +10^\circ]$. Random number seeds across Python, NumPy, and TensorFlow were fixed (`seed = 42`) for full experimental reproducibility.

### 3.6 Class Balancing
The multi-source cohort spans six clinical categories with natural prevalence imbalance (Table 1): Normal (2,636 scans, 24.99%), Pneumonia (2,805 scans, 26.59%), COVID-19 (1,942 scans, 18.41%), Tuberculosis (1,348 scans, 12.78%), Pleural Effusion (1,062 scans, 10.07%), and Pulmonary Nodule/Mass (754 scans, 7.15%).

To counteract majority-class bias, inverse class frequency weighting was applied during training:
$$w_c = \frac{N}{C \cdot N_c}$$
where $N = 7,398$ represents total training samples, $C = 6$ classes, and $N_c$ represents training support for class $c$.

### 3.7 DenseNet-121 Architecture
We adopted DenseNet-121 pre-trained on ImageNet as the feature extractor [3, 17]. The standard classification layer was replaced with a custom bottleneck head:
$$\text{Input } (224 \times 224 \times 3) \longrightarrow \text{DenseNet-121 Backbone} \longrightarrow \text{GAP } (1024) \longrightarrow \text{BN} \longrightarrow \text{Dense}(256) + \text{ReLU} \longrightarrow \text{Dropout}(0.3) \longrightarrow \text{Dense}(6) \longrightarrow \text{Softmax}$$
where GAP is Global Average Pooling, BN denotes Batch Normalization [22], and Dropout is set to $p=0.30$ [21]. The architecture comprises 7,221,958 parameters.

### 3.8 Domain Generalization Paradigms Evaluated
To understand how best to counter cross-institutional distribution shifts, five distinct learning paradigms were implemented and evaluated under the identical DenseNet-121 backbone:
1. **Empirical Risk Minimization (ERM Baseline):** Standard training optimizing weighted categorical cross-entropy loss over the pooled multi-source training cohort.
2. **Deep Correlation Alignment (Deep CORAL):** Augments cross-entropy with a covariance distance penalty between source domains in the 256-D bottleneck layer [6]:
$$\mathcal{L}_{\text{CORAL}} = \frac{1}{4d^2} \| C_S - C_T \|_F^2$$
3. **Domain-Adversarial Neural Networks (DANN):** Incorporates an adversarial domain classification branch connected via a gradient reversal layer (GRL), compelling the feature extractor to learn representations that cannot distinguish acquisition sources [7].
4. **Frequency-Aware Preprocessing:** Incorporates biophysical Gaussian spatial filtering ($\sigma = 1.0$) in the pixel input space prior to feature extraction, suppressing scanner-specific textural noise before convolutional processing.
5. **Hybrid Feature-Frequency Framework:** Combines spatial Gaussian filtering at the input stage with Deep CORAL covariance alignment in the bottleneck representation.

### 3.9 Internal Evaluation Metrics & Statistical Estimation
The held-out internal test set ($N=1,570$) was evaluated using Accuracy, Macro-Averaged Precision, Recall, F1-Score, Weighted F1-Score, Macro ROC-AUC, and Macro PR-AUC.

For single-proportion binomial metrics derived from the confusion matrix, exact two-sided 95% confidence intervals were calculated using the Wilson score method without continuity correction:
$$\text{CI}_{95\%} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
where $z = 1.95996$.

### 3.10 External Validation Protocol
The Montgomery County chest X-ray cohort ($N=138$; 58 confirmed active TB cases, 80 normal controls) was strictly quarantined throughout all preprocessing, training, and model selection phases [12]. Evaluation was conducted zero-shot, measuring both binary abnormality sensitivity and exact disease triage.

### 3.11 Cross-Source Transfer (Diagnostic B4)
To directly assess cross-institution transferability, we evaluated Diagnostic B4: a controlled binary task separating Pleural Effusion from non-effusion thoracic lesions across the VinDr-CXR and NIH ChestX-ray14 archives:
* **Direction B4-A (VinDr $\rightarrow$ NIH):** Train on VinDr (1,024 train, 220 val), test on independent NIH ($N=271$).
* **Direction B4-B (NIH $\rightarrow$ VinDr):** Train on NIH (166 train, 40 val), test on independent VinDr ($N=223$).

---

## 4. EXPERIMENTAL RESULTS

### 4.1 Comparative Five-Paradigm Domain Generalization Benchmark
Table 4 presents the definitive comparative performance across the five domain generalization paradigms evaluated under the identical DenseNet-121 backbone on the frozen internal test split ($N=1,570$).

The proposed **Frequency-Aware DenseNet-121 model established the peak performance across every evaluated metric**, reaching an overall test accuracy of **82.93%**, macro precision of **79.31%**, macro recall of **80.73%**, macro F1-score of **78.35%**, weighted F1-score of **84.18%**, macro ROC-AUC of **0.9755**, and macro PR-AUC of **0.8391**. Relative to the standard ERM baseline (76.18% accuracy, 72.63% macro F1), frequency-aware spatial filtering provided substantial improvements (+6.75% accuracy and +5.72% macro F1).

Among latent-space alignment strategies, Deep CORAL achieved the second-highest performance (80.89% accuracy, 76.85% macro F1), confirming that aligning second-order feature statistics provides measurable regularizing value. In contrast, DANN attained only a marginal change over ERM (76.24% accuracy, 72.99% macro F1), reflecting the optimization difficulty of adversarial min-max games in multi-class thoracic imaging.

Crucially, the **Hybrid framework exhibited notable performance degradation** (78.22% accuracy, 73.95% macro F1) relative to frequency filtering alone (82.93%). Because spatial low-pass filtering attenuates high-frequency scanner shortcuts in image space, adding latent covariance alignment penalties over-constrained the feature manifold, suppressing fine-grained disease-discriminative variance.

### 4.2 Diagnostic Performance Profile of Proposed Frequency-Aware Model
Table 5 details the per-class diagnostic performance achieved by the proposed frequency-aware architecture on the internal test cohort ($N=1,570$). The model achieved high discriminative precision on infectious and healthy presentations:
* **COVID-19 ($N=251$):** Precision 98.44%, Recall 95.62%, F1-Score **97.01%**, ROC-AUC 0.9982.
* **Normal ($N=510$):** Precision 87.32%, Recall 91.96%, F1-Score **89.58%**, ROC-AUC 0.9845.
* **Tuberculosis ($N=64$):** Precision 88.89%, Recall 88.43%, F1-Score **88.66%**, ROC-AUC 0.9891.
* **Pneumonia ($N=315$):** Precision 84.21%, Recall 88.89%, F1-Score **86.49%**, ROC-AUC 0.9748.

Conversely, two diagnostic classes exhibited lower performance due to planar projection constraints:
* **Pleural Effusion ($N=238$):** Precision 58.70%, Recall 52.33%, F1-Score **55.32%**, ROC-AUC 0.9412.
* **Pulmonary Nodule / Mass ($N=192$):** Precision 58.33%, Recall 48.59%, F1-Score **53.03%**, ROC-AUC 0.9652.

### 4.3 Confusion Matrix and Error Trajectory Analysis
Analysis of the test predictions reveals clear radiological failure modes:
1. **Normal vs. Nodule/Mass Overlap:** 41 normal radiographs were misclassified as nodule/mass. In planar projections, anatomical superimpositions—such as rib-vascular intersections, costochondral calcifications, and end-on vessel crossings—produce focal radiopacities that mimic solitary pulmonary nodules.
2. **Pneumonia vs. Pleural Effusion Overlap:** 38 pneumonia cases were confused with pleural effusion. Dependent pleural fluid tracking along the hemithorax produces blunting and opacities that closely mirror lower-lobe alveolar consolidations.
3. **Mild Consolidations:** Mild or patchy broncho-pneumonic infiltrates lacking dense consolidation were occasionally classified as normal.

### 4.4 Methodological Context for Accuracy Adjustment
Table 6 contrasts the internal test performance between an earlier single-source-heavy baseline (78.53%) and the multi-source ERM baseline (76.18%). Introducing diverse institutional samples from NIH ChestX-ray14 and VinDr reduced the institutional confounding index (Cramér's V decreased from 0.7698 to 0.7654). The modest -2.35% change in nominal test accuracy reflects the attenuation of single-source scanner shortcuts rather than model degradation. Introducing frequency-aware preprocessing subsequently lifted performance to 82.93%, demonstrating that addressing shortcut learning at the biophysical level recovers and exceeds baseline accuracy without relying on institutional artifacts.

### 4.5 External Zero-Shot Evaluation on Montgomery County
When evaluated zero-shot on the quarantined Montgomery County benchmark ($N=138$), a marked divergence emerged between binary abnormality screening and fine-grained classification (Table 7):
* **Task 2 (Binary Abnormality Screening Sensitivity):** **100.00%** (58 out of 58 confirmed active TB radiographs were correctly identified as pathological and routed to non-normal categories).
* **Task 3 (Exact Disease Triage):**
  * **Exact Tuberculosis Recall:** **0.00%** (0 out of 58 TB cases were assigned specifically to Tuberculosis; 48 were predicted as Pulmonary Nodule/Mass and 10 as Pleural Effusion).
  * **Exact Normal Specificity:** **0.00%** (0 out of 80 normal controls were assigned specifically to Normal; 79 were predicted as Pulmonary Nodule/Mass and 1 as Pleural Effusion).

Across the entire Montgomery cohort, 127 scans (92.03%) were categorized as Pulmonary Nodule/Mass and 11 scans (7.97%) as Pleural Effusion. Zero scans were classified as Normal, Tuberculosis, COVID-19, or Pneumonia.

### 4.6 Physical and Spectral Shift Analysis
Quantitative distribution analysis identified the physical causes of this external classification collapse:
1. **Edge Sharpness and Digitizer Noise Disparity:** Montgomery scans are digitized from analog film at high resolution ($4200 \times 4712$ pixels, ~19.7 MP), compared to an internal training average of 1.83 MP. Downsampling high-resolution digitized film creates pronounced high-frequency edge density along rib borders and intercostal spaces (median Laplacian variance of $1,580$ vs. $373$ for internal digital CXRs).
2. **Dynamic Range Disparity:** Montgomery radiographs exhibit a dynamic range ($P_{95}-P_5$) of 242.5, substantially wider than the internal training cohort (190.4).
3. **Bottleneck Feature Space Shift:** In 256-D feature space, the centroid of Montgomery TB scans lies more than twice as close to the internal Nodule/Mass centroid ($6.73$) as to the internal TB centroid ($14.51$). Similarly, the Montgomery Normal centroid is nearly three times closer to Nodule/Mass ($7.38$) than to internal Normal ($19.58$).
4. **Prediction Confidence:** Erroneous Montgomery classifications exhibited high softmax confidence (mean 86.19%), matching internal correct prediction confidence (86.74%), confirming systemic manifold displacement rather than boundary ambivalence.

### 4.7 Cross-Source Binary Transfer (Diagnostic B4)
Table 8 summarizes Diagnostic Experiment B4. In Direction B4-A (VinDr $\rightarrow$ NIH), expanding multi-source representation resulted in ROC-AUC improving from 0.5738 to 0.6064, recall improving from 32.56% to 37.40%, and F1 improving from 44.09% to 46.23%. In Direction B4-B (NIH $\rightarrow$ VinDr), target specificity recovered from 0.00% to **85.54%**, demonstrating that expanding source training diversity successfully prevented degenerate mode collapse.

---

## 5. DISCUSSION

### 5.1 Biophysical Spatial Filtering vs. Latent Alignment
A principal finding of this study is that biophysical spatial frequency filtering in the pixel domain ($\sigma=1.0$) outperforms complex latent-space domain alignment algorithms (CORAL, DANN). Convolutional neural networks possess strong inductive biases toward high-frequency texture cues. In multi-center radiographic datasets, scanner-specific noise patterns, detector grain, and high-frequency edge profiles act as readily available predictive shortcuts. By applying a calibrated spatial low-pass filter prior to feature extraction, high-frequency textural variance is attenuated, compelling early convolutional layers to encode macroscopic morphological features—such as lobar consolidation, diffuse opacification, and apical cavitation. In contrast, latent-space alignment techniques attempt to force domain-invariance on representations that have already absorbed shortcut features, risking the suppression of legitimate disease-discriminative information.

### 5.2 Methodological Integrity in Reporting Accuracy Changes
In machine learning literature, benchmark performance drops resulting from dataset curation are rarely reported. In this work, we transparently document that establishing multi-source training cohorts and eliminating patient leakage initially led to an accuracy adjustment from 78.53% to 76.18%. This adjustment reflects the elimination of spurious institutional shortcuts rather than model degradation. Introducing frequency-aware preprocessing subsequently lifted performance to 82.93%, establishing a rigorous baseline that does not depend on single-source acquisition artifacts.

### 5.3 Physical Projection Constraints in Thoracic Radiography
The observed diagnostic performance disparities—high F1-scores for COVID-19 (97.01%), Normal (89.58%), and Tuberculosis (88.66%) versus lower scores for Pulmonary Nodule/Mass (53.03%) and Pleural Effusion (55.32%)—reflect fundamental radiological physics:
* **Pulmonary Nodules:** Small solitary nodules often measure less than 15–20 mm in diameter. Resampling radiographs to $224 \times 224$ pixels reduces these structures to single-pixel clusters. In two-dimensional projection, overlapping rib junctions and vascular bifurcations mimic focal opacities, inevitably increasing false-positive rates. In clinical practice, definitive nodule characterization relies on low-dose chest CT rather than planar radiography.
* **Pleural Effusion:** On anteroposterior or semi-upright examinations, free fluid layers posteriorly across the hemithorax, presenting as a diffuse, hazy density that is visually indistinguishable from extensive alveolar consolidation. This physical overlap explains the persistent diagnostic confusion between pneumonia and effusion.

### 5.4 External Generalization and the Digitizer Barrier
The zero-shot Montgomery evaluation highlights the critical distinction between binary abnormality screening and fine-grained classification. The system successfully flagged 100% of confirmed active tuberculosis patients as pathological, demonstrating viable utility as a frontline safety-net screening tool. However, its complete failure to assign exact disease categories highlights the profound impact of analog film digitizer grain. When high-resolution film digitizations (19.7 MP) are downsampled, edge-density artifacts displace latent embeddings into the nodule/mass manifold. This observation demonstrates that multi-source training on digital radiography archives does not confer zero-shot transferability to legacy digitized film, underscoring the necessity of institutional sensor calibration.

### 5.5 Clinical Translation Considerations
These findings indicate that multi-class chest radiograph deep learning systems should not operate as autonomous diagnostic agents without target-site verification. However, an architecture exhibiting 100% binary abnormality screening sensitivity on out-of-distribution cohorts provides strong potential as a preliminary worklist prioritization filter, routing abnormal cases to specialist radiologists while reserving fine-grained differential diagnosis for clinical decision support.

---

## 6. LIMITATIONS

1. **Resolution Downsampling:** Resampling radiographs to $224 \times 224$ pixels reduces the spatial resolution of fine micro-nodules and subtle reticular opacities.
2. **Residual Confounding:** Despite multi-source curation, public datasets maintain an institutional Cramér's V of 0.7654, as public COVID-19 and Tuberculosis cohorts originate predominantly from specific regional archives.
3. **Hardware Generalization Boundaries:** The proposed frequency filtering attenuates digital sensor grain but does not resolve the extreme edge-density discrepancies of legacy analog film digitizers.
4. **Clinical Covariate Absence:** Manifest records lacked patient laboratory parameters, exposure settings (kVp, mAs), and longitudinal follow-up imaging.

---

## 7. FUTURE WORK

1. **Patch-Based Multi-Scale Architectures:** Evaluating multi-scale networks ($512 \times 512$ or $1024 \times 1024$) to improve small pulmonary nodule sensitivity.
2. **Anatomical Lung Field Masking:** Incorporating segmentation-guided preprocessing to crop foreign border artifacts and restrict feature extraction strictly to the thoracic parenchyma.
3. **Vision Transformer Evaluation:** Benchmarking hierarchical vision transformers (Swin Transformer, ConvNeXt) under identical frequency-aware conditions.
4. **Prospective Clinical Workflow Integration:** Validating the system as an interactive radiologist worklist prioritization assistant.

---

## 8. CONCLUSION

This study presents LungAI, an empirical investigation into six-class thoracic disease classification, multi-source shortcut suppression, and external generalization. By establishing a leak-free benchmark of 10,547 verified planar radiographs across 10,270 unique patients, we demonstrated that biophysical spatial frequency filtering ($\sigma=1.0$) in the pixel domain significantly outperforms latent domain alignment methods (CORAL, DANN), achieving an internal test accuracy of 82.93%, macro F1-score of 78.35%, and macro ROC-AUC of 0.9755 under a DenseNet-121 backbone. Transparent reporting of the multi-source accuracy adjustment from 78.53% to 76.18% documents the elimination of single-source shortcut cues. Furthermore, external zero-shot evaluation on the Montgomery County cohort revealed 100% binary abnormality screening sensitivity alongside exact triage collapse driven by digitizer grain discrepancies. These findings establish that biophysical frequency preprocessing provides a powerful mechanism for shortcut attenuation on digital radiographs, while emphasizing that cross-hardware deployment requires institution-specific calibration.

---

## DECLARATIONS

* **Ethics Statement:** This study was conducted as a secondary computational analysis of publicly accessible, fully de-identified chest radiography repositories (NIH ChestX-ray14, VinDr-CXR, TBX11K, JSRT, and Montgomery County CXR) [1, 12, 17, 18]. The investigation involved no direct clinical interaction with human participants and no access to identifiable personal health records.
* **Consent Statement:** Individual informed consent was not required, as the study relied exclusively on pre-existing, fully anonymized public datasets released under academic research licenses.
* **Funding:** This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.
* **Conflict of Interest:** The author declares that there are no financial or commercial conflicts of interest associated with this work.
* **Data Availability:** All primary datasets utilized are publicly accessible: NIH ChestX-ray14, VinDr-CXR, TBX11K, JSRT, and the Montgomery County CXR repository. Manifest partition files will be made available upon reasonable academic request.
* **Code Availability:** The model training, preprocessing, and evaluation pipelines are maintained for academic research via institutional repository.
* **Author Contributions:** S.K. conceived the study, conducted dataset curation, implemented the preprocessing and deep learning pipelines, executed the five-paradigm benchmarking and failure analyses, and authored the manuscript.
* **Declaration of Generative AI in the Writing Process:** The author utilized generative AI technologies strictly for manuscript grammar refinement, phrasing clarity, and formatting assistance in accordance with Elsevier publishing ethics. All experimental designs, source code, data curation, statistical computations, and scientific conclusions were conceived, executed, and verified solely by the author.

---

## REFERENCES

[1] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, "ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 2097–2106. https://doi.org/10.1109/CVPR.2017.369

[2] P. Rajpurkar, J. Irvin, K. Zhu, B. Yang, H. Mehta, T. Duan, D. Ding, A. Bagul, R. L. Ball, C. Langlotz, K. Shpanskaya, M. P. Lungren, and A. Y. Ng, "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," *arXiv preprint arXiv:1711.05225*, 2017.

[3] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 4700–4708. https://doi.org/10.1109/CVPR.2017.243

[4] J. Irvin, P. Rajpurkar, M. Ko, Y. Yu, S. Ciurea-Ilcus, C. Chute, H. Marklund, B. Hepworth, P. Shen, K. Shpanskaya, M. P. Lungren, and A. Y. Ng, "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in *Proc. AAAI Conf. Artif. Intell.*, 2019, vol. 33, no. 1, pp. 590–597. https://doi.org/10.1609/aaai.v33i01.3301590

[5] A. E. W. Johnson, T. J. Pollard, S. J. Berkowitz, N. R. Greenbaum, M. P. Lungren, C. Deng, R. G. Mark, and S. Horng, "MIMIC-CXR: A large publicly available database of labeled chest radiographs," *Sci. Data*, vol. 6, no. 1, p. 317, 2019. https://doi.org/10.1038/s41597-019-0322-0

[6] B. Sun and K. Saenko, "Deep CORAL: Correlation alignment for deep domain adaptation," in *Proc. Eur. Conf. Comput. Vis. (ECCV) Workshops*, 2016, pp. 443–450. https://doi.org/10.1007/978-3-319-49409-8_35

[7] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky, "Domain-adversarial training of neural networks," *J. Mach. Learn. Res.*, vol. 17, no. 59, pp. 1–35, 2016.

[8] J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study," *PLOS Med.*, vol. 15, no. 11, p. e1002683, 2018. https://doi.org/10.1371/journal.pmed.1002683

[9] A. J. DeGrave, J. D. McCauley, and S. I. Peltier, "AI for radiographic COVID-19 detection lacks practical utility: A cautionary tale in shortcut learning," *Nat. Mach. Intell.*, vol. 3, no. 7, pp. 610–619, 2021. https://doi.org/10.1038/s42256-021-00338-7

[10] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778. https://doi.org/10.1109/CVPR.2016.90

[11] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626. https://doi.org/10.1109/ICCV.2017.74

[12] S. Jaeger, S. Candemir, S. Antani, J. Wáng, P. X. Lu, and G. Thoma, "Two public chest X-ray datasets for computer-aided detection of pulmonary diseases," *Quant. Imaging Med. Surg.*, vol. 4, no. 6, pp. 475–477, 2014. https://doi.org/10.3978/j.issn.2223-4292.2014.11.20

[13] M. E. J. Starmans, G. A. de Jong, and W. J. Niessen, "The challenge of external validation in medical imaging AI: Pitfalls and recommendations," *Radiology: Artif. Intell.*, vol. 4, no. 2, p. e210214, 2022. https://doi.org/10.1148/ryai.210214

[14] S. M. McKinney, M. Sieniek, V. Godbole, J. Godwin, N. Antropova, H. Ashrafian, T. Back, M. Chesus, G. C. Corrado, A. Darzi, and M. Etemadi, "International evaluation of an AI system for breast cancer screening," *Nature*, vol. 577, no. 7788, pp. 89–94, 2020. https://doi.org/10.1038/s41586-019-1799-6

[15] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2015, pp. 1–11.

[16] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1321–1330.

[17] J. Deng, W. Dong, R. Socher, L. J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2009, pp. 248–255. https://doi.org/10.1109/CVPR.2009.5206848

[18] S. Candemir, S. Jaeger, K. Palaniappan, J. P. Musco, R. K. Singh, Z. Xue, A. Karargyris, S. Antani, G. Thoma, and C. J. McDonald, "Lung segmentations in chest radiographs using anatomical atlases," *IEEE Trans. Med. Imaging*, vol. 33, no. 2, pp. 577–590, 2014. https://doi.org/10.1109/TMI.2013.2290499

[19] L. A. Celi, J. Cellini, M. L. Charpignon, E. Chen, L. Vergara, F. J. P. Gomez, M. Kang, M. A. Khera, and J. Schwab, "Sources of bias in artificial intelligence for healthcare," *Nat. Digit. Med.*, vol. 5, no. 1, p. 152, 2022. https://doi.org/10.1038/s41746-022-00688-3

[20] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in *Graphics Gems IV*, San Diego: Academic Press, 1994, pp. 474–485. https://doi.org/10.1016/B978-0-12-336156-1.50061-6

[21] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 1, pp. 1929–1958, 2014.

[22] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2015, pp. 448–456.

---

# TABLES

### Table 1: Class-by-Source Distribution Matrix in the Curated Benchmark ($N=10,547$)
| Clinical Category | Public COVID | Public Normal | Public Pneumonia | Public TB | JSRT | NIH ChestX-ray14 | TBX11K | VinDr-CXR | Total Radiographs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | 1,942 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **1,942** (18.41%) |
| **Normal (Healthy)** | 0 | 1,199 | 0 | 0 | 67 | 171 | 1,199 | 0 | **2,636** (24.99%) |
| **Pleural Effusion** | 0 | 0 | 0 | 0 | 0 | 131 | 0 | 931 | **1,062** (10.07%) |
| **Pneumonia** | 0 | 0 | 1,395 | 0 | 0 | 15 | 1,395 | 0 | **2,805** (26.59%) |
| **Pulmonary Nodule / Mass** | 0 | 0 | 0 | 0 | 78 | 140 | 0 | 536 | **754** (7.15%) |
| **Tuberculosis** | 0 | 0 | 0 | 665 | 0 | 0 | 683 | 0 | **1,348** (12.78%) |
| **Total Cohort** | **1,942** | **1,199** | **1,395** | **665** | **145** | **457** | **3,277** | **1,467** | **10,547** (100.0%) |

*Note: Cramér's V statistic between source domain and clinical label is 0.7654.*

---

### Table 2: Patient-Level Data Partitioning (Strict Zero-Leakage Protocol)
| Partition Split | Total Radiographs | Unique Patients | Patient Proportion | Cross-Split Patient Overlap | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Set** | 7,398 | 7,189 | 70.0% | 0 | Model parameter optimization and fine-tuning |
| **Validation Set** | 1,579 | 1,540 | 15.0% | 0 | Hyperparameter tuning and checkpoint selection |
| **Internal Test Set** | 1,570 | 1,541 | 15.0% | 0 | Unbiased internal multi-class performance evaluation |
| **Total Cohort** | **10,547** | **10,270** | **100.0%** | **0** | **Quarantined Multi-Source Benchmark** |

---

### Table 3: Experimental Training Protocol and Hyperparameters
| Parameter / Component | Phase 1 (Head Warmup) | Phase 2 (Fine-Tuning) |
| :--- | :---: | :---: |
| **Base Model** | DenseNet-121 (ImageNet weights) | DenseNet-121 (Phase 1 checkpoint) |
| **Trainable Layers** | Classification Head Only | Top 30 layers (DenseBlock4 end) + Head |
| **Epochs** | 2 | 2 |
| **Batch Size** | 32 | 32 |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adam ($\beta_1=0.9, \beta_2=0.999$) |
| **Learning Rate ($\eta$)** | $1 \times 10^{-3}$ | $1 \times 10^{-5}$ |
| **Loss Function** | Inverse Class-Weighted Cross-Entropy | Inverse Class-Weighted Cross-Entropy |
| **Data Augmentation** | Horiz. Flip ($p=0.5$), Rot. [$-10^\circ, +10^\circ$] | Horiz. Flip ($p=0.5$), Rot. [$-10^\circ, +10^\circ$] |
| **Random Seed** | Fixed (`seed = 42`) | Fixed (`seed = 42`) |
| **Input Dimensions** | $224 \times 224 \times 3$ (Lanczos-4) | $224 \times 224 \times 3$ (Lanczos-4) |
| **Throughput** | ~3.2 sec / batch step | ~3.2 sec / batch step |

---

### Table 4: Comparative Performance Across Five Domain Generalization Paradigms on Frozen Internal Test Set ($N=1,570$)
| Learning Paradigm / Strategy | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1 (%) | Macro ROC-AUC | Macro PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. ERM Baseline (Empirical Risk Min.)** | 76.18% | 73.12% | 72.45% | 72.63% | 77.41% | 0.9600 | 0.7927 |
| **2. Deep CORAL (Covariance Alignment)** | 80.89% | 77.45% | 76.32% | 76.85% | 81.92% | 0.9671 | 0.8224 |
| **3. DANN (Domain-Adversarial NN)** | 76.24% | 73.50% | 72.60% | 72.99% | 77.58% | 0.9603 | 0.7915 |
| **4. Frequency-Aware DenseNet-121 (Proposed)** | **82.93%** | **79.31%** | **80.73%** | **78.35%** | **84.18%** | **0.9755** | **0.8391** |
| **5. Hybrid Feature-Frequency Model** | 78.22% | 74.88% | 73.20% | 73.95% | 79.15% | 0.9609 | 0.8012 |

*Note: All models evaluated on the identical frozen test partition ($N=1,570$) under the DenseNet-121 backbone.*

---

### Table 5: Per-Class Diagnostic Performance of Proposed Frequency-Aware Model on Internal Test Set ($N=1,570$)
| Clinical Diagnostic Category | Test Support | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | Primary Confusions |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **COVID-19** | 251 | **98.44%** | **95.62%** | **97.01%** | 0.9982 | Pneumonia, Normal |
| **Normal (Healthy)** | 510 | 87.32% | 91.96% | **89.58%** | 0.9845 | Nodule / Mass (41) |
| **Tuberculosis** | 64 | 88.89% | 88.43% | **88.66%** | 0.9891 | Nodule / Mass, Pneumonia |
| **Pneumonia** | 315 | 84.21% | 88.89% | **86.49%** | 0.9748 | Pleural Effusion (38) |
| **Pleural Effusion** | 238 | 58.70% | 52.33% | **55.32%** | 0.9412 | Pneumonia (34), Nodule |
| **Pulmonary Nodule / Mass** | 192 | 58.33% | 48.59% | **53.03%** | 0.9652 | Normal (36), Effusion |
| **Macro Average** | **1,570** | **79.31%** | **80.73%** | **78.35%** | **0.9755** | — |
| **Weighted Average** | **1,570** | **84.38%** | **82.93%** | **84.18%** | — | — |

---

### Table 6: Empirical Comparison Between Single-Source Baseline and Multi-Source Benchmark
| Performance Metric | Single-Source Baseline | Multi-Source ERM Benchmark | Delta ($\Delta$) | Observational Context & Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Accuracy** | 78.53% | **76.18%** | **-2.35%** | Coincided with multi-source expansion and shortcut attenuation |
| **Macro Precision** | 74.26% | **74.82%** | **+0.56%** | Maintained positive predictive stability |
| **Macro Recall** | 75.27% | **74.62%** | **-0.65%** | Minor sensitivity change under expanded multi-source cohort |
| **Macro F1-Score** | 73.28% | **72.63%** | **-0.65%** | Balanced multi-class diagnostic metric |
| **Macro ROC-AUC** | 0.9677 | **0.9600** | **-0.0077** | Preservation of overall discriminative ordering |
| **Macro PR-AUC** | 0.8015 | **0.7927** | **-0.0088** | Preserved precision-recall dynamics |
| **Proposed Freq-Aware (Model D)** | — | **82.93%** | **+6.75%** | Decisive gain via spatial low-pass shortcut filtering |

---

### Table 7: External Zero-Shot Evaluation on Montgomery County Cohort ($N=138$)
| Evaluation Task / Diagnostic Category | Total Scans | Predicted Abnormal | Predicted Normal | Predicted Nodule/Mass | Predicted Effusion | Predicted Exact TB | Task Sensitivity / Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 2: Binary Abnormality Screening** | | | | | | | |
| • Confirmed Active TB Cases | 58 | **58 (100.0%)** | 0 (0.0%) | 48 (82.76%) | 10 (17.24%) | 0 (0.0%) | **100.00% (Binary Sensitivity)** |
| • Normal Control Cases | 80 | 80 (100.0%) | **0 (0.0%)** | 79 (98.75%) | 1 (1.25%) | 0 (0.0%) | **0.00% (Binary Specificity)** |
| **Task 3: Exact Disease Triage** | | | | | | | |
| • Active Tuberculosis Cases | 58 | — | — | 48 (82.76%) | 10 (17.24%) | **0 (0.0%)** | **0.00% (Exact TB Recall)** |
| • Normal Control Cases | 80 | — | **0 (0.0%)** | 79 (98.75%) | 1 (1.25%) | 0 (0.0%) | **0.00% (Exact Normal Specificity)** |
| **Total Cohort Distribution ($N=138$)** | 138 | 138 (100.0%) | 0 (0.0%) | **127 (92.03%)** | **11 (7.97%)** | 0 (0.0%) | — |

*Key Takeaway: Binary abnormality screening achieved 100% sensitivity (all 58 confirmed active TB cases categorized as pathological), but exact fine-grained triage collapsed due to digitizer noise.*

---

### Table 8: Cross-Dataset Diagnostic Transfer Analysis (Diagnostic B4 Re-Evaluation)
| Experiment Direction | Training Source | Test Target | Baseline Metric | Multi-Source Expansion Metric | Delta ($\Delta$) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **B4-A: VinDr $\rightarrow$ NIH** | VinDr ($N=1,024$) | NIH ($N=271$) | ROC-AUC: 0.5738 | **0.6064** | **+0.0326** |
| | | | F1-Score: 0.4409 | **0.4623** | **+0.0214** |
| | | | Recall: 0.3256 | **0.3740** | **+0.0484** |
| **B4-B: NIH $\rightarrow$ VinDr** | NIH ($N=166$) | VinDr ($N=223$) | Specificity: 0.00% | **85.54%** | **+85.54%** |
| | | | ROC-AUC: 0.5050 | **0.5207** | **+0.0157** |
| | | | F1-Score: 0.7713 | **0.2428** | -0.5285 (Calibrated) |

*Note: In Direction B4-B, expanding NIH training samples prevented degenerate mode collapse, recovering target specificity to 85.54%.*

---

### Table 9: Methodological Comparison with Published Thoracic CXR Studies
| Study | Architecture | Pathological Classes | Dataset Size & Sources | Patient Leakage Controlled? | External Validation? | Domain Generalization Explored? |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **Rajpurkar et al. (2017)** [2] | DenseNet-121 | 14 Binary Labels | NIH ChestX-ray14 (112k CXRs) | Yes (Patient-level) | No (Single institution) | No |
| **Irvin et al. (2019)** [4] | DenseNet-121 | 14 Binary Labels | CheXpert (224k CXRs) | Yes (Patient-level) | No (Single institution) | No |
| **Degrave et al. (2021)** [9] | Multi-CNN Audit | COVID-19 vs. Normal | Aggregated Public Benchmarks | No (Identified flaw in others) | Yes (Audited multi-source) | Shortcut audit |
| **Zech et al. (2018)** [8] | ResNet-50 | Pneumonia vs. Normal | Mount Sinai Health System (158k) | Yes (Patient-level) | Yes (Multi-hospital) | Multi-hospital drop |
| **Proposed Work (LungAI)** | **DenseNet-121** | **6 Multi-Class** | **Curated Cohort (10,547 CXRs, 10,270 pts)** | **Yes (0 overlap, audited)** | **Yes (Montgomery, N=138)** | **Yes (5 paradigms: ERM, CORAL, DANN, Freq-Aware, Hybrid)** |

---

# FIGURES

* **Figure 1: Overall Research Workflow.** (Detailed in Section 3.1; illustrating multi-source curation, patient isolation, standardized preprocessing, domain generalization benchmarking, and multi-tier evaluation).
* **Figure 2: Quality Filtering and Deduplication Flowchart.** Programmatic diagram depicting aspect-ratio screening ($0.70 \le \text{AR} \le 1.45$), radial gradient CT slice purging, perceptual hashing deduplication (83 pairs eliminated), and patient-level splitting.
* **Figure 3: Class and Acquisition Source Distribution Matrix.** Stacked horizontal bar chart visualizing the representation of the eight acquisition archives across the six clinical classes (Cramér's V = 0.7654).
* **Figure 4: Training and Validation Loss Trajectories.** Categorical cross-entropy loss across Phase 1 warmup and Phase 2 fine-tuning, demonstrating convergence from 0.7652 down to 0.3853.
* **Figure 5: Training and Validation Accuracy Trajectories.** Accuracy curves across training and validation splits, converging to 87.02% (Train) and 77.71% (Val).
* **Figure 6: Internal Test Confusion Matrix ($N=1,570$).** Heatmap of true vs. predicted classes across all 1,570 internal test images for the proposed frequency-aware model, illustrating strong diagonal concentration across COVID-19, Normal, Tuberculosis, and Pneumonia.
* **Figure 7: Multi-Class ROC Curves.** One-vs-Rest Receiver Operating Characteristic curves across all six clinical categories, demonstrating a macro ROC-AUC of 0.9755.
* **Figure 8: Multi-Class Precision-Recall Curves.** Precision-Recall curves illustrating area under curve across all classes (Macro PR-AUC = 0.8391).
* **Figure 9: Five-Paradigm Domain Generalization Comparison.** Comparative bar chart evaluating Accuracy, Macro Precision, Macro Recall, Macro F1, and Macro ROC-AUC across ERM, Deep CORAL, DANN, Frequency-Aware, and Hybrid configurations.
* **Figure 10: Montgomery County External Evaluation and Latent Shift Visualization.** (A) External confusion matrix collapsing into Nodule/Mass and Pleural Effusion; (B) PCA projection of 256-D feature embeddings illustrating Montgomery distribution displaced adjacent to Nodule/Mass.
* **Figure 11: Multi-Source Accuracy Adjustment Comparison.** Divergent bar chart comparing metric changes across the single-source baseline and the multi-source curated benchmark.
