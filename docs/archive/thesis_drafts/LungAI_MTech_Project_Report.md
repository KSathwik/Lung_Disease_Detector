# LUNGAI: ROBUST MULTI-CLASS LUNG DISEASE DETECTION FROM CHEST X-RAY IMAGES USING DEEP LEARNING AND CROSS-SOURCE GENERALIZATION

**A Project Dissertation Submitted in Partial Fulfillment of the Requirements for the Award of the Degree of MASTER OF TECHNOLOGY in COMPUTER SCIENCE AND ENGINEERING**

**Candidate:** SATHWIK KATKAM (Roll No: 22011D0501)  
**Supervisor:** Internal Guide, Assistant Professor  
**Department:** DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING, JAWAHARLAL NEHRU TECHNOLOGICAL UNIVERSITY HYDERABAD  
**Academic Year:** 2025–2026

---

## CERTIFICATE

This is to certify that the project dissertation entitled "LUNGAI: ROBUST MULTI-CLASS LUNG DISEASE DETECTION FROM CHEST X-RAY IMAGES USING DEEP LEARNING AND CROSS-SOURCE GENERALIZATION", submitted by SATHWIK KATKAM (Roll No: 22011D0501) in partial fulfillment of the requirements for the award of the degree of MASTER OF TECHNOLOGY in COMPUTER SCIENCE AND ENGINEERING from JAWAHARLAL NEHRU TECHNOLOGICAL UNIVERSITY HYDERABAD, is an authentic record of original research and engineering work carried out under academic supervision and guidance during the academic year 2025–2026.

The results and conclusions embodied in this dissertation have reached a verified, reproducible, and frozen state and have not been submitted to any other university or institute for the award of any degree, diploma, or fellowship. All secondary datasets, clinical benchmarks, and open-source scientific software libraries have been explicitly acknowledged.

**Internal Guide**  
Assistant Professor, Department of Computer Science and Engineering  
Jawaharlal Nehru Technological University Hyderabad

---

## DECLARATION

I hereby declare that the project dissertation entitled "LUNGAI: ROBUST MULTI-CLASS LUNG DISEASE DETECTION FROM CHEST X-RAY IMAGES USING DEEP LEARNING AND CROSS-SOURCE GENERALIZATION" is an authentic record of my own research and engineering work carried out in the DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING, JAWAHARLAL NEHRU TECHNOLOGICAL UNIVERSITY HYDERABAD, under the supervision of Internal Guide, Assistant Professor.

I further declare that this dissertation has not been submitted previously, in part or in full, to any other institution or university for the award of any academic degree or diploma. I have adhered to all academic ethics and anti-plagiarism standards; all empirical findings, architectural formulations, and source-held-out evaluations reported herein represent genuine scientific investigations.

**SATHWIK KATKAM**  
Roll No: 22011D0501  
M.Tech (Computer Science and Engineering)  
Jawaharlal Nehru Technological University Hyderabad

---

## ACKNOWLEDGEMENT

I express my profound gratitude and sincere appreciation to my project supervisor, Internal Guide, Assistant Professor, for invaluable intellectual guidance, continuous encouragement, and rigorous scientific feedback throughout the conception, mathematical formulation, forensic dataset auditing, and software realization of this M.Tech dissertation.

I extend my heartfelt thanks to the Head of the Department, Professor & Head, and the esteemed faculty members of the Department of Computer Science and Engineering for providing the high-performance computing infrastructure, GPU cluster access, and stimulating research environment indispensable for executing multi-source deep learning experiments.

I gratefully acknowledge the curators and medical contributors of the open-access medical imaging databases—specifically the NIH Clinical Center (ChestX-ray14), VinBigData (VinDr-CXR), the Chinese University of Hong Kong (TBX11K), the Japanese Society of Radiological Technology (JSRT), the U.S. National Library of Medicine (Montgomery County benchmark), and the research groups of Chowdhury et al., Rahman et al., and Kermany et al.—whose public datasets enabled this empirical investigation.

Finally, I express my deepest gratitude to my parents, family, and colleagues for their constant encouragement, patience, and unwavering moral support throughout my academic tenure.

---

## EXECUTIVE ABSTRACT

Chest radiography (CXR) represents the universal first-line diagnostic imaging modality worldwide due to its low ionization dose, cost efficiency, and rapid acquisition throughput. However, standard planar radiographs collapse complex volumetric thoracic anatomy into two-dimensional planar projections, creating extensive anatomical superimposition. Compounding these physical imaging challenges, conventional clinical Picture Archiving and Communication Systems (PACS) manage incoming examinations through unprioritized First-In, First-Out (FIFO) worklists, precipitating critical diagnostic turnaround backlogs (24 to 72 hours) in resource-constrained facilities while overburdened radiologists exhibit inter-observer disagreement rates between 15% and 30%. While deep convolutional neural networks frequently report high classification accuracy on curated single-source benchmarks, they frequently overfit to acquisition-specific high-frequency shortcuts, suffering catastrophic generalization failure when deployed across independent clinical centers.

This M.Tech dissertation presents LungAI, an auditable, end-to-end medical AI decision-support platform addressing multi-class thoracic disease classification and cross-source domain generalization. The research formulates the central research question: Can a multi-source six-class chest X-ray classifier learn disease-relevant representations that generalize across independent acquisition sources, and can a domain-generalization strategy improve cross-source performance compared with a standard DenseNet-121 baseline? To address this, the investigation executes a structured forensic reconstruction of the multi-source corpus, yielding the canonical leak-free V5 dataset comprising 10,547 verified planar radiographs from 10,270 unique patients across eight international repositories, partitioned at the strict patient level (0% patient overlap) into 7,398 training (7,189 patients), 1,579 validation (1,540 patients), and 1,570 internal test scans (1,541 patients) across six diagnostic categories: COVID-19 (1,942), Normal (2,636), Pleural Effusion (1,062), Pneumonia (2,805), Pulmonary Nodule / Mass (754), and Tuberculosis (1,348). The sixth class represents radiographic nodule and mass findings rather than histologically confirmed lung malignancy. Severe source-label confounding is quantified via Cramér's V = 0.7654.

Under a controlled experimental protocol, five distinct learning paradigms are evaluated: (1) Empirical Risk Minimization (ERM) baseline (76.18% accuracy, 72.63% macro F1); (2) Deep CORAL latent covariance alignment (80.89% accuracy, 76.85% macro F1); (3) Domain-Adversarial Neural Networks (DANN) gradient reversal (76.24% accuracy, 72.99% macro F1); (4) Frequency-aware Gaussian low-pass spatial filtering at sigma=1.0 (82.93% accuracy, 78.35% macro F1, 0.9755 macro ROC-AUC, 0.8391 macro PR-AUC); and (5) Hybrid CORAL with frequency preprocessing (78.22% accuracy, 73.95% macro F1). Controlled empirical evidence designates Model D (DenseNet-121 with frequency-aware spatial filtering) as the production inference engine. On the internal held-out test split, Model D achieves per-class F1-scores of 97.01% for COVID-19, 89.58% for Normal, 88.66% for Tuberculosis, 86.49% for Pneumonia, 55.32% for Pleural Effusion, and 53.03% for Pulmonary Nodule / Mass. External generalization evaluation on the untouched Montgomery County benchmark (N=138, 58 TB, 80 Normal) demonstrates 100% binary abnormal sensitivity but complete fine-grained classification collapse (0% TB exact recall, 0% Normal specificity), with cases collapsing to Pulmonary Nodule / Mass. Biophysical forensic auditing links this collapse to substantial source-dependent distribution differences, including approximately 4x higher Laplacian high-frequency edge variance in digitized analog film compared to modern digital detectors (1,580 vs 373), proving that causal attribution is multi-factorial.

To protect downstream inference from non-radiographic contamination, LungAI deploys a two-stage defense-in-depth CXR input-validation gate combining Stage 1 biophysical/chromatic screening with Stage 2 DenseNet-121 anatomical verification operating at production threshold tau=0.83 (correcting an adversarial leak where cat images passed at tau=0.70). On a 125-image evaluation cohort, the gate achieved 100% sensitivity and 100% specificity (ROC-AUC 1.0000), which is disclosed as a functional verification estimate due to threshold-selection leakage. The complete inference pipeline—integrating input validation, Model D inference, Grad-CAM visual explainability, and deterministic clinical triage tiers (Emergency, Urgent, Routine)—is realized as an asynchronous FastAPI microservice backed by SQLAlchemy ORM persistence and a responsive React 18 single-page application. System reliability is verified via an automated 44-test verification suite achieving 100% pass rate. This dissertation establishes that rigorous multi-source dataset reconstruction, frequency-aware preprocessing, transparent external failure disclosure, and defensive input validation provide a reproducible engineering reference for clinical-grade decision support.

**Keywords:** *Chest Radiography, Deep Learning, DenseNet-121, Multi-Class Classification, Domain Generalization, Dataset Forensics, Frequency-Aware Preprocessing, Explainable AI, Grad-CAM, Chest X-Ray, Medical Decision Support.*

---

## TABLE OF CONTENTS

| Chapter | Title | Page |
| :---: | :--- | :---: |
| | **Certificate** | 2 |
| | **Declaration** | 3 |
| | **Acknowledgement** | 4 |
| | **Executive Abstract** | 5 |
| | **Table of Contents** | 7 |
| | **List of Figures** | 8 |
| | **List of Tables** | 10 |
| **Chapter 1** | **INTRODUCTION** | **—** |
| **Chapter 2** | **LITERATURE SURVEY** | **16** |
| **Chapter 3** | **EXISTING SYSTEM** | **33** |
| **Chapter 4** | **PROPOSED SYSTEM** | **37** |
| **Chapter 5** | **SYSTEM REQUIREMENTS** | **43** |
| **Chapter 6** | **SYSTEM DESIGN** | **48** |
| **Chapter 7** | **DATASET AND DATA PREPROCESSING** | **60** |
| **Chapter 8** | **MACHINE LEARNING / AI MODEL** | **68** |
| **Chapter 9** | **SYSTEM IMPLEMENTATION** | **75** |
| **Chapter 10** | **USER INTERFACE** | **79** |
| **Chapter 11** | **TESTING** | **84** |
| **Chapter 12** | **MODEL EVALUATION AND RESULTS** | **91** |
| **Chapter 13** | **SECURITY, PRIVACY AND RESPONSIBLE AI** | **105** |
| **Chapter 14** | **LIMITATIONS** | **108** |
| **Chapter 15** | **FUTURE ENHANCEMENTS** | **110** |
| **Chapter 16** | **CONCLUSION** | **112** |
| | **REFERENCES** | **115** |
| | **APPENDICES** | **118** |

---

## LIST OF FIGURES

| Figure No. | Caption / Description | Page |
| :---: | :--- | :---: |
| **Figure 3.1** | Conventional PACS FIFO reading workflow and triage delays | 33 |
| **Figure 4.1** | Proposed LungAI three-tier system architecture and data flow | 38 |
| **Figure 6.1** | DFD Level 0 (Context Diagram) of LungAI clinical ecosystem | 48 |
| **Figure 6.2** | DFD Level 1 (Functional Decomposition) of inference pipeline | 49 |
| **Figure 6.3** | DFD Level 2 (Inference Subsystem Decomposition) tensor flow | 50 |
| **Figure 6.4** | UML Use Case Diagram mapping clinical and administrative roles | 51 |
| **Figure 6.5** | UML Class Diagram delineating services and ORM data entities | 52 |
| **Figure 6.6** | UML Component Diagram illustrating decoupled microservice architecture | 52 |
| **Figure 6.7** | UML Deployment Diagram across client, backend, and database tiers | 53 |
| **Figure 6.8** | UML Sequence Diagram of asynchronous image evaluation cycle | 54 |
| **Figure 6.9** | UML Activity Diagram for gate validation and fallback pathways | 55 |
| **Figure 6.10** | Relational Entity-Relationship (ER) Diagram of persistence schema | 58 |
| **Figure 6.11** | Machine Learning Inference Pipeline and biophysical transformations | 58 |
| **Figure 7.1** | Deterministic Radiographic Preprocessing Pipeline visual progression | 67 |
| **Figure 10.1** | LungAI Home Portal displaying system status and operational metrics | 79 |
| **Figure 10.2** | Interactive Diagnostic Workspace prior to scan ingestion | 80 |
| **Figure 10.3** | Completed Diagnostic Analysis interface displaying 6-class confidence | 81 |
| **Figure 10.4** | Grad-CAM Visual Saliency Inspection over apical fibro-cavitary lesion | 81 |
| **Figure 10.5** | Scientific Metrics Dashboard with confusion matrices and ROC curves | 82 |
| **Figure 10.6** | Historical Scan Log and Audit Trail with diagnostic record history | 83 |
| **Figure 10.7** | Patient Management Portal displaying registered clinical profiles | 83 |
| **Figure 12.1** | Model D Confusion Matrix across 1,570-image internal test split | 96 |
| **Figure 12.2** | Multi-Class Receiver Operating Characteristic (ROC) Curves for Model D | 97 |
| **Figure 12.3** | Precision-Recall (PR) Curves for Model D across diagnostic classes | 98 |
| **Figure 12.4** | Confusion Matrix on Quarantined Montgomery Benchmark (N=138) | 100 |
| **Figure 12.5** | Biophysical Distribution Analysis (Analog Film vs. Digital Detectors) | 101 |
| **Figure 12.6** | Latent Feature Space Projection under Extreme Domain Shift | 102 |
| **Figure 12.7** | Saliency Failure Panel for Montgomery Normal Scan 0001 | 102 |
| **Figure 12.8** | t-SNE Latent Feature Projection of Model D on Internal Test Split | 103 |
| **Figure 12.9** | t-SNE Latent Feature Projection of Deep CORAL Domain Adaptation | 103 |
| **Figure 12.10** | Qualitative Grad-CAM Saliency Panel on Internal Digital Radiograph | 103 |

---

## LIST OF TABLES

| Table No. | Title / Description | Page |
| :---: | :--- | :---: |
| **Table 2.1** | Comparative Analysis of Existing Chest X-Ray Classification Studies and LungAI | 24 |
| **Table 2.2** | Research Differentiation of LungAI from Representative Existing Approaches | 28 |
| **Table 2.3** | Impact of Existing Approaches on the Design and Evaluation of LungAI | 30 |
| **Table 3.1** | Comparative Limitations Matrix of Existing Systems | 35 |
| **Table 4.1** | Comprehensive Comparison of Initial Baseline vs. Proposed LungAI System | 41 |
| **Table 5.1** | Cross-Functional System Requirements Matrix | 46 |
| **Table 6.1** | Relational Database Schema Specification | 55 |
| **Table 7.1** | Dataset Reconstruction and Forensic Evolution History (V1 to V5) | 60 |
| **Table 7.2** | Canonical Unified V5 Dataset Partition Distribution | 61 |
| **Table 7.3** | Diagnostic Class Distribution of the Canonical V5 Cohort | 63 |
| **Table 7.4** | Source Repository Contribution across Diagnostic Classes | 64 |
| **Table 7.5** | High-Frequency Energy Retention under Gaussian Filtering Grid | 65 |
| **Table 8.1** | Structural Hyperparameters for Five Evaluated Architectures | 69 |
| **Table 8.2** | Controlled Five-Paradigm Hyperparameter Configurations | 73 |
| **Table 11.1** | Automated Software Verification Test Suite Execution Matrix (44 Tests) | 84 |
| **Table 11.2** | Two-Stage CXR Validation Gate Performance Metrics | 88 |
| **Table 12.1** | Master Comparative Performance Matrix of Five Learning Paradigms | 92 |
| **Table 12.2** | Comprehensive Classification Metrics of Model D (Internal Test Split) | 94 |
| **Table 12.3** | Detailed Class-Wise Discriminative Performance of Model D | 98 |
| **Table 12.4** | External Generalization Performance on Montgomery Benchmark | 99 |
| **Table 13.1** | Clinical Urgency Stratification and Triaging Protocol | 106 |
| **Table 15.1** | Prioritized Roadmap for Future System Enhancements | 111 |

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Clinical Background & Problem Domain

Thoracic diseases constitute one of the most critical public health challenges of the twenty-first century, accounting for tens of millions of hospitalizations and fatalities across the globe annually [1]. Infectious pulmonary pathologies, such as acute community-acquired and nosocomial bacterial pneumonia, coronavirus disease 2019 (COVID-19), and pulmonary tuberculosis (Mycobacterium tuberculosis), present immediate, life-threatening risks of respiratory failure and acute respiratory distress syndrome (ARDS). Concurrently, chronic and focal thoracic conditions, including pleural effusion and solitary or multifocal pulmonary nodules and masses, represent frequent radiographic manifestations of cardiovascular compromise, severe systemic infection, or neoplastic thoracic malignancy. In acute respiratory decompensation, every hour of therapeutic delay substantially elevates mortality, whereas in chronic or progressive thoracic illnesses, timely detection of radiographic manifestations is the primary determinant of successful clinical management and long-term survival.

Chest radiography (CXR) serves as the primary, most accessible, and most universally utilized diagnostic imaging modality in clinical medicine. Its minimal ionizing radiation footprint (approximately 0.1 mSv per posteroanterior view, compared to 7.0–10.0 mSv for volumetric thoracic computed tomography [CT]), rapid image acquisition throughput, and low capital cost make it the standard diagnostic examination in emergency triage wards, intensive care units, outpatient clinics, and rural or resource-constrained healthcare environments worldwide. 

However, despite its ubiquitous clinical role, planar chest radiography poses immense perceptual and diagnostic challenges. Because planar radiography flattens a complex, dynamic three-dimensional anatomical volume into a single two-dimensional projection, overlying anatomical structures—such as anterior and posterior ribs, clavicles, mediastinal contours, cardiac borders, and pulmonary vascular branchings—superimpose upon one another. Consequently, subtle, low-contrast parenchymal opacities, ground-glass infiltrates, small apical cavities, and faint solitary nodules are frequently obscured by normal anatomical clutter. Discriminating early pathological patterns from benign anatomical variations requires years of specialized radiological training and sustained cognitive vigilance.

## 1.2 Radiographic Imaging Principles & Diagnostic Role

Planar chest radiograph formation is fundamentally governed by the physical principles of differential X-ray photon attenuation across human tissues of divergent atomic composition, physical thickness, and tissue density. In standard thoracic imaging, tissues are categorized into four canonical radiodensities:
1. **Air / Gas**: Possesses negligible physical density, allowing the vast majority of incident X-ray photons to pass through unattenuated to the digital receptor, producing low optical density (radiolucent, appearing black or dark charcoal). In a healthy subject, ventilated lung fields appear dark.
2. **Fat / Adipose Tissue**: Possesses intermediate low density, attenuating slightly more photons than air and appearing dark grey (such as subcutaneous fat layers).
3. **Soft Tissue / Fluid**: Comprising the heart, great vessels, diaphragm, blood, purulent exudate, and parenchymal tissue, attenuates photons moderately, manifesting as mid-tone grey or off-white opacities.
4. **Bone / Calcification**: Characterized by high calcium content and atomic number, strongly attenuates X-ray photons via photoelectric absorption, resulting in high optical radiopacity (bright white).

Pathological processes within the thoracic cage alter this normal distribution of photon attenuation:
- **Pneumonia**: Microbial infection incites acute alveolar inflammation, filling alveolar airspaces with purulent cellular exudate, fibrin, and erythrocytes. This alveolar consolidation replaces radiolucent air with soft-tissue density, generating patchy or confluent opacifications and classic air bronchograms.
- **COVID-19**: Viral alveolar injury and interstitial inflammation produce distinctive peripheral, subpleural, and bilateral ground-glass opacities (GGOs) and consolidative patches, frequently concentrated in lower lung zones.
- **Tuberculosis**: Mycobacterial infection triggers granulomatous immune reactions that manifest as apical fibro-cavitary lesions, patchy parenchymal infiltrations, Ghon complexes, or widespread miliary micronodules.
- **Pleural Effusion**: Pathological fluid collection within the pleural cavity blunts the normally sharp costophrenic and cardiophrenic sulci, producing homogeneous crescent-shaped opacities with distinctive meniscus contours.
- **Pulmonary Nodule / Mass**: Focal parenchymal tissue proliferation, inflammatory granulomas, or primary/metastatic neoplastic lesions manifest as discrete, rounded, or lobulated radiopaque opacities within the radiolucent lung fields.

Accurately recognizing and differentiating these overlapping radiographic opacities is an intellectually demanding cognitive process, heavily reliant on the clinician's pattern-recognition experience.

## 1.3 Workflow Challenges in Conventional Radiology

Contemporary healthcare systems worldwide face unprecedented systemic bottlenecks in radiological workflows, driven by three interrelated structural crises:

First, there exists an acute, widening global shortage and geographical maldistribution of certified diagnostic radiologists. In developing and lower-middle-income nations, the ratio of certified radiologists to population often drops below 1 per 100,000 citizens. Even across tertiary medical centers in high-income countries, annual imaging examination volumes have far outpaced the growth of the radiological workforce. Consequently, emergency department physicians, medical officers, and intensive care clinicians are routinely forced to make urgent diagnostic decisions without specialist radiological interpretation during nights, weekends, and high-volume clinical shifts.

Second, standard hospital Picture Archiving and Communication Systems (PACS) organize incoming imaging studies using unprioritized, First-In, First-Out (FIFO) worklists. Under this operational architecture, an acute emergency radiograph exhibiting massive bilateral consolidation, tension pneumothorax, or extensive infectious infiltrate sits in the reading queue in the exact sequence it was transmitted by the modality, grouped indiscriminately alongside routine pre-employment physical examinations and outpatient follow-ups. This structural absence of automated triage causes report turnaround delays extending from 24 to over 72 hours in high-volume public hospitals, deferring life-saving antimicrobial or supportive clinical interventions.

Third, manual human image interpretation is inherently susceptible to diagnostic errors, cognitive fatigue, and perceptual blind spots. Extended shifts, visual fatigue, perceptual distraction, and variations in subspecialty expertise produce documented inter-observer and intra-observer diagnostic disagreement rates between 15% and 30% in thoracic radiograph interpretation. These realities motivate the development of objective, automated, computer-aided detection and clinical triage algorithms.

## 1.4 Problem Statement & Research Question

The overarching research problem addressed in this dissertation is formulated as follows:

Existing computer-aided detection (CAD) systems and published deep learning classifiers frequently report impressive classification accuracy (>95%) on curated, single-source, or artificially blended benchmark datasets. However, when these models are subjected to rigorous forensic dataset audits, source-held-out evaluations, or independent external hospital validation, performance frequently degrades severely. This degradation stems from several critical factors:
1. **Shortcut Learning & Confounding**: Models overfit to dataset-specific digital watermarks, scanner high-frequency noise textures, patient demographics, or hospital-specific projection biases rather than true pathological manifestations.
2. **Dataset Contamination & Cross-Modality Contamination**: Prior public multi-class collections have inadvertently merged axial CT slices with planar CXRs, or allowed severe patient identity overlap across training and test splits.
3. **Severe Cross-Domain Sensor Shift**: Variations in radiographic acquisition hardware (e.g., digitized analog film scanners vs. modern direct digital radiography detectors) create insurmountable sensor domain shifts that disrupt standard deep convolutional feature extractors.
4. **Vulnerability to Out-of-Distribution Inputs**: Standard deep neural networks will confidently output high-probability thoracic disease predictions even when supplied with non-medical images, corrupt files, or non-radiographic scans.

To address these fundamental challenges, this dissertation investigates the primary research question:

> **"Can a multi-source six-class chest X-ray classifier learn disease-relevant representations that generalize across independent acquisition sources, and can a domain-generalization strategy improve cross-source performance compared with a standard DenseNet-121 baseline?"**

## 1.5 Research Objectives & Project Scope

To answer this research question, the dissertation establishes the following specific research and engineering objectives:

1. **Forensic Dataset Reconstruction & Harmonization**: Perform a comprehensive forensic audit of public chest radiography repositories to eliminate cross-modality contamination, duplicate scans, and patient overlap leakage. Construct a canonical, leak-free, multi-source six-class benchmark (the V5 dataset) partitioned strictly at the patient level across eight independent acquisition sources.
2. **Deterministic Biophysical Preprocessing Pipeline**: Formulate a reproducible preprocessing workflow combining chromatic-luminance decoupling (CIE LAB space), Contrast Limited Adaptive Histogram Equalization (CLAHE) on the luminance channel, high-order Lanczos-4 spatial resampling, and channel-wise standardization.
3. **Controlled Multi-Paradigm Domain Generalization Benchmarking**: Under an identical backbone (DenseNet-121) and identical experimental protocol, systematically train and benchmark five distinct learning paradigms:
   - Empirical Risk Minimization (ERM) baseline;
   - Deep Correlation Alignment (Deep CORAL) latent covariance alignment;
   - Domain-Adversarial Neural Networks (DANN) feature-level minimax alignment;
   - Frequency-aware Gaussian spatial low-pass filtering (Model D);
   - Hybrid frequency-domain filtering with latent covariance alignment.
4. **Transparent External Generalization Failure Audit**: Evaluate the trained architectures zero-shot on an untouched, external acquisition benchmark (the Montgomery County cohort) and execute biophysical forensic analyses to isolate the root causes of external domain transfer failure.
5. **Two-Stage Defense-in-Depth CXR Input-Validation Gate**: Design and implement a robust input screening pipeline integrating biophysical/chromatic filtering (Stage 1) with an anatomical DenseNet-121 verification classifier (Stage 2) operating at an empirically audited threshold (tau=0.83) to reject non-radiographic inputs.
6. **Explainable AI Integration**: Integrate Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the bottleneck dense feature representations to provide interpretable visual attention heatmaps confirming anatomical lesion localization.
7. **End-to-End Decision Support Microservice**: Implement and verify an auditable, asynchronous three-tier software architecture comprising a FastAPI ASGI backend, SQLAlchemy ORM database persistence, and an interactive React 18 single-page application equipped with automated clinical urgency triage stratification.

**Project Scope & Boundaries**: The machine learning classifier evaluates planar chest radiographs across exactly six active categories: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, and Tuberculosis. The sixth category represents radiographic nodule and mass findings rather than histologically confirmed lung cancer. The software system is strictly positioned as an academic research and decision-support prototype; it does not possess clinical regulatory clearance and is not designed to operate autonomously without human-in-the-loop expert radiological oversight.

## 1.6 Research Contributions of the Dissertation

The specific contributions of this M.Tech dissertation are not merely adopting an off-the-shelf convolutional backbone, but establishing a rigorous, reproducible, and transparent research methodology spanning data forensics, model selection, failure analysis, and system engineering:

1. **Forensic Dataset Reconstruction (V5 Cohort)**: Reconstructed and released the unified V5 dataset manifest (10,547 images, 10,270 unique patients) spanning eight international repositories, enforcing zero patient overlap across training (7,398 scans / 7,189 patients), validation (1,579 scans / 1,540 patients), and internal test (1,570 scans / 1,541 patients) splits.
2. **Empirical Quantification of Source-Label Confounding**: Systematically audited and documented multi-source dataset bias, demonstrating a Cramér's V statistic of 0.7654 across source and disease distributions, highlighting the danger of shortcut learning in multi-dataset benchmarks.
3. **Controlled Domain-Generalization Benchmarking**: Executed a controlled comparison of five distinct learning paradigms (ERM, Deep CORAL, DANN, Frequency-Aware, Hybrid) using an identical DenseNet-121 backbone, proving that frequency-aware spatial filtering achieves the strongest internal multi-source performance (+5.72% macro F1 delta over ERM).
4. **Evidence-Driven Model D Selection**: Established that the frequency-aware DenseNet-121 (Model D) achieves 82.93% internal test accuracy, 79.31% macro precision, 80.73% macro recall, 78.35% macro F1-score, 0.9755 macro ROC-AUC, and 0.8391 macro PR-AUC across the six classes, outperforming more complex adversarial and hybrid configurations.
5. **Rigorous Analysis of Negative Experimental Results**: Documented that adversarial domain adaptation (DANN) and hybrid methods failed to outperform simpler frequency filtering, providing scientifically valuable negative evidence that domain adaptation cannot be assumed to yield universal gains without domain compatibility.
6. **Transparent External Failure Analysis (The Montgomery Audit)**: Disclosed and analyzed the complete fine-grained classification collapse of all tested models on the external Montgomery benchmark (100% binary abnormal sensitivity, but 0% exact TB recall), demonstrating through edge-variance analysis (~4x higher Laplacian variance: 1,580 vs 373) that digitized analog film characteristics disrupt deep feature spaces.
7. **Two-Stage CXR Input-Validation Gate**: Formulated and deployed a defense-in-depth gate combining biophysical chromatic/spatial screening with a semantic DenseNet-121 classifier at tau=0.83, eliminating adversarial leaks (e.g., cat images passing at tau=0.70).
8. **Transparent Threshold-Selection Leakage Disclosure**: Explicitly disclosed that the CXR gate's 100% sensitivity, 100% specificity, and 1.0000 ROC-AUC on its 125-image evaluation cohort represent functional verification estimates rather than unbiased generalization metrics due to evaluation-set threshold selection.
9. **Visual Interpretability via Grad-CAM**: Integrated gradient-weighted class activation mapping directly onto the final dense convolutional concat layer (`conv5_block16_concat`), generating anatomically verifiable heatmaps for clinical explainability.
10. **Full-Stack Auditable Software Platform**: Delivered an end-to-end clinical triage web application (FastAPI + SQLAlchemy + React 18) featuring deterministic urgency stratification (Emergency, Urgent, Routine), patient EMR linking, and an automated 44-test verification suite with 100% pass rate.

## 1.7 Organization of the Dissertation

The remainder of this dissertation is structured as follows:
- **Chapter 2 (Literature Survey)** provides an exhaustive critical review of computer-aided detection in chest radiography, tracing the progression from classical texture descriptors to transfer learning, multi-class architectures, explainability, domain shift, and shortcut learning. It incorporates three mandatory comparative matrices.
- **Chapter 3 (Existing System)** examines conventional radiological reading workflows, PACS FIFO queuing, cognitive fatigue, and the fundamental limitations of single-source deep learning classifiers.
- **Chapter 4 (Proposed System)** presents the complete architectural vision of LungAI, detailing the multi-tier topology, CXR gate, Model D inference pipeline, Grad-CAM module, and clinical urgency stratification engine.
- **Chapter 5 (System Requirements)** delineates hardware, software, functional (FR1–FR10), and non-functional (NFR1–NFR8) requirements.
- **Chapter 6 (System Design)** provides comprehensive architectural modeling using Data Flow Diagrams (DFDs Level 0–2), UML structural and behavioral diagrams, the relational database ER diagram, and the ML inference pipeline.
- **Chapter 7 (Dataset and Data Preprocessing)** documents the forensic reconstruction history (V1 to V5), dataset distributions, patient-strict splitting, source confounding analysis, biophysical CLAHE, Gaussian spatial filtering, and intensity standardization.
- **Chapter 8 (Machine Learning / AI Model)** presents the mathematical foundations of DenseNet-121, Model D, loss formulations (weighted cross-entropy, CORAL, DANN, Gaussian low-pass), optimization schedules, and Grad-CAM mathematics.
- **Chapter 9 (System Implementation)** describes the technical implementation of the FastAPI backend, SQLAlchemy ORM persistence, singleton inference engine, and React 18 single-page application.
- **Chapter 10 (User Interface)** showcases the responsive clinical workspaces, DICOM viewport windowing, Grad-CAM visual overlays, and administrative dashboards.
- **Chapter 11 (Testing)** presents the software quality assurance framework, documenting the 44-test automated verification suite, CXR gate forensic testing, adversarial screening, and the threshold-selection leakage disclosure.
- **Chapter 12 (Model Evaluation and Results)** delivers the empirical centerpiece of the thesis, presenting the 5-paradigm comparison, Model D class-wise performance, source-level diagnostics, the Montgomery failure investigation, and Grad-CAM evaluations.
- **Chapter 13 (Security, Privacy and Responsible AI)** details ethical positioning, medical disclaimers, human-in-the-loop governance, patient data de-identification, and security controls.
- **Chapter 14 (Limitations)** comprehensively articulates all eighteen technical, methodological, and clinical limitations of the research.
- **Chapter 15 (Future Enhancements)** outlines a prioritized research roadmap for multi-center validation, calibration, DICOM PACS integration, and multi-label modeling.
- **Chapter 16 (Conclusion)** synthesizes research findings, answers the central research question, and delivers concluding remarks.
- **References & Appendices** provide the complete verified IEEE bibliography and supplementary engineering configurations.

---

# CHAPTER 2 — LITERATURE SURVEY

## 2.1 Evolution of Computer-Aided Diagnosis

Computer-Aided Diagnosis (CAD) in thoracic radiography has evolved over five decades from early heuristic image processing algorithms into modern deep representation learning architectures [28]. The fundamental ambition of CAD systems has consistently been to serve as an objective "second reader," reducing diagnostic miss rates and mitigating cognitive fatigue for clinical practitioners. 

Early CAD systems introduced in the late 1970s and 1980s relied upon rule-based edge detection, morphological filtering, and manual thresholding to identify gross anatomical deviations, such as cardiomegaly or large solitary pulmonary nodules. In the 1990s and 2000s, CAD evolved into the Computer-Aided Detection (CADe) and Computer-Aided Diagnosis (CADx) paradigms. These systems were characterized by hand-engineered feature extraction pipelines where computer vision experts extracted mathematical descriptors representing geometry, texture, and pixel intensity profiles, which were subsequently categorized using classical machine learning classifiers.

Despite substantial research efforts, first- and second-generation CAD systems suffered from persistent clinical limitations. They exhibited high false-positive rates (frequently generating 5 to 10 false alarms per radiograph), extreme sensitivity to variations in image acquisition hardware and exposure parameters, and brittle heuristic rules that could not adapt to subtle, polymorphous parenchymal infiltrates. Consequently, these systems failed to achieve widespread clinical trust, remaining largely confined to narrow screening tasks such as automated mammographic microcalcification detection or initial nodule candidate screening.

The emergence of large-scale annotated medical datasets, coupled with advancements in high-performance GPU computing and deep convolutional neural networks (CNNs), catalyzed a paradigm shift in medical image analysis [16], [28]. Deep learning eliminated the dependency on hand-engineered descriptors by learning hierarchical, end-to-end representations directly from raw pixel matrices, establishing unprecedented pattern-recognition capabilities across diagnostic imaging [29].

## 2.2 Traditional Machine Learning for CXR

Prior to the deep learning era, automated chest radiograph classification followed a sequential, decoupled pipeline: lung field segmentation, region-of-interest (ROI) detection, hand-engineered feature extraction, and statistical classification.

Classical feature extraction techniques focused on capturing mathematical characterizations of thoracic tissue texture:
- **Gray-Level Co-occurrence Matrix (GLCM)**: Computed second-order statistical measures—including angular second moment (energy), contrast, correlation, variance, inverse difference moment (homogeneity), and entropy—across spatial pixel relationships to describe parenchymal texture variations.
- **Local Binary Patterns (LBP)**: Captured micro-textural patterns by thresholding neighboring pixels against a central pixel, creating binary codes representing local edges, spots, and flat regions.
- **Scale-Invariant Feature Transform (SIFT) & Speeded-Up Robust Features (SURF)**: Extracted localized, scale- and rotation-invariant gradient descriptors around salient anatomical landmarks.
- **Wavelet & Gabor Filter Banks**: Decomposed the spatial frequency spectrum into directional sub-bands to capture multi-scale textural patterns corresponding to pulmonary opacities.

These extracted feature vectors were subsequently fed into statistical learning models:
- **Support Vector Machines (SVM)**: Constructed optimal hyperplanes maximizing margin separation between disease and normal feature spaces, frequently employing Radial Basis Function (RBF) kernels to handle non-linear boundaries.
- **Random Forests (RF) & Decision Trees**: Ensembled hundreds of randomized decision trees to capture complex decision boundaries while offering relative resilience against feature collinearity.
- **k-Nearest Neighbors (k-NN)**: Classifying unknown cases based on distance metrics in high-dimensional feature space.

While traditional machine learning workflows achieved moderate success on constrained, single-center binary datasets (e.g., distinguishing high-contrast nodules from normal lung tissue), they suffered from catastrophic failure modes:
1. **Decoupled Optimization**: Feature extraction was fundamentally disconnected from classification objective functions; handcrafted features optimized for generic texture description could not adapt to subtle, disease-specific biological variations.
2. **Segmentation Fragility**: Extraction algorithms critically depended on precise lung boundary segmentation. Severe consolidations, pleural effusions, or apical infiltrates obscured anatomical lung boundaries, causing segmentation algorithms to fail and corrupting downstream feature vectors.
3. **Inability to Model Multi-Disease Complexity**: Handcrafted descriptors lacked the expressive capacity to simultaneously distinguish multiple overlapping, low-contrast pathologies such as viral pneumonia, bacterial consolidation, and mycobacterial cavities.

## 2.3 CNN-Based CXR Classification

The successful application of deep convolutional neural networks (CNNs) revolutionized medical imaging by unifying feature extraction and classification into a single, end-to-end differentiable mathematical optimization framework. By stacking convolutional layers, non-linear activation functions (ReLU), spatial pooling operations, and batch normalization, CNNs autonomously discover hierarchical feature representations:
- Low-level layers capture primitive visual cues such as edge gradients, high-frequency boundaries, and local intensity contrasts.
- Mid-level layers synthesize these primitives into complex textural patterns, structural motifs, and localized shapes.
- High-level layers capture macroscopic anatomical context, spatial configurations of the thoracic cage, and diffuse parenchymal infiltration patterns.

Unlike classical algorithms, CNNs preserve the spatial topology of chest radiographs through translation-equivariant convolutional kernels. The publication of CheXNet by Rajpurkar et al. (2017) [5] marked a watershed moment in thoracic AI. Utilizing a 121-layer Densely Connected Convolutional Network (DenseNet-121) trained on 112,120 frontal chest radiographs from the NIH ChestX-ray14 dataset [6], CheXNet reported radiologist-level diagnostic performance in detecting pneumonia, achieving a ROC-AUC of 0.841 across the cohort. This demonstrated that deep convolutional architectures could extract subtle pathological features across massive, heterogeneous patient populations without requiring manual feature engineering.

Subsequent investigations rapidly applied deep CNNs to COVID-19 detection during the global pandemic, automated pulmonary tuberculosis screening, and multi-label thoracic disease diagnosis across public datasets including CheXpert [7] and PadChest [27].

## 2.4 Transfer Learning

Training deep neural networks containing tens of millions of parameters from scratch requires hundreds of thousands of meticulously annotated training samples to prevent severe overfitting. In clinical medicine, acquiring fully balanced, expert-annotated datasets of such scale is constrained by patient privacy regulations, intellectual property protections, and the immense labor expense of certified radiologist review.

To overcome the small-data bottleneck, **transfer learning** has emerged as the foundational paradigm in medical deep learning [16], [31]. Under this framework, a convolutional neural network is initially pre-trained on a massive natural vision dataset (most prominently ImageNet, comprising over 14 million images across 1,000 object categories) [31]. Through this pre-training, the network learns generic visual representations—such as Gabor-like directional filters, color boundaries, curvature transitions, and complex textural compositions.

When transferring to chest radiograph analysis, two canonical transfer strategies are employed:
1. **Linear Probing / Feature Extraction**: The pre-trained convolutional backbone weights are frozen, and only the final classification head (dense layers and Softmax projection) is optimized on the target medical dataset. While computationally efficient and immune to catastrophic forgetting, frozen backbones cannot adapt domain-specific representations to the subtle gray-scale textures of medical radiographs.
2. **Fine-Tuning**: The entire network (or a selected subset of high-level convolutional blocks) is initialized with ImageNet weights and updated with a low learning rate on the medical cohort. This enables the network's high-level feature extractors to warp from natural object representations toward specialized radiological opacities (e.g., ground-glass attenuations, consolidation patterns, and pleural blunting).

Extensive empirical literature confirms that fine-tuning pre-trained networks consistently yields faster convergence, superior generalization, and higher classification metrics compared to models trained randomly from scratch on medical imaging datasets [2], [3], [11].

## 2.5 DenseNet / ResNet / EfficientNet Comparative Studies

Selecting an optimal convolutional backbone is a critical architectural decision in medical image classification. Across published literature, three deep architectural families have dominated thoracic radiograph analysis:

### 2.5.1 Residual Networks (ResNet)
Introduced by He et al. (2016) [16], ResNet introduced identity shortcut connections that bypass parameterized convolutional layers:
$$x_{l+1} = \mathcal{H}(x_l) = \mathcal{F}(x_l, \mathcal{W}_l) + x_l$$
This formulation ensures that gradients can propagate directly through the identity mappings during backpropagation, eliminating the vanishing/exploding gradient problem and enabling the successful training of very deep networks (e.g., ResNet50, ResNet101, ResNet152). In CXR analysis, ResNet architectures have been widely utilized due to their stable convergence and strong local feature representation capabilities [9], [11], [13].

### 2.5.2 Densely Connected Networks (DenseNet)
Introduced by Huang et al. (2017) [15], DenseNet radically extended the shortcut concept by connecting each layer to every other layer in a feed-forward fashion within a dense block:
$$x_l = H_l([x_0, x_1, x_2, \dots, x_{l-1}])$$
where $[x_0, x_1, \dots, x_{l-1}]$ represents the concatenation of all feature maps produced in preceding layers. DenseNet introduces three distinct advantages for chest radiography:
1. **Maximum Feature Reuse**: Every layer has direct access to both low-level edge features and high-level abstract semantics, highly advantageous for detecting subtle parenchymal opacities that require both fine texture and macroscopic context.
2. **Parameter Efficiency**: Because feature maps are concatenated rather than summed, each layer needs to learn only a small number of new feature channels (controlled by the growth rate, typically $k=32$), reducing parameter count compared to wide residual networks.
3. **Implicit Deep Supervision & Gradient Flow**: Direct connections ensure short paths from the final classification loss to all earlier layers, bolstering gradient flow and mitigating overfitting on moderate-sized cohorts.

These characteristics led Rajpurkar et al. [5] and numerous subsequent researchers to adopt DenseNet-121 as the gold-standard architecture for chest radiograph CAD.

### 2.5.3 EfficientNet
Introduced by Tan and Le (2019) [17], EfficientNet introduced a systematic **compound scaling** method that uniformly scales network depth, width, and input image resolution using a fixed compound coefficient $\phi$:
$$	ext{depth: } d = lpha^\phi, \quad 	ext{width: } w = eta^\phi, \quad 	ext{resolution: } r = \gamma^\phi$$
subject to $lpha \cdot eta^2 \cdot \gamma^2 pprox 2$ and $lpha \ge 1, eta \ge 1, \gamma \ge 1$. Utilizing mobile inverted bottleneck convolutions (MBConv) with squeeze-and-excitation blocks, EfficientNet achieves high ImageNet accuracy with reduced computational FLOPs. In medical imaging, EfficientNet models (B0 through B7) have demonstrated competitive accuracy, though they sometimes exhibit training instability when fine-tuned on uncurated, highly imbalanced medical corpora.

Several published studies have compared these backbones on chest radiograph datasets:
- Fernando et al. (2022) [9] conducted a comparative study of MobileNetV2, ResNet50, InceptionV3, and Xception for 3-class CXR classification (Normal, Pneumonia, COVID-19), reporting that ResNet50 achieved the highest average accuracy (98.87%).
- Mahesh and Kumar (2023) [10] compared DenseNet121, ResNet50, and InceptionV3 on a 4-class Kaggle dataset (Pneumonia, COVID-19, TB, Healthy), showing that model checkpointing and learning rate decay were essential for stabilizing convergence across DenseNet and ResNet models.
- Charan et al. (2024) [11] investigated MobileNetV2, VGG16, InceptionNet, ResNet50, and EfficientNet with textural features (LBP) on a fused multi-class dataset, finding that ResNet50 (97.1%) and EfficientNet (96.3%) achieved leading performance.

## 2.6 Multi-Disease CXR Classification

While early research focused almost exclusively on isolated binary classification (e.g., Pneumonia vs. Normal [2], COVID-19 vs. Normal [3], or Tuberculosis vs. Normal [4]), clinical diagnostic practice demands simultaneous differential diagnosis across multiple thoracic pathologies. A patient presenting to an emergency triage department with acute cough, fever, and dyspnea may suffer from bacterial pneumonia, COVID-19, active pulmonary tuberculosis, or decompensated heart failure with pleural effusion. A binary model trained only to distinguish pneumonia from normal scans is clinically unsafe when confronted with tuberculosis or nodular malignancies, as it forces an incorrect binary choice.

Consequently, research transitioned toward multi-class and multi-label formulations. Prominent hospital-scale benchmarks—including NIH ChestX-ray14 (14 labels, weakly mined via NLP from radiology reports) [6], CheXpert (14 observations with explicit uncertainty labels) [7], and PadChest (over 160,000 radiographs with 174 radiographic findings) [27]—formalized multi-label prediction where an individual radiograph can exhibit multiple simultaneous findings (e.g., Pneumonia co-occurring with Pleural Effusion).

Concurrently, multi-class single-label formulations have been widely investigated for computer-aided clinical triage and primary screening:
- Chowdhury et al. (2020) [3] established the widely utilized COVID-19 Radiography Database, demonstrating that pre-trained CNNs could achieve up to 99.7% accuracy in distinguishing COVID-19, viral pneumonia, and normal scans.
- Rahman et al. (2020) [4] developed a comprehensive tuberculosis database, demonstrating 98.6% classification accuracy using segmented DenseNet-201 models.
- Deva and Dagur (2025) [13] developed a hybrid Vision Transformer (ViT) and ResNet architecture with CLAHE enhancement for 4-class multi-disease classification (COVID-19, Pneumonia, Lung Opacity, Normal), reporting 98.54% overall test accuracy.

However, a critical vulnerability of many published multi-disease studies is their reliance on simple dataset merging: images from independent, single-disease datasets are pooled into a single folder structure without verifying patient-level separation, without auditing for modality contamination (e.g., accidental inclusion of CT slices), and without investigating whether models learn disease pathology or acquisition-site shortcuts.

## 2.7 Explainable AI / Grad-CAM

Deep neural networks are notoriously characterized as "black boxes" due to their complex non-linear parameter spaces, creating severe barriers to clinical adoption. In safety-critical healthcare environments, clinicians cannot responsibly base therapeutic interventions solely on an opaque numerical probability score. If an algorithm predicts "COVID-19 with 99% confidence," the physician must verify whether the model is identifying genuine peripheral bilateral ground-glass opacities, or whether it has latched onto a hospital-specific lateral marker, a chest tube artifact, or scanner-specific border padding.

To establish clinical interpretability, Explainable Artificial Intelligence (XAI) techniques have become an essential requirement for medical CAD systems. Among visual attribution methods, **Gradient-weighted Class Activation Mapping (Grad-CAM)**, formulated by Selvaraju et al. (2017) [20], represents the dominant standard:
$$lpha_k^c = rac{1}{Z} \sum_{i} \sum_{j} rac{\partial Y^c}{\partial A_{i,j}^k}$$
$$L_{	ext{Grad-CAM}}^c = 	ext{ReLU}\left(\sum_k lpha_k^c A^k
ight)$$
where $Y^c$ is the pre-softmax score for class $c$, $A^k$ is the $k$-th feature activation map of a designated convolutional layer, and $Z$ is the spatial area ($U 	imes V$). By computing the gradient of the class score with respect to feature activation maps, Grad-CAM captures the importance weight $lpha_k^c$ of each channel. The rectified linear unit (ReLU) ensures that only features positively contributing to the target class are visualized.

In chest radiography:
- Dagnaw and El Mouthadi (2023) [12] implemented Score-CAM on lightweight CNNs with CLAHE preprocessing for pneumonia and tuberculosis classification, demonstrating that visual explanation is vital for verifying that predictions align with radiological lesion locations rather than artifactual noise.
- Deva and Dagur (2025) [13] integrated Grad-CAM into their ViT-ResNet hybrid framework to generate heatmaps highlighting inflammatory consolidations and ground-glass opacities.
- Selvaraju et al. [20] and Rajpurkar et al. [5] demonstrated that Grad-CAM maps from DenseNet architectures successfully localize consolidations, pulmonary masses, and pleural fluid blunting without requiring pixel-level bounding-box supervision during training.

## 2.8 Domain Shift and Cross-Dataset Generalization

A profound, widely acknowledged crisis in medical artificial intelligence is the failure of models to generalize across independent clinical acquisition environments—a phenomenon known as **domain shift** or **distribution shift**. 

Formally, domain shift occurs when the joint distribution of inputs $X$ and labels $Y$ differs between the training source domain $\mathcal{D}_S = \{X_S, P_S(X)\}$ and the clinical deployment target domain $\mathcal{D}_T = \{X_T, P_T(X)\}$, such that $P_S(X) 
e P_T(X)$ (covariate shift) or $P_S(Y|X) 
e P_T(Y|X)$ (concept shift).

In chest radiography, domain shift is driven by extensive physical and clinical variations:
1. **Acquisition Hardware**: Direct digital radiography (DR) detectors exhibit radically different dynamic ranges, spatial resolutions, modulation transfer functions, and quantum noise characteristics compared to computed radiography (CR) photostimulable phosphor plates or scanned analog film digitizers.
2. **Technique & Exposure Parameters**: Tube kilovoltage (kVp, typically 90–125 kVp for adults), exposure time (mAs), beam filtration, and anti-scatter grid usage vary substantially between clinical centers and portable bedside vs. fixed radiographic rooms.
3. **Patient Positioning & Projection**: Differences between posteroanterior (PA) erect views and anteroposterior (AP) supine views alter cardiac magnification, vascular engorgement, and rib cage orientation.
4. **Demographics & Disease Prevalence**: Differences in patient age distributions, underlying comorbidities, disease severity, and regional pathogen prevalence induce substantial distribution discrepancies.

To counter domain shift, two major algorithmic strategies have been explored:
- **Latent Covariance Alignment (Deep CORAL)**: Sun and Saenko (2016) [18] proposed minimizing the Frobenius norm distance between the second-order feature covariance matrices of source and target domains, aligning latent feature distributions without requiring target-domain class labels.
- **Domain-Adversarial Neural Networks (DANN)**: Ganin et al. (2016) [19] introduced a minimax adversarial framework where a feature extractor is trained to simultaneously maximize classification accuracy while confusing a domain discriminator via a Gradient Reversal Layer (GRL), compelling the network to learn domain-invariant representations.
- **Hybrid Domain-Adversarial Architectures**: Akyol and Bilgin (2025) [14] proposed a hybrid CNN-Transformer domain-adversarial framework for robust pneumonia classification across heterogeneous CXR datasets, demonstrating that domain adaptation can reduce performance degradation between disparate multi-center cohorts.

## 2.9 Dataset Bias and Shortcut Learning

In a landmark investigation published in *Nature Machine Intelligence*, DeGrave, Janizek, and Lee (2021) [8] audited multiple high-performing deep learning models trained to detect COVID-19 from chest radiographs. Utilizing saliency mapping, generative adversarial counterfactuals, and cross-dataset testing, the authors uncovered that deep neural networks were not learning the immunological manifestations of coronavirus infection. Instead, the models were relying on **shortcut learning**:
- Models learned to exploit radiopaque lateral markers (e.g., metal "L" and "R" anatomical tags) whose font, size, and positioning uniquely identified specific hospitals where COVID-19 patients were treated.
- Networks identified patient positioning differences (supine bedside AP views for acute, bedbound COVID-19 patients vs. ambulatory erect PA views for normal controls).
- Models activated on the border cropping, dark corner padding, and high-frequency edge textures unique to specific digitizer models.

When these high-performing models were evaluated on chest radiographs from independent hospitals where the shortcuts were absent, their diagnostic sensitivity collapsed to near random guessing.

Shortcut learning presents a severe threat in multi-dataset research: when researchers merge single-source datasets (e.g., Kaggle pneumonia, GitHub COVID-19, and JSRT nodule scans), the acquisition site becomes heavily correlated with the disease label. A standard deep CNN will naturally take the path of least mathematical resistance, learning the high-frequency scanner signature of the repository rather than the underlying disease morphology. This reality necessitates rigorous dataset forensic audits, source-held-out diagnostics, and biophysical frequency filtering.

## 2.10 Comparative Analysis of Existing Studies

To rigorously contextualize LungAI within published literature, Table 2.1 provides a structured, evidence-based comparative analysis of representative peer-reviewed thoracic classification studies.

### Table 2.1: Comparative Analysis of Existing Chest X-Ray Classification Studies and LungAI

| Study / Authors | Year | Dataset | Disease Classes | Algorithm | Reported Accuracy / AUC | Evaluation Strategy | Explainability | Domain-Generalization Evaluation | Key Limitation | Relevance to LungAI |
| :--- | :---: | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| **Kermany et al.** [2] | 2018 | Guangzhou Women and Children's Medical Center (5,856 CXRs) | Binary: Normal vs. Pneumonia | Inception-v3 Transfer Learning | 92.8% Accuracy, 96.8% Sensitivity | 90/10 Train/Test Split | Occlusion Testing | None (single pediatric center) | Confined to pediatric population; binary scope; single clinical site | Established benchmark for pediatric transfer learning; source of initial pneumonia scans |
| **Chowdhury et al.** [3] | 2020 | COVID-19 Radiography Database | 3-Class: COVID-19, Normal, Viral Pneumonia | SqueezeNet, MobileNet, ResNet18, DenseNet201 | 99.7% Accuracy (DenseNet201) | 5-Fold Cross-Validation | None | None (single blended repository) | Highly curated, small initial COVID cohort; risk of repository shortcut learning | Motivated multiclass COVID evaluation; source of COVID-19 training data |
| **Rahman et al.** [4] | 2020 | Multi-source TB Dataset (7,000 CXRs) | Binary: Normal vs. Tuberculosis | 9 CNNs + U-Net Lung Segmentation | 98.6% Accuracy (DenseNet201 + Segmented) | 80/20 Train/Test Split | Grad-CAM | None (pooled data) | Dependent on robust lung field segmentation; binary evaluation | Demonstrated DenseNet efficacy for TB; motivated U-Net roadmap |
| **Rajpurkar et al. (CheXNet)** [5] | 2017 | NIH ChestX-ray14 (112,120 CXRs) | 14 Thoracic Diseases (Multi-label) | DenseNet-121 (121 Layers) | 0.841 ROC-AUC (Pneumonia); radiologist comparison | Random Train/Val/Test Split | CAM (Class Activation Mapping) | None (single institution: NIH Clinical Center) | Weakly supervised NLP-mined labels; ~10% label noise; single acquisition institution | Established DenseNet-121 as gold-standard CXR architecture; informed backbone selection |
| **Fernando et al.** [9] | 2022 | Kaggle CXR Collections | 3-Class: Normal, Pneumonia, COVID-19 | ResNet50, MobileNetV2, InceptionV3, Xception | 98.87% Accuracy (ResNet50), 98.54% Recall | 5-Fold Cross-Validation | None | None (single blended dataset) | Small 3-class setup; no patient-level splitting documented; no cross-acquisition evaluation | Demonstrates that standard CNN transfer learning easily achieves >98% on simple pooled sets |
| **Mahesh & Kumar** [10] | 2023 | Kaggle Respiratory Collection | 4-Class: Pneumonia, COVID-19, TB, Healthy | DenseNet121, ResNet50, InceptionV3 | Competitive multiclass accuracy | Hold-out evaluation with LR decay | None | None (pooled dataset) | Single-source/pooled data; no cross-center testing; no explainability integration | Compares DenseNet and ResNet under learning rate decay and model checkpointing |
| **Charan et al.** [11] | 2024 | Fused Multi-Class CXR Data | Multi-Class: TB, COVID-19, Pneumonia, Normal | MobileNetV2, ResNet50, EfficientNet + LBP Texture | 97.1% Accuracy (ResNet50), 96.3% (EfficientNet) | Train/Test Split on Fused Data | None | None (fused data without source hold-out) | Fused datasets without controlling for source-class confounding; no cross-acquisition testing | Shows benefits of combining textural features with CNNs on fused datasets |
| **Dagnaw & El Mouthadi** [12] | 2023 | Kaggle CXR Repositories | 3-Class: Pneumonia, TB, Normal | Lightweight CNN + CLAHE | >96% Precision / Recall | Train/Val/Test Split | Score-CAM | None (single-center splits) | Unbalanced small cohort; no cross-domain testing | Justifies CLAHE preprocessing and CAM-based saliency mapping for clinical auditability |
| **Deva & Dagur** [13] | 2025 | Multi-Class CXR Collection | 4-Class: COVID-19, Pneumonia, Lung Opacity, Normal | Hybrid Vision Transformer (ViT) + ResNet | 98.54% Internal Accuracy, 94.07% External Cross-Dataset | Internal Split + External Cross-Dataset | Grad-CAM | Yes (Cross-dataset evaluation showed 4.47% accuracy drop) | High computational complexity; lacks fine-grained nodule/mass differentiation | Proves hybrid model efficacy and illustrates domain degradation across external datasets |
| **Akyol & Bilgin** [14] | 2025 | Heterogeneous Multi-Center CXR | Binary: Pneumonia vs. Normal | Hybrid CNN–Transformer DANN | Improved cross-domain ROC-AUC | Multi-source domain adaptation | Attention maps | Yes (evaluated cross-dataset domain adaptation) | Binary classification only; complex multi-stage minimax training | Directly corroborates LungAI's exploration of domain-adversarial adaptation (DANN) |
| **LungAI (Model D)** *(This Thesis)* | 2026 | Unified V5 Corpus (10,547 CXRs / 10,270 Patients from 8 Sources) | 6-Class: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, TB | DenseNet-121 + Gaussian Spatial Low-Pass Filter ($\sigma=1.0$) | 82.93% Accuracy, 78.35% Macro F1, 0.9755 ROC-AUC, 0.8391 PR-AUC | Patient-Strict 70/15/15 Split + Quarantined External Montgomery Benchmark | Grad-CAM (`conv5_block16_concat`) | Yes (Complete 5-paradigm comparison: ERM, CORAL, DANN, Frequency, Hybrid + Montgomery failure audit) | Montgomery fine-grained collapse (0% exact TB recall); nodule/effusion diagnostic challenge; research-prototype scope | Comprehensive multi-source dataset audit, biophysical frequency filtering, transparent external failure disclosure, and full-stack software integration |

> *Note: Reported metrics are not directly comparable across studies because datasets, class definitions, prevalence, partition strategies, preprocessing pipelines, and evaluation protocols differ substantially.*

---

### Table 2.2: Research Differentiation of LungAI from Representative Existing Approaches

| Dimension | Existing Studies | Typical Limitation | LungAI Approach | Research Significance |
| :--- | :--- | :--- | :--- | :--- |
| **1. Dataset Construction** | Ad-hoc downloading of public folders without verification | Blends incompatible modalities, formats, and conflicting annotations | Reconstructed V5 canonical cohort ($N=10,547$ scans) from 8 international repositories | Provides a verified, reproducible multi-source benchmark for thoracic deep learning |
| **2. Dataset Forensic Audit** | Unchecked merging of Kaggle/GitHub repositories | Accidental inclusion of CT slices, duplicates, and corrupt files | Complete hash-level duplicate audit, perceptual hash deduplication, and cross-modality purge | Disclosed and eliminated 692 CT slices and duplicate patient entries from historical iterations |
| **3. Patient-Level Splitting** | Random shuffle split across image filenames | Images from the same patient appear in train and test splits (identity leakage) | Strict patient-level metadata partitioning ($N=10,270$ patients, 0% patient overlap) | Guarantees test metrics evaluate true clinical generalization rather than patient re-identification |
| **4. Multi-Source Composition** | Single-center datasets (e.g., Kermany [2]) or 2-source pools | Models overfit to center-specific acquisition protocols and patient demographics | Harmonized 8 distinct international acquisition sources (TBX11K, NIH, VinDr, JSRT, etc.) | Exposes models to multi-institutional sensor distributions during representation learning |
| **5. Source/Class Confounding** | Assumed independence between data source and disease label | Pathologies originate from single centers, creating complete source-label confounding | Computed Cramér's V statistic (0.7654), documenting severe confounding across public pools | Academically proves that naive multi-dataset models learn repository shortcuts rather than pathology |
| **6. External-Domain Evaluation** | Internal test evaluation only (e.g., [3], [4], [9], [10], [11]) | High reported accuracies (>98%) collapse under real-world clinical deployment | Quarantined Montgomery County benchmark ($N=138$) evaluated completely zero-shot | Transparently demonstrates real-world clinical deployment boundaries of deep models |
| **7. Source-Held-Out Diagnostics** | Aggregate pooled metrics reported without source breakdown | Hidden failure modes across specific acquisition centers are masked | Evaluated per-source classification accuracy and class-specific recalls across all 8 sources | Identifies institutional performance disparities (e.g., VinDr 63.18% vs. Existing Normal 95.56%) |
| **8. Domain-Generalization Experiments** | Single baseline model trained via standard empirical risk minimization | Unclear whether domain adaptation strategies provide measurable clinical utility | Controlled benchmark of ERM, Deep CORAL, DANN, Frequency-Aware, and Hybrid models | Provides rigorous empirical comparison of representation learning vs. input-space filtering |
| **9. Frequency-Aware Preprocessing** | Standard spatial resizing (bilinear/bicubic) without frequency filtering | Deep convolutional kernels latch onto high-frequency digitizer and sensor noise shortcuts | Formulated Gaussian spatial low-pass filtering ($\sigma=1.0$) prior to Lanczos-4 resampling | Filters sensor grain, improving internal multi-source accuracy from 76.18% to 82.93% (+5.72% F1) |
| **10. Controlled Ablation** | Varying multiple hyperparameter components simultaneously | Inability to attribute performance gains to specific architectural interventions | Isolated backbone (DenseNet-121), learning rate, and schedule across all 5 paradigms | Validates that performance deltas originate strictly from the tested domain-generalization mechanisms |
| **11. Negative / Failed Experiments** | Withholding failed experiments (publication bias) | Skews scientific literature toward unrealistic claims of universal algorithmic success | Fully reported DANN failure (76.24%) and Hybrid underperformance (78.22%) | Provides critical scientific evidence that complex adversarial alignment can degrade feature quality |
| **12. Explainability** | Opaque black-box models or disconnected post-hoc analyses | Clinicians cannot audit whether predictions originate from genuine thoracic lesions | Integrated Grad-CAM directly on `conv5_block16_concat` with normalized heatmap rendering | Enables real-time radiological verification of parenchymal lesion localization in the UI |
| **13. CXR Input Validation** | Absence of input screening; models process any uploaded image | Non-CXR images, corrupted streams, and out-of-distribution files produce false diagnoses | Two-stage defense-in-depth gate: Stage 1 biophysical screening + Stage 2 DenseNet-121 classifier | Prevents out-of-distribution input contamination from reaching the disease classification engine |
| **14. Non-CXR Rejection** | Arbitrary confidence thresholds or unhandled runtime exceptions | Adversarial images (e.g., domestic animals, documents, CT slices) masquerade as CXR | Evaluated against 7 adversarial non-CXR categories; production threshold audited at $\tau=0.83$ | Successfully rejects 100% of tested adversarial inputs in production testing |
| **15. Application Integration** | Offline Jupyter notebooks or theoretical research scripts | Algorithms remain inaccessible to practicing medical professionals | Asynchronous three-tier clinical platform: FastAPI REST service + React 18 single-page app | Bridges deep learning theory with functional, responsive clinical decision-support engineering |
| **16. Database / Auditability** | Transient in-memory execution without persistent audit trails | Regulatory violation; diagnostic records cannot be retroactively audited | Relational SQLAlchemy persistence (SQLite/PostgreSQL) tracking patient UUIDs, scans, and results | Ensures complete clinical traceability and EMR integration compliance |
| **17. Reproducibility** | Undisclosed manifests, missing split definitions, unseeded code | Experiments cannot be reproduced or verified by independent researchers | Published unified manifest (`unified_manifest_v5.csv`), cryptographic hashes, and frozen scripts | Guarantees complete end-to-end scientific auditability and experimental repeatability |
| **18. Limitations Disclosure** | Unsubstantiated claims of "clinical readiness" and "superiority" | Generates dangerous over-reliance in medical software without clinical validation | Comprehensive disclosure of 18 limitations, Montgomery failure, and gate selection leakage | Upholds academic honesty and establishes realistic expectations for clinical decision support |

---

### Table 2.3: Impact of Existing Approaches on the Design and Evaluation of LungAI

| Existing Approach | Algorithm | Reported Result | What the Approach Demonstrates | Potential Vulnerability / Limitation | How It Influenced LungAI |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **ResNet Transfer Learning** (e.g., He et al. [16], Fernando et al. [9]) | ResNet50 with ImageNet Initialization | >95% accuracy on 3-class pooled cohorts | Pre-trained residual representations converge rapidly and capture strong local visual features | Identity summation can suppress low-contrast parenchymal details; does not prevent domain overfitting | Motivated baseline transfer learning comparisons; established that high pooled accuracy does not guarantee multi-source robustness |
| **DenseNet Feature Concatenation** (e.g., Huang et al. [15], Rajpurkar et al. [5]) | DenseNet-121 with Dense Concatenation | 0.841 ROC-AUC on NIH ChestX-ray14; radiologist-level | Dense feature reuse preserves multi-scale representations (fine textures and global context) | High internal accuracy can coexist with severe external scanner degradation | Selected as the canonical backbone for all 5 paradigms; validated feature reuse for subtle pulmonary lesions |
| **Vision Transformer / Hybrid Networks** (e.g., Deva & Dagur [13]) | ViT + ResNet Fusion with CLAHE | 98.54% internal accuracy; 94.07% cross-dataset | Self-attention captures long-range anatomical dependencies across bilateral lung fields | High computational complexity, large memory footprint, and data hunger on moderate medical cohorts | Demonstrates architectural alternatives; motivated investigating whether simpler frequency filtering could achieve robust gains |
| **Explainable AI / Visual Attribution** (e.g., Selvaraju et al. [20], Dagnaw & El Mouthadi [12]) | Grad-CAM, Score-CAM on CNN layers | Qualitative localization of consolidations | Gradient backpropagation provides interpretable visual heatmaps without bounding-box labels | Saliency maps can be visually plausible even when models rely on co-occurring background shortcuts | Directly integrated Grad-CAM on DenseNet-121's bottleneck concat layer to enable real-time visual inspection in the UI |
| **Latent Covariance Alignment** (e.g., Sun & Saenko [18]) | Deep CORAL Feature Alignment | Reduced classification error on Office-31 benchmark | Minimizing second-order covariance distance aligns source and target latent feature manifolds | Target domain feature distribution must be known during training; cannot resolve pixel-level sensor artifacts | Implemented in Phase 4B; demonstrated internal multi-source gain (80.89% vs. 76.18%), proving covariance alignment benefits |
| **Domain-Adversarial Training** (e.g., Ganin et al. [19], Akyol & Bilgin [14]) | DANN with Gradient Reversal Layer | Domain-invariant representations on multi-source data | Minimax optimization forces the feature extractor to discard source-specific discriminative cues | Adversarial training instability, mode collapse, and loss of class-discriminative pathological features | Implemented in Phase 4C; empirical evaluation proved DANN failed to improve over ERM (76.24%), preventing arbitrary architectural selection |
| **Frequency-Aware & Biophysical Preprocessing** (e.g., Zuiderveld [22], DeGrave et al. [8]) | CLAHE Contrast Equalization & Frequency Filtering | Enhanced local contrast and suppression of high-frequency noise | Normalizing optical dynamic range and filtering high-frequency noise attenuates scanner grain shortcuts | Over-filtering can blur subtle micro-nodules or faint ground-glass margins | Directly inspired Phase 4D; systematic validation grid established $\sigma=1.0$ as optimal, producing final Model D (82.93% accuracy) |

---

## 2.14 Identified Research Gaps & Synthesis

Synthesizing the surveyed literature reveals four critical research gaps that define the necessity and scope of the LungAI project:

1. **The Gap Between Pooled Benchmark Accuracy and Cross-Source Generalization**: Published literature extensively reports near-perfect classification accuracies (>95%–99%) on curated, single-source or blended multi-class datasets. However, these studies rarely conduct rigorous source-held-out diagnostics or zero-shot external hospital evaluations. Consequently, high reported performance often masks severe overfitting to repository-specific scanner signatures, as evidenced by DeGrave et al. [8].
2. **Lack of Controlled Domain Generalization Comparisons in Multi-Class CXR**: While individual domain adaptation techniques (CORAL, DANN) and frequency filtering have been studied in computer vision, controlled head-to-head comparisons on identical multi-source chest radiograph benchmarks remain exceedingly scarce. The literature lacks empirical clarity on whether algorithmic domain adaptation or biophysical frequency preprocessing provides superior clinical utility.
3. **Absence of Defensive Out-of-Distribution Input Verification**: Existing medical CAD literature almost universally operates on the unverified assumption that inputs provided to the classifier are valid chest radiographs. In real-world software deployments, uploading a non-radiographic photo, an axial CT slice, or a corrupted file results in confident, misleading thoracic disease predictions, presenting an unaddressed patient safety hazard.
4. **Disconnection Between Algorithmic Models and Clinical Decision Support**: The vast majority of published studies terminate at theoretical offline metrics (accuracy, ROC-AUC) without implementing end-to-end software integration. Practical deployment requires bridging the gap between raw Softmax probability distributions and deterministic clinical triage prioritization, persistent relational auditability, and ergonomic radiological viewport visualization.

LungAI directly bridges these gaps through forensic dataset harmonization, controlled multi-paradigm benchmarking, transparent external failure auditing, defense-in-depth input validation, and full-stack software integration.

---

# CHAPTER 3 — EXISTING SYSTEM

## 3.1 Conventional Clinical Triage & FIFO PACS Queues

In standard hospital radiological workflows, digital imaging modalities (Computed Radiography [CR] and Direct Digital Radiography [DR] units) capture thoracic radiographs and transmit them via the Digital Imaging and Communications in Medicine (DICOM) network protocol to an institutional Picture Archiving and Communication System (PACS). Once ingested, examinations populate an unprioritized, First-In, First-Out (FIFO) reading worklist accessed by on-duty radiologists.

Under this operational topology, an acute emergency radiograph exhibiting massive bilateral consolidation, tension pneumothorax, or extensive infectious infiltrate sits in the reading queue in the exact sequence it was transmitted by the modality, grouped indiscriminately alongside routine pre-employment physical examinations and outpatient follow-ups. In resource-constrained public hospitals, district healthcare centers, and rural health clinics, this structural absence of automated triage causes report turnaround delays extending from 24 to over 72 hours. In acute respiratory failure or fulminant bacterial pneumonia, therapeutic delays of even a few hours can substantially elevate patient morbidity and mortality.

Figure 3.1 illustrates the structural delays and cognitive bottlenecks inherent in the conventional radiological workflow.

![Conventional Radiological Workflow](docs/generated_figures/fig_3_1_existing_workflow.png)
*Figure 3.1: Conventional PACS FIFO reading workflow, highlighting the absence of automated triage, sequential turnaround delays, and vulnerability to cognitive fatigue.*

## 3.2 Inter-Observer Variability & Cognitive Fatigue

Visual interpretation of chest radiographs is an intrinsically complex cognitive task bounded by human perceptual sensitivity. Planar CXR compresses three-dimensional thoracic anatomy into a single two-dimensional projection, resulting in extensive anatomical superimposition. Discriminating subtle, low-contrast opacities (such as early viral ground-glass attenuations, small apical tuberculous infiltrates, or faint solitary pulmonary nodules) from overlying rib structures, clavicular junctions, and normal pulmonary vascular markings requires continuous, intense visual scrutiny.

In clinical practice, diagnostic performance is heavily degraded by cognitive fatigue, extended work shifts, high examination volumes, and perceptual distraction. Extensive clinical literature documents inter-observer and intra-observer diagnostic disagreement rates between 15% and 30% among certified radiologists interpreting thoracic radiographs. Emergency department physicians and non-specialist resident medical officers—who are frequently required to make initial therapeutic decisions during night shifts without immediate radiologist consultation—exhibit even higher error rates. These realities underscore the acute need for an objective, automated decision-support system capable of pre-screening examinations at the moment of digital acquisition.

## 3.3 Rule-Based & First-Generation CAD Systems

Early computer-aided detection (CAD) systems developed in the 1980s and 1990s relied upon deterministic rule-based algorithms, edge-detection filters, and morphological operations. Second-generation CAD systems in the 2000s introduced hand-engineered textural feature extraction (e.g., Gray-Level Co-occurrence Matrices [GLCM], Local Binary Patterns [LBP], and Gabor filter banks) coupled with classical statistical classifiers (Support Vector Machines [SVM] and Random Forests).

While these traditional CAD systems represented pioneering efforts, they suffered from crippling operational limitations:
1. **Excessive False-Positive Rates**: Heuristic edge detectors and textural filters consistently triggered false alarms on normal anatomical variations (such as rib crossings, vascular bifurcations, and skin folds), generating between 5 and 10 false-positive marks per radiograph. Clinicians quickly experienced "alert fatigue" and routinely disabled the software.
2. **Extreme Sensitivity to Exposure Variations**: Handcrafted descriptors assumed standardized pixel intensity distributions. Variations in tube kilovoltage (kVp), milliampere-seconds (mAs), or patient body habitus severely altered GLCM and LBP feature values, causing dramatic classification failure.
3. **Inability to Model Multi-Pathology Interactions**: Rule-based systems could only evaluate narrow, isolated binary conditions (e.g., nodule presence vs. absence). They were fundamentally incapable of performing multi-class differential diagnosis across overlapping inflammatory, infectious, and consolidative thoracic diseases.

## 3.4 Limitations of Single-Source Deep Learning Classifiers

The advent of deep convolutional neural networks (CNNs) eliminated the dependency on handcrafted features, demonstrating remarkable classification performance on published benchmarks. However, the vast majority of existing academic and commercial deep learning CAD systems are trained and evaluated on homogeneous, single-institution datasets or blindly blended multi-dataset pools.

When deployed in clinical environments, single-source deep learning classifiers exhibit severe structural vulnerabilities:
1. **Shortcut Learning & Confounder Overfitting**: As demonstrated by DeGrave et al. [8], deep neural networks frequently achieve high benchmark accuracy by memorizing dataset-specific digital shortcuts—such as the unique typography of radiopaque anatomical markers ("L" vs. "R"), digital border cropping, hospital-specific patient positioning, or scanner-specific noise textures—rather than learning genuine disease pathology.
2. **Vulnerability to Out-of-Distribution Inputs**: Standard deep neural networks inherently lack defensive input validation. If an operator inadvertently uploads a lateral radiograph, an axial CT slice, a photograph of documentation, or a non-medical image, the softmax layer will output a high-confidence thoracic disease prediction (e.g., "98% Pneumonia"), creating a dangerous patient safety hazard.
3. **Lack of Relational Auditability**: Many academic CAD models operate as isolated computational scripts without persistent relational databases, patient entity linking, or auditable logging, violating fundamental clinical governance requirements.

## 3.5 The Inevitability of Sensor & Domain Shift in Healthcare

In clinical healthcare systems, hardware heterogeneity is unavoidable. Hospitals operate radiographic equipment acquired across decades, encompassing:
- Modern Direct Digital Radiography (DR) flat-panel detectors (amorphous silicon or selenium);
- Computed Radiography (CR) systems utilizing photostimulable phosphor plates scanned by laser digitizers;
- Legacy analog film radiographs scanned via optical film digitizers for archival purposes.

These divergent hardware modalities produce radical variations in spatial resolution, dynamic range, contrast resolution, modulation transfer function, and high-frequency noise textures. A deep neural network trained exclusively on modern digital DR scans will encounter severe distribution shift when presented with a digitized film scan, frequently causing feature representations to collapse.

## 3.6 Existing System Limitations Matrix

Table 3.1 synthesizes the operational, clinical, and architectural limitations of existing radiological workflows and conventional CAD systems.

### Table 3.1: Comparative Limitations Matrix of Existing Systems

| Workflow / System Dimension | Conventional Radiological Workflow | Traditional CAD Systems (GLCM / SVM) | Standard Single-Source Deep Learning CAD |
| :--- | :--- | :--- | :--- |
| **Triage Mechanism** | Unprioritized FIFO queue; no automated urgency sorting | None; post-acquisition secondary reader | Raw softmax output; no structured clinical urgency tiers |
| **Diagnostic Turnaround** | 24 to 72+ hours in public/rural hospitals | Immediate, but ignored due to high false-alarm rate | Fast inference, but lacks workflow integration |
| **Inter-Observer Agreement** | 15% to 30% disagreement due to cognitive fatigue | Inconsistent; high sensitivity to exposure parameters | Moderate consistency, but vulnerable to domain shift |
| **False-Positive Burden** | Baseline human error rate | Severe (5 to 10 false alarms per scan; alert fatigue) | Low on internal data; highly unpredictable externally |
| **Input Validation & Safety** | Manual verification by radiologic technologist | None; crashes or corrupts on invalid input | Absent; outputs high confidence on non-CXR inputs |
| **Domain Generalization** | Human visual cortex adapts naturally to hardware shifts | Fails catastrophically across different scanner hardware | Severe performance drop (>20% accuracy loss) on external sites |
| **Multi-Class Capability** | Full differential diagnosis (subject to fatigue) | Limited strictly to isolated binary tasks (e.g., nodule CAD) | Frequently trained on 2–3 classes; rarely audits 6+ classes |
| **Explainability** | Detailed narrative radiological report | Bounding circles around heuristic threshold triggers | Often absent; requires separate Grad-CAM implementation |
| **Clinical Auditability** | Standard PACS archiving | Transient workstation overlays; no EMR integration | Usually standalone Jupyter script; no relational persistence |

---

# CHAPTER 4 — PROPOSED SYSTEM

## 4.1 Architectural Vision & System Philosophy

To overcome the systemic failures of conventional PACS reading queues and the domain-shift vulnerabilities of single-source classifiers, this dissertation proposes **LungAI**—an end-to-end, auditable, and domain-generalized clinical decision-support and triage platform. 

The core architectural philosophy of LungAI is governed by four design principles:
1. **Rigorous Multi-Source Domain Generalization**: Moving beyond the illusion of high single-dataset accuracy by engineering representations that generalize across heterogeneous acquisition sources through biophysical frequency filtering and rigorous forensic data auditing.
2. **Defensive Input Verification (Defense-in-Depth)**: Enforcing strict biophysical and anatomical validation gates to intercept and reject out-of-distribution, corrupt, or non-radiographic uploads before they can contaminate downstream inference.
3. **Actionable Clinical Urgency Stratification**: Translating continuous multi-class softmax probabilities into deterministic clinical triage tiers (Emergency, Urgent, Routine), transforming an algorithmic score into a practical tool for prioritizing hospital worklists.
4. **Auditable Three-Tier Enterprise Topology**: Decoupling the machine learning inference engine behind a high-performance asynchronous REST API, backed by persistent relational electronic health records (EHR) and an ergonomic, responsive clinical viewport interface.

## 4.2 Multi-Tier Decoupled System Architecture

LungAI is realized as an asynchronous three-tier distributed software platform:
- **Presentation Tier (Frontend SPA)**: Engineered in React 18, delivering an ergonomic radiological workspace with interactive DICOM-style viewport windowing (pan, zoom, contrast adjustment), real-time Grad-CAM visual overlay inspection, patient demographic management, and automated structured report generation.
- **Application & Service Tier (FastAPI REST Backend)**: An asynchronous ASGI microservice built with FastAPI and Uvicorn, exposing structured RESTful endpoints under `/api/v1/`. The backend orchestrates request validation, the two-stage CXR gate, the singleton inference engine, Grad-CAM generation, and clinical urgency calculation.
- **Persistence & Data Tier (Relational Storage)**: Utilizing SQLAlchemy Object-Relational Mapping (ORM) with SQLite (dialect-compatible with PostgreSQL/MySQL), managing normalized relational schemas for patients, radiographic scans, multi-class predictions, Grad-CAM heatmaps, and audit logs.

Figure 4.1 details the multi-tier system architecture and operational data flow of the proposed LungAI platform.

![Proposed System Architecture](docs/diagrams/rendered/system_architecture.png)
*Figure 4.1: Proposed LungAI system architecture, illustrating the presentation, application, and persistence tiers, the two-stage CXR validation gate, Model D inference pipeline, and Grad-CAM explainability engine.*

## 4.3 Two-Stage CXR Input-Validation Gate

A critical innovation of the proposed LungAI platform is the integration of an automated, two-stage defense-in-depth CXR input-validation gate (`backend/ml/cxr_gate.py`) positioned directly at the API perimeter:

- **Stage 1 (Biophysical & Chromatic Screening)**: Intercepts raw byte streams and evaluates fundamental physical image properties:
  - Minimum spatial dimensions ($\ge 32 	imes 32$ pixels);
  - Anatomical aspect ratio bounds ($\le 2.2:1$ width-to-height ratio);
  - Luminance standard deviation threshold ($\sigma_L \ge 10.0$) to reject flat, blank, or low-contrast synthetic documents;
  - Polychromatic saturation screening in HSV color space: identifies and rejects photographic images exhibiting $>15\%$ high-saturation pixels across $\ge 3$ distinct hue bands, filtering out natural color photographs, selfies, landscapes, and vehicle images before executing deep neural inference.
- **Stage 2 (Semantic & Anatomical Deep Classifier)**: For grayscale or near-grayscale inputs that pass Stage 1 (such as axial CT slices, abdominal ultrasound images, or black-and-white natural photographs), Stage 2 deploys a dedicated binary DenseNet-121 classifier trained to distinguish planar frontal chest radiographs from non-CXR medical and non-medical images. Operating at an empirically audited production threshold of $	au = 0.83$, Stage 2 accurately verifies the presence of canonical thoracic anatomical landmarks (bilateral lung fields, mediastinum, cardiac silhouette, and rib cage).
- **Fallback Verification**: If the deep gate model weights are unavailable, the system automatically falls back to an algorithmic bilateral thoracic symmetry scoring function operating at threshold 0.45.

Inputs failing either stage are immediately isolated and rejected with structured HTTP 422 Unprocessable Entity responses, preventing misleading predictions.

## 4.4 Model D: DenseNet-121 with Frequency Preprocessing

The primary disease classification engine of LungAI is **Model D**—a 121-layer Densely Connected Convolutional Network (DenseNet-121) coupled with frequency-aware biophysical preprocessing:

1. **Preprocessing Pipeline**:
   - Chromatic-luminance decoupling via CIE LAB color space conversion;
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) applied exclusively to the luminance ($L$) channel (clip limit 2.0, $8 	imes 8$ grid), sharpening low-contrast parenchymal opacities while avoiding chromatic distortion;
   - **Gaussian Spatial Low-Pass Filtering** ($\sigma = 1.0$, $3 	imes 3$ kernel): selectively attenuates high-frequency scanner noise and digitized film grain shortcuts, compelling the convolutional kernels to focus on macroscopic anatomical lesion structures;
   - High-order Lanczos-4 spatial resampling to $224 	imes 224$ pixels;
   - Channel-wise ImageNet standardization.
2. **Backbone & Dense Bottleneck Architecture**:
   - Pre-trained DenseNet-121 convolutional feature extractor capitalizing on dense feature reuse across four dense blocks (6, 12, 24, and 16 dense layers, growth rate $k=32$);
   - Global Average Pooling (GAP) reducing spatial feature maps to a 1,024-dimensional feature vector;
   - Batch Normalization stabilizing latent feature variance;
   - Fully connected Dense projection layer (256 units, ReLU activation) establishing a compact, 256-dimensional latent embedding space;
   - Dropout layer ($p=0.3$) mitigating co-adaptation;
   - Final Dense classification layer (6 units) with Softmax activation generating the probability simplex $\hat{y} \in \Delta^5$ over the six diagnostic categories.

## 4.5 Visual Saliency & Explainability with Grad-CAM

To satisfy the ethical and clinical requirements of Explainable AI (XAI), LungAI integrates real-time Gradient-weighted Class Activation Mapping (Grad-CAM). Grad-CAM intercepts feature activations and backpropagated gradients at the final convolutional concatenation layer of DenseNet-121 (`conv5_block16_concat`):
1. Gradients of the predicted class score $Y^c$ are computed with respect to all 1,024 feature activation maps of `conv5_block16_concat`.
2. Global average pooling calculates neuron importance weights $lpha_k^c$.
3. The weighted linear combination is passed through a ReLU activation to isolate positive visual evidence.
4. The resulting $7 	imes 7$ saliency map is bilinearly upsampled to $224 	imes 224$, normalized between 0.0 and 1.0, and converted to a perceptually uniform pseudo-color heatmap (OpenCV COLORMAP_JET).
5. The heatmap is alpha-blended ($lpha = 0.45$) over the standardized radiograph, producing an interpretable visual overlay displayed directly in the web viewport.

## 4.6 Deterministic Clinical Urgency Triage Engine

Rather than expecting busy clinical practitioners to interpret raw probability vectors, LungAI synthesizes algorithmic outputs into an automated clinical urgency triage tier $U \in \{	ext{Routine}, 	ext{Urgent}, 	ext{Emergency}\}$ based on deterministic, clinically informed rules:

1. **Emergency Tier (Priority 1 — Immediate Evaluation Required)**:
   - Triggered if the predicted class is `Pneumonia` with confidence $p \ge 0.70$ (indicating extensive consolidative pneumonia with risk of respiratory decompensation);
   - OR if `COVID-19` with confidence $p \ge 0.70$ (indicating acute bilateral viral pneumonia);
   - OR if any active acute pathology exhibits confidence $p \ge 0.85$.
2. **Urgent Tier (Priority 2 — Priority Clinical Review Required)**:
   - Triggered if the predicted class is `Tuberculosis` with confidence $p \ge 0.50$ (warranting immediate infection control, sputum testing, and isolation);
   - OR if `Pleural Effusion` with confidence $p \ge 0.50$ (warranting diagnostic thoracentesis or diuretic management);
   - OR if `Pulmonary Nodule / Mass` with confidence $p \ge 0.40$ (warranting priority oncological workup, contrast CT, and biopsy planning);
   - OR if `Pneumonia` / `COVID-19` exhibit intermediate confidence $0.40 \le p < 0.70$.
3. **Routine Tier (Priority 3 — Standard Workflow)**:
   - Triggered if the predicted class is `Normal` with confidence $p \ge 0.60$ (indicating an absence of acute radiographic consolidations);
   - OR when predictions exhibit high diagnostic entropy across classes, prompting elective radiologist review.

This rule-based stratification enables hospital PACS workflows to dynamically prioritize reading queues, ensuring that acute, life-threatening thoracic opacities are reviewed within minutes of acquisition.

## 4.7 Electronic Health Record Persistence & Audit Trails

To comply with medical software engineering standards, LungAI manages all patient data and analytical records through a normalized relational schema:
- **Patients Entity**: Stores patient demographics, assigned unique identifiers (UUIDv4), medical record numbers, age, biological sex, and clinical history.
- **Scans Entity**: Manages raw image uploads, file hashes (SHA-256 for cryptographic tamper-evidence), acquisition timestamps, and physical image metadata.
- **Predictions Entity**: Persists primary predicted disease, full 6-class softmax probability distributions, calculated urgency triage tier, Grad-CAM visualization file references, and processing latency.
- **Reports Entity**: Stores auto-generated, structured clinical reports incorporating radiologist review notes, timestamped validation signatures, and exportable PDF summaries.

## 4.8 Architectural Comparison: Baseline vs. Proposed

Table 4.1 summarizes the architectural and scientific differences between the initial baseline approach and the proposed LungAI platform.

### Table 4.1: Comprehensive Comparison of Initial Baseline vs. Proposed LungAI System

| Architectural Feature | Initial Baseline System | Proposed LungAI System (Model D) |
| :--- | :--- | :--- |
| **Active Class Taxonomy** | 5 Classes (COVID-19, Normal, Pneumonia, TB, Lung Cancer) | **6 Classes** (COVID-19, Normal, Pleural Effusion, Pneumonia, **Pulmonary Nodule / Mass**, TB) |
| **Oncological Labeling** | Confounded "Lung Cancer" (included 692 axial CT slices) | **Pulmonary Nodule / Mass** (strictly planar CXR nodule/mass findings; no cross-modality contamination) |
| **Dataset Size & Scope** | 10,864 scans (unverified patient partitioning) | **10,547 scans from 10,270 patients** across 8 international repositories (V5 unified manifest) |
| **Patient Leakage** | Unknown / unverified patient overlap | **Strict 0.0% patient leakage** across train, validation, and test partitions |
| **Input Validation** | None (processed any uploaded file or non-CXR image) | **Two-Stage Defense-in-Depth Gate** (Stage 1 biophysical + Stage 2 DenseNet-121 at $\tau=0.83$) |
| **Core Architecture** | Custom CNN baseline vs. ResNet50 | **DenseNet-121 with Dense Bottleneck (256-D)** + frequency-aware spatial filtering |
| **Preprocessing Pipeline** | Standard CLAHE and resizing | **CIE LAB CLAHE + Gaussian spatial low-pass filter ($\sigma=1.0$)** + Lanczos-4 resampling |
| **Domain Generalization** | Assumed from ImageNet pre-training | **Systematically evaluated across 5 paradigms** (ERM, CORAL, DANN, Frequency, Hybrid) |
| **External Benchmark** | None (evaluated only on internal held-out split) | **Quarantined Montgomery County cohort ($N=138$)** evaluated zero-shot with forensic failure analysis |
| **Internal Test Performance**| 95.21% accuracy on blended 5-class split | **82.93% accuracy, 78.35% macro F1, 0.9755 ROC-AUC, 0.8391 PR-AUC** on leak-free 6-class split |
| **Explainability (XAI)** | Offline experimental visualization | **Integrated real-time Grad-CAM** on `conv5_block16_concat` rendered dynamically in UI |
| **Clinical Triage** | Basic probability thresholding | **Deterministic 3-tier urgency stratification** (Emergency, Urgent, Routine) |
| **Software Verification** | Historical 22-test test suite | **Comprehensive 44-test automated test suite** achieving 100% pass rate |

---

# CHAPTER 5 — SYSTEM REQUIREMENTS

## 5.1 Hardware Specifications

The computational demands of the LungAI platform are divided into training/experimental requirements and production inference/deployment requirements.

### 5.1.1 Training and Experimental Environment
- **Central Processing Unit (CPU)**: AMD Ryzen 9 5900X (12 Cores, 24 Threads, base clock 3.7 GHz, boost clock 4.8 GHz) or Intel Core i9-12900K. High thread count is essential for multi-threaded OpenCV image decoding, CLAHE enhancement, and on-the-fly spatial data augmentation during tensor batch generation.
- **Graphical Processing Unit (GPU)**: NVIDIA GeForce RTX 3080 (10 GB GDDR6X VRAM, 8,704 CUDA cores, 272 Tensor Cores, memory bandwidth 760 GB/s) with CUDA Compute Capability 8.6. Dedicated GPU hardware is mandatory for accelerating tensor backpropagation, batch matrix multiplications, and domain adaptation gradient reversal operations.
- **System Memory (RAM)**: 32 GB DDR4-3600 MHz dual-channel memory, providing sufficient bandwidth to cache preprocessed image tensors, dataset manifests, and intermediate activation tensors in RAM during multi-epoch training.
- **Storage Subsystem**: 1 TB NVMe PCIe Gen 4.0 Solid-State Drive (sequential read/write speeds up to 5,000 MB/s), critical for low-latency random I/O when streaming 10,547 high-resolution radiograph files during dataset compilation.

### 5.1.2 Production Inference and Deployment Environment
- **CPU**: 4-Core x86-64 processor (Intel Xeon, AMD EPYC, or modern desktop processor) running at $\ge 2.5$ GHz.
- **GPU (Optional / Recommended)**: NVIDIA T4 Tensor Core GPU (16 GB) or NVIDIA RTX 3060 for high-throughput batch inference; CPU-only execution is fully supported for edge clinics, achieving single-scan inference latencies under 450 ms.
- **RAM**: 8 GB minimum (16 GB recommended) to support concurrent ASGI worker threads, in-memory image tensor allocations, and OpenCV matrix transformations.
- **Storage**: 50 GB solid-state drive space for application binaries, serialized model checkpoints (`.h5` weights $pprox 30$ MB), SQLite/PostgreSQL relational database files, and encrypted local radiograph caches.

## 5.2 Software Environment & Open-Source Stack

The software architecture leverages an established, open-source scientific computing and web application stack:
- **Operating System**: Microsoft Windows 11 Enterprise (x86-64) / Ubuntu 22.04 LTS Linux.
- **Programming Language**: Python 3.10+ / 3.11 for all backend, machine learning, and data engineering subsystems. Node.js (v18+ LTS) with NPM for the frontend single-page application.
- **Deep Learning Frameworks**: TensorFlow 2.15+ and Keras 2.15+, utilizing cuDNN 8.9 and CUDA 12.2 for GPU-accelerated tensor computations.
- **Computer Vision & Image Processing**: OpenCV (v4.8+) for high-speed multi-threaded image decoding, color-space conversions (BGR to CIE LAB), CLAHE enhancement, and Gaussian spatial filtering. NumPy (v1.24+) for vectorized tensor operations. SciPy (v1.11+) for Fourier transform analysis and statistical distance calculations.
- **Backend API & Concurrency**: FastAPI (v0.104+) ASGI web framework, Uvicorn (v0.24+) asynchronous web server, and Pydantic (v2.4+) for rigorous runtime request validation and serialization.
- **Database & Persistence**: SQLAlchemy (v2.0+) Object-Relational Mapping (ORM) with aiosqlite for asynchronous SQLite operations (compatible with PostgreSQL via asyncpg).
- **Frontend Framework**: React 18 single-page application engineered with Vite build tooling, Axios for asynchronous HTTP communication, Lucide React for modern iconography, and Vanilla CSS for high-performance styling without CSS utility library bloat.
- **Verification & Testing**: Pytest (v7.4+) and HTTPX test client for automated unit, integration, and API contract verification.

## 5.3 Functional Requirements

The functional requirements specify the active capabilities that the LungAI system must provide to clinical operators:

- **FR1 (Image Ingestion & Format Decoding)**: The system shall accept planar chest radiograph uploads in JPEG, PNG, and DICOM formats up to a maximum payload ceiling of 10 MB via multipart/form-data HTTP POST requests.
- **FR2 (Automated Input Screening & Non-CXR Rejection)**: The system shall screen all incoming uploads through a two-stage gate. Inputs failing biophysical thresholds or achieving semantic CXR confidence below $	au = 0.83$ shall be rejected with an HTTP 422 response and an informative error message.
- **FR3 (Deterministic Image Preprocessing)**: The system shall automatically preprocess accepted radiographs through CIE LAB CLAHE (clip limit 2.0), Gaussian spatial low-pass filtering ($\sigma=1.0$), Lanczos-4 spatial resampling to $224 	imes 224 	imes 3$, and ImageNet normalization.
- **FR4 (Six-Class Disease Inference)**: The system shall execute Model D inference, returning continuous probability distributions summing to $1.0 \pm 10^{-4}$ across the six active categories: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, and Tuberculosis.
- **FR5 (Real-Time Grad-CAM Saliency Generation)**: The system shall calculate gradient-weighted class activation maps targeting layer `conv5_block16_concat`, generate a normalized color-mapped heatmap, composite the heatmap over the radiograph, and return both image references.
- **FR6 (Deterministic Urgency Triage Stratification)**: The system shall map predicted probabilities into an actionable clinical triage tier: Emergency (Priority 1), Urgent (Priority 2), or Routine (Priority 3).
- **FR7 (Structured Clinical Report Generation)**: The system shall automatically synthesize a structured clinical report containing patient details, top predicted findings, confidence percentages, urgency tier, Grad-CAM visual evidence, and a mandatory academic disclaimer.
- **FR8 (Electronic Health Record & Patient Linking)**: The system shall support patient record management, enabling scans and predictions to be persistently linked to unique patient profiles (UUIDv4).
- **FR9 (Historical Record Auditing & Log Inspection)**: The system shall maintain an auditable, timestamped log of all processed scans, predictions, execution latencies, and triage assignments.
- **FR10 (Scientific Metrics & Model Transparency Dashboard)**: The system shall expose interactive dashboards displaying verified confusion matrices, ROC/PR curves, t-SNE feature projections, and source-held-out metrics.

## 5.4 Non-Functional Requirements

- **NFR1 (Inference Latency & Responsiveness)**: The end-to-end processing latency—encompassing Stage 1/Stage 2 gate verification, preprocessing, Model D inference, Grad-CAM generation, and database persistence—shall not exceed 1,500 milliseconds per radiograph on GPU hardware (and $\le 3,000$ milliseconds on modern multi-core CPU hardware).
- **NFR2 (System Availability & Reliability)**: The API microservice shall maintain $\ge 99.5\%$ operational uptime during normal clinical shifts, featuring stateless ASGI execution and automated process recovery.
- **NFR3 (Defensive Exception Handling & Stability)**: The system shall gracefully intercept all corrupted image streams, truncated byte payloads, and unsupported file extensions without crashing, returning standardized JSON error payloads.
- **NFR4 (Reproducibility & Determinism)**: Image preprocessing, tensor normalization, and model inference must produce bitwise identical output tensors given identical input byte streams.
- **NFR5 (Data Integrity & Relational Consistency)**: Database transactions shall adhere to strict ACID principles, enforcing foreign key integrity across patients, scans, and prediction tables.
- **NFR6 (Security & Privacy Governance)**: The system shall anonymize all stored patient records using non-sequential UUIDv4 identifiers, strip private clinical metadata from uploaded scans, and prevent unauthorized file system traversal.
- **NFR7 (Ergonomic Usability & Accessibility)**: The web interface shall deliver a clean, responsive viewport providing contrast windowing, zooming, panning, and clear visual hierarchy optimized for clinical monitors.
- **NFR8 (Maintainability & Test Coverage)**: The codebase shall maintain comprehensive automated test coverage ($\ge 90\%$ backend code paths) with an automated test suite verifying all system invariants before production deployment.

## 5.5 System Requirements Matrix

Table 5.1 provides a cross-functional mapping of all system requirements against their architectural realization and verification mechanisms.

### Table 5.1: Cross-Functional System Requirements Matrix

| Req ID | Requirement Type | Functional Description | Implementing Component | Verification Mechanism |
| :---: | :--- | :--- | :--- | :--- |
| **FR1** | Functional | Ingestion & validation of JPEG/PNG/DICOM byte streams up to 10MB | `backend/routers/predict.py` | Automated payload test suite (HTTP 413 on >10MB) |
| **FR2** | Functional | Two-stage screening; rejection of non-CXR inputs at $\tau=0.83$ | `backend/ml/cxr_gate.py` | 125-image evaluation cohort & adversarial test cases |
| **FR3** | Functional | Deterministic CIE LAB CLAHE + Gaussian LP $\sigma=1.0$ + Lanczos-4 | `backend/ml/preprocess.py` | Unit test verifying output shape $(1, 224, 224, 3)$ |
| **FR4** | Functional | Model D 6-class softmax probability inference | `backend/ml/inference.py` | Prediction test verifying sum of probabilities $\approx 1.0$ |
| **FR5** | Functional | Grad-CAM saliency generation on `conv5_block16_concat` | `backend/ml/gradcam.py` | Saliency overlay generation test & non-zero heatmap check |
| **FR6** | Functional | Deterministic 3-tier clinical urgency triage stratification | `backend/ml/inference.py` | Unit tests covering Emergency, Urgent, and Routine rules |
| **FR7** | Functional | Automated structured clinical PDF/JSON report generation | `backend/routers/reports.py` | Report generation API endpoint test |
| **FR8** | Functional | Relational persistence of patient profiles (UUIDv4) | `backend/models/patient.py` | CRUD integration test suite across patient endpoints |
| **FR9** | Functional | Auditable historical scan log with timestamped records | `backend/models/prediction.py` | Relational query test verifying foreign key constraints |
| **FR10**| Functional | Frontend scientific dashboard for metrics and t-SNE | `frontend/src/pages/MetricsPage.jsx` | UI rendering test and visual verification |
| **NFR1**| Non-Functional | Total latency $\le 1,500$ ms (GPU) / $\le 3,000$ ms (CPU) | `InferenceEngine` singleton | Automated performance benchmarking test |
| **NFR2**| Non-Functional | System availability $\ge 99.5\%$ with stateless ASGI worker | `backend/main.py` + Uvicorn | Health-check endpoint `/api/v1/health` monitoring |
| **NFR3**| Non-Functional | Defensive exception handling on corrupt/truncated streams | FastAPI exception handlers | Tests simulating truncated streams and corrupt headers |
| **NFR4**| Non-Functional | Bitwise reproducibility across deterministic preprocessing | `backend/ml/preprocess.py` | Bitwise hash comparison on repeated preprocessing runs |
| **NFR5**| Non-Functional | ACID relational integrity and cascading foreign keys | SQLAlchemy ORM & SQLite engine | Database rollback and transaction integrity test suite |
| **NFR6**| Non-Functional | Privacy-conscious de-identification and UUID handling | `Patient` model & file sanitizer | Multi-pattern secret scanner & metadata stripping test |
| **NFR7**| Non-Functional | Diagnostic viewport ergonomics (zoom, pan, windowing) | `frontend/src/components/Viewport.jsx` | End-to-end browser walkthrough test |
| **NFR8**| Non-Functional | Automated verification suite with 100% pass rate | `backend/tests/test_*.py` | 44/44 automated Pytest execution suite |

---

# CHAPTER 6 — SYSTEM DESIGN

## 6.1 Data Flow Diagrams (DFDs)

Data Flow Diagrams (DFDs) provide a hierarchical, graphical representation of information flow through the LungAI system, tracing raw radiograph bytes from external client upload through multi-stage validation, inference, explainability, and database persistence.

### 6.1.1 DFD Level 0 (Context-Level Diagram)
The Context-Level DFD establishes the operational boundary of the LungAI platform. The primary external entities are the **Clinical Operator / Radiologist** and the **System Administrator**. 

The clinical operator submits raw chest radiographs and patient identifiers, receiving validated disease classifications, Grad-CAM heatmaps, urgency triage levels, and structured clinical documentation. The administrator oversees model checkpoints, performance logs, and relational audit trails.

![DFD Level 0](docs/generated_figures/fig_6_1_dfd_level_0.png)
*Figure 6.1: DFD Level 0 (Context Diagram), delineating external interactions between clinical operators, administrative entities, and the LungAI platform.*

### 6.1.2 DFD Level 1 (Macro Subsystem Process Decomposition)
DFD Level 1 decomposes the platform into five core operational processes:
- **Process 1.0 (Input Ingestion & Validation Gate)**: Intercepts raw multipart payloads, executes Stage 1 biophysical filtering and Stage 2 DenseNet-121 anatomical verification, isolating invalid files.
- **Process 2.0 (Deterministic Radiographic Preprocessing)**: Performs CIE LAB conversion, luminance CLAHE enhancement, Gaussian low-pass spatial filtering ($\sigma=1.0$), Lanczos-4 resampling, and ImageNet standardization.
- **Process 3.0 (Model D Deep Inference Engine)**: Executes the 121-layer Densely Connected Network to generate continuous 6-class softmax probabilities.
- **Process 4.0 (Visual Explainability & Triage Synthesis)**: Calculates Grad-CAM heatmaps on `conv5_block16_concat` and evaluates deterministic clinical urgency rules.
- **Process 5.0 (Persistence & Record Management)**: Coordinates relational ORM transactions across Patients, Scans, Predictions, and Reports data stores.

![DFD Level 1](docs/generated_figures/fig_6_2_dfd_level_1.png)
*Figure 6.2: DFD Level 1, detailing the decomposition of input ingestion, preprocessing, deep inference, explainability generation, and relational persistence.*

### 6.1.3 DFD Level 2 (Inference & Explainability Detail)
DFD Level 2 provides granular decomposition of Process 3.0 and Process 4.0, detailing the internal flow of activation tensors, backpropagated class gradients, Grad-CAM alpha compositing, and triage decision trees.

![DFD Level 2](docs/generated_figures/fig_6_3_dfd_level_2.png)
*Figure 6.3: DFD Level 2, decomposing the internal tensor operations, feature map extraction, gradient pooling, and urgency logic.*

## 6.2 UML Structural Modeling

Unified Modeling Language (UML) structural diagrams define the static organization of software entities, modules, and hardware deployment nodes comprising LungAI.

### 6.2.1 Use Case Diagram
The Use Case Diagram models functional interactions supported by the platform:
- Uploading and validating chest radiographs;
- Inspecting multi-class prediction probabilities and confidence distributions;
- Visualizing Grad-CAM attention heatmaps;
- Adjusting DICOM viewport windowing and contrast;
- Generating and exporting structured clinical PDF reports;
- Registering and managing patient demographic profiles;
- Reviewing historical scan logs and audit records;
- Auditing scientific metrics and confusion matrices.

![UML Use Case Diagram](docs/generated_figures/fig_6_4_use_case_diagram.png)
*Figure 6.4: UML Use Case Diagram, mapping clinical and administrative user roles to platform capabilities.*

### 6.2.2 Class Diagram
The Class Diagram models the object-oriented structure of the backend application:
- `InferenceEngine`: Implements the Singleton pattern, managing model loading, GPU memory allocation, and thread-safe forward passes.
- `CXRGate`: Encapsulates Stage 1 biophysical screening and Stage 2 deep anatomical verification.
- `GradCAMGenerator`: Computes feature activations, backpropagated gradients, and alpha-blended heatmap overlays.
- Relational ORM Entities: `Patient`, `Scan`, `Prediction`, and `Report` classes defining database attributes and entity relationships.
- Service Routers: API controllers handling HTTP serialization, validation, and response formatting.

![UML Class Diagram](docs/generated_figures/fig_6_5_class_diagram.png)
*Figure 6.5: UML Class Diagram, delineating class relationships, service singletons, and ORM entity models.*

### 6.2.3 Component Diagram
The Component Diagram details modular encapsulation across presentation, microservice, and database tiers.

![UML Component Diagram](docs/generated_figures/fig_6_8_component_diagram.png)
*Figure 6.6: UML Component Diagram, illustrating decoupled software modules, interfaces, and API contracts.*

### 6.2.4 Deployment Diagram
The Deployment Diagram models the physical and network infrastructure of the platform, showing client web browsers communicating via TLS-encrypted HTTP/2 with the ASGI web server, backend Python runtime, GPU acceleration libraries (CUDA/cuDNN), and persistent storage volumes.

![UML Deployment Diagram](docs/diagrams/rendered/deployment_diagram.png)
*Figure 6.7: UML Deployment Diagram, illustrating distributed deployment across client browsers, FastAPI ASGI server, GPU compute node, and relational database.*

## 6.3 UML Behavioral Modeling

Behavioral models document dynamic operational workflows, object collaborations, and state transitions during system execution.

### 6.3.1 Sequence Diagram
The Sequence Diagram traces the end-to-end temporal execution flow triggered when a clinician uploads a radiograph:
1. `Client Viewport` dispatches multipart image stream to `POST /api/v1/predict`.
2. `PredictRouter` intercepts request and passes payload to `CXRGate`.
3. `CXRGate` executes Stage 1 biophysical checks. If valid, executes Stage 2 deep semantic screening.
4. Upon gate approval, payload passes to `PreprocessModule` for CLAHE, Gaussian low-pass filtering, and normalization.
5. Standardized tensor is dispatched to the `InferenceEngine` singleton.
6. Model D computes forward pass; softmax probabilities are routed to `GradCAMGenerator` and `TriageEngine`.
7. `GradCAMGenerator` computes gradient backpropagation on `conv5_block16_concat` and generates image overlay.
8. `DatabaseManager` executes an atomic transaction persisting scan, prediction, and triage records.
9. Structured JSON response is returned to `Client Viewport` for rendering.

![UML Sequence Diagram](docs/diagrams/rendered/application_sequence.png)
*Figure 6.8: UML Sequence Diagram, detailing asynchronous object interactions from client upload to persistent report delivery.*

### 6.3.2 Activity Diagram
The Activity Diagram details control logic, conditional branches, validation loops, and defensive error handlers governing request execution.

![UML Activity Diagram](docs/generated_figures/fig_6_7_activity_diagram.png)
*Figure 6.9: UML Activity Diagram, showing decision pathways for gate validation, fallback routines, and urgency calculation.*

## 6.4 Relational Database Schema & Entity-Relationship Design

The persistence tier is engineered using a normalized relational schema (Third Normal Form, 3NF) managed via SQLAlchemy ORM. Table 6.1 defines the relational entities, primary keys, foreign keys, and integrity constraints.

### Table 6.1: Relational Database Schema Specification

| Table Name | Column Name | Data Type | Constraints | Description / Business Logic |
| :--- | :--- | :--- | :--- | :--- |
| **patients** | `id` | VARCHAR(36) | PRIMARY KEY | Unique patient identifier (UUIDv4) |
| | `patient_id` | VARCHAR(64) | UNIQUE, NOT NULL | Hospital-assigned medical record number (MRN) |
| | `name` | VARCHAR(128) | NOT NULL | Patient legal name (de-identified in public exports) |
| | `age` | INTEGER | CHECK (age >= 0 AND age <= 125) | Patient age at registration |
| | `gender` | VARCHAR(16) | NOT NULL | Biological sex (Male, Female, Other) |
| | `created_at` | DATETIME | NOT NULL, DEFAULT UTC_NOW | Timestamp of patient profile creation |
| **scans** | `id` | VARCHAR(36) | PRIMARY KEY | Unique scan identifier (UUIDv4) |
| | `patient_id` | VARCHAR(36) | FOREIGN KEY -> patients(id) | Associated patient profile (ON DELETE CASCADE) |
| | `file_path` | VARCHAR(512) | NOT NULL | File system path to stored radiograph |
| | `file_hash` | VARCHAR(64) | NOT NULL | SHA-256 cryptographic hash of image bytes |
| | `width` | INTEGER | NOT NULL | Native image pixel width |
| | `height` | INTEGER | NOT NULL | Native image pixel height |
| | `uploaded_at` | DATETIME | NOT NULL, DEFAULT UTC_NOW | Image upload timestamp |
| **predictions** | `id` | VARCHAR(36) | PRIMARY KEY | Unique prediction identifier (UUIDv4) |
| | `scan_id` | VARCHAR(36) | FOREIGN KEY -> scans(id) | Analyzed radiograph (ON DELETE CASCADE) |
| | `model_name` | VARCHAR(64) | NOT NULL | Inference engine identifier (e.g., Model_D_DenseNet121) |
| | `predicted_class`| VARCHAR(64) | NOT NULL | Primary diagnostic category with top confidence |
| | `confidence` | FLOAT | CHECK (confidence >= 0.0 AND <= 1.0) | Top class probability score |
| | `probabilities` | JSON | NOT NULL | Complete 6-class softmax probability distribution |
| | `urgency_tier` | VARCHAR(32) | NOT NULL | Stratified clinical priority (Emergency, Urgent, Routine) |
| | `gradcam_path` | VARCHAR(512) | NULLABLE | File path to rendered Grad-CAM visual overlay |
| | `execution_time`| FLOAT | NOT NULL | Total end-to-end processing latency in milliseconds |
| | `created_at` | DATETIME | NOT NULL, DEFAULT UTC_NOW | Timestamp of prediction execution |
| **reports** | `id` | VARCHAR(36) | PRIMARY KEY | Unique clinical report identifier (UUIDv4) |
| | `prediction_id`| VARCHAR(36) | FOREIGN KEY -> predictions(id) | Source prediction record (ON DELETE CASCADE) |
| | `radiologist_notes`| TEXT | NULLABLE | Clinical review notes added by practitioner |
| | `is_validated` | BOOLEAN | NOT NULL, DEFAULT FALSE | Flag indicating certified human radiologist sign-off |
| | `pdf_path` | VARCHAR(512) | NULLABLE | File path to generated clinical PDF summary |
| | `generated_at` | DATETIME | NOT NULL, DEFAULT UTC_NOW | Timestamp of report generation |

Figure 6.10 shows the complete Entity-Relationship (ER) diagram illustrating cardinality and cascading delete constraints.

![Entity Relationship Diagram](docs/diagrams/rendered/database_er.png)
*Figure 6.10: Relational Entity-Relationship (ER) Diagram, detailing normalized schemas for patients, scans, predictions, and reports.*

## 6.5 Machine Learning Inference Pipeline Architecture

Figure 6.11 presents the end-to-end machine learning inference pipeline architecture, illustrating how raw radiograph bytes flow sequentially through the two-stage CXR validation gate, biophysical preprocessing, Model D forward inference, Grad-CAM saliency computation, and clinical urgency stratification.

![ML Inference Pipeline](docs/diagrams/rendered/ml_inference_pipeline.png)
*Figure 6.11: Machine Learning Inference Pipeline, showing sequential data transformations from raw byte ingestion to clinical urgency stratification.*

---

# CHAPTER 7 — DATASET AND DATA PREPROCESSING

## 7.1 Multi-Source Corpus Reconstruction History

The empirical foundation of this dissertation underwent a multi-stage forensic evolution. In the initial phase of the research, an uncurated five-class dataset was compiled by combining public collections from Kaggle and GitHub repositories. However, comprehensive forensic auditing uncovered critical methodological vulnerabilities:
1. **Cross-Modality Contamination**: The fifth class ("Lung Cancer") had inadvertently incorporated 692 axial Computed Tomography (CT) slices from the IQ-OTH/NCCD dataset. Merging cross-sectional CT slices with planar projection radiographs introduced an impermissible physical confounder: neural networks learned to identify CT acquisition geometry (circular scan borders and soft-tissue windowing) rather than pulmonary oncological pathology.
2. **Patient Identity Leakage**: Initial partitions performed random image-level splits without patient de-identification metadata. Consequently, multiple radiographs from the same patient appeared simultaneously in training and test splits, causing test metrics to evaluate patient re-identification rather than true pathological generalization.
3. **Severe Sensor Overfitting**: Zero-shot auditing on independent hospital datasets revealed catastrophic performance collapse, caused by models learning repository-specific high-frequency noise textures and edge padding.

To resolve these vulnerabilities, the dataset was systematically audited and reconstructed across five major iterations (V1 through V5), culminating in the frozen canonical V5 benchmark. Table 7.1 details the reconstruction history and forensic interventions executed across each iteration.

### Table 7.1: Dataset Reconstruction and Forensic Evolution History

| Dataset Version | Total Scans | Diagnostic Classes | Data Sources | Key Forensic Problem Identified | Corrective Intervention Executed |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **V1** | 13,102 | 6 Classes | 7 Repositories | Contained 692 axial CT slices; high image duplication; unverified patient IDs | Flagged cross-modality contamination; initiated hash-level deduplication |
| **V2** | 11,845 | 6 Classes | 7 Repositories | Retained CT slices; severe source-label confounding across Kaggle pools | Purged 1,257 duplicate scans; standardized file naming conventions |
| **V3** | 12,410 | 6 Classes | 8 Repositories | JSRT included benign solitary nodules; identity leakage across patient series | Purged 100 benign JSRT cases; integrated TBX11K tuberculosis cohort |
| **V4** | 10,864 | 5 Classes | 6 Repositories | Temporary 5-class baseline; still contained CT slices labeled as "Lung Cancer" | Completely purged all 692 axial CT slices; eliminated cross-modality confounder |
| **V5 (Final Canonical)** | **10,547** | **6 Classes** | **8 Repositories** | Multi-source imbalance; Cramér's V = 0.7654; potential digitizer shift | Replaced "Lung Cancer" with verified CXR **Pulmonary Nodule / Mass**; integrated NIH & VinDr; enforced **strict patient-level splitting (10,270 unique patients, 0% patient leakage)**; quarantined Montgomery cohort ($N=138$) |

## 7.2 Unified V5 Cohort Composition

The final canonical V5 dataset comprises **10,547 verified planar chest radiographs** acquired from **10,270 unique clinical patients** across eight international medical imaging repositories. 

To eliminate data leakage, the cohort was partitioned strictly at the patient level using an MD5-hashed patient identifier allocation. Radiographs from any single patient were assigned exclusively to a single split, guaranteeing zero identity overlap. Table 7.2 presents the official partition distribution.

### Table 7.2: Canonical Unified V5 Dataset Partition Distribution

| Dataset Partition | Total Radiographs | Unique Patients | Proportion (Scans) | Proportion (Patients) | Patient Leakage Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | 7,398 | 7,189 | 70.14% | 70.00% | **0.0% (Zero Overlap)** |
| **Validation Split** | 1,579 | 1,540 | 14.97% | 15.00% | **0.0% (Zero Overlap)** |
| **Internal Test Split**| 1,570 | 1,541 | 14.89% | 15.00% | **0.0% (Zero Overlap)** |
| **Total V5 Cohort** | **10,547** | **10,270** | **100.0%** | **100.0%** | **0.0% Verified** |
| *External Benchmark (Montgomery)* | *138* | *138* | *Quarantined* | *Quarantined* | *100% Isolated* |

## 7.3 Diagnostic Class Definitions & Taxonomy

The V5 taxonomy categorizes radiographs into six clinically distinct, non-overlapping diagnostic findings:
1. **COVID-19 (1,942 scans)**: Radiographs exhibiting documented manifestations of coronavirus disease 2019, including bilateral peripheral ground-glass opacities, vascular thickening, and lower-zone consolidation patterns.
2. **Normal (2,636 scans)**: Healthy control radiographs verified as free from active consolidations, acute infiltrates, pleural effusions, pneumothorax, or parenchymal masses.
3. **Pleural Effusion (1,062 scans)**: Radiographs exhibiting blunting of the lateral or anterior costophrenic angles, meniscus signs, or large homogeneous basal opacifications representing pathological fluid accumulation in the pleural space.
4. **Pneumonia (2,805 scans)**: Bacterial, viral (non-COVID), or fungal pulmonary infections manifesting as localized lobar airspace consolidation, bronchopneumonia patches, or interstitial reticular opacities.
5. **Pulmonary Nodule / Mass (754 scans)**: Planar radiographs exhibiting discrete, well-defined or irregular parenchymal opacities ($<3$ cm for nodules, $>3$ cm for masses). **Crucially, this class is formally designated as "Pulmonary Nodule / Mass" rather than "Lung Cancer"** because planar radiography detects optical density alterations; histological confirmation of bronchogenic malignancy requires tissue biopsy or cytological examination.
6. **Tuberculosis (1,348 scans)**: Active pulmonary tuberculosis displaying classic upper-lobe fibro-cavitary lesions, apical parenchymal infiltrates, mediastinal/hilar lymphadenopathy, or widespread miliary patterns.

Table 7.3 summarizes the distribution across the six active diagnostic categories.

### Table 7.3: Diagnostic Class Distribution of the Canonical V5 Cohort

| Class Index | Diagnostic Category | Total Scans | Proportion (%) | Primary Radiographic Characteristics |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **COVID-19** | 1,942 | 18.41% | Bilateral, peripheral, subpleural ground-glass opacities and patchy consolidations |
| **1** | **Normal** | 2,636 | 24.99% | Clear lung fields, sharp costophrenic angles, normal cardiothoracic ratio |
| **2** | **Pleural Effusion** | 1,062 | 10.07% | Costophrenic sulcus blunting, fluid meniscus sign, homogeneous basal opacity |
| **3** | **Pneumonia** | 2,805 | 26.60% | Lobar or segmental alveolar consolidation, air bronchograms, patchy infiltrates |
| **4** | **Pulmonary Nodule / Mass** | 754 | 7.15% | Well-circumscribed or lobulated radiopaque soft-tissue lesions ($<3$ cm / $>3$ cm) |
| **5** | **Tuberculosis** | 1,348 | 12.78% | Apical fibro-cavitary lesions, parenchymal consolidation, miliary micronodules |
| **—** | **Total Cohort** | **10,547** | **100.0%** | **Comprehensive multi-class thoracic representation** |

## 7.4 Multi-Source Provenance & Distribution

To ensure broad geographic and hardware diversity, the V5 dataset harmonizes scans from eight independent international acquisition sources:
- **TBX11K (3,277 scans)**: High-resolution digital radiograph collection compiled for tuberculosis research, providing Normal, Pneumonia, and Tuberculosis cases.
- **Existing COVID-19 Collection (1,942 scans)**: Curated multi-center open-access repository of viral COVID-19 radiographs.
- **VinBigData VinDr-CXR (1,467 scans)**: High-quality Vietnamese multi-hospital clinical dataset annotated by seventeen certified radiologists, providing verified Pleural Effusion and Pulmonary Nodule / Mass cases.
- **Existing Pneumonia Collection (1,395 scans)**: Pediatric and adult clinical pneumonia scans from public repositories.
- **Existing Normal Collection (1,199 scans)**: Healthy control radiographs.
- **Existing Tuberculosis Collection (665 scans)**: Clinical tuberculosis cases.
- **NIH ChestX-ray14 (457 scans)**: Frontal radiographs from the NIH Clinical Center, providing supplemental Pleural Effusion and Nodule cases.
- **JSRT (145 scans)**: Japanese Society of Radiological Technology standard digital database, providing verified malignant and benign solitary pulmonary nodules with normal controls.

Table 7.4 details the multi-source composition across classes and acquisition repositories.

### Table 7.4: Multi-Source Repository Cross-Tabulation Matrix

| Acquisition Source Repository | COVID-19 | Normal | Pleural Effusion | Pneumonia | Pulmonary Nodule / Mass | Tuberculosis | Total Scans |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TBX11K** | 0 | 1,176 | 0 | 1,410 | 0 | 691 | **3,277** |
| **Existing COVID-19** | 1,942 | 0 | 0 | 0 | 0 | 0 | **1,942** |
| **VinBigData VinDr-CXR** | 0 | 0 | 903 | 0 | 564 | 0 | **1,467** |
| **Existing Pneumonia** | 0 | 0 | 0 | 1,395 | 0 | 0 | **1,395** |
| **Existing Normal** | 0 | 1,199 | 0 | 0 | 0 | 0 | **1,199** |
| **Existing Tuberculosis** | 0 | 0 | 0 | 0 | 0 | 665 | **665** |
| **NIH ChestX-ray14** | 0 | 211 | 159 | 0 | 87 | 0 | **457** |
| **JSRT (Japan)** | 0 | 50 | 0 | 0 | 95 | 0 | **145** |
| **Total Cohort** | **1,942** | **2,636** | **1,062** | **2,805** | **754** | **1,348** | **10,547** |

## 7.5 Source-Class Confounding Analysis

A critical scientific finding documented in Table 7.4 is the presence of severe **source-class confounding**. In multi-source medical image collections, certain disease categories originate predominantly or exclusively from specific repositories. For example, all 1,942 COVID-19 cases originate from the Existing COVID-19 repository, while all Pleural Effusion cases originate from VinBigData (903) and NIH (159).

To quantify the degree of association between the data source variable $S$ and the disease category variable $Y$, the contingency table was evaluated using **Cramér's V** statistic:
$$V = \sqrt{rac{\chi^2}{N \cdot \min(r-1, c-1)}}$$
where $\chi^2$ is the chi-square statistic from the $r 	imes c$ contingency table ($r=8$ sources, $c=6$ classes) and $N=10,547$.

The evaluation yielded:
$$\chi^2 = 30,891.4, \quad V = 0.7654$$
In statistical theory, a Cramér's V value exceeding 0.50 denotes an exceptionally strong correlation. At $V = 0.7654$, the data source and the disease label are heavily confounded. This proves that an unconstrained deep convolutional network trained via standard empirical risk minimization will inevitably learn source-specific shortcuts (e.g., repository border padding, resolution differences, or digitizer noise) rather than genuine pathology, rigorously motivating the domain generalization and frequency-filtering investigations of Phase 4.

## 7.6 Forensic Data Quality & Leakage Audit

Table 7.5 summarizes the comprehensive data quality control and forensic leakage audit executed across the cohort.

### Table 7.5: Data Quality Control and Forensic Audit Summary

| Forensic Audit Dimension | Pre-Audit Finding | Audit Methodology | Corrective Action & Verification | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Cross-Modality Contamination** | 692 axial CT slices in Lung Cancer class | File header inspection & aspect-ratio screening | Completely purged all 692 CT slices; redefined class as Pulmonary Nodule / Mass | **PASS** |
| **Exact Duplicate Scans** | 1,257 identical image files across folders | SHA-256 cryptographic hashing | Deduplicated; only unique cryptographic instances retained | **PASS** |
| **Perceptual Duplicate Scans** | Rescaled/re-compressed copies of identical scans | Difference Hashing (dHash) & structural similarity | Flagged and removed near-identical scans ($d_{	ext{Hamming}} \le 2$) | **PASS** |
| **Patient Identity Leakage** | Patient series split randomly between train/test | Metadata string parsing & cross-split matching | Enforced strict patient-level partition ($N=10,270$, 0% patient overlap) | **PASS** |
| **Benign Nodule Contamination** | 100 benign nodules in JSRT collection | Clinical metadata review | Purged 100 benign cases to preserve radiographic consistency | **PASS** |
| **External Cohort Isolation** | Risk of Montgomery scans contaminating V5 | Filename and hash matching against V5 | Quarantined Montgomery ($N=138$) completely; zero overlap with V5 | **PASS** |

## 7.7 Quarantined External Benchmark: Montgomery County

To evaluate true zero-shot external domain generalization, the **Montgomery County Chest X-ray Dataset** was established as a quarantined, out-of-distribution benchmark. Collected in collaboration with the Department of Health and Human Services of Montgomery County, Maryland, USA [23], the cohort consists of **138 frontal chest radiographs** (58 active pulmonary tuberculosis cases and 80 healthy normal controls).

The Montgomery cohort was physically and logically quarantined from all training, validation, and hyperparameter tuning pipelines:
- Zero Montgomery scans appear in the V5 manifest.
- All 138 scans are digitized analog radiographs acquired using an optical film digitizer, creating substantial hardware and sensor distribution shifts relative to modern digital detectors.
- This cohort serves strictly as a clinical boundary test to determine whether models can generalize to an unseen health jurisdiction.

## 7.8 Deterministic Radiographic Preprocessing Pipeline

To prepare raw, uncalibrated radiographs for deep neural feature extraction while suppressing high-frequency scanner shortcuts, a deterministic biophysical preprocessing pipeline was engineered:

1. **Chromatic-Luminance Decoupling**: Raw byte streams are decoded to BGR format and converted to the CIE LAB color space ($L$: lightness/luminance, $A$: green-red opponent, $B$: blue-yellow opponent). Decoupling luminance from chrominance ensures that contrast enhancement operations operate strictly on physical photon attenuation levels without introducing false color fringe artifacts.
2. **Contrast Limited Adaptive Histogram Equalization (CLAHE)**: Planar radiographs frequently suffer from underexposed retrocardiac zones or overexposed peripheral lung apices. Standard global histogram equalization over-amplifies noise in homogeneous regions. CLAHE operates on localized $8 	imes 8$ contextual tiles. A contrast clip limit of 2.0 is enforced: local histogram bins exceeding the ceiling are clipped and uniformly redistributed across the grayscale histogram prior to computing the cumulative distribution function (CDF). Bilinear interpolation between neighboring tiles eliminates boundary artifacts.
3. **Gaussian Spatial Low-Pass Filtering ($\sigma=1.0$)**: To decouple convolutional feature learning from high-frequency film digitizer grain and scanner-specific sensor noise, a 2D Gaussian low-pass spatial kernel is applied:
   $$G(x, y) = rac{1}{2\pi\sigma^2} \exp\left(-rac{x^2 + y^2}{2\sigma^2}
ight)$$
   Systematic validation grid tuning established that $\sigma = 1.0$ (using a $3 	imes 3$ kernel) successfully attenuates high-frequency noise shortcuts without blurring clinically significant anatomical structures (such as parenchymal consolidations, cavity walls, and nodule margins).
4. **Lanczos-4 Spatial Resampling**: Images are resized to the standardized input resolution of $224 	imes 224$ pixels using high-order Lanczos-4 interpolation (based on an 8-lobed sinc filter), preserving smooth boundary transitions and minimizing aliasing artifacts.
5. **Two-Step Intensity Normalization**: Pixel values are mapped from $[0, 255]$ to the interval $[0.0, 1.0]$ and standardized channel-wise using ImageNet population statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$):
   $$	ilde{X}_{c} = rac{X_c / 255.0 - \mu_c}{\sigma_c}, \quad c \in \{R, G, B\}$$

Figure 7.1 illustrates the sequential visual transformations of the preprocessing pipeline.

![Preprocessing Pipeline](docs/generated_figures/fig_7_1_preprocessing_pipeline.png)
*Figure 7.1: Deterministic Radiographic Preprocessing Pipeline, showing raw ingestion, CIE LAB CLAHE enhancement, Gaussian low-pass spatial filtering, Lanczos-4 resampling, and channel standardization.*

---

# CHAPTER 8 — MACHINE LEARNING / AI MODEL

## 8.1 Theoretical Foundations of Deep Convolutional Networks

Convolutional Neural Networks (CNNs) model high-dimensional image tensors by enforcing two core architectural inductive biases: **local receptive fields** and **weight sharing** (spatial translation equivariance). 

For an input feature map $X \in \mathbb{R}^{H 	imes W 	imes C_{	ext{in}}}$, a convolutional layer applies a bank of $C_{	ext{out}}$ learnable kernels $W \in \mathbb{R}^{K 	imes K 	imes C_{	ext{in}} 	imes C_{	ext{out}}}$ and bias terms $b \in \mathbb{R}^{C_{	ext{out}}}$:
$$Z(i, j, k) = \sum_{m=-M}^M \sum_{n=-N}^N \sum_{c=1}^{C_{	ext{in}}} X(i+m, j+n, c) \cdot W(m, n, c, k) + b(k)$$
followed by an element-wise non-linear activation function $A = \sigma(Z)$ (typically Rectified Linear Unit, $	ext{ReLU}(z) = \max(0, z)$). By stacking convolutional operations hierarchically, early layers capture high-frequency edges and gradients, while deeper layers synthesize these features into macroscopic thoracic anatomical representations.

## 8.2 DenseNet-121 Backbone & Feature Reuse

LungAI adopts the 121-layer Densely Connected Convolutional Network (**DenseNet-121**) [15] as its canonical deep feature extraction backbone. 

In traditional architectures (e.g., AlexNet or VGG), layer $l$ receives the output of layer $l-1$: $x_l = H_l(x_{l-1})$. In ResNet [16], an additive identity bypass is introduced: $x_l = H_l(x_{l-1}) + x_{l-1}$. While residual connections preserve gradient flow, summation can impede information flow across disparate abstraction levels.

DenseNet resolves this by establishing direct feed-forward connections from every layer to all subsequent layers within a dense block:
$$x_l = H_l([x_0, x_1, x_2, \dots, x_{l-1}])$$
where $[x_0, x_1, \dots, x_{l-1}]$ represents the channel-wise concatenation of feature maps produced by all preceding layers $0, \dots, l-1$. The non-linear transformation $H_l(\cdot)$ is defined as a composite function of three consecutive operations: Batch Normalization (BN), Rectified Linear Unit (ReLU), and a $3 	imes 3$ Convolution (Conv), preceded by a $1 	imes 1$ Convolution bottleneck to reduce channel dimensionality.

DenseNet-121 consists of four dense blocks containing 6, 12, 24, and 16 dense layers, respectively, separated by transition layers comprising a $1 	imes 1$ convolution and $2 	imes 2$ average pooling with a compression factor $	heta = 0.5$. This design yields three decisive advantages for thoracic radiography:
1. **Multi-Scale Feature Persistence**: Low-level edge features (critical for recognizing sharp nodule margins and pleural fluid lines) remain accessible alongside high-level semantic abstractions (critical for broad lobar consolidations).
2. **Mitigation of Vanishing Gradients**: Direct connections ensure that error gradients propagate directly from the loss function to early convolutional layers during backpropagation.
3. **Substantial Parameter Efficiency**: Dense concatenation prevents the need to relearn redundant feature maps, enabling DenseNet-121 (approximately 7.0 million base parameters) to outperform substantially heavier networks (e.g., ResNet50 at 23.5 million parameters) on medical imaging tasks.

## 8.3 Model D Architectural Specification

Model D integrates the ImageNet pre-trained DenseNet-121 backbone with a specialized classification head designed for regularized representation learning. Table 8.1 delineates the structural layer specifications of Model D.

### Table 8.1: Model D Architectural Layer Specifications

| Layer Index | Component / Layer Name | Output Tensor Shape | Parameter Count | Operational Function & Activation |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **Input Tensor** | $(224, 224, 3)$ | 0 | Preprocessed standardized 3-channel radiograph |
| **1** | **DenseNet-121 Base** | $(7, 7, 1024)$ | 7,037,504 | Pre-trained feature extractor (Dense Blocks 1–4) |
| **—** | *Layer `conv5_block16_concat`* | $(7, 7, 1024)$ | — | *Target layer for Grad-CAM visual explainability* |
| **2** | **Global Average Pooling** | $(1024)$ | 0 | Spatial dimensional reduction across $7 	imes 7$ feature maps |
| **3** | **Batch Normalization** | $(1024)$ | 4,096 | Normalizes latent feature activations; stabilizes training |
| **4** | **Dense Projection Layer** | $(256)$ | 262,400 | Dimensionality reduction to 256-D latent space; ReLU |
| **5** | **Dropout Layer ($p=0.3$)** | $(256)$ | 0 | Regularization; randomly drops 30% of activations |
| **6** | **Dense Classification Head**| $(6)$ | 1,542 | Linear transformation to 6 disease logits |
| **7** | **Softmax Activation** | $(6)$ | 0 | Normalizes logits into probability simplex $\hat{y} \in \Delta^5$ |
| **Total** | **Model D Full Architecture**| — | **7,305,542** | **Complete end-to-end inference network (28.71 MB)** |

## 8.4 Mathematical Formulations

### 8.4.1 Categorical Cross-Entropy Loss
For a multi-class classification problem with $K = 6$ classes, let $y \in \{0, 1\}^K$ denote the one-hot ground-truth indicator vector, and $\hat{y} \in [0, 1]^K$ denote the predicted probability distribution output by the softmax function:
$$\hat{y}_k = rac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}$$
The empirical categorical cross-entropy loss $\mathcal{L}_{	ext{CE}}$ over a mini-batch of $N$ samples is:
$$\mathcal{L}_{	ext{CE}} = -rac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{i,k} \log(\hat{y}_{i,k})$$

### 8.4.2 Inverse-Frequency Class Weighting
To counter severe class imbalance (e.g., 2,805 Pneumonia vs. 754 Pulmonary Nodule / Mass scans), cost-sensitive loss weighting is incorporated:
$$w_k = rac{N_{	ext{total}}}{K \cdot N_k}$$
$$\mathcal{L}_{	ext{WCE}} = -rac{1}{N} \sum_{i=1}^N \sum_{k=1}^K w_k \cdot y_{i,k} \log(\hat{y}_{i,k})$$
where $N_k$ is the total number of training samples belonging to class $k$.

### 8.4.3 Gaussian Spatial Low-Pass Filtering
The continuous isotropic 2D Gaussian kernel is defined as:
$$G(x, y; \sigma) = rac{1}{2\pi\sigma^2} \exp\left(-rac{x^2 + y^2}{2\sigma^2}
ight)$$
In discrete spatial convolution, the filtered image tensor $I_{	ext{filtered}}$ is computed via 2D spatial convolution over local window $\Omega$:
$$I_{	ext{filtered}}(u, v) = \sum_{m=-1}^1 \sum_{n=-1}^1 I(u+m, v+n) \cdot G(m, n; \sigma=1.0)$$
In the spatial frequency domain (via 2D Discrete Fourier Transform), this operation acts as a smooth low-pass transfer function:
$$H(u, v) = \exp\left(-rac{u^2 + v^2}{2\sigma_f^2}
ight)$$
which exponentially attenuates high-frequency spectral components ($u^2 + v^2 > \omega_c$), effectively filtering out film grain digitizer noise.

### 8.4.4 Deep CORAL Covariance Alignment Loss
In the Deep CORAL domain adaptation paradigm (Phase 4B) [18], the network minimizes the distance between the second-order statistical covariances of source domain features $D_S$ and target domain features $D_T$ in the 256-dimensional latent space:
$$C_S = rac{1}{N_S - 1} \left(F_S^T F_S - rac{1}{N_S} (\mathbf{1}^T F_S)^T (\mathbf{1}^T F_S)
ight)$$
$$C_T = rac{1}{N_T - 1} \left(F_T^T F_T - rac{1}{N_T} (\mathbf{1}^T F_T)^T (\mathbf{1}^T F_T)
ight)$$
where $F_S, F_T \in \mathbb{R}^{N 	imes d}$ ($d=256$) are latent activation matrices. The CORAL loss is formulated as the squared Frobenius norm distance between covariance matrices:
$$\mathcal{L}_{	ext{CORAL}} = rac{1}{4d^2} \|C_S - C_T\|_F^2 = rac{1}{4d^2} \sum_{i=1}^d \sum_{j=1}^d (C_{S, i,j} - C_{T, i,j})^2$$
The total objective function optimized during CORAL training is:
$$\mathcal{L}_{	ext{total}} = \mathcal{L}_{	ext{class}}(X_S, Y_S) + \lambda \cdot \mathcal{L}_{	ext{CORAL}}(X_S, X_T)$$
with adaptation weight $\lambda = 0.5$.

### 8.4.5 DANN Domain-Adversarial Objective
In the Domain-Adversarial Neural Network paradigm (Phase 4C) [19], a domain discriminator $G_d$ parameterized by $	heta_d$ is connected to the feature extractor $G_f$ via a Gradient Reversal Layer (GRL). The GRL acts as an identity mapping during forward propagation, but negates gradients during backward propagation:
$$\mathcal{R}(x) = x, \quad rac{d\mathcal{R}}{dx} = -\lambda \mathbf{I}$$
The minimax optimization objective is formulated as:
$$E(	heta_f, 	heta_y, 	heta_d) = rac{1}{N_S} \sum_{i=1}^{N_S} \mathcal{L}_{	ext{class}}(G_y(G_f(x_i^S)), y_i^S) - \lambda \left[ rac{1}{N_S} \sum_{i=1}^{N_S} \mathcal{L}_{	ext{domain}}(G_d(G_f(x_i^S)), 0) + rac{1}{N_T} \sum_{j=1}^{N_T} \mathcal{L}_{	ext{domain}}(G_d(G_f(x_j^T)), 1) 
ight]$$

### 8.4.6 Grad-CAM Mathematical Formulation
To compute class activation maps for class $c$, the gradient of the class score $Y^c$ (before softmax) with respect to feature activation map $A^k$ of layer `conv5_block16_concat` is evaluated:
$$lpha_k^c = rac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V rac{\partial Y^c}{\partial A_{i,j}^k}$$
where $Z = U 	imes V = 7 	imes 7 = 49$. The localization heatmap $L_{	ext{Grad-CAM}}^c$ is computed as:
$$L_{	ext{Grad-CAM}}^c(x, y) = 	ext{ReLU}\left(\sum_{k=1}^{1024} lpha_k^c A^k(x, y)
ight)$$

## 8.5 Training Strategy & Phased Optimization Schedule

To maximize representation transfer while preventing catastrophic forgetting of ImageNet features, a disciplined two-stage optimization schedule was executed across all experimental paradigms:
- **Phase 1 (Frozen Backbone Warmup, Epochs 1–2)**: The DenseNet-121 convolutional base was frozen ($7,037,504$ parameters non-trainable). The Adam optimizer ($eta_1=0.9, eta_2=0.999, \epsilon=10^{-7}$) updated strictly the newly initialized Dense Bottleneck, Batch Normalization, and Softmax layers ($268,038$ trainable parameters) with a learning rate $\eta = 10^{-3}$.
- **Phase 2 (Fine-Tuning High-Level Layers, Epochs 3–4)**: The top 30 layers of Dense Block 4 were unfrozen. Training continued with a reduced learning rate $\eta = 10^{-5}$ and weight decay $10^{-4}$ to fine-tune high-level thoracic representations without disrupting foundational edge detectors.

Table 8.2 summarizes the formal hyperparameter configuration.

### Table 8.2: Experimental Hyperparameter Configuration Matrix

| Hyperparameter / Training Attribute | Configured Setting | Operational Rationale |
| :--- | :--- | :--- |
| **Base Architecture** | DenseNet-121 (ImageNet initialized) | Optimal parameter efficiency and feature reuse for CXR |
| **Input Tensor Resolution** | $224 	imes 224 	imes 3$ | Standardized ImageNet spatial dimensions |
| **Mini-Batch Size** | 32 samples | Balances gradient stability with GPU VRAM allocation |
| **Loss Function** | Weighted Categorical Cross-Entropy | Compensates for empirical class imbalance across 6 classes |
| **Optimizer** | Adam ($eta_1=0.9, eta_2=0.999$) | Adaptive learning rate optimization |
| **Initial Learning Rate (Stage 1)** | $\eta = 1 	imes 10^{-3}$ (Epochs 1–2, base frozen) | Rapid convergence of new dense classification head |
| **Fine-Tuning Learning Rate (Stage 2)**| $\eta = 1 	imes 10^{-5}$ (Epochs 3–4, top 30 unfrozen) | Gentle parameter adjustment preventing catastrophic forgetting |
| **Regularization** | Dropout ($p=0.3$) + L2 Weight Decay ($10^{-4}$) | Prevents latent co-adaptation and overfitting |
| **Data Augmentation (Train Only)** | Random rotation $\pm 10^\circ$, horizontal flip | Simulates patient positioning variations in clinical wards |
| **Random Seed** | 42 (fixed across NumPy, Python, TF) | Guarantees exact experimental reproducibility |

## 8.6 Evaluation Metrics Formulations

Model performance is evaluated across seven standard statistical metrics:
- **Accuracy**: $rac{	ext{TP} + 	ext{TN}}{	ext{TP} + 	ext{TN} + 	ext{FP} + 	ext{FN}}$
- **Per-Class Precision**: $rac{	ext{TP}}{	ext{TP} + 	ext{FP}}$
- **Per-Class Recall (Sensitivity)**: $rac{	ext{TP}}{	ext{TP} + 	ext{FN}}$
- **Per-Class Specificity**: $rac{	ext{TN}}{	ext{TN} + 	ext{FP}}$
- **Per-Class F1-Score**: $2 \cdot rac{	ext{Precision} \cdot 	ext{Recall}}{	ext{Precision} + 	ext{Recall}}$
- **Macro-Averaged F1-Score**: $rac{1}{K} \sum_{k=1}^K 	ext{F1}_k$ (gives equal weight to each disease class regardless of prevalence)
- **Macro ROC-AUC**: Area under the Receiver Operating Characteristic curve macro-averaged across the six one-vs-rest binary curves.
- **Macro PR-AUC**: Area under the Precision-Recall curve macro-averaged across classes, essential for evaluating performance under class imbalance.

---

# CHAPTER 9 — SYSTEM IMPLEMENTATION

## 9.1 Backend Microservice Architecture

The backend of LungAI is engineered as an asynchronous Application Server Gateway Interface (ASGI) microservice utilizing Python 3.10+, FastAPI, and Uvicorn. FastAPI was selected for its native asynchronous event loop (`async/await`), high concurrency throughput, automatic OpenAPI documentation, and strict runtime type verification via Pydantic schemas.

The backend service is structured into modular layers:
- `backend/main.py`: ASGI application factory configuring Cross-Origin Resource Sharing (CORS) middleware, lifespan event handlers for model preloading, global exception handlers, and routing mounts.
- `backend/routers/`: Modular controllers managing API endpoints:
  - `predict.py`: Handles scan ingestion, gate verification, inference orchestration, and triage assignment.
  - `patients.py`: Provides CRUD endpoints for patient profile management.
  - `reports.py`: Generates and streams structured clinical PDF and JSON reports.
  - `health.py`: Exposes system health checks, GPU memory status, and model metadata under `/api/v1/health`.
- `backend/ml/`: Encapsulates machine learning components, ensuring complete isolation between deep learning inference logic and HTTP transport layers.

## 9.2 Asynchronous Relational Persistence

Data persistence is managed via SQLAlchemy 2.0+ ORM configured with `aiosqlite` for asynchronous I/O operations (fully compatible with PostgreSQL via `asyncpg` for high-throughput enterprise deployments). 

Database connections are managed using an asynchronous session factory (`AsyncSession`). To ensure data consistency and prevent race conditions during high-concurrency clinical uploads, all multi-table mutations (such as creating a scan record and immediately linking a prediction record) are wrapped within atomic transaction blocks with automated rollback on failure:

```python
async with async_session() as session:
    async with session.begin():
        scan = Scan(patient_id=patient_uuid, file_path=saved_path, file_hash=sha256_hash)
        session.add(scan)
        await session.flush()
        
        prediction = Prediction(
            scan_id=scan.id,
            model_name="DenseNet121_Frequency_V5",
            predicted_class=top_class,
            confidence=top_prob,
            probabilities=prob_dict,
            urgency_tier=urgency
        )
        session.add(prediction)
```

## 9.3 High-Performance Singleton Inference Engine

To optimize memory efficiency and eliminate model loading overhead on concurrent requests, the inference service is implemented as a thread-safe **Singleton** pattern (`backend/ml/inference.py`).

Upon application startup (within the FastAPI lifespan event), the `InferenceEngine` initializes:
1. Verifies the cryptographic hash and file integrity of the serialized weights artifact (`models/densenet121_frequency_v5.h5`, 28.71 MB).
2. Allocates GPU tensor memory and compiles the model execution graph.
3. Pre-warms the computational graph by executing a dummy forward pass on a zero-tensor of shape $(1, 224, 224, 3)$.
4. Retains the compiled graph in memory for the lifespan of the server process.

Subsequent inference requests reuse the pre-warmed singleton instance, executing forward passes in under 120 ms on GPU hardware (and $pprox 380$ ms on multi-core CPU).

## 9.4 Real-Time Grad-CAM Saliency Generation

The Grad-CAM generation engine (`backend/ml/gradcam.py`) is directly integrated into the inference execution pipeline:
1. Intercepts the preprocessed input tensor and accesses the target layer `conv5_block16_concat`.
2. Utilizes `tf.GradientTape()` to record forward activations and compute exact gradients of the predicted class logit with respect to the 1,024 feature activation channels.
3. Computes channel-wise importance weights via global average pooling across spatial dimensions $(7, 7)$.
4. Generates a weighted linear combination of feature maps, applies ReLU activation, and upsamples the spatial matrix to $224 	imes 224$ pixels.
5. Converts the normalized saliency matrix to an RGB pseudo-color heatmap using OpenCV's `COLORMAP_JET`.
6. Alpha-blends the heatmap over the original radiograph with a 45% transparency coefficient ($lpha = 0.45$) and encodes the composite image to PNG format on disk, returning the relative URI to the frontend viewport.

## 9.5 CXR Validation Gate Implementation

The two-stage CXR validation gate (`backend/ml/cxr_gate.py`) provides an active perimeter defense:
- **Stage 1 (Biophysical Screening)**: Evaluates image dimensions, aspect ratio, luminance variance, and polychromatic saturation across HSV channels. If an uploaded image exhibits excessive chromatic saturation across multiple hue bands (indicative of a natural color photo) or insufficient luminance variance, Stage 1 immediately isolates the payload.
- **Stage 2 (Semantic Classifier)**: Standardizes accepted grayscale inputs to $224 	imes 224 	imes 3$ and executes forward inference through a dedicated binary DenseNet-121 classifier trained to recognize thoracic skeletal and mediastinal anatomy.
- If the semantic probability is $\ge 0.83$ ($	au = 0.83$), the upload is designated as a valid chest radiograph and routed to Model D.
- If the probability is $< 0.83$, the upload is rejected with a structured HTTP 422 Unprocessable Entity error.

## 9.6 Frontend Single-Page Application

The frontend client is engineered as a responsive single-page application (SPA) using React 18, Vite, and modern ES6+ JavaScript. To maximize rendering speed, minimize bundle size, and maintain complete control over visual presentation, the user interface is styled using Vanilla CSS and custom CSS design tokens rather than bloated CSS utility frameworks.

Core architectural components include:
- **AnalyzeWorkspace**: The primary diagnostic interface featuring drag-and-drop file ingestion, real-time upload progress, side-by-side radiograph and Grad-CAM viewports, and interactive windowing sliders.
- **ViewportController**: Delivers radiological viewing controls—including zoom ($	imes 1$ to $	imes 4$), panning, inversion, and brightness/contrast adjustments—operating directly on HTML5 canvas elements for hardware-accelerated rendering.
- **TriageCard**: Displays the calculated clinical urgency tier with high-visibility color coding (Emergency: Red, Urgent: Amber, Routine: Emerald) and continuous probability progress bars across all six classes.
- **PatientManager**: Facilitates patient registration, profile editing, and search filtering across stored electronic health records.
- **MetricsDashboard**: Renders interactive performance charts, confusion matrices, ROC/PR curves, and source-level generalization metrics.

## 9.7 Automated Clinical Report Generator

The reporting subsystem (`backend/routers/reports.py`) dynamically synthesizes structured clinical documentation from database entities:
1. Queries patient demographics, scan metadata, and Model D prediction records.
2. Embeds the original radiograph and the Grad-CAM visual overlay side by side.
3. Formulates structured sections: Clinical Indication, Radiographic Findings, Algorithmic Assessment, Urgency Tier, and Actionable Clinical Recommendations.
4. Generates an exportable PDF document stamped with a unique report identifier, cryptographic hash, and a mandatory academic disclaimer emphasizing the necessity of certified radiological review.

---

# CHAPTER 10 — USER INTERFACE

## 10.1 Design Principles & Diagnostic Ergonomics

The user interface of LungAI is designed according to established medical human-computer interaction (HCI) standards. In clinical radiology environments, interfaces must prioritize visual clarity, minimize cognitive distraction, and adhere to strict ergonomic principles:
- **High-Contrast Dark Theme**: The diagnostic viewport employs a specialized dark palette (background `#0B132B`, slate cards `#1C2541`, accent `#48CAE4`) designed to prevent ocular fatigue in darkened radiological reading rooms and maximize the perceptual visibility of subtle grayscale parenchymal lesions.
- **Clear Information Hierarchy**: Urgent diagnostic findings and clinical triage tiers are presented with bold, unmistakable visual cues, ensuring that acute conditions immediately capture the clinician's attention.
- **Side-by-Side Visual Verification**: The original raw radiograph and the Grad-CAM saliency overlay are positioned side by side with synchronized zooming and panning, enabling practitioners to instantaneously cross-reference heatmaps against anatomical landmarks.

## 10.2 Home / Landing Viewport

The home interface serves as the primary clinical portal, presenting an overview of platform capabilities, operational status of the inference engine, recent scan throughput, and one-click navigation to core workspaces.

![Home Page](docs/screenshots/application/01_home.png)
*Figure 10.1: LungAI Home Portal, displaying system status, operational metrics, and navigation pathways.*

## 10.3 Interactive Scan Analysis & Diagnostic Workspace

The Scan Analysis workspace is the primary diagnostic workstation for attending clinicians. Clinicians can drag and drop a planar radiograph or select an existing patient record. Upon upload, the interface displays real-time progress indicators as the image traverses Stage 1/Stage 2 gate validation, Model D inference, and Grad-CAM generation.

Figure 10.2 illustrates the empty analysis workspace awaiting image upload.

![Analyze Page](docs/screenshots/application/02_analyze.png)
*Figure 10.2: Interactive Diagnostic Workspace prior to scan ingestion, highlighting drag-and-drop ingestion and patient selection.*

Upon completion of inference, the workspace populates with comprehensive diagnostic findings:
- Primary predicted disease finding with top confidence percentage;
- Full 6-class probability distribution displayed via calibrated progress bars;
- Stratified clinical urgency tier badge;
- Interactive DICOM-style canvas viewport with contrast windowing, brightness adjustment, zoom, and pan controls.

Figure 10.3 shows the completed analysis of a clinical radiograph correctly identified as Tuberculosis with high confidence and Urgent triage prioritization.

![Prediction Result](docs/screenshots/application/03_prediction.png)
*Figure 10.3: Completed Diagnostic Analysis interface, displaying primary classification, 6-class confidence bars, and urgent triage badge.*

## 10.4 Grad-CAM Visual Saliency Inspection

The Grad-CAM inspection view enables clinicians to visually audit the neural network's focus. The interface renders an alpha-blended heatmap directly over the patient's radiograph, highlighting the specific anatomical lung fields driving the prediction.

Figure 10.4 showcases the Grad-CAM saliency view for a patient presenting with active apical pulmonary tuberculosis. The attention heatmap correctly concentrates on the apical fibro-cavitary infiltrates in the upper right lung field, verifying that the model is responding to true pathological opacities rather than background artifacts.

![Grad-CAM Overlay](docs/screenshots/application/04_gradcam.png)
*Figure 10.4: Grad-CAM Visual Saliency Inspection, showing focal attention localized over apical fibro-cavitary parenchymal infiltrates.*

## 10.5 Model Evaluation & Scientific Metrics Dashboard

To maintain transparency and foster clinical trust, LungAI incorporates a dedicated Metrics Dashboard. Practicing clinicians and hospital administrators can inspect the empirical performance of Model D across the leak-free V5 benchmark.

The dashboard renders:
- Comprehensive 6-class confusion matrices with row-normalized percentages;
- Multi-class ROC curves and Precision-Recall curves;
- Class-wise performance breakdowns (Precision, Recall, Specificity, F1);
- Latent feature space projections (t-SNE) illustrating class separation;
- Transparent disclosure of external generalization performance on the Montgomery benchmark.

Figure 10.5 illustrates the scientific evaluation dashboard.

![Metrics Dashboard](docs/screenshots/application/05_metrics.png)
*Figure 10.5: Scientific Metrics Dashboard, presenting verified confusion matrices, ROC/PR curves, and class-wise performance statistics.*

## 10.6 Historical Records & Scan Audit Log

The History interface provides a fully searchable, sortable audit trail of all processed radiographic examinations. Clinicians can filter past scans by date range, patient identifier, predicted pathology, or clinical urgency tier, facilitating longitudinal patient monitoring and retrospective quality assurance audits.

Figure 10.6 displays the historical records interface.

![History Page](docs/screenshots/application/06_history.png)
*Figure 10.6: Historical Scan Log and Audit Trail, detailing past examinations, timestamps, predictions, and urgency tiers.*

## 10.7 Patient Management & EHR Linking

The Patient Management interface enables healthcare facilities to maintain persistent demographic records. Clinicians can register new patients, view historical imaging examinations linked to specific patient UUIDs, and review longitudinal diagnostic trajectories.

Figure 10.7 illustrates the patient management portal.

![Patients Page](docs/screenshots/application/07_patients.png)
*Figure 10.7: Patient Management Portal, showing registered patient profiles, medical record numbers, and linked imaging histories.*

---

# CHAPTER 11 — TESTING

## 11.1 Verification Strategy & Quality Assurance Framework

In medical software engineering, comprehensive automated testing is an essential prerequisite for ensuring patient safety, data integrity, and operational reliability. Software failures in clinical environments can delay critical care or corrupt electronic health records.

To guarantee system stability, LungAI was subjected to a disciplined, multi-layered automated verification protocol comprising:
1. **Unit Testing**: Verifying isolated functions, image preprocessing transformations, tensor normalization invariants, and algorithmic fallback routines.
2. **Integration Testing**: Verifying interactions between the FastAPI service layer, the SQLAlchemy ORM persistence engine, and the filesystem.
3. **End-to-End API Testing**: Simulating client HTTP requests across all exposed REST endpoints using an asynchronous test client (`httpx`).
4. **CXR Input Gate Forensic Testing**: Evaluating the two-stage validation gate against genuine radiographs, corrupt files, and adversarial non-radiographic images.

## 11.2 Comprehensive 44-Test Automated Verification Matrix

The test suite consists of **44 automated tests** executed via Pytest. In the final system audit, the test suite achieved a **100% pass rate (44 passed, 0 failures, 0 errors)**. 

Table 11.1 details the complete 44-test verification matrix across functional subsystems.

### Table 11.1: Automated Software Verification Matrix (44 Tests, 100% Pass Rate)

| Test Module / Suite | Test Case Identifier | Verification Objective & Assertion | Expected Outcome | Actual Result |
| :--- | :--- | :--- | :---: | :---: |
| **Preprocessing** (`test_preprocess.py`) | `test_preprocess_valid_image` | Standardizes valid radiograph to shape $(1, 224, 224, 3)$ | Shape match, float32 | **PASS** |
| | `test_preprocess_corrupt_bytes` | Throws `ValueError` when decoding truncated/corrupt stream | Exception caught | **PASS** |
| | `test_preprocess_empty_bytes` | Throws `ValueError` on 0-byte payload | Exception caught | **PASS** |
| | `test_clahe_enhancement_effect` | Luminance dynamic range increases post-CLAHE | $\sigma_{	ext{post}} > \sigma_{	ext{pre}}$ | **PASS** |
| | `test_gaussian_lowpass_filter` | High-frequency Fourier energy attenuates at $\sigma=1.0$ | Energy ratio $< 0.85$ | **PASS** |
| | `test_imagenet_normalization` | Standardized tensor mean $pprox 0.0$ and std $pprox 1.0$ | Invariants hold | **PASS** |
| **Model & Inference** (`test_inference.py`) | `test_model_loading_singleton` | Weights load into memory; output shape is $(None, 6)$ | Model initialized | **PASS** |
| | `test_prediction_probabilities` | 6-class softmax probabilities sum to $1.0 \pm 10^{-4}$ | $\sum p_i = 1.0$ | **PASS** |
| | `test_prediction_top_class` | Top predicted class matches maximum probability index | Index matches | **PASS** |
| | `test_urgency_tier_emergency` | Pneumonia $p \ge 0.70$ triggers Emergency tier | Tier == Emergency | **PASS** |
| | `test_urgency_tier_urgent_tb` | Tuberculosis $p \ge 0.50$ triggers Urgent tier | Tier == Urgent | **PASS** |
| | `test_urgency_tier_urgent_nodule`| Pulmonary Nodule $p \ge 0.40$ triggers Urgent tier | Tier == Urgent | **PASS** |
| | `test_urgency_tier_routine` | Normal $p \ge 0.60$ triggers Routine tier | Tier == Routine | **PASS** |
| | `test_gradcam_tensor_generation`| Generates $224 	imes 224$ heatmap with non-zero activations | Shape match, $\max > 0$| **PASS** |
| | `test_gradcam_overlay_creation` | Alpha blending produces valid 3-channel RGB overlay | Valid image array | **PASS** |
| **CXR Validation Gate** (`test_gate.py`) | `test_stage1_dimensions_pass` | Accepts image with width $\ge 32$ and height $\ge 32$ | Pass Stage 1 | **PASS** |
| | `test_stage1_aspect_ratio_fail` | Rejects extreme aspect ratio $> 2.2:1$ (e.g., panoramic) | Fail Stage 1 | **PASS** |
| | `test_stage1_low_luminance_fail`| Rejects flat, blank, or low-contrast synthetic documents | Fail Stage 1 | **PASS** |
| | `test_stage1_color_photo_fail` | Rejects polychromatic color image (car, landscape) | Fail Stage 1 | **PASS** |
| | `test_stage2_valid_cxr_pass` | Authentic CXR passes semantic gate with score $\ge 0.83$ | Pass Stage 2 | **PASS** |
| | `test_stage2_non_cxr_rejection` | Grayscale non-CXR (CT, ultrasound) scores $< 0.83$ | Rejected ($<0.83$) | **PASS** |
| | `test_gate_adversarial_cat` | Cat photo (scored 0.822) correctly rejected at $	au=0.83$ | Rejected | **PASS** |
| | `test_fallback_symmetry_scoring`| Bilateral symmetry function executes when model absent | Valid score [0, 1] | **PASS** |
| **API Endpoints** (`test_api.py`) | `test_health_endpoint` | `GET /api/v1/health` returns HTTP 200 and "healthy" | HTTP 200 OK | **PASS** |
| | `test_predict_valid_upload` | `POST /api/v1/predict` returns 200, prediction, triage | Valid JSON payload | **PASS** |
| | `test_predict_invalid_extension`| Uploading `.txt` or `.exe` returns HTTP 400 Bad Request | HTTP 400 | **PASS** |
| | `test_predict_empty_file` | Uploading 0-byte file returns HTTP 400 Bad Request | HTTP 400 | **PASS** |
| | `test_predict_corrupt_file` | Uploading invalid byte stream returns HTTP 422 | HTTP 422 | **PASS** |
| | `test_predict_non_cxr_rejection`| Uploading non-CXR image triggers HTTP 422 with message | HTTP 422 Rejected | **PASS** |
| | `test_predict_payload_too_large`| Uploading $>10$ MB image returns HTTP 413 Ceiling | HTTP 413 | **PASS** |
| **Patient Management** (`test_patients.py`) | `test_create_patient_success` | `POST /api/v1/patients/` creates profile with UUIDv4 | HTTP 201 Created | **PASS** |
| | `test_create_patient_duplicate` | Creating duplicate MRN throws HTTP 409 Conflict | HTTP 409 | **PASS** |
| | `test_get_patient_by_id` | `GET /api/v1/patients/{id}` returns profile | HTTP 200 OK | **PASS** |
| | `test_list_patients_pagination` | `GET /api/v1/patients/` returns paginated list | Correct count | **PASS** |
| | `test_update_patient_profile` | `PUT /api/v1/patients/{id}` updates attributes | Updated successfully| **PASS** |
| | `test_delete_patient_cascade` | Deleting patient cascades to linked scans/predictions | Records purged | **PASS** |
| **Database & Reports** (`test_reports.py`) | `test_prediction_persistence` | Scan and prediction persist in relational database | Foreign keys match | **PASS** |
| | `test_prediction_history_query` | `GET /api/v1/predictions/` retrieves historical records | List matches DB | **PASS** |
| | `test_report_generation_json` | `GET /api/v1/reports/{id}` returns structured report | Schema valid | **PASS** |
| | `test_report_generation_pdf` | `GET /api/v1/reports/{id}/pdf` generates valid PDF stream| Valid PDF header | **PASS** |
| | `test_db_transaction_rollback` | Simulated crash rolls back transaction cleanly | DB remains clean | **PASS** |
| | `test_cors_headers_present` | API responses include permissive CORS headers | Headers verified | **PASS** |
| | `test_execution_latency_logged`| Prediction record includes positive execution time | $t_{	ext{exec}} > 0$ | **PASS** |
| | `test_concurrent_inference` | 5 concurrent requests execute without thread collision | All return 200 OK | **PASS** |

## 11.3 Unit Testing of Preprocessing & Model Invariants

Unit tests verify that individual mathematical components function strictly according to design specifications. In `test_preprocess.py`, deterministic invariants were rigorously asserted:
- **Spatial Dimensions**: Asserted that diverse raw input formats (ranging from $512 	imes 512$ to $3000 	imes 3000$) consistently yield tensors of exact shape $(1, 224, 224, 3)$.
- **Dynamic Range & Standardization**: Asserted that preprocessed image pixels mapped to $[-3.0, 3.0]$ conforming to ImageNet population distributions.
- **Bitwise Determinism**: Asserted that repeated preprocessing runs on identical image byte arrays produced bitwise identical tensors (0-byte deviation across float32 arrays).

## 11.4 Integration & REST API Contract Verification

Integration tests verified asynchronous database session lifecycles, connection pooling, and cascading foreign key constraints. Using the `httpx.AsyncClient`, the API test suite simulated concurrent client interactions, asserting that multipart image uploads, asynchronous database inserts, and JSON serialization executed without connection leaks or orphaned database rows.

## 11.5 CXR Input-Validation Gate Verification Suite

The two-stage CXR validation gate was subjected to an extensive forensic evaluation to quantify its screening capabilities against out-of-distribution inputs.

The evaluation suite utilized a dedicated held-out cohort of **125 evaluation images**:
- **48 Authentic Chest Radiographs** (sampled across the 6 disease classes from the V5 training split);
- **77 Non-CXR Images** comprising:
  - 100 axial chest CT slices;
  - Natural color photographs (vehicles, landscapes, animals, food, cityscapes);
  - Grayscale non-medical photographs;
  - Synthetic documents, text screenshots, and charts.

Table 11.2 details the forensic performance of the gate across screening stages.

### Table 11.2: Two-Stage CXR Input-Validation Gate Forensic Evaluation

| Evaluation Cohort / Category | Total Images Tested | Stage 1 (Biophysical) Result | Stage 2 (Semantic at $\tau=0.83$) Result | Combined Final Decision |
| :--- | :---: | :---: | :---: | :---: |
| **Authentic Chest Radiographs** | 48 | 48 Passed (0 False Rejections) | 48 Passed ($\ge 0.83$) | **48 / 48 Accepted (100% Sensitivity)** |
| **Axial Chest CT Slices** | 35 | 35 Passed (grayscale appearance)| 35 Rejected (Semantic $< 0.83$) | **35 / 35 Rejected (100% Specificity)** |
| **Natural Color Photos (Car, Food)**| 20 | 20 Rejected (High saturation) | Not executed (Intercepted Stage 1) | **20 / 20 Rejected (100% Specificity)** |
| **Grayscale Natural Photos** | 12 | 12 Passed (low saturation) | 12 Rejected (Semantic $< 0.83$) | **12 / 12 Rejected (100% Specificity)** |
| **Documents & Synthetic Screenshots**| 10 | 10 Rejected (Low $\sigma_L$ variance)| Not executed (Intercepted Stage 1) | **10 / 10 Rejected (100% Specificity)** |
| **Total Evaluation Cohort** | **125** | — | — | **100% Sensitivity, 100% Specificity** |

## 11.6 Adversarial Non-CXR Screening & The Cat Test Case

During forensic security auditing, the validation gate was tested against seven specialized adversarial non-CXR categories designed to bypass naive heuristic filters:
1. Domestic animal photographs (color and grayscale);
2. Axial thoracic CT scans (sharing thoracic tissue density);
3. Abdominal ultrasound images;
4. Synthetic medical charts and ECG waveforms;
5. Microscopic histology slides;
6. Scanned text documents;
7. Solid black and solid white blank frames.

### The Cat Test Case and Threshold Correction
A critical defect was uncovered regarding the production threshold $	au$:
- In earlier system drafts, the Stage 2 gate classifier utilized a default threshold $	au = 0.70$.
- When evaluated against a grayscale photograph of a domestic cat, the Stage 2 classifier output a semantic score of **0.822**.
- Under $	au = 0.70$, the cat image achieved $0.822 \ge 0.70$ and successfully bypassed the validation gate, proceeding to Model D, which generated a confident disease prediction.
- This represented an active security leak. 

To eliminate this vulnerability, a comprehensive threshold sweep was conducted across the 125-image evaluation cohort. At $	au = 0.83$, the cat image scored $0.822 < 0.83$ and was decisively rejected. Furthermore, all 48 genuine CXRs scored $\ge 0.842$, preserving 100% genuine CXR pass-through. Consequently, the production threshold was officially changed and frozen at **$	au = 0.83$** in `backend/ml/cxr_gate.py`.

In combined functional testing:
- Stage 1 rejected 5 of 7 adversarial categories directly at the perimeter.
- Stage 2 rejected the remaining 2 grayscale categories (CT slices and the cat test case).
- Combined, the two-stage gate successfully intercepted and rejected **13 of 13 representative test inputs** (6/6 genuine CXRs accepted, 7/7 adversarial non-CXRs rejected).

## 11.7 Critical Disclosure: Threshold-Selection Leakage Analysis

In accordance with rigorous academic and scientific standards, a critical methodological limitation must be explicitly disclosed regarding the reported gate metrics:

> **Important Scientific Disclosure**: The production threshold $	au = 0.83$ was selected and optimized using the same 125-image evaluation cohort upon which test metrics were evaluated. In statistical learning theory, selecting an operational decision threshold on the test cohort introduces **threshold-selection data leakage**. Consequently, the reported 100% sensitivity, 100% specificity, and ROC-AUC of 1.0000 must be interpreted strictly as **functional verification estimates** of system calibration rather than unbiased estimates of real-world generalization performance. Independent multi-institutional validation on a separate, unobserved test set is required to establish unbiased operational generalization.

---

# CHAPTER 12 — MODEL EVALUATION AND RESULTS

## 12.1 Experimental Setup & Evaluation Protocol

The empirical investigation of this dissertation evaluates the central research question: whether deep convolutional networks can learn disease-relevant representations that generalize across independent acquisition sources, and whether domain-generalization strategies improve cross-source performance compared to a standard baseline.

To eliminate confounding experimental variables, all comparative evaluations were executed under a strictly controlled scientific protocol:
- **Canonical Dataset Cohort**: All models were trained, validated, and evaluated on the identical patient-strict V5 cohort ($N=10,547$ radiographs from 10,270 patients) with 0% patient leakage.
- **Identical Backbone Architecture**: Every tested model utilizes the pre-trained DenseNet-121 backbone, identical global average pooling, batch normalization, 256-dimensional dense latent projection bottleneck, and 6-class softmax classification head.
- **Identical Optimization Schedule**: All models followed the two-phase convergence schedule: 2 epochs with frozen backbone ($\eta = 10^{-3}$) followed by 2 epochs of fine-tuning the top 30 layers ($\eta = 10^{-5}$) using Adam optimization and inverse-frequency class weighting.
- **Quarantined External Benchmark**: All models were evaluated zero-shot against the untouched Montgomery County benchmark ($N=138$) without any fine-tuning or threshold re-calibration.

## 12.2 Master Five-Paradigm Benchmark Comparison

Five distinct algorithmic and biophysical paradigms were investigated:
1. **Phase 3B — Empirical Risk Minimization (ERM) Baseline**: Standard DenseNet-121 trained using weighted categorical cross-entropy without domain adaptation or frequency filtering.
2. **Phase 4B — Deep CORAL Latent Covariance Alignment**: DenseNet-121 trained with the composite objective $\mathcal{L}_{	ext{total}} = \mathcal{L}_{	ext{class}} + 0.5 \cdot \mathcal{L}_{	ext{CORAL}}$, minimizing the second-order covariance distance between multi-source feature manifolds in the 256-D latent bottleneck.
3. **Phase 4C — Domain-Adversarial Neural Networks (DANN)**: DenseNet-121 coupled with a domain discriminator and Gradient Reversal Layer (GRL), trained minimax to discard domain-predictive features.
4. **Phase 4D — Frequency-Aware Gaussian Spatial Filtering (Model D)**: DenseNet-121 preceded by a Gaussian spatial low-pass filter ($\sigma = 1.0$) on standardized CIE LAB CLAHE radiographs, attenuating high-frequency scanner shortcuts in the pixel input space.
5. **Phase 4E — Hybrid Domain Adaptation Synthesis**: DenseNet-121 combining both Gaussian low-pass spatial filtering ($\sigma=1.0$) and Deep CORAL latent covariance alignment.

Table 12.1 summarizes the master comparative results across the five paradigms on the internal held-out test split ($N=1,570$) and the quarantined external Montgomery benchmark ($N=138$).

### Table 12.1: Master Five-Paradigm Experimental Comparison

| Phase | Scientific Paradigm | Preprocessing Pipeline | Domain Strategy | Internal Test Accuracy | Internal Macro F1 | Internal Macro ROC-AUC | Internal Macro PR-AUC | Montgomery External TB Recall | Primary Role in Research Progression |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **3B** | **ERM Baseline** | Standard CLAHE | None (Empirical Risk) | 76.18% | 72.63% | 0.9600 | 0.7927 | 0.00% (0/58) | Scientific baseline; exposed sensor overfitting |
| **4B** | **Deep CORAL** | Standard CLAHE | Covariance Alignment | 80.89% | 76.85% | 0.9671 | 0.8224 | 1.72% (1/58) | Latent alignment study; +4.22% F1 over baseline |
| **4C** | **DANN Adversarial** | Standard CLAHE | Minimax Adversarial | 76.24% | 72.99% | 0.9603 | 0.7915 | 0.00% (0/58) | Adversarial adaptation; failed to improve over ERM |
| **4D** | **Frequency-Aware (Model D)**| **Gaussian LP ($\sigma=1.0$)** | **Input Frequency Filter**| **82.93%** | **78.35%** | **0.9755** | **0.8391** | **0.00% (0/58)** | **Selected Production Engine (Peak Internal F1)** |
| **4E** | **Hybrid Synthesis** | Gaussian LP ($\sigma=1.0$) | LP + Deep CORAL | 78.22% | 73.95% | 0.9609 | — | 0.00% (0/58) | Synthesis study; did not outperform frequency alone |

## 12.3 Empirical Analysis of Baseline vs. Domain Adaptation Paradigms

A critical analysis of Table 12.1 yields fundamental scientific insights:

1. **Efficacy of Deep CORAL**: Minimizing latent covariance distance between source repositories produced decisive performance gains over ERM baseline: internal accuracy rose from 76.18% to 80.89% (+4.71%), macro F1 improved from 72.63% to 76.85% (+4.22%), and macro ROC-AUC reached 0.9671. Aligning second-order feature statistics successfully penalized the network from organizing latent space around repository-specific centroids.
2. **Failure of Adversarial DANN**: In contrast to CORAL, adversarial domain adaptation (Phase 4C) failed to produce meaningful gains over the ERM baseline (76.24% accuracy vs. 76.18%; 72.99% F1 vs. 72.63%). Minimax adversarial optimization proved unstable on this multi-source medical cohort: the gradient reversal layer forced the feature extractor to discard subtle high-level textural features that were simultaneously discriminative for disease classification and correlated with source repositories. This represents a vital negative scientific finding, demonstrating that adversarial domain adaptation cannot be assumed to yield universal benefits in heterogeneous medical imaging.
3. **Decisive Superiority of Frequency-Aware Preprocessing (Model D)**: Applying a biophysical Gaussian spatial low-pass filter ($\sigma = 1.0$) in the pixel input space produced the strongest internal performance among all five tested paradigms: **82.93% accuracy (+6.75% delta over ERM), 78.35% macro F1 (+5.72% delta over ERM), 0.9755 macro ROC-AUC, and 0.8391 macro PR-AUC**. By directly filtering out high-frequency digitizer noise and scanner grain in the pixel space before feature extraction, the convolutional kernels were naturally compelled to focus on macroscopic anatomical lesion structures.
4. **Underperformance of Hybrid Synthesis**: Combining Gaussian spatial filtering with Deep CORAL (Phase 4E) resulted in 78.22% accuracy and 73.95% macro F1—substantially lower than frequency filtering alone (82.93%). This occurs because spatial low-pass filtering already removes the high-frequency sensor discrepancies that CORAL seeks to align; adding covariance penalties on smoothed representations over-constrained the latent space, suppressing fine disease-discriminative variance.

Consequently, **Model D was selected as the final production inference engine through evidence-driven empirical arbitration**, proving that simpler, mathematically principled biophysical interventions can outperform complex adversarial architectures.

## 12.4 Final Model D Internal Test Performance

On the quarantined internal test split of 1,570 clinical radiographs, Model D achieved an overall test accuracy of **82.93%**, macro precision of **79.31% (0.7931)**, macro recall of **80.73% (0.8073)**, macro F1-score of **78.35% (0.7835)**, weighted F1-score of **84.18% (0.8418)**, macro ROC-AUC of **0.9755**, and macro PR-AUC of **0.8391**.

Table 12.2 provides the complete class-wise performance breakdown for Model D.

### Table 12.2: Final Model D Class-Wise Performance Metrics (Internal Test Split, $N=1,570$)

| Class Index | Diagnostic Category | Support ($N$) | Precision | Recall (Sensitivity) | Specificity | F1-Score | Primary Diagnostic Finding |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **COVID-19** | 292 | **99.64%** | **94.52%** | **99.92%** | **97.01%** | Near-perfect bilateral ground-glass detection |
| **1** | **Normal** | 394 | **90.86%** | **88.32%** | **97.02%** | **89.58%** | High specificity on healthy control lung fields |
| **2** | **Pleural Effusion** | 152 | **60.00%** | **51.32%** | **96.33%** | **55.32%** | Moderate recall; obscured by basal consolidations |
| **3** | **Pneumonia** | 423 | **94.92%** | **79.43%** | **98.43%** | **86.49%** | Strong precision; some confusion with effusion/TB |
| **4** | **Pulmonary Nodule / Mass** | 108 | **38.49%** | **85.19%** | **89.95%** | **53.03%** | High sensitivity (catches nodules), lower precision |
| **5** | **Tuberculosis** | 201 | **91.98%** | **85.57%** | **98.90%** | **88.66%** | Robust identification of apical fibro-cavities |
| **—** | **Macro Average / Total** | **1,570** | **79.31%** | **80.73%** | **96.76%** | **78.35%** | **Macro ROC-AUC: 0.9755 \| Macro PR-AUC: 0.8391** |
| **—** | **Weighted Average** | **1,570** | **85.82%** | **82.93%** | — | **84.18%** | **Overall Internal Accuracy: 82.93%** |

## 12.5 Per-Class Diagnostic Performance Analysis

Analyzing the per-class metrics in Table 12.2 reveals key clinical strengths and diagnostic boundaries:
- **COVID-19 (F1: 97.01%)**: Achieved exceptional precision (99.64%) and sensitivity (94.52%), misclassifying only 16 of 292 cases. The bilateral, peripheral distribution of viral ground-glass opacities creates distinctive macroscopic representations that DenseNet-121 readily separates from bacterial pneumonia.
- **Normal (F1: 89.58%)**: Demonstrated 90.86% precision and 97.02% specificity, successfully recognizing uncompromised parenchymal aeration and sharp costophrenic angles.
- **Tuberculosis (F1: 88.66%)**: Achieved 91.98% precision and 85.57% recall, accurately localizing apical consolidations and fibro-cavitary architectural distortions.
- **Pneumonia (F1: 86.49%)**: Showed 94.92% precision and 79.43% recall. False negatives occurred primarily in cases with concomitant pleural fluid blunting, where the model predicted Pleural Effusion.
- **Pleural Effusion (F1: 55.32%)**: Exhibited moderate recall (51.32%). In planar radiography, pleural fluid collections frequently co-occur with or mask underlying lower-lobe consolidations, creating diagnostic ambiguity.
- **Pulmonary Nodule / Mass (F1: 53.03%)**: Exhibited high sensitivity (85.19%), successfully detecting 92 of 108 subtle parenchymal nodules. However, precision was lower (38.49%) due to false-positive alarms on rib crossings and vascular superimpositions. Clinically, high sensitivity is advantageous for oncological triage, ensuring suspicious lesions are flagged for follow-up CT evaluation.

Figure 12.1 presents the confusion matrix for Model D on the internal test split.

![Model D Confusion Matrix](docs/figures/performance/model_d_confusion_matrix.png)
*Figure 12.1: Model D Confusion Matrix across the 1,570-image internal test split, demonstrating strong diagonal concentration across COVID-19, Normal, Pneumonia, and TB.*

Figure 12.2 and Figure 12.3 present the Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves, respectively.

![Model D ROC Curves](docs/figures/performance/model_d_roc_curves.png)
*Figure 12.2: Multi-class ROC curves for Model D, highlighting exceptional discrimination across COVID-19 (AUC 0.999), Normal (AUC 0.985), and TB (AUC 0.986).*

![Model D PR Curves](docs/figures/performance/model_d_pr_curves.png)
*Figure 12.3: Precision-Recall curves for Model D, confirming high area under PR curves despite class prevalence variations.*

## 12.6 Source-Level Generalization Diagnostics

To evaluate whether Model D learned representations that generalize across acquisition centers, classification performance was audited across the individual source repositories comprising the test split. Table 12.3 details the source-level diagnostics.

### Table 12.3: Source-Level Classification Performance Breakdown (Model D)

| Source Repository | Test Samples ($N$) | Source Accuracy | Dominant Class Recalls | Operational Observation |
| :--- | :---: | :---: | :--- | :--- |
| **Existing COVID-19** | 292 | **94.52%** | COVID-19: 94.52% | High fidelity on viral ground-glass features |
| **Existing Normal** | 180 | **95.56%** | Normal: 95.56% | High baseline specificity on healthy controls |
| **Existing Tuberculosis**| 99 | **94.95%** | Tuberculosis: 94.95% | Excellent identification of mycobacterial cavities |
| **Existing Pneumonia** | 210 | **88.10%** | Pneumonia: 88.10% | Strong recognition of lobar alveolar consolidations |
| **TBX11K (Hong Kong)** | 492 | **81.91%** | Normal: 96.67%, Pneumonia: 71.90%, TB: 76.47% | High normal specificity; moderate cross-disease confusion |
| **VinBigData (Vietnam)** | 220 | **63.18%** | Nodule / Mass: 82.05%, Effusion: 52.82% | Difficult multi-label cohort; high nodule sensitivity |
| **JSRT (Japan)** | 21 | **52.38%** | Nodule / Mass: 100.0%, Normal: 0.0% | Normal scans confused with subtle solitary nodules |
| **NIH ChestX-ray14 (USA)**| 56 | **39.29%** | Nodule: 89.47%, Effusion: 30.00%, Normal: 8.33% | Severe multi-label co-occurrence and low-contrast labels |

The source diagnostics reveal that while single-condition repositories achieved accuracies exceeding 88%–95%, multi-label hospital collections (NIH and VinBigData) exhibited lower performance. This stems from label co-occurrence: many NIH patients exhibit both pneumonia and effusion simultaneously, which a single-label multi-class network must arbitrate into a single class.

## 12.7 Quarantined External Generalization Failure: The Montgomery Audit

To rigorously evaluate out-of-distribution generalization, Model D was deployed zero-shot on the quarantined Montgomery County benchmark ($N=138$, 58 active TB cases, 80 normal controls). Table 12.4 details the external evaluation results.

### Table 12.4: Zero-Shot External Evaluation on Quarantined Montgomery Benchmark

| Metric / Attribute | Recorded Value | Clinical / Statistical Interpretation |
| :--- | :---: | :--- |
| **Total Scans Evaluated** | 138 | 58 active tuberculosis cases, 80 healthy normal controls |
| **Binary Abnormal Sensitivity** | **100.0% (58 / 58)** | Every single abnormal TB scan was flagged as pathological (0 false negatives) |
| **Exact Tuberculosis Recall** | **0.00% (0 / 58)** | Zero TB cases were assigned the exact label "Tuberculosis" |
| **Exact Normal Specificity** | **0.00% (0 / 80)** | Zero normal cases were assigned the exact label "Normal" |
| **TB Predictions Breakdown** | Nodule / Mass: 50, Effusion: 8 | TB cases collapsed predominantly into Pulmonary Nodule / Mass (86.2%) |
| **Normal Predictions Breakdown**| Nodule / Mass: 80 | 100% of healthy normal scans collapsed into Pulmonary Nodule / Mass |
| **Prediction Confidence** | Mean Max Prob: 0.8691 | Model output high confidence (median 0.9034) on incorrect classifications |

Figure 12.4 displays the Montgomery confusion matrix, illustrating the complete collapse into the Pulmonary Nodule / Mass category.

![Montgomery Confusion Matrix](docs/figures/failure_analysis/phase4a_montgomery_confusion_matrix.png)
*Figure 12.4: Confusion Matrix on Quarantined Montgomery Benchmark ($N=138$), displaying complete fine-grained classification collapse into Pulmonary Nodule / Mass.*

## 12.8 Biophysical Root-Cause Investigation: High-Frequency Sensor Grain & Edge Variance

To understand why all tested models (ERM, CORAL, DANN, Model D, Hybrid) experienced fine-grained collapse on Montgomery scans, a forensic biophysical analysis was executed comparing Montgomery radiographs against internal V5 digital radiographs:

1. **Optical Density & Contrast Discrepancies**: The Montgomery cohort consists of analog film radiographs digitized using an optical film digitizer, whereas V5 scans originate from modern direct digital radiography (DR) detectors. Digitized analog films exhibit severe non-linear optical density curves and compressed dynamic ranges.
2. **Laplacian High-Frequency Edge Variance Analysis**: The spatial Laplacian variance $\sigma^2(
abla^2 I)$—a standard computer vision metric for high-frequency sharpness and noise grain—was computed across cohorts:
   - Internal V5 Digital CXRs: Mean Laplacian variance = **373.4**
   - Montgomery Scanned Film CXRs: Mean Laplacian variance = **1,580.2**
   Digitized analog film exhibited approximately **$4	imes$ higher high-frequency edge variance** due to physical film grain, optical scanner sensor noise, and dust refraction.
3. **Parenchymal Feature Confusion**: When DenseNet-121's deep convolutional kernels processed Montgomery scans, the high-frequency film grain textures triggered dense feature activations that closely matched the texture of nodular parenchymal opacities, causing the softmax layer to systematically assign high probability to Pulmonary Nodule / Mass.

Figure 12.5 and Figure 12.6 illustrate the image distribution and feature space collapse.

![Image Distribution Analysis](docs/figures/failure_analysis/phase4a_image_distribution_analysis.png)
*Figure 12.5: Biophysical distribution analysis, demonstrating the $4	imes$ edge variance discrepancy between digitized film and digital detectors.*

![Phase 4A Feature Space](docs/figures/failure_analysis/phase4a_feature_space.png)
*Figure 12.6: Latent feature space projection under domain shift, illustrating the severe separation between internal digital scans and external Montgomery scans.*

> **Critical Scientific Caveat**: The thesis explicitly emphasizes that *substantial source-dependent distribution differences were associated with the observed external-domain failure*. Because multiple factors (resolution, digitizer grain, dynamic range, projection angles, and regional demographic prevalence) vary simultaneously, **causal attribution to any single physical factor is not established**.

Figure 12.7 shows representative failure panels on Montgomery scans, illustrating spurious activations driven by high-frequency grain.

![Montgomery Failure Panel](docs/figures/failure_analysis/panel_montgomery_normal_0001.png)
*Figure 12.7: Saliency failure panel for Montgomery normal scan 0001, demonstrating dense focal activation on film grain texture leading to a false nodule prediction.*

## 12.9 t-SNE Feature Space Representation Analysis

To examine how different learning paradigms organize representations in the 256-dimensional latent space, t-Distributed Stochastic Neighbor Embedding (t-SNE) was performed.

Figure 12.8 presents the latent feature distribution for Model D. The projection demonstrates distinct, compact clusters for COVID-19, Normal, Pneumonia, and Tuberculosis, reflecting strong class-discriminative representation learning. In contrast, Pulmonary Nodule / Mass and Pleural Effusion display broader dispersion, consistent with their lower classification F1-scores.

![Model D t-SNE Feature Space](docs/figures/model/model_d_tsne_feature_space.png)
*Figure 12.8: t-SNE latent feature projection of Model D on the internal test split, showing distinct clustering of infectious pathologies.*

Comparative t-SNE projections for Deep CORAL (Figure 12.9), DANN (Figure 12.10), and Hybrid (Figure 12.11) confirm that while CORAL tightened latent covariance clusters, DANN caused cluster boundary diffusion, explaining its lower empirical accuracy.

![CORAL t-SNE Feature Space](docs/figures/model/coral_tsne_feature_space.png)
*Figure 12.9: t-SNE feature projection of Deep CORAL, illustrating covariance alignment across source domains.*

## 12.10 Qualitative Saliency Analysis with Grad-CAM

Figure 12.12 presents qualitative Grad-CAM saliency evaluations across authentic internal radiographs:
- **Pneumonia**: Attention heatmaps concentrate densely over unilateral lower-lobe alveolar consolidations.
- **COVID-19**: Heatmaps exhibit bilateral, peripheral multifocal distributions corresponding to ground-glass opacities.
- **Tuberculosis**: Heatmaps localize sharply over apical fibro-cavitary architectural distortions.

![Internal Saliency Panel](docs/figures/failure_analysis/panel_internal_pneumonia.png)
*Figure 12.10: Qualitative Grad-CAM saliency panel on an internal digital pneumonia radiograph, showing precise anatomical localization over dense lobar consolidation.*

## 12.11 Synthesis of Scientific Findings & Experimental Evidence

The empirical investigations establish four overarching conclusions:
1. **Multi-Source Benchmarking Reveals Hidden Sensitivities**: Naive empirical risk minimization achieves moderate accuracy (76.18%) on multi-source data, but fails to disentangle source-specific noise shortcuts.
2. **Frequency Preprocessing Outperforms Complex Adversarial Adaptation**: Intervening directly in the pixel space via Gaussian low-pass spatial filtering ($\sigma=1.0$) provided the strongest internal performance (+5.72% F1 over ERM), whereas adversarial adaptation (DANN) degraded class-discriminative representations.
3. **Domain Generalization Remains Unsolved Across Sensor Modalities**: While Model D eliminated internal sensor grain overfitting, the zero-shot external collapse on Montgomery scans proves that digitized film vs. direct digital sensor shift remains a fundamental barrier requiring target-domain calibration or lung field segmentation.
4. **Input Verification is Essential for Safe Deployment**: Integrating a two-stage CXR validation gate at $	au = 0.83$ provides a critical defensive safety layer, intercepting 100% of tested non-radiographic uploads.

---

# CHAPTER 13 — SECURITY, PRIVACY AND RESPONSIBLE AI

## 13.1 Academic Research Status & Clinical Positioning

LungAI is conceptualized, engineered, and evaluated strictly as an academic research and clinical decision-support prototype. It is not an autonomous medical device, does not possess clearance or approval from medical regulatory authorities (such as the United States Food and Drug Administration [FDA], the European Medicines Agency [EMA], or the Central Drugs Standard Control Organisation [CDSCO] of India), and cannot legally or ethically serve as a sole primary diagnostic instrument.

The system is positioned as an assistive second reader designed to aid clinical triage in high-volume, resource-constrained emergency wards by pre-screening incoming examinations, prioritizing reading queues, and highlighting suspicious anatomical regions for certified radiological review.

## 13.2 Mandatory Medical Disclaimer & Human-in-the-Loop Protocol

To ensure ethical and responsible artificial intelligence deployment, the platform enforces a strict **Human-in-the-Loop (HITL)** operational paradigm:
1. **Mandatory Interface Disclaimer**: Every viewport, diagnostic summary card, and exported clinical report displays a permanent, high-visibility medical disclaimer:
   > *"MEDICAL DISCLAIMER: LungAI is an academic clinical decision-support prototype engineered strictly for research and triage prioritization. It does not provide certified medical diagnoses. All algorithmic predictions, confidence scores, and Grad-CAM visualizations must be independently interpreted and verified by a board-certified radiologist or licensed medical practitioner prior to clinical intervention."*
2. **Mandatory Human Sign-Off**: The reporting engine requires a licensed physician to review the auto-generated findings, append narrative notes, and validate the report before it can be exported or committed to an institutional hospital record.

## 13.3 Absence of Regulatory Clearance & Clinical Validation Bounds

Because the system has been validated exclusively on retrospective, open-access imaging cohorts and quarantined public benchmarks, its diagnostic performance under prospective clinical workflows remains unestablished. Algorithmic outputs cannot substitute for laboratory confirmation (e.g., RT-PCR for COVID-19, sputum smear microscopy or GeneXpert for tuberculosis, or histological tissue biopsy for suspected pulmonary nodules).

## 13.4 CXR Input Gate as a Safety Guardrail

Standard deep learning architectures will output confident diagnostic predictions on arbitrary images (e.g., classifying a domestic animal or a document screenshot as "Pneumonia with 98% confidence"). LungAI mitigates this severe safety vulnerability through its two-stage perimeter validation gate (`backend/ml/cxr_gate.py`):
- Stage 1 biophysical checks eliminate non-image payloads, corrupt byte streams, extreme aspect ratios, and polychromatic photographic images.
- Stage 2 semantic screening enforces an empirically audited confidence threshold ($	au = 0.83$) against thoracic skeletal and mediastinal anatomy.
- Non-CXR inputs are immediately isolated, preventing out-of-distribution contamination from reaching the disease classifier.

## 13.5 Data Protection, Patient UUIDs & De-Identification Principles

While formal compliance with health data privacy regulations (such as HIPAA in the United States or the Digital Personal Data Protection Act [DPDP] in India) requires institutional infrastructure audits, administrative policies, and physical safeguards beyond the scope of this software prototype, LungAI incorporates privacy-conscious engineering principles:
- **De-Identification & Entity Masking**: Patient records are identified within the persistence tier using cryptographically generated Universally Unique Identifiers (UUIDv4), preventing sequential enumeration attacks.
- **Client-Side Metadata Stripping**: Uploaded image payloads have all DICOM patient metadata headers (Patient Name, Date of Birth, Social Security Number, Medical Record Number) stripped in memory prior to file persistence.
- **Isolated File Storage**: Raw radiographs and rendered Grad-CAM overlays are stored in secured local directories with access restricted to the backend service process.

## 13.6 Relational Auditability & Diagnostic Persistence

To support retrospective clinical audits and medico-legal accountability, the SQLAlchemy persistence engine logs comprehensive transaction metadata for every analyzed radiograph:
- Cryptographic SHA-256 hash of the uploaded image;
- Exact timestamp of ingestion and inference execution;
- Inference latency in milliseconds;
- Full 6-class softmax probability simplex;
- Assigned clinical urgency triage tier;
- File system reference to the rendered Grad-CAM visualization.

Table 13.1 synthesizes the security, privacy, and responsible AI governance controls implemented across the platform.

### Table 13.1: Security, Privacy, and Responsible AI Governance Matrix

| Governance Dimension | Specific Risk Addressed | Implementing Control Mechanism | Clinical & Regulatory Impact |
| :--- | :--- | :--- | :--- |
| **Clinical Misuse** | Clinicians treating AI predictions as autonomous diagnoses | Permanent banner disclaimers & mandatory radiologist sign-off | Enforces Human-in-the-Loop oversight; mitigates liability |
| **Out-of-Distribution Inputs** | Non-CXR images generating misleading thoracic diagnoses | Two-stage defense-in-depth gate at $	au=0.83$ | Intercepts 100% of tested non-radiographic uploads |
| **Patient Identity Disclosure** | Exposure of protected health information (PHI) | UUIDv4 primary keys; DICOM header sanitization | Protects patient confidentiality in research environments |
| **Tamper & Integrity Risk** | Modification of stored radiographic images | SHA-256 cryptographic hashing upon ingestion | Enables forensic verification of image integrity |
| **Regulatory Misrepresentation**| Users assuming formal FDA/CE clearance | Explicit documentation of academic research status | Establishes legally defensible research boundaries |
| **Model Inscrutability** | Opaque black-box predictions generating distrust | Integrated Grad-CAM saliency heatmaps on `conv5_block16` | Facilitates radiological verification of lesion anatomy |

---

# CHAPTER 14 — LIMITATIONS

In accordance with rigorous academic and scientific standards, this chapter details the technical, methodological, and clinical limitations of the LungAI research project:

1. **Catastrophic Fine-Grained Collapse on External Digitized Film (The Montgomery Benchmark)**: While Model D achieved 100% binary abnormal sensitivity on the quarantined Montgomery County benchmark ($N=138$), it exhibited complete fine-grained classification collapse (0% exact TB recall and 0% Normal specificity), systematically assigning cases to Pulmonary Nodule / Mass. This establishes that zero-shot transfer across radical hardware shifts (analog film digitizers vs. modern digital detectors) remains an unsolved challenge.
2. **Severe Source-Class Confounding across Public Benchmarks**: Statistical auditing demonstrated severe source-label confounding across public repositories (Cramér's V = 0.7654). Certain disease categories (e.g., COVID-19) originate predominantly from single collections, creating an inherent risk that convolutional kernels capture repository-specific artifacts despite low-pass filtering.
3. **Severe Class Imbalance**: The canonical V5 cohort exhibits substantial class prevalence disparities, ranging from 2,805 Pneumonia scans to 754 Pulmonary Nodule / Mass scans. While cost-sensitive class weighting was applied, minority classes (Nodule/Mass and Pleural Effusion) exhibited lower F1-scores (53.03% and 55.32%).
4. **Diagnostic Ambiguity of Planar Nodules and Effusions**: Solitary pulmonary nodules and pleural fluid blunting are notoriously difficult to evaluate on two-dimensional planar radiographs due to overlying rib crossings and cardiac silhouettes. High-sensitivity detection of nodules (85.19% recall) was accompanied by low precision (38.49%) due to false alarms on normal skeletal structures.
5. **Lack of Complete Patient Metadata for External Subsets**: Several open-access collections (e.g., Kaggle pneumonia and COVID-19 subsets) lack comprehensive patient de-identification metadata. While exhaustive cryptographic and perceptual deduplication was executed, absolute zero patient overlap cannot be mathematically guaranteed for un-annotated subsets.
6. **Absence of Lateral Projection Radiographs**: All training and evaluation scans are frontal radiographs (PA or AP views). In clinical practice, evaluating retrocardiac consolidations, posterior costophrenic effusions, and mediastinal masses frequently requires lateral projection views, which are absent from the current pipeline.
7. **Single-Label Multi-Class Modeling Assumption**: LungAI enforces a single-label multi-class Softmax formulation. In real-world clinical radiology, thoracic pathologies frequently co-occur (e.g., a patient presenting with both bacterial pneumonia and pleural effusion). Forcing a single-label prediction oversimplifies multi-pathology co-occurrences.
8. **Threshold-Selection Data Leakage in CXR Gate**: The production threshold $	au = 0.83$ for the Stage 2 CXR validation gate was selected and optimized on the same 125-image evaluation cohort used for reporting metrics. Consequently, the reported 100% sensitivity, 100% specificity, and 1.0000 ROC-AUC represent functional verification estimates rather than unbiased generalization metrics.
9. **Small Gate Evaluation Cohort**: The CXR gate was evaluated on 125 images (48 CXR, 77 non-CXR). While effective for initial functional verification, a cohort of this size cannot capture the full spectrum of global non-radiographic medical and non-medical images.
10. **Absence of Empirical Probability Calibration**: Modern deep neural networks frequently produce overconfident probability distributions near 0.0 or 1.0. The Softmax probabilities output by Model D were not subjected to post-hoc calibration techniques (such as Platt scaling, temperature scaling, or isotonic regression).
11. **Heuristic Clinical Urgency Triage Rules**: The triage stratification rules (Emergency, Urgent, Routine) represent application-level research logic based on probability thresholds rather than prospectively validated clinical practice guidelines.
12. **Absence of Prospective Clinical Trial Validation**: All empirical findings reported in this dissertation are based on retrospective datasets. Prospective clinical trials in real hospital emergency departments are required to evaluate clinical efficacy and patient outcomes.
13. **Absence of Radiologist Multi-Reader Study**: The diagnostic accuracy of Model D was not evaluated head-to-head against human radiologists on the identical V5 test cohort in a controlled multi-reader, multi-case (MRMC) study.
14. **Lack of Anatomical Lung Field Segmentation**: Model D processes the entire cropped radiograph without explicit lung field segmentation masking. Consequently, background areas (neck soft tissues, abdomen, lateral markers) could theoretically influence feature activations.
15. **Fixed Spatial Input Resolution ($224 	imes 224$)**: Resampling high-resolution medical radiographs ($2000 	imes 2000+$ pixels) to $224 	imes 224$ pixels reduces high-frequency spatial detail, potentially obscuring subtle micro-nodules or fine apical cavitation lines.
16. **Lack of Direct PACS / DICOM Protocol Integration**: The current platform ingests files via HTTP multipart REST endpoints. In enterprise hospital environments, native DICOM C-STORE and C-FIND network protocols are required for seamless PACS communication.
17. **Absence of Formal HIPAA / GDPR Certification**: While the platform incorporates de-identification and access control principles, it has not undergone formal regulatory third-party audits for HIPAA, GDPR, or ISO 27001 certification.
18. **Unresolved Domain Shift Across Sensor Technologies**: Although frequency-aware spatial filtering significantly improved internal multi-source performance (from 76.18% to 82.93%), it did not resolve the profound domain shift encountered on external analog film digitizers, demonstrating the fundamental boundaries of pixel-level filtering.

---

# CHAPTER 15 — FUTURE ENHANCEMENTS

Based on the empirical insights and limitations established in this dissertation, a prioritized research and engineering roadmap is formulated across six domains:

## 15.1 Independent Multi-Center Gate Validation
To establish unbiased operational generalization, future work will assemble a multi-institutional out-of-distribution evaluation benchmark containing over 5,000 diverse images—including cross-sectional CT slices, MRI scans, ultrasound, nuclear medicine, fluoroscopy, mobile phone camera photos, and digitized documents. The decision threshold $	au$ will be strictly optimized on a separate validation partition to eliminate threshold-selection data leakage.

## 15.2 Post-Hoc Probability Calibration & Uncertainty Estimation
To transform raw Softmax probabilities into true posterior class likelihoods, future work will integrate:
- **Temperature Scaling**: Learning a single scalar parameter $T > 0$ on the validation split to rescale logits before softmax: $\hat{p}_k = rac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$ [21], minimizing Expected Calibration Error (ECE);
- **Monte Carlo Dropout & Evidential Deep Learning**: Estimating epistemic (model) and aleatoric (data) uncertainty during inference to explicitly flag low-confidence or out-of-distribution scans for mandatory human review.

## 15.3 Multi-Label Formulation & Lesion Localization
To model complex multi-pathology presentations:
- Transition from multi-class categorical cross-entropy to **binary cross-entropy across multi-label targets**, allowing independent probability estimation for co-occurring conditions (e.g., Pneumonia + Pleural Effusion);
- Integrate **weakly supervised lesion localization** using Grad-CAM++ or bounding-box regression heads (e.g., YOLOv8-CXR) to provide precise spatial coordinates for suspicious parenchymal nodules.

## 15.4 Anatomical Lung Field Segmentation Masking
To insulate convolutional feature extractors from peripheral scanner artifacts and lateral markers, a pre-inference anatomical segmentation network (e.g., U-Net or SegFormer) will be trained to generate binary masks of the bilateral lung fields, isolating thoracic parenchyma prior to disease classification.

## 15.5 Prospective Multi-Reader Clinical Study
A multi-center, multi-reader study will be conducted pairing certified radiologists, emergency physicians, and resident medical officers evaluating a balanced cohort with and without LungAI decision support. Measuring diagnostic sensitivity, specificity, and report turnaround times will clinically quantify the impact of automated triage.

## 15.6 Federated Learning & Enterprise PACS Integration
To enable continuous model training across multiple hospital networks without centralizing sensitive patient radiographs, future development will deploy privacy-preserving **Federated Learning** (e.g., FedAvg with differential privacy). Concurrently, native DICOM C-STORE and DIMSE network adapters will be implemented to integrate directly with hospital PACS archives.

Table 15.1 summarizes the prioritized future enhancements roadmap.

### Table 15.1: Prioritized Future Research and Engineering Enhancements Roadmap

| Priority Tier | Research / Engineering Domain | Specific Technical Intervention | Expected Scientific / Clinical Impact |
| :---: | :--- | :--- | :--- |
| **Tier 1 (Immediate)** | **Gate Validation** | Independent 5,000-image multi-center gate validation with split-isolated threshold | Eliminates threshold leakage; verifies real-world non-CXR rejection |
| **Tier 1 (Immediate)** | **Calibration** | Temperature scaling & Expected Calibration Error (ECE) optimization | Calibrates confidence scores; aligns probabilities with true clinical prevalence |
| **Tier 2 (Near-Term)** | **Multi-Label Modeling** | Binary cross-entropy multi-label classification head | Enables simultaneous detection of co-occurring conditions (e.g., Pneumonia + Effusion) |
| **Tier 2 (Near-Term)** | **Lung Segmentation** | U-Net / SegFormer lung field masking prior to inference | Suppresses peripheral shortcuts, lateral markers, and diaphragm artifacts |
| **Tier 3 (Medium-Term)**| **Lesion Localization** | Grad-CAM++ & weakly supervised bounding-box generation | Provides millimeter-level spatial bounding coordinates for solitary nodules |
| **Tier 3 (Medium-Term)**| **Clinical Reader Study**| Prospective multi-reader multi-case (MRMC) trial with certified radiologists | Clinically proves diagnostic turnaround acceleration and error reduction |
| **Tier 4 (Long-Term)** | **Enterprise Deployment**| Native DICOM C-STORE/PACS protocol integration & Federated Learning | Enables multi-hospital privacy-preserving continuous learning |

---

# CHAPTER 16 — CONCLUSION

## 16.1 Executive Summary of Research Investigation

This M.Tech dissertation presented the comprehensive research, empirical benchmarking, and software engineering realization of **LungAI**—an auditable, end-to-end medical AI decision-support platform designed for robust multi-class thoracic disease detection and cross-source domain generalization from chest radiographs.

Motivated by critical global radiologist shortages, systemic PACS FIFO queuing delays, and the high inter-observer variability of manual radiograph reading, the research addressed the fundamental failure modes of contemporary thoracic CAD systems: shortcut learning, dataset contamination, cross-modality confounders, and severe sensor domain shift.

Through disciplined dataset auditing, the project reconstructed the canonical, leak-free **V5 multi-source cohort** comprising 10,547 verified planar radiographs from 10,270 unique clinical patients across eight international repositories, enforcing strict patient-level partitioning (0% patient overlap) across six diagnostic categories: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, and Tuberculosis.

## 16.2 Resolution of Core Research Question

The research was formulated around the central research question:

> *"Can a multi-source six-class chest X-ray classifier learn disease-relevant representations that generalize across independent acquisition sources, and can a domain-generalization strategy improve cross-source performance compared with a standard DenseNet-121 baseline?"*

The empirical investigations deliver a definitive, nuanced resolution:

1. **Internal Multi-Source Generalization is Decisively Achievable**:
   - The standard Empirical Risk Minimization baseline achieved an internal test accuracy of 76.18% and a macro F1 of 72.63%.
   - Latent covariance alignment (Deep CORAL) successfully improved multi-source performance to 80.89% accuracy and 76.85% macro F1.
   - Biophysical frequency-aware preprocessing (Gaussian spatial low-pass filtering at $\sigma=1.0$) achieved the decisive peak internal performance: **82.93% test accuracy, 79.31% macro precision, 80.73% macro recall, 78.35% macro F1-score, 0.9755 macro ROC-AUC, and 0.8391 macro PR-AUC** across the six disease categories.
   - Consequently, **domain-generalization strategies—specifically input frequency filtering and latent covariance alignment—significantly improve internal multi-source performance compared to a standard DenseNet-121 baseline**.
2. **Adversarial Adaptation Does Not Yield Universal Gains**:
   - Domain-Adversarial Neural Networks (DANN) failed to outperform the baseline (76.24% vs. 76.18%), proving that minimax gradient reversal can degrade class-discriminative representations on heterogeneous medical datasets.
3. **Cross-Modality Sensor Shift Remains a Fundamental Clinical Boundary**:
   - Zero-shot evaluation on the quarantined Montgomery County benchmark ($N=138$) revealed complete fine-grained classification collapse (0% exact TB recall), caused by approximately $4	imes$ higher high-frequency edge variance in digitized analog film. This proves that while frequency filtering successfully resolves internal multi-source variations, radical sensor shifts across disparate hardware technologies require target-domain calibration or lung field segmentation.
4. **Evidence-Driven Model D Selection**:
   - Model D was selected as the final production engine through controlled empirical arbitration, demonstrating that mathematically principled biophysical interventions can outperform more complex architectures.

## 16.3 Key Empirical & Engineering Insights

The investigation establishes four enduring engineering principles for medical artificial intelligence:
1. **Dataset Forensics Precedes Algorithmic Optimization**: Deep learning models will infallibly exploit cross-modality contamination (e.g., CT slices in CXR datasets) and patient identity leakage. Rigorous cryptographic hashing and patient-strict partitioning are mandatory foundations for scientific validity.
2. **Defensive Input Verification is Mandatory for Clinical Safety**: Unconstrained classifiers inevitably output false predictions on out-of-distribution files. The two-stage CXR validation gate operating at $	au=0.83$ proves that biophysical and semantic screening can intercept 100% of tested non-radiographic uploads.
3. **Transparent Disclosure of Failure Modes Builds Scientific Trust**: Acknowledging the Montgomery generalization collapse, disclosure of threshold-selection leakage, and documenting negative results (DANN and Hybrid) elevate this dissertation from an optimistic proof-of-concept into a serious scientific investigation.
4. **Full-Stack Software Realization Bridges Theory and Practice**: Implementing an asynchronous FastAPI microservice, SQLAlchemy ORM persistence, and a responsive React 18 interface with deterministic urgency triage proves that deep learning models can be transformed into ergonomic, auditable clinical decision-support systems.

## 16.4 Final Concluding Remarks

In conclusion, LungAI establishes a verified, reproducible, and transparent reference architecture for multi-source computer-aided thoracic diagnosis. By proving that frequency-aware spatial filtering strengthens multi-source representation learning while candidly delineating the boundaries of external domain transfer, this dissertation contributes foundational methodologies toward the realization of safe, robust, and clinically trustworthy artificial intelligence in diagnostic radiology.

---

# REFERENCES

* [1] World Health Organization, 'The top 10 causes of death,' WHO Global Health Estimates, Geneva, Switzerland, Dec. 2020. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death.
* [2] D. S. Kermany, M. Goldbaum, W. Cai, C. C. Valentim, H. Liang, S. L. Baxter, A. McKeown, G. Yang, X. Wu, F. Yan, and J. Dong, 'Identifying medical diagnoses and treatable diseases by image-based deep learning,' Cell, vol. 172, no. 5, pp. 1122–1131, Feb. 2018. DOI: 10.1016/j.cell.2018.02.010.
* [3] M. E. H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M. A. Kadir, Z. B. Mahbub, K. R. Islam, M. S. Khan, A. Iqbal, N. Al Emadi, M. B. I. Reaz, and M. T. Islam, 'Can AI help in screening viral and COVID-19 pneumonia?,' IEEE Access, vol. 8, pp. 132665–132676, 2020. DOI: 10.1109/ACCESS.2020.3010287.
* [4] T. Rahman, A. Khandakar, M. A. Kadir, K. R. Islam, K. F. Islam, R. Mazhar, T. Hamid, M. T. Islam, S. Kashem, Z. B. Mahbub, M. A. Ayari, and M. E. H. Chowdhury, 'Reliable tuberculosis detection using chest X-ray with deep learning, segmentation and visualization,' IEEE Access, vol. 8, pp. 191586–191601, 2020. DOI: 10.1109/ACCESS.2020.3031384.
* [5] P. Rajpurkar, J. Irvin, K. Zhu, B. Yang, H. Mehta, T. Duan, D. Ding, A. Bagul, R. L. Ball, C. Langlotz, K. Shpanskaya, M. P. Lungren, and A. Y. Ng, 'CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning,' arXiv:1711.05225, 2017.
* [6] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, 'ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases,' in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 2097–2106. DOI: 10.1109/CVPR.2017.369.
* [7] J. Irvin, P. Rajpurkar, M. Ko, Y. Yu, S. Ciurea-Ilcus, C. Chute, H. Marklund, B. Hepworth, P. Shen, K. Shpanskaya, M. P. Lungren, and A. Y. Ng, 'CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison,' in Proc. AAAI Conf. Artif. Intell., vol. 33, no. 1, 2019, pp. 590–597. DOI: 10.1609/aaai.v33i01.3301590.
* [8] A. J. DeGrave, J. D. Janizek, and S.-I. Lee, 'AI for radiographic COVID-19 detection selects shortcuts over signal,' Nature Machine Intelligence, vol. 3, no. 7, pp. 610–619, May 2021. DOI: 10.1038/s42256-021-00338-7.
* [9] C. Fernando, S. Kolonne, H. Kumarasinghe, and D. Meedeniya, 'Chest Radiographs Classification Using Multi-model Deep Learning: A Comparative Study,' in Proc. 2nd Int. Conf. Adv. Res. Comput. (ICARC), 2022, pp. 165–170. DOI: 10.1109/ICARC54489.2022.9753811.
* [10] A. Mahesh and N. Kumar, 'Respiratory Diseases Detection Using Deep Learning Methods,' in Proc. 2023 2nd Int. Conf. Trends Electr., Electron. Comput. (TEECCON), 2023, pp. 1–6. DOI: 10.1109/TEECCON59234.2023.10335887.
* [11] K. S. Charan, O. V. Krishna, P. V. Sai, and A. K. Ilavarasi, 'Transfer Learning Based Multi-Class Lung Disease Prediction Using Textural Features Derived From Fusion Data,' IEEE Access, vol. 12, pp. 108248–108262, 2024. DOI: 10.1109/ACCESS.2024.3435680.
* [12] G. H. Dagnaw and M. El Mouthadi, 'Towards Explainable Artificial Intelligence for Pneumonia and Tuberculosis Classification from Chest X-Ray,' in Proc. 2023 Int. Conf. ICT Dev. Africa (ICT4DA), 2023, pp. 69–74. DOI: 10.1109/ICT4DA59526.2023.10302183.
* [13] R. Deva and A. Dagur, 'ViT-ResNet Fusion: An Explainable Hybrid Framework for High-Accuracy Multiclass Lung Disease Classification in Chest X-Rays,' IEEE Access, vol. 13, 2025. DOI: 10.1109/ACCESS.2025.3649109.
* [14] F. B. Akyol and G. Bilgin, 'A Hybrid CNN–Transformer Domain-Adversarial Neural Network for Robust Pneumonia Classification in Heterogeneous Chest X-Ray Datasets,' in Proc. 10th Int. Conf. Comput. Sci. Eng. (UBMK), Sept. 2025.
* [15] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, 'Densely connected convolutional networks,' in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 4700–4708. DOI: 10.1109/CVPR.2017.243.
* [16] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep residual learning for image recognition,' in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778. DOI: 10.1109/CVPR.2016.90.
* [17] M. Tan and Q. Le, 'EfficientNet: Rethinking model scaling for convolutional neural networks,' in Proc. 36th Int. Conf. Mach. Learn. (ICML), 2019, pp. 6105–6114.
* [18] B. Sun and K. Saenko, 'Deep CORAL: Correlation alignment for deep domain adaptation,' in Proc. Eur. Conf. Comput. Vis. (ECCV) Workshops, 2016, pp. 443–450. DOI: 10.1007/978-3-319-49409-8_35.
* [19] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky, 'Domain-adversarial training of neural networks,' Journal of Machine Learning Research, vol. 17, no. 59, pp. 1–35, 2016.
* [20] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, 'Grad-CAM: Visual explanations from deep networks via gradient-based localization,' in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 618–626. DOI: 10.1109/ICCV.2017.74.
* [21] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, 'On calibration of modern neural networks,' in Proc. 34th Int. Conf. Mach. Learn. (ICML), 2017, pp. 1321–1330.
* [22] K. Zuiderveld, 'Contrast limited adaptive histogram equalization,' in Graphics Gems IV, P. S. Heckbert, Ed. San Diego, CA: Academic Press, 1994, pp. 474–485. DOI: 10.1016/B978-0-12-336156-1.50061-6.
* [23] S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wáng, P.-X. Lu, and G. Thoma, 'Two public chest X-ray datasets for computer-aided screening of pulmonary diseases,' Quantitative Imaging in Medicine and Surgery, vol. 4, no. 6, pp. 475–477, Dec. 2014. DOI: 10.3978/j.issn.2223-4292.2014.11.20.
* [24] J. Shiraishi, S. Katsuragawa, J. Ikezoe, T. Matsumoto, T. Kobayashi, K. Komatsu, M. Matsui, H. Fujita, Y. Kodera, and K. Doi, 'Development of a digital image database for chest radiographs with and without a lung nodule: Receiver operating characteristic analysis of radiologists' detection of pulmonary nodules,' American Journal of Roentgenology, vol. 174, no. 1, pp. 71–74, Jan. 2000. DOI: 10.2214/ajr.174.1.1740071.
* [25] H. Q. Nguyen, K. Lam, L. T. Le, P. H. Pham, H. N. Tran, D. B. Nguyen, D. D. Le, C. M. Pham, H. T. T. Tong, D. H. Dinh, C. D. Do, L. T. Doan, C. X. Nguyen, B. Q. Nguyen, B. V. Nguyen, N. B. Dang, B. D. Nguyen, M. T. Dang, and H. T. Nguyen, 'VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations,' Scientific Data, vol. 9, Art. no. 429, Jul. 2022. DOI: 10.1038/s41597-022-01498-w.
* [26] Y. Liu, Y. Wu, Y. Ban, H. Wang, and M. Cheng, 'TBX11K: A large-scale dataset for tuberculosis detection and evaluation,' in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020.
* [27] A. Bustos, A. Pertusa, J.-M. Salinas, and M. de la Iglesia-Vayá, 'PadChest: A large chest X-ray image dataset with multi-label annotations along with associated raw medical image reports and technical information,' Medical Image Analysis, vol. 66, Art. no. 101797, Dec. 2020. DOI: 10.1016/j.media.2020.101797.
* [28] G. Litjens, T. Kooi, B. E. Bejnordi, A. A. A. Setio, F. Ciompi, M. Ghafoorian, J. A. van der Laak, B. van Ginneken, and C. I. Sánchez, 'A survey on deep learning in medical image analysis,' Medical Image Analysis, vol. 42, pp. 60–88, Dec. 2017. DOI: 10.1016/j.media.2017.07.005.
* [29] A. Esteva, B. Kuprel, R. A. Novoa, J. Ko, S. M. Swetter, H. M. Blau, and S. Thrun, 'Dermatologist-level classification of skin cancer with deep neural networks,' Nature, vol. 542, no. 7639, pp. 115–118, Feb. 2017. DOI: 10.1038/nature21056.
* [30] C. Shorten and T. M. Khoshgoftaar, 'A survey on image data augmentation for deep learning,' Journal of Big Data, vol. 6, no. 1, Art. no. 60, Jul. 2019. DOI: 10.1186/s40537-019-0197-0.
* [31] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, 'ImageNet: A large-scale hierarchical image database,' in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2009, pp. 248–255. DOI: 10.1109/CVPR.2009.5206848.
* [32] D. P. Kingma and J. Ba, 'Adam: A method for stochastic optimization,' in Proc. 3rd Int. Conf. Learn. Represent. (ICLR), 2015, pp. 1–15.
* [33] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, 'Dropout: A simple way to prevent neural networks from overfitting,' Journal of Machine Learning Research, vol. 15, no. 56, pp. 1929–1958, 2014.
* [34] S. Ioffe and C. Szegedy, 'Batch normalization: Accelerating deep network training by reducing internal covariate shift,' in Proc. 32nd Int. Conf. Mach. Learn. (ICML), 2015, pp. 448–456.
* [35] M. Abadi et al., 'TensorFlow: A system for large-scale machine learning,' in Proc. 12th USENIX Conf. Oper. Syst. Des. Implementation (OSDI), 2016, pp. 265–283.
* [36] S. Ramírez, 'FastAPI: Modern, fast, high-performance web framework for building APIs with Python,' 2018. [Online]. Available: https://github.com/fastapi/fastapi.

---

# APPENDICES

## Appendix A: Prototype Runtime Environment Configuration Parameters

Table A.1 lists the operational runtime configuration parameters of the LungAI prototype.

### Table A.1: Prototype Runtime Environment Configuration

| Parameter / Configuration Key | Configured Production Value | Functional Role in System |
| :--- | :--- | :--- |
| `API_HOST` / `API_PORT` | `0.0.0.0 : 8000` | Asynchronous ASGI network listener binding for local and containerized access |
| `DATABASE_URL` | `sqlite+aiosqlite:///./lung_disease.db` | Asynchronous database connection URI (compatible with PostgreSQL via asyncpg) |
| `PRODUCTION_MODEL_PATH` | `models/densenet121_frequency_v5.h5` | Serialized HDF5 weights artifact for Model D (DenseNet-121 Frequency V5) |
| `PRODUCTION_CXR_GATE_PATH`| `models/cxr_gate_model.h5` | Serialized HDF5 weights artifact for Stage 2 semantic CXR gate classifier |
| `PRODUCTION_CXR_THRESHOLD`| `0.83` | Audited decision threshold for semantic CXR verification |
| `MAX_UPLOAD_SIZE_BYTES` | `10,485,760` (10 MB) | Defensive upload ceiling for multipart image payloads |
| `IMAGE_TARGET_SIZE` | `(224, 224)` | Standardized spatial tensor resolution for deep neural input |
| `ACTIVE_CLASSES` | `COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, Tuberculosis` | 6-Class unified diagnostic taxonomy (backend/ml/class_mapping.json) |
| `GRADCAM_TARGET_LAYER` | `conv5_block16_concat` | Final dense concatenation layer targeted for gradient backpropagation |
| `TRIAGE_CONFIDENCE_CEILING`| `0.70` (Pneumonia/COVID-19) | Deterministic confidence threshold triggering Emergency clinical triage tier |

## Appendix B: Core Preprocessing & Inference Implementation Listing

```python
# Model D Preprocessing and Singleton Inference Implementation (Python / OpenCV / TensorFlow)
import cv2
import numpy as np
import tensorflow as tf

def preprocess_cxr(image_bytes: bytes) -> np.ndarray:
    """Transforms raw radiograph byte stream into standardized Model D input tensor."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes into valid radiograph.")
    
    # 1. Color space conversion & Luminance CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lab = cv2.merge((clahe.apply(l), a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 2. Gaussian Spatial Low-Pass Filtering (sigma = 1.0)
    # Filters out high-frequency scanner grain and digitizer noise shortcuts
    filtered = cv2.GaussianBlur(enhanced_bgr, (3, 3), 1.0)
    enhanced_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    
    # 3. High-Order Lanczos-4 Spatial Resampling to 224x224
    resized = cv2.resize(enhanced_rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    
    # 4. Two-Step Intensity Standardization (ImageNet population statistics)
    scaled = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    standardized = (scaled - mean) / std
    
    return np.expand_dims(standardized, axis=0)
```

## Appendix C: CXR Input-Validation Gate Implementation Listing

```python
# Two-Stage CXR Input-Validation Gate Implementation (backend/ml/cxr_gate.py)
import cv2
import numpy as np

PRODUCTION_CXR_THRESHOLD = 0.83

def evaluate_stage1_biophysical(img_bgr: np.ndarray) -> tuple[bool, str]:
    """Performs Stage 1 biophysical and chromatic screening."""
    h, w = img_bgr.shape[:2]
    if h < 32 or w < 32:
        return False, "Image dimensions below minimum resolution (32x32)"
    
    aspect_ratio = max(w / h, h / w)
    if aspect_ratio > 2.2:
        return False, f"Aspect ratio {aspect_ratio:.2f} exceeds anatomical bounds (<=2.2)"
    
    # Luminance variance check
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 10.0:
        return False, "Low luminance variance; possible synthetic or blank document"
    
    # Polychromatic saturation screening (filters out color photographs)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    high_sat_pct = float(np.mean(sat > 60))
    if high_sat_pct > 0.15:
        # Check hue distribution diversity
        hues = hsv[:, :, 0][sat > 60]
        hist, _ = np.histogram(hues, bins=6, range=(0, 180))
        active_bins = int(np.sum(hist > len(hues) * 0.05))
        if active_bins >= 3:
            return False, "Polychromatic content detected; input is a color photograph"
            
    return True, "Passed Stage 1 biophysical screening"
```

## Appendix D: Automated Verification Suite Summary (44/44 Passed)

The automated verification suite was executed under Python 3.10 and Pytest 7.4. All 44 test cases passed with 0 failures, 0 errors, and 0 warnings:
- `backend/tests/test_preprocess.py`: 6 passed
- `backend/tests/test_inference.py`: 9 passed
- `backend/tests/test_gate.py`: 8 passed
- `backend/tests/test_api.py`: 7 passed
- `backend/tests/test_patients.py`: 6 passed
- `backend/tests/test_reports.py`: 8 passed
- **Total Test Result**: **44 / 44 PASSED (100% Pass Rate)**

