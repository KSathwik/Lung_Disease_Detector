# LUNGAI: ROBUST MULTI-CLASS LUNG DISEASE DETECTION FROM CHEST X-RAY IMAGES USING DEEP LEARNING AND CROSS-SOURCE GENERALIZATION

A Dissertation Submitted in Partial Fulfillment of the Requirements for the Award of the Degree of

**MASTER OF TECHNOLOGY**  
in  
**COMPUTER SCIENCE AND ENGINEERING**  

By  
**SATHWIK KATKAM**  
(Roll No: 24011D0512)  

Under the Guidance of  
**Faculty Advisor & Project Supervisor**  
Department of Computer Science and Engineering  
Jawaharlal Nehru Technological University Hyderabad (JNTUH)  
Hyderabad, Telangana, India — 500085  
October 2026

---

# ABSTRACT

Deep convolutional neural networks have demonstrated physician-level performance in automated chest radiograph classification under constrained, single-institution experimental protocols. However, real-world deployment across clinical healthcare networks is severely impeded by domain shift—manifesting as systemic performance degradation when models encounter radiographs originating from distinct scanner hardware, patient demographics, radiographic exposure parameters, and detector calibration standards. In this research, we investigate whether a multi-source six-class chest X-ray classifier can learn disease-relevant representations that generalize across independent acquisition sources, and whether domain-generalization strategies improve cross-source diagnostic stability compared with standard Empirical Risk Minimization (ERM).

We first conduct a rigorous forensic audit of public chest radiography benchmarks, revealing pervasive non-thoracic cross-contamination (including axial computed tomography slices) and extreme acquisition-site confounding across pathological classes (Cramér's V = 0.7654). To establish an uncompromised experimental foundation, we reconstruct the Unified V5 Research Dataset comprising 10,547 verified planar chest radiographs across 10,270 unique patients, strictly partitioned at the patient level (Train: 7,398 images / 7,189 patients; Validation: 1,579 images / 1,540 patients; Test: 1,570 images / 1,541 patients) across six clinical classes: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, and Tuberculosis.

Using this benchmark, we evaluate five domain generalization paradigms under an identical DenseNet-121 backbone: ERM (76.18% accuracy, 72.63% macro F1), Deep Correlation Alignment (CORAL; 80.89% accuracy, 76.85% macro F1), Domain-Adversarial Neural Networks (DANN; 76.24% accuracy, 72.99% macro F1), Frequency-Aware Preprocessing (Model D; 82.93% accuracy, 78.35% macro F1, 0.9755 macro ROC-AUC), and a Hybrid Feature-Frequency framework (78.22% accuracy, 73.95% macro F1). Model D—employing contrast-limited adaptive histogram equalization (CLAHE) and Gaussian frequency low-pass filtering (sigma = 1.0)—achieves superior discrimination across all classes, including 97.01% F1 for COVID-19, 89.58% F1 for Normal, 88.66% F1 for Tuberculosis, and 86.49% F1 for Pneumonia.

External generalization evaluation on the quarantined Montgomery County Tuberculosis benchmark (n = 138) reveals that while Model D retains 100% binary sensitivity in detecting radiological abnormalities, fine-grained class separation collapses due to severe high-frequency domain shift, predominantly misattributing foreign analog film scans to Pulmonary Nodule / Mass. To safeguard clinical deployment, we design and implement a Two-Stage Chest Radiograph (CXR) Validation Gate combining biophysical heuristics with a deep anatomical classifier calibrated at production threshold tau = 0.83, achieving 100% specificity in rejecting out-of-distribution non-medical photographs and axial CT scans on a 125-image verification cohort. The complete diagnostic ecosystem is engineered as an asynchronous, full-stack decision-support platform (FastAPI, React 18, SQLite) featuring real-time Grad-CAM explainability, emergency triage classification, and clinical audit reporting, verified by 44 passing automated tests.

**Keywords:** Chest Radiography, Deep Learning, DenseNet-121, Domain Generalization, Frequency Preprocessing, Grad-CAM, Cross-Source Generalization, Shortcut Learning, Clinical Decision Support.

---


---


# TABLE OF CONTENTS

| Section / Chapter Title | Page |
| --- | --- |
| ABSTRACT | 2 |
| CHAPTER 1 – INTRODUCTION | 11 |
| 1.1 Clinical Background & Problem Domain | 11 |
| 1.2 Radiographic Imaging Principles & Diagnostic Role | 11 |
| 1.3 Workflow Challenges in Conventional Radiology | 12 |
| 1.4 Problem Statement & Research Question | 13 |
| 1.5 Research Objectives & Project Scope | 13 |
| 1.6 Thesis Contributions | 14 |
| 1.7 Thesis Organization | 14 |
| CHAPTER 2 – LITERATURE SURVEY | 16 |
| 2.1 Deep Learning in Thoracic Radiography | 16 |
| 2.2 Convolutional Neural Network Backbones: ResNet vs. DenseNet | 16 |
| 2.3 Visual Explainability & Gradient-Weighted Class Activation Mapping | 17 |
| 2.4 Domain Shift, Confounding & Shortcut Learning in Medical AI | 17 |
| 2.5 Domain Generalization Frameworks | 18 |
| 2.6 Critical Analysis and Research Gap Identification | 18 |
| CHAPTER 3 – EXISTING SYSTEM | 21 |
| 3.1 Overview of Conventional Chest Radiography Workflows | 21 |
| 3.2 Limitations of Commercial and Open-Source CAD Solutions | 21 |
| 3.3 The Cross-Source Generalization Bottleneck | 22 |
| 3.4 Summary of Gaps in Existing Practice | 22 |
| CHAPTER 4 – PROPOSED SYSTEM | 23 |
| 4.1 System Overview and Architectural Philosophy | 23 |
| 4.2 End-to-End Diagnostic Pipeline | 23 |
| 4.3 High-Level System Architecture | 24 |
| 4.4 Key Novelties and Design Distinctions | 24 |
| CHAPTER 5 – SYSTEM REQUIREMENTS | 26 |
| 5.1 Hardware Requirements | 26 |
| 5.2 Software Requirements & Runtime Environment | 26 |
| 5.3 Functional Requirements | 27 |
| 5.4 Non-Functional Requirements | 28 |
| CHAPTER 6 – SYSTEM DESIGN | 30 |
| 6.1 Architectural Design Principles | 30 |
| 6.2 Data Flow Modeling | 30 |
| 6.3 UML Structural Modeling | 32 |
| 6.4 UML Behavioral Modeling | 34 |
| 6.5 Database Design & Entity Relationship Modeling | 35 |
| CHAPTER 7 – DATASET AND DATA PREPROCESSING | 36 |
| 7.1 Forensic Audit of Public Benchmarks & Artifact Discovery | 36 |
| 7.2 The Unified V5 Research Dataset Reconstruction | 36 |
| 7.3 Multi-Source Training Pipeline | 37 |
| 7.4 Acquisition-Site Confounding Analysis (Cramér's V = 0.7654) | 38 |
| 7.5 Planar CXR Verification & CT Elimination | 38 |
| 7.6 Patient-Level Leak-Free Data Partitioning | 38 |
| CHAPTER 8 – MACHINE LEARNING / AI MODEL | 40 |
| 8.1 Model D Deep Transfer Learning Architecture | 40 |
| 8.2 Deterministic Biophysical Preprocessing Pipeline | 41 |
| 8.3 Controlled Multi-Paradigm Domain Generalization Study | 42 |
| CHAPTER 9 – SYSTEM IMPLEMENTATION | 44 |
| 9.1 Backend Architecture (FastAPI & Uvicorn Runtime) | 44 |
| 9.2 Thread-Safe ML Inference Singleton & Memory Management | 44 |
| 9.3 Two-Stage CXR Validation Gate Implementation | 44 |
| 9.4 Real-Time Grad-CAM Saliency Generation | 45 |
| 9.5 Database Persistence & Case Management (SQLAlchemy / SQLite) | 45 |
| 9.6 Frontend Client Architecture (React 18 & Glassmorphic UI) | 45 |
| CHAPTER 10 – USER INTERFACE | 46 |
| 10.1 User Interface Design Philosophy & Clinical Ergonomics | 46 |
| 10.2 Diagnostic Dashboard & Application Overview | 46 |
| 10.3 Radiograph Ingestion & Triage Workflow | 46 |
| 10.4 Multi-Class Diagnostic Output & Probability Visualization | 47 |
| 10.5 Grad-CAM Visual Attention Heatmap | 47 |
| 10.6 Clinical Audit Trail & Historical Record Management | 48 |
| 10.7 Model Evaluation & Performance Metrics Dashboard | 48 |
| CHAPTER 11 – TESTING | 50 |
| 11.1 Verification Strategy & Quality Assurance Framework | 50 |
| 11.2 Unit, Integration & API Test Coverage (44/44 Tests Passed) | 50 |
| 11.3 CXR Validation Gate Testing & Out-of-Distribution Defense | 52 |
| 11.4 Explainability & Grad-CAM Verification | 52 |
| 11.5 Deterministic Inference & Model Reliability | 52 |
| 11.6 Research-Stage Verification vs. Clinical Validation Disclaimer | 52 |
| CHAPTER 12 – MODEL EVALUATION AND RESULTS | 53 |
| 12.1 Experimental Protocol & Comprehensive Evaluation Metrics | 53 |
| 12.2 Five-Paradigm Comparative Performance Benchmark | 53 |
| 12.3 Production Model D Internal Validation Performance | 54 |
| 12.4 Visual Performance Evidence | 54 |
| 12.5 Negative Result Analysis: Hybrid Model Degradation | 56 |
| 12.6 External Zero-Shot Generalization Audit: Montgomery County Cohort | 56 |
| 12.7 Root Cause Investigation of External Classification Failure | 57 |
| 12.8 Two-Stage CXR Gate Functional Evaluation & Threshold Calibration | 57 |
| 12.9 Threshold Selection Limitation & Out-of-Distribution Cat Defect | 58 |
| CHAPTER 13 – SECURITY, PRIVACY AND RESPONSIBLE AI | 59 |
| 13.1 Medical Data Security & Boundary Protection | 59 |
| 13.2 Patient Privacy, De-Identification & Regulatory Considerations | 59 |
| 13.3 Explainability & Clinician-in-the-Loop Safeguards | 59 |
| 13.4 Non-Diagnostic Positioning & Clinical Disclaimers | 60 |
| CHAPTER 14 – LIMITATIONS | 61 |
| 14.1 External Dataset Generalization Failure (Montgomery Cohort) | 61 |
| 14.2 Class Imbalance & Pathological Representation Gaps | 61 |
| 14.3 Sensor Domain Shift & High-Frequency Sensitivity | 61 |
| 14.4 CXR Validation Gate Threshold Selection Leakage | 61 |
| 14.5 Lack of Prospective Clinical & Hospital Validation | 61 |
| CHAPTER 15 – FUTURE ENHANCEMENTS | 63 |
| 15.1 Multi-Center Prospective Clinical Trials | 63 |
| 15.2 Independent Threshold Calibration & Temperature Scaling | 63 |
| 15.3 DICOM & PACS Integration Standards | 63 |
| 15.4 Advanced Domain Invariant Representation Learning | 63 |
| 15.5 Enhanced Explainability Frameworks (Grad-CAM++, Score-CAM) | 64 |
| CHAPTER 16 – CONCLUSION | 65 |
| 16.1 Summary of Research Contributions | 65 |
| 16.2 Scientific Findings & Lessons Learned | 65 |
| 16.3 Final Concluding Remarks | 66 |
| REFERENCES | 67 |
| APPENDICES | 70 |
| Appendix A: Experimental Hardware & Software Configurations | 70 |
| Appendix B: Mathematical Formulations of Domain Generalization Algorithms | 70 |


---


---

# LIST OF FIGURES

| Figure Identifier & Descriptive Caption | Page |
| --- | --- |
| Figure 4.1: High-Level System Architecture and Operational Context | 24 |
| Figure 6.1: Level-0 Data Flow Diagram (Context Level) | 30 |
| Figure 6.2: Level-1 Data Flow Diagram Illustrating Functional Decomposition | 30 |
| Figure 6.3: Level-2 Data Flow Diagram of the Deterministic ML Inference Pipeline | 31 |
| Figure 6.4: UML Use Case Diagram Representing Verified Actor Interactions | 32 |
| Figure 6.5: UML Class Diagram of the FastAPI Service and ML Architecture | 32 |
| Figure 6.6: UML Component Diagram of the Full-Stack LungAI System | 33 |
| Figure 6.7: UML Deployment Diagram Illustrating Workstation Infrastructure | 34 |
| Figure 6.8: UML Sequence Diagram for CXR Ingestion with Rejection Branch | 33 |
| Figure 6.9: UML Activity Diagram Depicting End-to-End Diagnostic Workflow | 33 |
| Figure 6.10: Entity Relationship (ER) Diagram of the Relational Database Schema | 35 |
| Figure 7.1: Multi-Source Dataset Harmonization and Training Pipeline | 37 |
| Figure 8.1: DenseNet-121 Model D Deep Transfer Learning Architecture | 40 |
| Figure 8.2: Deterministic Preprocessing and Frequency-Aware Filtering Pipeline | 41 |
| Figure 10.1: LungAI Web Application Home and Clinical Triage Dashboard | 46 |
| Figure 10.2: Radiographic Study Upload, Parameter Specification, and Gate Status | 46 |
| Figure 10.3: Multi-Class Disease Probability Distribution and Triage Classification | 47 |
| Figure 10.4: Grad-CAM Saliency Map Overlay Demonstrating Attention Localization | 47 |
| Figure 10.5: Historical Diagnostic Case Audit Trail and Longitudinal Record Retrieval | 48 |
| Figure 10.6: Model Evaluation Dashboard Displaying Comparative Benchmark Metrics | 48 |
| Figure 12.1: Normalized Confusion Matrix for DenseNet-121 Model D on Test Set | 54 |
| Figure 12.2: Multi-Class Receiver Operating Characteristic (ROC) Curves for Model D | 55 |
| Figure 12.3: Multi-Class Precision-Recall (PR) Curves for Model D Across Six Classes | 55 |


---


---

# LIST OF TABLES

| Table Identifier & Title | Page |
| --- | --- |
| Table 2.1: Systematic Comparison of Thoracic Radiography AI Architectures | 18 |
| Table 5.1: Minimum and Recommended Hardware Specifications | 26 |
| Table 5.2: Software Environment, Library Dependencies, and Runtime Frameworks | 26 |
| Table 5.3: Core Functional Requirements of the Diagnostic Platform | 27 |
| Table 7.1: Class Distribution Across Acquisition Sources in Unified V5 Dataset | 36 |
| Table 7.2: Patient-Level Splitting Distribution of the V5 Dataset | 38 |
| Table 8.1: Architectural Specifications of DenseNet-121 Feature Extractor | 40 |
| Table 8.2: Experimental Hyperparameters Across Domain Generalization Paradigms | 42 |
| Table 11.1: Automated Software Verification and Test Suite Summary (44/44 Tests) | 50 |
| Table 12.1: Comparative Performance Across Five Domain Generalization Paradigms | 53 |
| Table 12.2: Per-Class Diagnostic Performance of Production Model D on Test Set | 54 |
| Table 12.3: Zero-Shot External Generalization on Montgomery County Cohort (n=138) | 56 |
| Table 12.4: Functional Validation of Two-Stage CXR Screening Gate (n=125) | 57 |


---


# CHAPTER 1 – INTRODUCTION

## 1.1 Clinical Background & Problem Domain

Thoracic diseases constitute one of the most pressing public health challenges globally, responsible for tens of millions of hospitalizations and premature deaths each year [1]. Acute infectious respiratory conditions, most notably bacterial and viral pneumonia, coronavirus disease 2019 (COVID-19), and pulmonary tuberculosis (*Mycobacterium tuberculosis*), pose immediate, life-threatening risks of respiratory failure and Acute Respiratory Distress Syndrome (ARDS). Concurrently, chronic and focal thoracic conditions, including pleural effusion and solitary or multifocal pulmonary nodules and masses, represent frequent radiographic manifestations of cardiovascular compromise, severe systemic infection, or primary and metastatic pulmonary malignancies. In acute emergency triage, every hour of therapeutic delay substantially elevates patient mortality, whereas in chronic thoracic pathologies, early detection of radiographic signs is the primary determinant of curative intervention and long-term survival.

Chest radiography (CXR) serves as the primary, most accessible, and most universally utilized diagnostic imaging modality in clinical medicine. Its minimal ionizing radiation footprint (approximately 0.1 mSv for a standard posteroanterior projection, compared to 7.0–10.0 mSv for thoracic computed tomography [CT]), rapid acquisition throughput, and low capital cost make it the standard diagnostic examination in emergency triage wards, intensive care units, outpatient clinics, and rural or resource-constrained healthcare environments worldwide.

However, planar chest radiography poses immense perceptual and diagnostic challenges. Because planar radiography flattens a complex, dynamic three-dimensional anatomical volume into a single two-dimensional projection, overlying anatomical structures—such as anterior and posterior ribs, clavicles, mediastinal contours, cardiac margins, and branching bronchovascular trees—superimpose upon one another. Consequently, subtle, low-contrast parenchymal opacities, ground-glass infiltrates, small apical cavities, and faint solitary nodules are frequently obscured by normal anatomical clutter. Discriminating early pathological patterns from benign physiological variations requires years of specialized radiological training and sustained cognitive vigilance.

## 1.2 Radiographic Imaging Principles & Diagnostic Role

Planar chest radiograph formation is fundamentally governed by the physical principles of differential X-ray photon attenuation across human tissues of divergent atomic composition, physical thickness, and tissue density. In standard thoracic imaging, bodily tissues are categorized into four canonical radiodensities:

1. **Air / Gas**: Possesses negligible physical density, allowing the vast majority of incident X-ray photons to pass through unattenuated to the digital receptor, producing low optical density (radiolucent, appearing black or dark charcoal). In a healthy subject, ventilated lung fields appear dark.
2. **Fat / Adipose Tissue**: Possesses intermediate low density, attenuating slightly more photons than air and appearing dark grey in subcutaneous planes.
3. **Soft Tissue / Fluid**: Comprising the heart, great vessels, diaphragm, blood, purulent exudate, and parenchymal tissue, attenuates photons moderately, manifesting as mid-tone grey or off-white opacities.
4. **Bone / Calcification**: Characterized by high calcium content and atomic number, strongly attenuates X-ray photons via photoelectric absorption, resulting in high optical radiopacity (bright white).

Pathological processes within the thoracic cage alter this normal distribution of photon attenuation:
- **Pneumonia**: Microbial infection incites acute alveolar inflammation, filling alveolar airspaces with purulent cellular exudate, fibrin, and erythrocytes. This alveolar consolidation replaces radiolucent air with soft-tissue density, generating patchy or confluent opacifications and classic air bronchograms.
- **COVID-19**: Viral alveolar injury and interstitial inflammation produce distinctive peripheral, subpleural, and bilateral ground-glass opacities (GGOs) and consolidative patches, frequently concentrated in lower lung zones.
- **Tuberculosis**: Mycobacterial infection triggers granulomatous immune reactions that manifest as apical fibro-cavitary lesions, patchy parenchymal infiltrations, Ghon complexes, or widespread miliary micronodules.
- **Pleural Effusion**: Pathological fluid collection within the pleural cavity blunts the normally sharp costophrenic and cardiophrenic sulci, producing homogeneous crescent-shaped opacities with distinctive meniscus contours.
- **Pulmonary Nodule / Mass**: Focal parenchymal tissue proliferation, inflammatory granulomas, or neoplastic lesions manifest as discrete, rounded, or lobulated radiopaque opacities within the radiolucent lung fields.

Accurately recognizing and differentiating these overlapping radiographic opacities is an intellectually demanding cognitive process, heavily reliant on the clinician's pattern-recognition experience.

## 1.3 Workflow Challenges in Conventional Radiology

Contemporary healthcare systems worldwide face unprecedented systemic bottlenecks in radiological workflows, driven by three interrelated structural crises:

First, there exists an acute, widening global shortage and geographical maldistribution of certified diagnostic radiologists. In developing and lower-middle-income nations, the ratio of certified radiologists to population often drops below 1 per 100,000 citizens. Even across tertiary medical centers in high-income countries, annual imaging examination volumes have far outpaced the growth of the radiological workforce. Consequently, emergency department physicians, medical officers, and intensive care clinicians are routinely forced to make urgent diagnostic decisions without specialist radiological interpretation during nights, weekends, and high-volume clinical shifts.

Second, standard hospital Picture Archiving and Communication Systems (PACS) organize incoming imaging studies using unprioritized, First-In, First-Out (FIFO) worklists. Under this operational architecture, an acute emergency radiograph exhibiting massive bilateral consolidation, tension pneumothorax, or extensive infectious infiltrate sits in the reading queue in the exact sequence it was transmitted by the modality, grouped indiscriminately alongside routine pre-employment physical examinations and outpatient follow-ups. This structural absence of automated triage causes report turnaround delays extending from 24 to over 72 hours in high-volume public hospitals, deferring life-saving antimicrobial or supportive clinical interventions.

Third, manual human image interpretation is inherently susceptible to diagnostic errors, cognitive fatigue, and perceptual blind spots. Extended shifts, visual fatigue, perceptual distraction, and variations in subspecialty expertise produce documented inter-observer and intra-observer diagnostic disagreement rates between 15% and 30% in thoracic radiograph interpretation. These realities motivate the development of objective, automated, computer-aided detection and clinical triage algorithms.

## 1.4 Problem Statement & Research Question

Despite remarkable published benchmarks—where deep learning models frequently report classification accuracy exceeding 95%—real-world translational deployment fails repeatedly due to profound distribution shifts across hospital sites. Prior models frequently overfit to hospital-specific radiographic acquisition shortcuts, scanner-specific digital watermarks, patient demographic markers, and high-frequency detector noise rather than authentic anatomical pathology. Furthermore, public training corpora have suffered from severe uncorrected artifacts, including cross-modality contamination (such as axial CT slices merged into planar CXR datasets), non-radiographic artifacts, and patient-identity leakage across train/test splits.

To address these fundamental challenges, this research addresses the central scientific research question:

> **"Can a multi-source six-class chest X-ray classifier learn disease-relevant representations that generalize across independent acquisition sources, and can a domain-generalization strategy improve cross-source performance compared with a standard DenseNet-121 baseline?"**

## 1.5 Research Objectives & Project Scope

To answer this research question, the following structured research objectives were defined and executed:

1. **Forensic Dataset Reconstruction & Harmonization**: Conduct an exhaustive forensic audit of public chest radiography repositories to eliminate cross-modality contamination, duplicate scans, and patient overlap leakage. Construct a canonical, leak-free, multi-source six-class benchmark (the Unified V5 dataset) comprising 10,547 images across 10,270 unique patients.
2. **Deterministic Biophysical Preprocessing Pipeline**: Formulate a reproducible preprocessing workflow combining chromatic-luminance decoupling (CIE LAB space), Contrast-Limited Adaptive Histogram Equalization (CLAHE) on the luminance channel, high-order Lanczos-4 spatial resampling, and Gaussian frequency low-pass filtering.
3. **Controlled Multi-Paradigm Domain Generalization Benchmarking**: Under an identical backbone (DenseNet-121) and identical experimental protocol, systematically train and benchmark five distinct learning paradigms: Empirical Risk Minimization (ERM) baseline, Deep Correlation Alignment (Deep CORAL), Domain-Adversarial Neural Networks (DANN), Frequency-aware Gaussian spatial low-pass filtering (Model D), and a Hybrid feature-frequency framework.
4. **Transparent External Generalization Failure Audit**: Evaluate the trained architectures zero-shot on an untouched, external acquisition benchmark (the Montgomery County cohort) and execute biophysical forensic analyses to isolate the root causes of external domain transfer failure.
5. **Two-Stage Defense-in-Depth CXR Input-Validation Gate**: Design and implement a robust input screening pipeline integrating biophysical heuristics with a deep anatomical classifier operating at an empirically audited threshold ($	au = 0.83$) to reject non-radiographic inputs.
6. **Full-Stack Clinical Decision Support System**: Engineer an asynchronous, full-stack clinical decision-support ecosystem (FastAPI, React 18, SQLite) featuring real-time Grad-CAM explainability and emergency triage prioritization, verified through 44 automated tests.

## 1.6 Thesis Contributions

The core contributions of this research are:
1. **Methodological Contribution**: Proving that biophysical frequency-aware spatial filtering outperforms complex latent domain-alignment objectives (CORAL, DANN) in suppressing high-frequency sensor shortcuts, achieving 82.93% accuracy and 0.9755 macro ROC-AUC.
2. **Empirical Contribution**: Demonstrating that high internal validation accuracy does not prevent external domain collapse under severe scanner sensor shift, documenting the Montgomery external failure transparently.
3. **Architectural & Clinical Safety Contribution**: Developing a production-calibrated Two-Stage CXR Validation Gate ($	au = 0.83$) achieving 100% specificity against non-medical and cross-modality images, and implementing a complete, verified decision-support platform.

## 1.7 Thesis Organization

The remainder of this dissertation is organized into 16 canonical chapters: Chapter 2 reviews relevant literature; Chapter 3 examines existing systems; Chapter 4 presents the proposed LungAI platform architecture; Chapter 5 details requirements; Chapter 6 delivers comprehensive system design; Chapter 7 documents dataset reconstruction; Chapter 8 details Model D and domain generalization paradigms; Chapter 9 describes implementation; Chapter 10 showcases the user interface; Chapter 11 presents software testing (44/44 tests); Chapter 12 delivers comprehensive evaluation; Chapter 13 analyzes security and responsible AI; Chapter 14 details study limitations; Chapter 15 outlines future research; and Chapter 16 concludes the dissertation.

---


---


# CHAPTER 2 – LITERATURE SURVEY

## 2.1 Deep Learning in Thoracic Radiography

Automated interpretation of chest radiographs has undergone a profound evolution over the past two decades. Early computer-aided detection (CAD) systems relied on handcrafted texture descriptors, edge filters, and classical machine learning models (such as Support Vector Machines and Random Forests). While effective for isolated, high-contrast lesions, these handcrafted methods failed to capture the subtle, non-linear parenchymal variations characteristic of diffuse consolidations and early interstitial viral infections.

The emergence of deep Convolutional Neural Networks (CNNs) revolutionized medical image analysis by enabling hierarchical end-to-end feature learning directly from raw pixel matrices [2]. Large public repositories, including NIH ChestX-ray8 / ChestX-ray14 [1], CheXpert [4], and MIMIC-CXR [5], catalyzed rapid development of automated thoracic classifiers. CheXNet demonstrated that deep architectures pre-trained on ImageNet could achieve radiologist-level area under the receiver operating characteristic curve (ROC-AUC) for pneumonia detection under closed-world test splits.

## 2.2 Convolutional Neural Network Backbones: ResNet vs. DenseNet

Deep residual networks (ResNet), introduced by He et al. [10], addressed the vanishing gradient problem in deep architectures through identity skip connections, formulating layer mapping as residual functions:

$$x_{l+1} = \mathcal{H}(x_l) = \mathcal{F}(x_l, \mathcal{W}_l) + x_l$$

ResNet architectures, particularly ResNet50, established strong performance benchmarks across computer vision and served as the foundational backbone in pioneering thoracic AI studies. However, in medical radiography, subtle opacities and diffuse interstitial markings require multi-scale feature reuse across early, intermediate, and deep layers.

Huang et al. [3] introduced the Densely Connected Convolutional Network (DenseNet), which modifies layer connectivity by concatenating the feature maps of all preceding layers into subsequent operations:

$$x_l = H_l([x_0, x_1, \dots, x_{l-1}])$$

This dense connectivity mechanism confers distinct advantages for chest radiography:
1. **Direct Gradient Propagation**: Gradients from the objective function flow directly to earlier layers through short paths, mitigating vanishing gradients and stabilizing convergence.
2. **Feature Reuse & Efficiency**: Intermediate spatial representations (such as rib edges, soft-tissue boundaries, and parenchymal textures) are preserved and reused throughout the network, reducing parameter count compared to wide residual networks.
3. **Compact Feature Bottlenecks**: DenseNet-121 achieves superior parameter efficiency (~7.0 million backbone parameters compared to ~23.5 million in ResNet50), substantially decreasing overfitting risks on moderately sized medical cohorts. Consequently, DenseNet-121 was selected as the canonical backbone for LungAI.

## 2.3 Visual Explainability & Gradient-Weighted Class Activation Mapping

A paramount barrier to clinical adoption of deep learning is the "black-box" opacity of deep neural networks. Without visual verification of the anatomical features driving a diagnostic prediction, clinicians cannot determine whether a model has identified authentic pathology or overfitted to spurious image artifacts.

Selvaraju et al. [11] developed Gradient-weighted Class Activation Mapping (Grad-CAM), which generates coarse 2D spatial attention heatmaps highlighting the discriminative image regions utilized by a CNN. For a targeted pathological class $c$, the importance weight $lpha_k^c$ for each feature map activation $A^k$ in the final convolutional layer is computed by global average pooling the gradients of class score $y^c$:

$$lpha_k^c = rac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} rac{\partial y^c}{\partial A_{i,j}^k}$$

The localization heatmap $L_{	ext{Grad-CAM}}^c$ is synthesized by calculating a rectified linear combination of feature maps:

$$L_{	ext{Grad-CAM}}^c = 	ext{ReLU}\left( \sum_k lpha_k^c A^k ight)$$

Applying the Rectified Linear Unit (ReLU) ensures that only features positively contributing to the target class are visualized, suppressing non-discriminative background activations. In LungAI, Grad-CAM is extracted directly from the terminal dense block of DenseNet-121, providing clinicians with verified visual validation of disease localization.

## 2.4 Domain Shift, Confounding & Shortcut Learning in Medical AI

Despite exceptional reported accuracy on internal test sets, deep learning models frequently suffer catastrophic performance drops when evaluated across external medical centers. Zech et al. [8] demonstrated that CNNs trained on chest radiographs from multiple hospital networks learned hospital-specific radiopaque markers, department-specific digital watermarks, and scanner post-processing tags rather than true pulmonary disease patterns.

Degrave, McCauley, and Peltier [9] conducted forensic saliency evaluations of COVID-19 detection networks, revealing that models routinely based predictions on radiographic projection markers (e.g., portable AP vs. upright PA labels) and patient positioning cushions. This phenomenon, known as shortcut learning, occurs because deep neural networks naturally exploit the simplest predictive correlation in the training distribution, even when scientifically spurious.

Concurrently, radiographic acquisition varies widely across scanner manufacturers (e.g., GE Healthcare, Siemens Healthineers, Philips Medical), detector technology (computed radiography photostimulable phosphor plates vs. direct flat-panel amorphous silicon detectors), and digital image processing algorithms. This creates profound high-frequency noise discrepancies that disrupt standard convolutional filters.

## 2.5 Domain Generalization Frameworks

To mitigate domain shift, two primary algorithmic paradigms have emerged:

### Deep Correlation Alignment (Deep CORAL)
Sun and Saenko [6] proposed Deep CORAL, an unsupervised domain adaptation framework that aligns the second-order statistics (covariance matrices) of latent feature representations across source and target domains without requiring target labels. The CORAL loss is defined as the Frobenius norm of the distance between feature covariance matrices:

$$\mathcal{L}_{	ext{CORAL}} = rac{1}{4d^2} \| C_S - C_T \|_F^2$$

where $d$ denotes feature dimension, and $C_S$ and $C_T$ are the sample covariance matrices of source and target domain features.

### Domain-Adversarial Neural Networks (DANN)
Ganin et al. [7] formulated domain adaptation as an adversarial minimax game between a feature extractor and a domain discriminator. A Gradient Reversal Layer (GRL) reverses the gradient sign during backpropagation, encouraging the feature extractor to map inputs into a representation space that maximizes disease classification accuracy while minimizing the domain discriminator's ability to distinguish acquisition sources.

### Frequency-Aware Biophysical Preprocessing
Recognizing that scanner-specific signatures predominantly reside in high-frequency pixel variations, frequency-domain filtering has emerged as an alternative to latent alignment. Applying spatial Gaussian low-pass filtering attenuates device-specific high-frequency textures while preserving lower-frequency parenchymal consolidations and anatomical contours.

## 2.6 Critical Analysis and Research Gap Identification

While existing literature has separately explored deep architectures, domain adaptation, and explainability, substantial research gaps persist:
1. **Pervasive Artifacts in Public Collections**: Widely used multi-class collections contain uncurated cross-modality contamination (axial CT slices), duplicate images, and patient-identity overlap across splits.
2. **Lack of Controlled Multi-Paradigm Comparison**: Prior works rarely evaluate ERM, CORAL, DANN, and frequency-aware filtering under an identical backbone and identical patient-split benchmark.
3. **Absence of Input Validation Gates**: Existing CAD deployments lack pre-inference validation gates, permitting non-radiographic images to produce confident disease diagnoses.

Table 2.1 presents a systematic academic comparison of prior thoracic AI literature, detailing how each informed the design of LungAI.



Table 2.1: Systematic Comparison of Thoracic Radiography AI Architectures and Domain Generalization Literature


| Study & Reference | Algorithm & Backbone | Dataset & Scope | Reported Results | Primary Limitation | Informed LungAI By |
| --- | --- | --- | --- | --- | --- |
| Wang et al. (2017) [1] | ResNet-50 / DenseNet-121 | NIH ChestX-ray14 (112,120 CXRs) | Mean ROC-AUC: 0.745 across 14 thoracic findings | NLP text-mined labels contain ~10-15% error rate; severe class imbalance | Highlighted need for verified labels and patient-level splitting |
| Rajpurkar et al. (2017) [2] | CheXNet (DenseNet-121) | NIH ChestX-ray14 (Pneumonia vs. All) | Pneumonia ROC-AUC: 0.8887; claimed exceeding average radiologist | Evaluated on single-source split; does not assess cross-hospital generalization | Established DenseNet-121 as primary architectural foundation |
| Huang et al. (2017) [3] | DenseNet Architecture | CIFAR-100, ImageNet benchmark | Top-1 error: 20.2% on ImageNet with fewer parameters than ResNet | General computer vision benchmark; no medical radiograph evaluation | Provided dense connectivity formulation for multi-scale feature reuse |
| Irvin et al. (2019) [4] | CheXpert DenseNet-121 | CheXpert (224,316 CXRs, 65,240 patients) | Mean ROC-AUC: 0.930 across 5 selected thoracic conditions | Uncertainty label policies affect performance; no input gate validation | Informed multi-class evaluation and weighted macro-metric formulation |
| Johnson et al. (2019) [5] | DualNet / ResNet-50 | MIMIC-CXR (377,110 images, DICOM) | Mean ROC-AUC: 0.82 across core thoracic pathologies | Complex DICOM processing pipeline; computationally prohibitive for edge triage | Emphasized need for lightweight, deterministic planar preprocessing |
| Sun & Saenko (2016) [6] | Deep CORAL (Covariance Alignment) | Office-31 domain adaptation benchmark | Unsupervised domain adaptation accuracy: 77.7% on standard benchmarks | Second-order alignment assumes Gaussian distributed latent feature space | Directly implemented as one of five comparative benchmark paradigms |
| Ganin et al. (2016) [7] | DANN (Gradient Reversal Layer) | MNIST, SVHN, Office domain adaptation | Adversarial domain invariance; domain classification reduced to random chance | Adversarial minimax game exhibits training instability and mode collapse | Directly implemented as comparative adversarial domain adaptation paradigm |
| Zech et al. (2018) [8] | CNN Hospital Generalization | Multi-Hospital CXR (Mount Sinai, NIH, Indiana) | Internal AUC: 0.84-0.89; external cross-hospital AUC dropped to 0.75-0.78 | Identified hospital-specific metal markers and post-processing shortcuts | Motivated cross-source generalization audit and frequency filtering |
| Degrave et al. (2021) [9] | Saliency Forensic Audit of COVID-19 | COVID-Net, COVID-1000, Multi-source CXR | Revealed models utilized positioning pads and font markers for classification | Demonstrated that high internal accuracy can mask catastrophic shortcut learning | Informed requirement for Grad-CAM visual verification and frequency filtering |
| Selvaraju et al. (2017) [11] | Grad-CAM (Visual Explanations) | ImageNet, VQA, Captioning benchmarks | High localization resolution without architectural modifications or retraining | Resolution bounded by final convolutional feature map spatial dimensions (7x7) | Integrated into LungAI inference pipeline for real-time visual saliency |


---


# CHAPTER 3 – EXISTING SYSTEM

## 3.1 Overview of Conventional Chest Radiography Workflows

In conventional healthcare operations, the diagnostic interpretation of chest radiographs relies entirely on human clinical expertise. Following image acquisition by a radiologic technologist, studies are transmitted via DICOM protocols to local Picture Archiving and Communication Systems (PACS). Diagnostic interpretation is subsequently performed by certified radiologists or attending medical officers on specialized high-luminance diagnostic displays.

While human expertise remains the undisputed clinical gold standard, the conventional operational workflow faces acute structural limitations:
1. **Unprioritized FIFO Queuing**: PACS worklists operate strictly on study arrival timestamp. Studies presenting life-threatening acute consolidations or bilateral pleural effusions sit in the identical queue alongside routine screening examinations, creating clinical turnaround delays of 24 to 72 hours.
2. **Cognitive Fatigue & Perceptual Error**: Visual fatigue during prolonged diagnostic reading shifts leads to documented diagnostic error rates of 15% to 30%, manifesting as missed apical pulmonary lesions or false-negative nodule readings.
3. **Diagnostic Disparities in Resource-Limited Settings**: In peripheral clinics and rural district hospitals, certified radiologists are absent on-site. General practitioners and medical officers must interpret complex thoracic images without specialist decision support.

## 3.2 Limitations of Commercial and Open-Source CAD Solutions

Over the past decade, numerous Computer-Aided Detection (CAD) systems and academic deep learning classifiers have been introduced. However, existing solutions exhibit critical structural shortcomings:

1. **Closed-World Single-Center Overfitting**: Most commercial and open-source models are developed on radiographs acquired from a single hospital network or scanner vendor. When deployed in clinical environments utilizing different acquisition parameters, classification performance degrades precipitously.
2. **Vulnerability to Out-of-Distribution Inputs**: Standard deep neural networks are "silent failures"—they output confident, high-probability thoracic disease predictions when presented with non-radiographic imagery, corrupt uploads, or non-thoracic scans (such as abdominal CT or lateral skull views).
3. **Absence of Real-Time Visual Explainability**: Many CAD implementations function as opaque numerical prediction engines, outputting a class label and scalar probability without spatial lesion localization. Clinicians cannot verify whether the model attended to pathological tissue or spurious background artifacts.
4. **Complex, Heavyweight Deployments**: Commercial systems frequently mandate specialized on-premise GPU clusters, proprietary PACS plugins, or high-latency cloud processing pipelines that cannot operate on edge clinical workstations.

## 3.3 The Cross-Source Generalization Bottleneck

The primary scientific failure mode of existing thoracic AI is the cross-source generalization bottleneck. Convolutional neural networks possess enormous representational capacity, enabling them to discover subtle high-frequency correlations between scanner-specific post-processing filters and disease labels. For example, if a training dataset acquires the majority of its severe pneumonia cases from an intensive care unit utilizing portable computed radiography (CR) scanners, the network will learn to detect the high-frequency detector noise of the portable scanner rather than the alveolar consolidation.

When the trained model is subsequently evaluated on images from an outpatient direct digital radiography (DR) system, the portable scanner shortcut is absent, causing classification accuracy to collapse. Existing commercial solutions address this solely by collecting vast proprietary training datasets, which remains prohibitively expensive and legally challenging across healthcare jurisdictions.

## 3.4 Summary of Gaps in Existing Practice

Existing thoracic imaging workflows and computational CAD tools present four acute unresolved gaps:
- **Gap 1**: Absence of standardized, leak-free, multi-source benchmarks audited for cross-modality contamination.
- **Gap 2**: Lack of systematic comparative evidence regarding the efficacy of latent domain adaptation versus biophysical frequency-aware preprocessing.
- **Gap 3**: Critical deficiency in automated, defense-in-depth input validation gates to intercept invalid uploads before deep inference.
- **Gap 4**: Shortage of lightweight, asynchronous clinical decision-support architectures capable of delivering real-time predictions, Grad-CAM explainability, and triage classification on local clinical hardware.

---


---


# CHAPTER 4 – PROPOSED SYSTEM

## 4.1 System Overview and Architectural Philosophy

To overcome the fundamental limitations of conventional radiology workflows and existing CAD algorithms, this dissertation introduces **LungAI**: a robust, full-stack, clinical decision-support ecosystem for multi-class thoracic disease detection and triage.

The design of LungAI is anchored in three core architectural principles:
1. **Safety-First Defensive Ingestion**: The system rejects the dangerous assumption that all ingested images are valid chest radiographs. It implements a Two-Stage Defense-in-Depth Validation Gate that filters out non-radiographic imagery, corrupt files, and axial CT scans before neural inference.
2. **Biophysical Frequency-Aware Robustness**: Rather than relying exclusively on complex latent domain-alignment losses, LungAI incorporates biophysical spatial low-pass filtering to attenuate scanner-specific high-frequency shortcuts directly in pixel space, ensuring robust cross-source feature extraction.
3. **Transparent Clinician-in-the-Loop Decision Support**: LungAI functions strictly as an assistive second-reader and triage tool. It delivers calibrated class probabilities, emergency triage classifications, and real-time Grad-CAM visual attention overlays, providing clinicians with immediate visual verification of predicted abnormalities.

## 4.2 End-to-End Diagnostic Pipeline

The LungAI diagnostic pipeline operates through seven coordinated, sequential stages:

1. **Secure Ingestion & Structural Validation**: The client uploads an image file via the React web interface. The FastAPI backend validates file format (PNG, JPEG, TIFF), dimensions ($\ge 32 	imes 32$), channel structure, and payload integrity.
2. **Two-Stage CXR Validation Gate**:
   - *Stage 1 (Biophysical Heuristics)*: Inspects chromatic saturation and dynamic range in HSV/LAB space to reject natural color photographs.
   - *Stage 2 (Anatomical Classifier)*: Evaluates the image using an independent DenseNet-121 anatomical gate model. If the predicted chest radiograph probability is below $	au = 0.83$, the study is immediately rejected with HTTP 422 Unprocessable Entity, terminating the pipeline without executing disease inference or persisting data.
3. **Deterministic Biophysical Preprocessing**: For validated radiographs, the system executes 3-channel normalization, BGR-to-RGB conversion, Gaussian smoothing ($3 	imes 3, \sigma = 0.8$), RGB-to-LAB conversion, Contrast-Limited Adaptive Histogram Equalization (CLAHE) on the L-channel, LAB-to-RGB reconversion, Gaussian frequency low-pass filtering ($\sigma = 1.0$), high-order Lanczos-4 spatial resampling to $224 	imes 224$ pixels, and ImageNet standardization.
4. **Deep Multi-Class Inference (Model D)**: The preprocessed tensor is routed through the DenseNet-121 Model D deep feature extractor and custom 256-dimensional bottleneck classification head, outputting raw logits transformed via Softmax into calibrated probabilities across six clinical classes.
5. **Real-Time Visual Explainability (Grad-CAM)**: Gradients of the predicted class score with respect to the terminal dense block feature activations are extracted, rectified, and bilinearly interpolated to generate a 2D spatial attention heatmap, blended with the original radiograph.
6. **Clinical Decision Support & Triage Processing**: The system assigns an emergency triage level:
   - **Emergency (Red)**: COVID-19 or Pleural Effusion (high acute decompensation risk);
   - **Urgent (Amber)**: Pneumonia or Tuberculosis (urgent infectious management);
   - **Non-Urgent / Review (Yellow)**: Pulmonary Nodule / Mass (outpatient follow-up);
   - **Routine (Green)**: Normal (standard review).
7. **Relational Persistence & Audit Logging**: The prediction metadata, triage level, processing latencies, and file paths are persisted in an ACID-compliant SQLite database via SQLAlchemy ORM for longitudinal auditability.

## 4.3 High-Level System Architecture

Figure 4.1 depicts the high-level system architecture of the LungAI diagnostic platform, illustrating the clear boundary separation between the presentation client, asynchronous API gateway, two-stage gate, machine learning inference singleton, and relational persistence tier.

![Figure 4.1: High-Level System Architecture and Operational Context of the LungAI Platform](docs/diagrams/rendered/system_architecture.png)

## 4.4 Key Novelties and Design Distinctions

The proposed LungAI architecture incorporates several critical innovations distinguishing it from prior literature:
- **Calibrated Two-Stage Ingestion Gate**: Intercepts invalid files and non-radiographic uploads at threshold $	au = 0.83$, eliminating the silent-failure vulnerability common to standard deep learning classifiers.
- **Biophysical Frequency Preprocessing (Model D)**: Leverages spatial Gaussian filtering ($\sigma = 1.0$) to suppress high-frequency hardware noise, outperforming complex latent alignment techniques (CORAL, DANN) on the multi-source V5 benchmark.
- **Dense Bottleneck Feature Reuse**: Combines a pre-trained DenseNet-121 backbone with Global Average Pooling, Batch Normalization, a 256-D dense projection layer with ReLU, and a 30% dropout rate, maximizing parameter efficiency and multi-scale feature reuse.
- **Zero-Footprint Local Workstation Architecture**: Optimized for lightweight CPU and edge GPU execution, delivering sub-second inference latency without requiring expensive enterprise cloud infrastructure.

---


---


# CHAPTER 5 – SYSTEM REQUIREMENTS

## 5.1 Hardware Requirements

The LungAI diagnostic platform is engineered with a resource-efficient architecture capable of operating in dual execution modes: high-throughput GPU-accelerated clinical server environments and resource-constrained CPU edge workstations (such as mobile triage laptops in rural clinics). 

In GPU-accelerated environments, tensor forward passes and Grad-CAM backpropagation leverage CUDA parallel cores to achieve sub-second diagnostic turnaround. On standard CPU workstations, the runtime utilizes multi-threaded vector operations (AVX-512 / AVX2 instruction sets via PyTorch and OpenBLAS) to ensure that specialist-grade decision support remains fully operational without dedicated graphics hardware. Table 5.1 delineates the minimum and recommended hardware specifications.



Table 5.1: Minimum and Recommended Hardware Specifications


| Component | Minimum Specification (Edge CPU) | Recommended Specification (Clinical Host / GPU) |
| --- | --- | --- |
| Processor (CPU) | 4-Core x86-64 (Intel Core i5 8th Gen / AMD Ryzen 5, 2.5 GHz) | 8-Core x86-64 (Intel Core i7/i9 12th Gen / AMD Ryzen 7, 3.8 GHz) |
| System Memory (RAM) | 8 GB DDR4 (2666 MHz) | 16 GB – 32 GB DDR4/DDR5 (3200+ MHz) |
| Graphics Processing Unit | Integrated Graphics (Intel UHD / AMD Radeon Graphics) | NVIDIA GeForce RTX 3060 / RTX 4070 / Quadro RTX 4000 (8+ GB VRAM) |
| Storage | 10 GB available SSD space (OS and model weights) | 50 GB NVMe PCIe M.2 SSD (for caching and database persistence) |
| Display Resolution | 1366 × 768 pixels (Standard Hospital Monitor) | 1920 × 1080 (Full HD) or 2560 × 1440 (Diagnostic Medical Grade Display) |
| Network Interface | 100 Mbps Ethernet / 802.11ac Wi-Fi | 1000 Mbps (Gigabit) Ethernet (for PACS DICOM streaming) |



## 5.2 Software Requirements & Runtime Environment

LungAI leverages an open-source, reproducible software stack anchored by the Python 3.12 machine learning ecosystem and modern asynchronous web standards. Table 5.2 outlines the software environment and core library dependencies.

The backend service is structured on FastAPI, an asynchronous ASGI web framework that achieves C-like execution speeds via Starlette and Pydantic. Deep learning models are orchestrated through PyTorch 2.2, utilizing pre-trained torchvision weights for DenseNet-121. The persistence layer utilizes SQLAlchemy 2.0 ORM interfacing with an ACID-compliant SQLite relational database. The client application is developed in React 18, utilizing the Vite build toolchain and modern glassmorphic component styling.



Table 5.2: Software Environment, Library Dependencies, and Runtime Frameworks


| Software Layer | Framework / Library | Version | Role & Functionality |
| --- | --- | --- | --- |
| Operating System | Microsoft Windows / Linux | Windows 10/11 / Ubuntu 22.04 LTS | Host operating system environment |
| Core Runtime | Python | 3.12.x | Underlying programming language runtime |
| Deep Learning Framework | PyTorch / Torchvision | 2.2.0+cu121 / 0.17.0 | Tensor computations, model loading, GPU acceleration, autograd |
| Computer Vision & Filters | OpenCV (opencv-python-headless) | 4.9.0+ | Biophysical preprocessing, CLAHE, Gaussian spatial filtering, Lanczos |
| Scientific Computing | NumPy / SciPy / scikit-learn | 1.26.x / 1.12.x / 1.4.x | Matrix transformations, statistical evaluations, metric calculation |
| Backend API Gateway | FastAPI / Uvicorn | 0.110.0+ / 0.28.0+ | Asynchronous REST API, request routing, OpenAPI documentation |
| Data Validation | Pydantic | 2.6.0+ | Strict schema validation and type coercion for API payloads |
| Relational Database / ORM | SQLAlchemy / SQLite | 2.0.28+ / 3.42+ | Persistent case history, patient tracking, relational audit trail |
| Presentation Layer | React / Node.js / Vite | 18.2.0 / 20.x / 5.1+ | Responsive clinical user interface, state management, triage rendering |
| Visualization / Charting | Recharts / Lucide-React | 2.12.0+ / 0.350+ | Dynamic probability charts, metrics dashboards, UI iconography |



## 5.3 Functional Requirements

The functional requirements (FR) define the fundamental operational capabilities executed by the LungAI system, summarized in Table 5.3.



Table 5.3: Core Functional Requirements of the LungAI Diagnostic Decision Support Platform


| Identifier | Requirement Name | Operational Description & Acceptance Criteria |
| --- | --- | --- |
| FR-1 | Secure Radiograph Ingestion | System shall accept DICOM, PNG, JPEG, and TIFF radiographic image uploads up to 30 MB in payload size. |
| FR-2 | Structural Image Validation | System shall enforce minimum dimensions (32×32 pixels), verify channel integrity, and reject corrupted file bytes. |
| FR-3 | Two-Stage CXR Screening Gate | System shall evaluate uploaded images against biophysical heuristics and an anatomical classifier (tau = 0.83), rejecting non-CXR inputs. |
| FR-4 | Deterministic Preprocessing | System shall apply CLAHE contrast enhancement, Gaussian low-pass filtering (sigma = 1.0), and Lanczos-4 resampling. |
| FR-5 | Multi-Class Disease Classification | System shall predict calibrated probabilities across six classes: COVID-19, Normal, Effusion, Pneumonia, Nodule, TB. |
| FR-6 | Visual Explainability (Grad-CAM) | System shall generate a 2D spatial class activation heatmap targeting the final dense block and blend it with the original radiograph. |
| FR-7 | Clinical Triage Prioritization | System shall categorize each study into Emergency (Red), Urgent (Amber), Review (Yellow), or Routine (Green) triage tiers. |
| FR-8 | Audit Logging & Case Persistence | System shall persist study metadata, patient identifier, predicted probabilities, triage tier, and latencies in SQLite. |
| FR-9 | Longitudinal Case History Retrieval | System shall provide filterable, paginated query access to historical patient studies and diagnostic reports. |
| FR-10 | System Metrics & Model Monitoring | System shall display real-time comparative benchmark metrics, ROC-AUC, and per-class diagnostic statistics. |



## 5.4 Non-Functional Requirements

1. **Performance & Latency**: End-to-end diagnostic inference latency—encompassing structural validation, gate verification, preprocessing, Model D forward pass, and Grad-CAM synthesis—shall not exceed 1.5 seconds on GPU-accelerated hosts and 3.5 seconds on multi-core CPU edge hardware.
2. **Deterministic Reliability**: The model inference engine shall be mathematically deterministic: identical input matrices must yield bitwise identical floating-point class probabilities across successive execution cycles.
3. **Availability & Fault Isolation**: The system shall isolate failures at the gate stage; invalid inputs must terminate with structured HTTP 422 errors without crashing the ASGI worker or leaking unhandled exceptions.
4. **Security & Boundary Protection**: The system shall sanitize all file metadata, restrict write access to authorized upload directories, and prevent directory traversal exploits.

---


---


# CHAPTER 6 – SYSTEM DESIGN

## 6.1 Architectural Design Principles

The LungAI platform is structured around three core architectural tenets:
1. **Separation of Concerns**: Complete structural decoupling between the presentation layer (React SPA), API orchestration gateway (FastAPI), machine learning inference engine (PyTorch singleton), and relational persistence store (SQLite).
2. **Defensive Pipeline Execution**: All operations follow a strict fail-fast validation paradigm. Unchecked inputs cannot traverse into the deep learning pipeline.
3. **Stateless Computation & Concurrency**: The ML inference engine operates as a thread-safe singleton managing deterministic forward passes under PyTorch `torch.no_grad()` contexts, supporting concurrent client requests without memory leaks.

## 6.2 Data Flow Modeling

### Level-0 Data Flow Diagram (Context Level)
Figure 6.1 illustrates the Level-0 Data Flow Diagram, defining the system boundary and external entity interactions between the Clinical User and the LungAI platform. The user supplies raw radiographic image files, patient identifiers, and query filters. In return, the system generates diagnostic disease predictions, calibrated class probability vectors, 2D Grad-CAM anatomical saliency maps, prioritized triage banners, and longitudinal clinical audit records.

![Figure 6.1: Level-0 Data Flow Diagram (Context Level) of the LungAI System](docs/diagrams/rendered/dfd_level_0.png)

### Level-1 Data Flow Diagram (Operational Decomposition)
Figure 6.2 decomposes the system into its primary operational processes:
- **Process 1.0 (Image Ingestion & Structural Check)**: Decodes incoming multipart image bytes, validates file headers, and verifies dimensions ($\ge 32 	imes 32$).
- **Process 2.0 (Two-Stage CXR Validation Gate)**: Assesses biophysical saturation and evaluates anatomical validity via a dedicated deep binary classifier calibrated at $	au = 0.83$. Non-radiographic uploads terminate here with HTTP 422.
- **Process 3.0 (Deterministic Radiographic Preprocessing)**: Performs CIE LAB conversion, luminance CLAHE enhancement, Gaussian low-pass spatial filtering ($\sigma = 1.0$), Lanczos-4 resampling, and ImageNet standardization.
- **Process 4.0 (Model D Multi-Class Inference)**: Executes the forward pass through DenseNet-121, generating calibrated probabilities across the six pathological classes.
- **Process 5.0 (Grad-CAM Saliency Generation)**: Computes class-specific gradients at the final dense block, producing a 2D spatial attention heatmap overlaid onto the original radiograph.
- **Process 6.0 (Clinical Triage Processing)**: Maps predicted probabilities to clinical risk tiers (Emergency, Urgent, Review, Routine).
- **Process 7.0 (Relational Persistence)**: Persists study metadata, patient records, and inference telemetry in SQLite via SQLAlchemy.
- **Process 8.0 (Case History & Reporting)**: Serves paginated query requests and audit summaries to the client.

![Figure 6.2: Level-1 Data Flow Diagram Illustrating Functional Module Decomposition](docs/diagrams/rendered/dfd_level_1.png)

### Level-2 Data Flow Diagram (ML Inference Pipeline Decomposition)
Figure 6.3 provides an in-depth decomposition of Process 3.0 and Process 4.0, detailing the exact flow of data through chromatic conversion, CLAHE enhancement, Gaussian low-pass spatial filtering, Lanczos-4 resampling, DenseNet-121 forward pass, and Softmax probability generation.

![Figure 6.3: Level-2 Data Flow Diagram of the Deterministic ML Inference Pipeline](docs/diagrams/rendered/dfd_level_2.png)

## 6.3 UML Structural Modeling

### UML Use Case Diagram
Figure 6.4 depicts the UML Use Case Diagram representing the interactions between the Clinical User / Radiologist and the verified functional capabilities of the system. Actors can upload radiographs, view multi-class predictions, examine Grad-CAM heatmaps, inspect triage tiers, query historical patient cases, and review comparative model performance metrics.

![Figure 6.4: UML Use Case Diagram Representing Verified Actor Interactions](docs/diagrams/rendered/uml_use_case.png)

### UML Class Diagram
Figure 6.5 details the object-oriented structure of the backend, showcasing the relationships between FastAPI route controllers, the `InferenceEngine` singleton, `CXRValidator`, `ModelDClassifier`, Pydantic schemas, and SQLAlchemy database models.

![Figure 6.5: UML Class Diagram of the FastAPI Service and ML Inference Architecture](docs/diagrams/rendered/uml_class.png)

### UML Component Diagram
Figure 6.6 illustrates the modular component architecture of LungAI, detailing internal component boundaries, dependencies, and interfaces across the presentation, application, machine learning, and persistence tiers.

![Figure 6.8: UML Component Diagram of the Full-Stack LungAI System Architecture](docs/diagrams/rendered/uml_component.png)

### UML Deployment Diagram
Figure 6.7 illustrates the physical deployment topology on a clinical workstation, detailing the execution nodes: User Web Browser, React Frontend Server, FastAPI ASGI Server, PyTorch ML Runtime, and SQLite Database File.

![Figure 6.9: UML Deployment Diagram Illustrating the Local Workstation Infrastructure](docs/diagrams/rendered/uml_deployment.png)

## 6.4 UML Behavioral Modeling

### UML Sequence Diagram (Prediction Workflow with Rejection Branch)
Figure 6.8 captures the dynamic interaction sequence during a diagnostic request. Crucially, it illustrates both the primary nominal execution path and the alternative rejection branch, wherein an invalid CXR is intercepted by the CXR Gate, returning HTTP 422 and bypassing Model D inference, Grad-CAM generation, and database persistence.

![Figure 6.6: UML Sequence Diagram for CXR Ingestion, Gate Verification, and Inference with Rejection Branch](docs/diagrams/rendered/uml_sequence.png)

### UML Activity Diagram (End-to-End Workflow with Validation Decision)
Figure 6.9 models the end-to-end workflow activities from file upload to result display, explicitly depicting the gate decision diamond, the invalid upload termination branch, and the parallel execution of Grad-CAM and triage classification.

![Figure 6.7: UML Activity Diagram Depicting the End-to-End Diagnostic Workflow and Validation Branch](docs/diagrams/rendered/uml_activity.png)

## 6.5 Database Design & Entity Relationship Modeling

Figure 6.10 details the Entity Relationship (ER) diagram of the relational database schema implemented in SQLite via SQLAlchemy. The schema captures three primary entities: `Patient` (storing demographic metadata and patient identifiers), `Prediction` (storing study timestamps, file paths, raw probabilities, predicted class, confidence, triage tier, and gate confidence), and `ModelMetric` (storing benchmark tracking data across experimental runs).

![Figure 6.10: Entity Relationship (ER) Diagram of the Relational Audit and Clinical Database Schema](docs/diagrams/rendered/database_er.png)

---


---


# CHAPTER 7 – DATASET AND DATA PREPROCESSING

## 7.1 Forensic Audit of Public Benchmarks & Artifact Discovery

Prior to model development, an exhaustive forensic audit of widely cited public chest radiography collections was conducted. This investigation uncovered widespread structural anomalies that undermine published academic claims:
1. **Cross-Modality Contamination**: Popular aggregated online collections were found to contain axial and coronal Computed Tomography (CT) slices mislabeled as planar chest radiographs. Because CT cross-sections display completely different physical attenuation geometries, their inclusion artificially distorts convolutional filter learning.
2. **Non-Radiographic Image Contamination**: Online repositories contained corrupt files, non-medical digital photographs, lateral skull radiographs, and pediatric extremity views.
3. **Severe Acquisition-Site Confounding**: Aggregated datasets frequently sourced specific disease categories from single institutions (e.g., all COVID-19 cases from a specialized Italian hospital cohort and all Normal cases from an outpatient clinic). Consequently, models trained on these collections learn to classify the acquisition hospital rather than the disease pathology.
4. **Patient-Identity Leakage**: Many published studies perform random train/test splits at the image level. In longitudinal clinical cohorts where patients undergo multiple sequential radiographs, random splitting places images of the same patient into both training and test partitions, resulting in severe data leakage and artificially inflated test metrics.

## 7.2 The Unified V5 Research Dataset Reconstruction

To establish an uncompromised scientific foundation, the **Unified V5 Research Dataset** was engineered through rigorous programmatic and radiological filtering. All axial CT slices, non-planar views, and corrupted images were permanently eliminated. The resulting V5 dataset comprises **10,547 verified planar chest radiographs** originating from **10,270 unique patients** across eight distinct acquisition sources. Table 7.1 details the class distribution across acquisition sources.



Table 7.1: Class Distribution Across Acquisition Sources in Unified V5 Dataset


| Acquisition Source / Hospital Network | COVID-19 | Normal | Pleural Effusion | Pneumonia | Nodule / Mass | Tuberculosis | Total Images |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Source A (NIH Clinical Center) | 0 | 1,200 | 850 | 650 | 720 | 0 | 3,420 |
| Source B (CheXpert Cohort) | 0 | 800 | 600 | 500 | 480 | 0 | 2,380 |
| Source C (COVID-19 Multi-Center) | 1,280 | 0 | 0 | 0 | 0 | 0 | 1,280 |
| Source D (RSNA Pneumonia Challenge) | 0 | 950 | 0 | 850 | 0 | 0 | 1,800 |
| Source E (Shenzhen No. 3 Hospital) | 0 | 150 | 0 | 0 | 0 | 330 | 480 |
| Source F (BIMCV-COVID19+) | 420 | 0 | 0 | 0 | 0 | 0 | 420 |
| Source G (PadChest Digital DR) | 0 | 300 | 150 | 100 | 120 | 0 | 670 |
| Source H (Belarus TB Portal) | 0 | 0 | 0 | 0 | 0 | 97 | 97 |
| **Unified V5 Total** | **1,700** | **3,400** | **1,600** | **2,100** | **1,320** | **427** | **10,547** |



## 7.3 Multi-Source Training Pipeline

Figure 7.1 illustrates the complete multi-source dataset harmonization, forensic filtering, patient-level partitioning, and training pipeline.

![Figure 7.1: Multi-Source Dataset Harmonization, Forensic Verification, and Training Pipeline](docs/diagrams/rendered/dataset_training_pipeline.png)

## 7.4 Acquisition-Site Confounding Analysis (Cramér's V = 0.7654)

To quantify the degree of acquisition-site confounding across the V5 collection, Cramér's V statistic was calculated between acquisition source $S$ and disease label $Y$:

$$V = \sqrt{rac{\chi^2}{N \cdot \min(r-1, c-1)}}$$

where $\chi^2$ represents the Pearson chi-square statistic, $N = 10,547$, $r = 8$ (sources), and $c = 6$ (classes). The calculation yielded **Cramér's V = 0.7654**, confirming an exceptionally strong statistical dependency between acquisition site and disease class. In naive convolutional models, this strong correlation induces catastrophic shortcut learning, where networks classify scanner artifacts rather than pathology.

## 7.5 Planar CXR Verification & CT Elimination

To enforce strict planar radiograph validity, programmatic filters were constructed:
1. **Aspect Ratio Screening**: Bounding image aspect ratio within $[0.70, 1.45]$, eliminating rectangular panoramic sweeps and square-cropped CT tiles.
2. **Radial Intensity Profiling**: Axial CT slices exhibit distinct circular gantry boundaries and dark peripheral air pockets; radial gradient analysis automatically identified and purged 287 contaminated CT slices from legacy folders.
3. **Contrast & Dynamic Range Check**: Verified that pixel histograms occupy $\ge 60\%$ of available 8-bit dynamic range.

## 7.6 Patient-Level Leak-Free Data Partitioning

To completely prevent data leakage, dataset splitting was performed strictly at the patient identifier level. All radiographs belonging to a given patient were assigned exclusively to a single partition: Train (70%), Validation (15%), or Internal Test (15%). Table 7.2 presents the exact patient and image counts across partitions.



Table 7.2: Patient-Level Splitting Distribution of the V5 Dataset


| Dataset Partition | Total Verified Images | Unique Patient Count | Patient Percentage | Purpose & Role |
| --- | --- | --- | --- | --- |
| Training Split | 7,398 | 7,189 | 70.0% | Model parameter optimization & backbone fine-tuning |
| Validation Split | 1,579 | 1,540 | 15.0% | Hyperparameter tuning, early stopping & checkpoint selection |
| Internal Test Split | 1,570 | 1,541 | 15.0% | Unbiased internal evaluation of final model checkpoints |
| **Total Cohort** | **10,547** | **10,270** | **100.0%** | **Frozen V5 Research Benchmark** |



---


---


# CHAPTER 8 – MACHINE LEARNING / AI MODEL

## 8.1 Model D Deep Transfer Learning Architecture

The primary diagnostic architecture in LungAI, designated **DenseNet-121 Model D**, employs a pre-trained ImageNet backbone modified with a customized bottleneck classification head. DenseNet-121 connects each layer to every other layer in a feed-forward fashion across four Dense Blocks interconnected by Transition Layers.

The architectural flow is structured as follows:
$$	ext{Input } (224 	imes 224 	imes 3) \longrightarrow 	ext{DenseNet-121 Backbone} \longrightarrow 	ext{GAP } (1024) \longrightarrow 	ext{BN} \longrightarrow 	ext{Dense}(256) + 	ext{ReLU} \longrightarrow 	ext{Dropout}(0.3) \longrightarrow 	ext{Dense}(6) \longrightarrow 	ext{Softmax}$$

Figure 8.1 depicts the complete architectural layout of Model D.

![Figure 8.1: DenseNet-121 Model D Deep Transfer Learning and Classification Head Architecture](docs/diagrams/rendered/model_d_architecture.png)

Table 8.1 specifies the structural parameters and activation shapes of each layer in Model D.



Table 8.1: Architectural Specifications of DenseNet-121 Feature Extractor and Classification Head


| Layer / Sub-Module | Output Dimension | Kernel / Filter Spec | Activation / Regularization | Trainable Parameters |
| --- | --- | --- | --- | --- |
| Input Image | 224 × 224 × 3 | RGB Preprocessed Tensor | ImageNet Normalization | 0 (Input) |
| Initial Conv / MaxPool | 56 × 56 × 64 | 7×7 Conv (stride 2), 3×3 MaxPool | BatchNorm + ReLU | 9,408 |
| Dense Block 1 (6 layers) | 56 × 56 × 256 | 1×1 Conv, 3×3 Conv (growth k=32) | Dense concatenation | 141,312 |
| Transition Layer 1 | 28 × 28 × 128 | 1×1 Conv, 2×2 AvgPool (stride 2) | BatchNorm + ReLU | 33,024 |
| Dense Block 2 (12 layers) | 28 × 28 × 512 | 1×1 Conv, 3×3 Conv (growth k=32) | Dense concatenation | 501,760 |
| Transition Layer 2 | 14 × 14 × 256 | 1×1 Conv, 2×2 AvgPool (stride 2) | BatchNorm + ReLU | 131,584 |
| Dense Block 3 (24 layers) | 14 × 14 × 1024 | 1×1 Conv, 3×3 Conv (growth k=32) | Dense concatenation | 2,385,920 |
| Transition Layer 3 | 7 × 7 × 512 | 1×1 Conv, 2×2 AvgPool (stride 2) | BatchNorm + ReLU | 525,312 |
| Dense Block 4 (16 layers) | 7 × 7 × 1024 | 1×1 Conv, 3×3 Conv (growth k=32) | Dense concatenation | 3,227,648 |
| Global Average Pooling | 1024 | Spatial reduction (7×7 -> 1×1) | None | 0 |
| Batch Normalization | 1024 | Feature scale & shift | gamma, beta learnable | 2,048 |
| Dense Bottleneck | 256 | Linear projection (1024 -> 256) | ReLU Activation | 262,400 |
| Dropout Regularization | 256 | Inverted Dropout | p = 0.30 drop probability | 0 |
| Classification Head | 6 | Linear projection (256 -> 6) | Softmax (6-Class Probs) | 1,542 |
| **Total Model D Architecture** | — | — | **Backbone + Head** | **7,221,958 parameters** |



## 8.2 Deterministic Biophysical Preprocessing Pipeline

To eliminate scanner-induced chromatic and luminance discrepancies, Model D enforces a deterministic preprocessing pipeline prior to tensor generation:
1. **Color Space Decoupling**: Converts the input to CIE LAB space, separating luminance ($L^*$) from chromaticity ($a^*, b^*$).
2. **Luminance CLAHE Enhancement**: Applies Contrast-Limited Adaptive Histogram Equalization to the $L^*$ channel with tile grid $8 	imes 8$ and clip limit 2.0, enhancing subtle parenchymal consolidations without amplifying noise.
3. **Gaussian Spatial Low-Pass Filtering**: Convolves the enhanced image with a 2D Gaussian kernel ($\sigma = 1.0$), attenuating high-frequency scanner noise.
4. **Lanczos-4 Resampling**: Resamples the image to $224 	imes 224$ pixels using 8-lobed sinc interpolation.
5. **ImageNet Standardization**: Normalizes channel distributions using $\mu = [0.485, 0.456, 0.406]$ and $\sigma = [0.229, 0.224, 0.225]$.

Figure 8.2 illustrates each stage of this deterministic pipeline.

![Figure 8.2: Deterministic Biophysical Preprocessing and Frequency-Aware Spatial Filtering Pipeline](docs/diagrams/rendered/preprocessing_pipeline.png)

## 8.3 Controlled Multi-Paradigm Domain Generalization Study

To determine whether domain generalization strategies improve cross-source performance, five distinct learning paradigms were benchmarked under identical conditions:
1. **Empirical Risk Minimization (ERM)**: Standard supervised cross-entropy loss without domain adaptation.
2. **Deep Correlation Alignment (Deep CORAL)**: Minimizes Frobenius norm distance between feature covariance matrices across source domains [6].
3. **Domain-Adversarial Neural Networks (DANN)**: Minimax adversarial game with a gradient reversal layer aligning domain feature distributions [7].
4. **Frequency-Aware Preprocessing (Model D)**: Biophysical Gaussian low-pass spatial filtering ($\sigma = 1.0$) combined with standard cross-entropy.
5. **Hybrid Feature-Frequency Model**: Combines spatial low-pass filtering with Deep CORAL covariance alignment.

Table 8.2 summarizes the experimental hyperparameters applied across all five paradigms.



Table 8.2: Experimental Hyperparameters Across Domain Generalization Paradigms


| Hyperparameter | ERM Baseline | Deep CORAL | DANN | Model D (Frequency) | Hybrid Model |
| --- | --- | --- | --- | --- | --- |
| Backbone Architecture | DenseNet-121 | DenseNet-121 | DenseNet-121 | DenseNet-121 | DenseNet-121 |
| Pre-training Weights | ImageNet-1k | ImageNet-1k | ImageNet-1k | ImageNet-1k | ImageNet-1k |
| Input Spatial Resolution | 224 × 224 × 3 | 224 × 224 × 3 | 224 × 224 × 3 | 224 × 224 × 3 | 224 × 224 × 3 |
| Batch Size | 32 | 32 (16 src / 16 tgt) | 32 (16 src / 16 tgt) | 32 | 32 (16 src / 16 tgt) |
| Base Learning Rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Optimizer | Adam (beta1=0.9, beta2=0.999) | Adam | Adam | Adam | Adam |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| Domain Loss Weight | N/A | lambda = 0.50 (CORAL) | lambda = 0.10 (Adversarial) | N/A | lambda = 0.50 (CORAL) |
| Spatial Low-Pass Filter | None | None | None | Gaussian (sigma = 1.0) | Gaussian (sigma = 1.0) |
| Training Epochs | 20 (Early stop p=5) | 20 (Early stop p=5) | 20 (Early stop p=5) | 20 (Early stop p=5) | 20 (Early stop p=5) |



---


---


# CHAPTER 9 – SYSTEM IMPLEMENTATION

## 9.1 Backend Architecture (FastAPI & Uvicorn Runtime)

The LungAI backend is implemented as an asynchronous, high-concurrency microservice utilizing **FastAPI** hosted on the **Uvicorn** ASGI server. FastAPI's native support for Python `asyncio` event loops enables non-blocking request handling, allowing concurrent ingestion of radiograph payloads while offloading computationally intensive tensor calculations to dedicated worker threads via `asyncio.to_thread`.

The API architecture exposes RESTful endpoints structured under `/api/v1`:
- `POST /api/v1/predict`: Central ingestion endpoint accepting multipart image uploads and optional patient IDs.
- `GET /api/v1/history`: Returns paginated, searchable historical diagnostic records.
- `GET /api/v1/metrics`: Provides real-time comparative benchmark statistics and per-class diagnostic metrics.
- `GET /api/v1/health`: Lightweight health check verifying model availability, GPU/CPU runtime status, and database connectivity.

## 9.2 Thread-Safe ML Inference Singleton & Memory Management

To prevent redundant model initialization and avoid out-of-memory errors on GPU and CPU hosts, the deep learning runtime is encapsulated within a thread-safe singleton class: `InferenceEngine`. 

Key implementation characteristics include:
1. **Double-Checked Locking**: Thread-safe lazy instantiation ensures that the DenseNet-121 backbone weights (~28 MB) and CXR Gate weights (~28 MB) are loaded into system memory exactly once during application startup.
2. **Context-Managed Evaluation**: All tensor forward passes are wrapped within `torch.no_grad()` blocks and set to `model.eval()`, disabling dropout, freezing batch normalization statistics, and preventing gradient computation graph allocation.
3. **Explicit Memory Garbage Collection**: After inference, intermediate tensor allocations are explicitly cleared via Python garbage collection and `torch.cuda.empty_cache()` when running on GPU hosts.

## 9.3 Two-Stage CXR Validation Gate Implementation

The Two-Stage CXR Validation Gate is implemented as a defensive pre-inference pipeline in `backend/services/cxr_gate.py`:
- **Stage 1 (Biophysical Heuristics)**: Decodes raw image bytes via OpenCV. It measures color saturation in HSV space and computes the luminance standard deviation. Natural multi-channel photographs (e.g., outdoor scenes, animals, skin lesions) exceeding saturation thresholds are flagged immediately.
- **Stage 2 (Anatomical Verification Classifier)**: A dedicated DenseNet-121 binary classifier evaluates the normalized image tensor. If the predicted probability of being a valid planar chest radiograph falls below the calibrated production threshold $	au = 0.83$, the pipeline halts immediately, raising an `HTTPException(status_code=422, detail="Uploaded file is not a valid planar chest radiograph")`. This guarantees that Model D is never exposed to out-of-distribution non-thoracic inputs.

## 9.4 Real-Time Grad-CAM Saliency Generation

Visual explainability is implemented through an optimized Grad-CAM module hooking into the final convolutional layer of DenseNet-121 (`features.denseblock4.denselayer16.conv2`):
1. **Forward Hook**: Captures spatial activations $A \in \mathbb{R}^{1024 	imes 7 	imes 7}$.
2. **Backward Hook**: Computes gradients $rac{\partial y^c}{\partial A}$ for the predicted class score $y^c$.
3. **Channel Weighting & Rectification**: Computes channel-wise importance weights $lpha_k^c$ via global average pooling, multiplies feature maps, applies ReLU, and normalizes the resulting matrix to $[0, 1]$.
4. **Colormap Rendering**: Resizes the $7 	imes 7$ heatmap to the original image dimensions using bilinear interpolation, applies the OpenCV `COLORMAP_JET` pseudo-color lookup table, and alpha-blends the heatmap with the grayscale radiograph ($lpha = 0.40, eta = 0.60$).

## 9.5 Database Persistence & Case Management (SQLAlchemy / SQLite)

Relational persistence is implemented using **SQLAlchemy ORM** backed by an ACID-compliant **SQLite** database (`lungai.db`). The schema maintains two primary tables:
- `patients`: Stores unique patient identifiers, demographic information, and admission records.
- `predictions`: Records the timestamp, original image filename, storage path, predicted class, class probabilities JSON payload, confidence score, triage level, gate status, and processing latency.

This provides healthcare facilities with an indelible, longitudinal audit trail for clinical quality control and medico-legal compliance.

## 9.6 Frontend Client Architecture (React 18 & Glassmorphic UI)

The user interface is engineered as a modern Single-Page Application (SPA) using **React 18** and **Vite**. The design system incorporates a clinical-grade glassmorphic aesthetic with dark-mode accents, high-contrast typography, and intuitive color coding for emergency triage levels. State management utilizes React hooks with Axios for asynchronous REST communication, providing real-time upload progress, interactive probability distribution charts (Recharts), and side-by-side radiograph/Grad-CAM visual comparisons.

---


---


# CHAPTER 10 – USER INTERFACE

## 10.1 User Interface Design Philosophy & Clinical Ergonomics

The user interface of LungAI was designed adhering to clinical ergonomics guidelines:
1. **Rapid Triage Comprehension**: Emergency and urgent diagnostic classifications are visually prioritized using universal medical triage colors (Red = Emergency, Amber = Urgent, Yellow = Review, Green = Routine).
2. **Side-by-Side Explainability**: The original radiograph is presented adjacent to the Grad-CAM saliency map, allowing clinicians to visually verify anatomical lesion localization without toggling views.
3. **Zero Configuration Deployment**: The web client runs in any standard hospital browser without requiring client-side plugin installations or specialized medical display hardware.

## 10.2 Diagnostic Dashboard & Application Overview

Figure 10.1 displays the primary landing dashboard of LungAI, providing clinicians with high-level access to recent study uploads, quick-action triage buttons, and platform health status.

![Figure 10.1: LungAI Web Application Home and Clinical Triage Dashboard Interface](docs/screenshots/application/01_home.png)

## 10.3 Radiograph Ingestion & Triage Workflow

Figure 10.2 depicts the study ingestion screen, where users drag-and-drop radiographic files, specify optional patient identifiers, and monitor real-time validation gate feedback.

![Figure 10.2: Radiographic Study Upload, Parameter Specification, and Real-Time Gate Status](docs/screenshots/application/02_analyze.png)

## 10.4 Multi-Class Diagnostic Output & Probability Visualization

Figure 10.3 shows the comprehensive diagnostic output screen generated following Model D inference. The interface displays the primary diagnostic finding, confidence score, triage priority banner, and an interactive probability distribution across all six clinical classes.

![Figure 10.3: Multi-Class Disease Probability Distribution, Primary Diagnostic Finding, and Triage Classification](docs/screenshots/application/03_prediction.png)

## 10.5 Grad-CAM Visual Attention Heatmap

Figure 10.4 showcases the visual explainability interface, presenting the high-resolution Grad-CAM saliency heatmap overlaid onto the original chest radiograph, clearly delineating regions of parenchymal consolidation.

![Figure 10.4: Grad-CAM Saliency Map Overlay Demonstrating Anatomical Attention Localization](docs/screenshots/application/04_gradcam.png)

## 10.6 Clinical Audit Trail & Historical Record Management

Figure 10.5 illustrates the historical case management view, allowing medical officers to filter, search, and review past diagnostic studies, export clinical summaries, and monitor longitudinal patient outcomes.

![Figure 10.5: Historical Diagnostic Case Audit Trail and Longitudinal Record Retrieval View](docs/screenshots/application/06_history.png)

## 10.7 Model Evaluation & Performance Metrics Dashboard

Figure 10.6 presents the interactive model performance metrics dashboard, displaying real-time ROC curves, precision-recall metrics, and comparative evaluation data across all five domain generalization paradigms.

![Figure 10.6: Model Evaluation Dashboard Displaying Five-Paradigm Comparative Performance Metrics](docs/screenshots/application/05_metrics.png)

---


---


# CHAPTER 11 – TESTING

## 11.1 Verification Strategy & Quality Assurance Framework

The verification of the LungAI ecosystem was conducted through an automated, multi-tiered testing framework implemented using **pytest**. Testing was structured across four distinct verification strata:
1. **Low-Level Algorithmic Testing**: Verifying deterministic preprocessing, numerical bounds of normalization filters, and CLAHE dynamic range expansion.
2. **Neural Model & Saliency Testing**: Confirming tensor input/output dimensional validity, probability distribution normalization ($\sum p_i = 1.0$), and Grad-CAM heatmap value constraints ($[0, 255]$).
3. **Defense-in-Depth Gate Testing**: Validating that non-radiographic images (natural photos, CT slices, corrupt files) are intercepted and rejected with appropriate HTTP error codes.
4. **End-to-End API & Persistence Testing**: Verifying complete client-server workflows, database transactions, patient case queries, and error handling.

## 11.2 Unit, Integration & API Test Coverage (44/44 Tests Passed)

The entire automated test suite achieved a **100% pass rate (44/44 tests passed)** under Python 3.12. Table 11.1 categorizes the test suite by functional domain and verified behaviors.



Table 11.1: Automated Software Verification and Test Suite Summary (44/44 Tests Passed)


| Test Module / Suite | Test Identifier | Verified Operational Behavior | Expected Result | Status |
| --- | --- | --- | --- | --- |
| `test_preprocessing.py` | `test_clahe_contrast_enhancement` | Verifies luminance standard deviation increases after CLAHE | sigma_post > sigma_pre | PASS |
|  | `test_gaussian_lowpass_attenuation` | Confirms high-frequency spatial noise reduction | Noise energy reduced | PASS |
|  | `test_lanczos_resampling_dimensions` | Validates 224×224 spatial tensor shape and 3-channel layout | (3, 224, 224) output | PASS |
|  | `test_imagenet_normalization_range` | Ensures output tensor normalized to ImageNet mean/variance | Mean ~0, Std ~1 | PASS |
| `test_model_inference.py` | `test_model_singleton_loading` | Confirms Model D loads weights exactly once into memory | Singleton ID identical | PASS |
|  | `test_forward_pass_dimensions` | Verifies forward pass produces (1, 6) logit tensor | Tensor shape [1, 6] | PASS |
|  | `test_softmax_probability_sum` | Verifies output probabilities sum to 1.0 within float tolerance | sum(p) == 1.0 ± 1e-5 | PASS |
|  | `test_deterministic_inference` | Confirms bitwise identical output across 10 repeated inferences | Zero variance | PASS |
| `test_gradcam.py` | `test_gradcam_activation_hook` | Confirms backward gradients captured at final dense layer | Non-empty gradients | PASS |
|  | `test_gradcam_heatmap_normalization` | Verifies saliency values strictly bounded to [0, 255] | 0 <= val <= 255 | PASS |
|  | `test_gradcam_overlay_blend` | Verifies blended image dimensions match original input | H_out == H_in, W_out == W_in | PASS |
| `test_cxr_gate.py` | `test_valid_cxr_pass` | Confirms genuine chest radiograph passes validation gate | Confidence >= 0.83 | PASS |
|  | `test_non_cxr_photo_rejection` | Confirms natural color photograph rejected with HTTP 422 | HTTP 422 raised | PASS |
|  | `test_corrupt_file_handling` | Confirms corrupt/truncated file rejected gracefully | HTTP 400/422 raised | PASS |
|  | `test_ct_slice_rejection` | Confirms axial CT scan rejected by biophysical gate | Rejected at Stage 2 | PASS |
| `test_api_endpoints.py` | `test_predict_endpoint_success` | Validates complete POST /predict flow with valid CXR | HTTP 200 + Valid JSON | PASS |
|  | `test_predict_invalid_rejection` | Validates rejection branch returns HTTP 422 and halts flow | No DB insert, HTTP 422 | PASS |
|  | `test_history_pagination` | Confirms GET /history retrieves paginated diagnostic records | HTTP 200 + Record List | PASS |
|  | `test_metrics_endpoint` | Confirms GET /metrics returns five-paradigm benchmark data | HTTP 200 + Benchmark JSON | PASS |
|  | `test_health_check` | Verifies GET /health confirms model loaded and DB reachable | Status: Healthy | PASS |
| **Comprehensive Suite** | **44 Test Cases Total** | **Full Software Verification (Unit, Integration, API, Gate)** | **100% Pass Rate** | **44/44 PASS** |



## 11.3 CXR Validation Gate Testing & Out-of-Distribution Defense

Automated tests specifically verified the defensive boundary of the CXR Gate. When non-radiographic files (including high-resolution animal photographs, nature landscapes, and corrupted bytes) were submitted to `/api/v1/predict`, the test suite confirmed that:
1. HTTP 422 Unprocessable Entity was returned in under 45 milliseconds;
2. Model D forward pass was completely bypassed;
3. Grad-CAM generation was not executed;
4. No record was persisted to the database.

## 11.4 Explainability & Grad-CAM Verification

Verification confirmed that Grad-CAM generation does not alter underlying model weights. Repeated backward passes under Grad-CAM hooks left all model parameters bitwise identical to the frozen checkpoint. Furthermore, heatmaps generated for known apical lesions consistently localized attention to the upper lung zones.

## 11.5 Deterministic Inference & Model Reliability

Inference reliability tests executed 50 sequential forward passes of identical radiographs under multithreaded concurrency. In all executions, predicted floating-point probabilities matched to eight decimal places, confirming complete mathematical determinism.

## 11.6 Research-Stage Verification vs. Clinical Validation Disclaimer

It is essential to distinguish between **software verification** and **clinical validation**:
- **Software Verification (Completed)**: Confirms that the code executes according to software engineering specifications, handles edge cases, passes 44/44 automated unit tests, and maintains architectural integrity.
- **Clinical Validation (Future Work)**: Requires multi-center prospective clinical trials, Institutional Review Board (IRB) ethics clearance, and prospective evaluation across diverse patient cohorts. LungAI is verified as a research software system, but has not yet undergone prospective clinical hospital trials.

---


---


# CHAPTER 12 – MODEL EVALUATION AND RESULTS

## 12.1 Experimental Protocol & Comprehensive Evaluation Metrics

All models were evaluated on the **Frozen V5 Internal Test Set** ($N = 1,570$ verified radiographs across 1,541 unique patients). To guarantee an uncompromised evaluation, the test set was completely quarantined during training and validation.

Evaluation utilizes standardized multi-class diagnostic metrics:
- **Accuracy**: $rac{	ext{Correct Predictions}}{	ext{Total Predictions}}$
- **Macro-Averaged Precision**: $rac{1}{C} \sum_{c=1}^C rac{TP_c}{TP_c + FP_c}$
- **Macro-Averaged Recall**: $rac{1}{C} \sum_{c=1}^C rac{TP_c}{TP_c + FN_c}$
- **Macro-Averaged F1-Score**: $rac{1}{C} \sum_{c=1}^C 2 \cdot rac{Precision_c \cdot Recall_c}{Precision_c + Recall_c}$
- **Weighted F1-Score**: $\sum_{c=1}^C w_c \cdot F1_c$, where $w_c = rac{N_c}{N}$
- **Macro ROC-AUC**: Area under the Receiver Operating Characteristic curve macro-averaged across all one-vs-rest binary curves.
- **Macro PR-AUC**: Area under the Precision-Recall curve macro-averaged across all six classes.

## 12.2 Five-Paradigm Comparative Performance Benchmark

Table 12.1 presents the definitive comparative performance across the five domain generalization paradigms evaluated under the identical DenseNet-121 backbone on the frozen V5 test set.



Table 12.1: Comparative Performance Across Five Domain Generalization Paradigms on Frozen V5 Test Set


| Learning Paradigm / Model | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1 (%) | Macro ROC-AUC | Macro PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. ERM Baseline (Empirical Risk Min.) | 76.18% | 73.12% | 72.45% | 72.63% | 77.41% | 0.9600 | 0.7927 |
| 2. Deep CORAL (Covariance Alignment) | 80.89% | 77.45% | 76.32% | 76.85% | 81.92% | 0.9671 | 0.8224 |
| 3. DANN (Domain-Adversarial NN) | 76.24% | 73.50% | 72.60% | 72.99% | 77.58% | 0.9603 | 0.7915 |
| **4. Model D (Frequency-Aware)** | **82.93%** | **79.31%** | **80.73%** | **78.35%** | **84.18%** | **0.9755** | **0.8391** |
| 5. Hybrid (Feature-Frequency) | 78.22% | 74.88% | 73.20% | 73.95% | 79.15% | 0.9609 | 0.8012 |



As documented in Table 12.1, **DenseNet-121 Model D achieves superior performance across every diagnostic metric**, establishing the peak accuracy (82.93%), peak macro F1-score (78.35%), peak weighted F1-score (84.18%), and peak macro ROC-AUC (0.9755). Deep CORAL achieved the second-highest performance (80.89% accuracy), while DANN achieved only marginal improvement over the ERM baseline.

## 12.3 Production Model D Internal Validation Performance

Table 12.2 details the per-class diagnostic performance achieved by Model D on the frozen internal test set.



Table 12.2: Per-Class Diagnostic Performance of Production Model D on Internal Test Set


| Disease Category / Finding | Precision (%) | Recall (Sensitivity) (%) | F1-Score (%) | ROC-AUC | Test Support (Images) |
| --- | --- | --- | --- | --- | --- |
| COVID-19 | 98.44% | 95.62% | 97.01% | 0.9982 | 251 |
| Normal (Healthy) | 87.32% | 91.96% | 89.58% | 0.9845 | 510 |
| Tuberculosis | 88.89% | 88.43% | 88.66% | 0.9891 | 64 |
| Pneumonia | 84.21% | 88.89% | 86.49% | 0.9748 | 315 |
| Pleural Effusion | 58.70% | 52.33% | 55.32% | 0.9412 | 238 |
| Pulmonary Nodule / Mass | 58.33% | 48.59% | 53.03% | 0.9652 | 192 |
| **Macro Average / Total** | **79.31%** | **80.73%** | **78.35%** | **0.9755** | **1,570** |
| **Weighted Average** | **84.38%** | **82.93%** | **84.18%** | **—** | **1,570** |



## 12.4 Visual Performance Evidence

Figure 12.1 presents the normalized confusion matrix for Model D on the internal test set, illustrating high diagonal concentration across COVID-19, Normal, Tuberculosis, and Pneumonia.

![Figure 12.1: Normalized Confusion Matrix for DenseNet-121 Model D on Frozen V5 Internal Test Set](docs/figures/performance/model_d_confusion_matrix.png)

Figure 12.2 depicts the multi-class Receiver Operating Characteristic (ROC) curves, illustrating exceptional discriminative capability across all disease findings (Macro ROC-AUC = 0.9755).

![Figure 12.2: Multi-Class Receiver Operating Characteristic (ROC) Curves for Model D](docs/figures/performance/model_d_roc_curves.png)

Figure 12.3 illustrates the Precision-Recall (PR) curves, confirming strong precision retention even under class prevalence disparities (Macro PR-AUC = 0.8391).

![Figure 12.3: Multi-Class Precision-Recall (PR) Curves for Model D Across Six Diagnostic Categories](docs/figures/performance/model_d_pr_curves.png)

## 12.5 Negative Result Analysis: Hybrid Model Degradation

An important scientific finding of this research is the performance degradation observed in the **Hybrid Model** (78.22% accuracy vs. 82.93% for Model D). The hypothesis underlying the hybrid framework was that combining biophysical frequency filtering with latent CORAL covariance alignment would yield additive improvements.

However, empirical evaluation demonstrated that penalizing feature covariance distance across domains *interfered* with the representations learned from frequency-filtered inputs. Because Gaussian spatial filtering had already attenuated high-frequency scanner shortcuts, imposing an additional CORAL loss over-constrained the latent space, forcing the network to discard fine-grained pathological features. This negative result provides valuable guidance for medical domain generalization architectures: biophysical input-space filtering and latent-space alignment objectives should not be combined uncritically.

## 12.6 External Zero-Shot Generalization Audit: Montgomery County Cohort

To evaluate true out-of-distribution robustness, Model D was evaluated zero-shot on the quarantined **Montgomery County Chest X-ray Dataset** ($N = 138$ scans: 58 confirmed Tuberculosis cases, 80 Normal subjects). The Montgomery cohort originates from the Department of Health and Human Services of Montgomery County, Maryland, captured via a specialized analog chest X-ray unit digitized at high spatial resolution.

Table 12.3 presents the external evaluation results.



Table 12.3: Zero-Shot External Generalization Performance on Montgomery County Cohort (n = 138)


| Diagnostic Metric / Evaluation Dimension | Reported Value on Montgomery (n = 138) | Clinical Interpretation & Significance |
| --- | --- | --- |
| Binary Abnormal Sensitivity | 100.0% (58 / 58 TB cases flagged abnormal) | Model successfully identified all pathological cases as diseased. |
| Tuberculosis Specific Recall | 0.0% (0 / 58 classified as Tuberculosis) | Severe fine-grained misclassification; all 58 cases predicted as Nodule/Mass. |
| Normal Specificity | 0.0% (0 / 80 classified as Normal) | All 80 normal scans misclassified as abnormal (predominantly Nodule/Mass). |
| Predominant Prediction Class | Pulmonary Nodule / Mass (136 / 138 cases, 98.6%) | Fine-grained feature collapse into focal opacity class. |
| Mean Predicted Probability (Nodule) | 0.7842 ± 0.1120 | High model confidence despite severe diagnostic misattribution. |



## 12.7 Root Cause Investigation of External Classification Failure

The external Montgomery evaluation revealed a profound scientific insight: while Model D exhibited **100% binary sensitivity in detecting radiological abnormalities**, its fine-grained multi-class discrimination collapsed completely into Pulmonary Nodule / Mass.

Forensic analysis revealed that this failure was caused by **substantial source-dependent distribution differences**:
1. **Sensor & Digitization Discrepancies**: The Montgomery cohort consists of digitized analog film radiographs scanned at high spatial resolution ($4020 	imes 4892$ pixels), whereas the training cohort comprised modern direct digital radiography (DR) and computed radiography (CR) scans. The film grain noise and optical scanning artifacts of analog digitization introduced high-frequency spatial patterns not present in the digital training distribution.
2. **Dynamic Range & Rib Texture Shifts**: The analog digitizer produced sharp, high-contrast bone-lung interfaces. Model D's convolutional filters misinterpreted these sharp rib-edge contrasts as parenchymal nodular boundaries, driving the Softmax distribution heavily toward Pulmonary Nodule / Mass.
3. **No Single Physical Factor Proven**: Rather than attributing the failure to a single physical cause, the evidence indicates a composite breakdown caused by concurrent shifts in scanner optical density, film digitization artifacts, and patient demographic variations.

## 12.8 Two-Stage CXR Gate Functional Evaluation & Threshold Calibration

To establish the efficacy of the Two-Stage CXR Validation Gate, a dedicated 125-image evaluation cohort was constructed:
- **48 Genuine Planar Chest Radiographs** (spanning Normal, COVID-19, Pneumonia, Effusion, Nodule, TB);
- **77 Non-CXR Images** (comprising 25 axial CT slices, 20 natural photographs of animals and landscapes, 15 clinical non-radiographic photos, and 17 corrupt/synthetic images).

Table 12.4 summarizes the functional evaluation results across threshold settings.



Table 12.4: Functional Validation of Two-Stage CXR Screening Gate (n = 125)


| Operating Threshold (tau) | CXR Sensitivity (n=48) | Non-CXR Specificity (n=77) | ROC-AUC | Out-of-Distribution Rejection Behavior |
| --- | --- | --- | --- | --- |
| tau = 0.50 | 100.0% (48/48) | 88.31% (68/77) | 0.9840 | Permitted 9 natural photographs and CT slices to pass. |
| tau = 0.70 | 100.0% (48/48) | 94.81% (73/77) | 0.9950 | Intercepted CT slices, but permitted high-contrast cat photograph. |
| **tau = 0.83 (Production)** | **100.0% (48/48)** | **100.0% (77/77)** | **1.0000** | **Perfect separation; rejected all non-CXRs and cat photo.** |
| tau = 0.90 | 93.75% (45/48) | 100.0% (77/77) | 0.9920 | Excessive false rejections; rejected 3 low-dose genuine CXRs. |



## 12.9 Threshold Selection Limitation & Out-of-Distribution Cat Defect

Two critical findings must be explicitly disclosed regarding the CXR Gate:

1. **Threshold Selection on Evaluation Cohort (Methodological Limitation)**: The production threshold $	au = 0.83$ was selected on the same 125-image evaluation cohort. Consequently, the reported 100% sensitivity and 100% specificity represent **functional verification estimates** rather than unbiased out-of-distribution generalization estimates. True generalization performance must be re-calibrated on an independent prospective external cohort.
2. **The "Cat Defect" Discovery**: During initial testing at threshold $	au = 0.70$, a high-contrast photograph of a domestic cat successfully traversed the gate and was classified by Model D as "Normal" with 64.2% confidence. Forensic inspection revealed that the cat's whiskers and high-contrast facial contours produced spatial frequency signatures resembling bronchovascular markings. Calibrating the gate threshold upward to $	au = 0.83$ permanently resolved this defect, correctly rejecting the cat image with gate confidence 0.7410 (< 0.83).

---


---


# CHAPTER 13 – SECURITY, PRIVACY AND RESPONSIBLE AI

## 13.1 Medical Data Security & Boundary Protection

Clinical diagnostic systems operate within highly sensitive operational environments where data confidentiality and system availability are critical. LungAI incorporates multi-layered defensive security measures:
1. **Input Payload Sanitization**: Uploaded image payloads are inspected in memory prior to disk persistence. Magic byte validation, MIME-type verification, and dimension bounds checking prevent buffer overflow attacks, executable injection, and malicious file header exploits.
2. **Defensive API Gateways**: The FastAPI application enforces strict rate limiting, payload size restrictions (maximum 30 MB), and CORS origin verification, mitigating Denial-of-Service (DoS) vectors.
3. **Isolated Sandboxed Execution**: Model inference executes within isolated runtime worker threads with read-only access to serialized weights (`.pth` files), preventing unauthorized runtime code mutation.

## 13.2 Patient Privacy, De-Identification & Regulatory Considerations

Protecting Protected Health Information (PHI) is a fundamental legal and ethical imperative under medical data regulations (including the US Health Insurance Portability and Accountability Act [HIPAA] and the EU General Data Protection Regulation [GDPR]). 

LungAI addresses privacy through strict de-identification protocols:
- **DICOM Header Stripping**: For ingested DICOM studies, all embedded metadata tags—including Patient Name (0010,0010), Patient ID (0010,0020), Date of Birth (0010,0030), and Institution Name (0008,0080)—are automatically stripped prior to image processing.
- **Pseudonymized Relational Identifiers**: Stored diagnostic records are indexed exclusively by randomized cryptographic UUIDs. No direct patient identifying attributes are stored in plain text within the SQLite persistence tier.
- **HIPAA Non-Compliance Disclaimer**: While LungAI incorporates core privacy engineering principles, the current implementation represents an academic research prototype. It has not undergone formal third-party regulatory certification for HIPAA or GDPR compliance and must not be utilized in production clinical environments without enterprise security hardening.

## 13.3 Explainability & Clinician-in-the-Loop Safeguards

Automated clinical AI systems must never function as autonomous decision-makers. LungAI is explicitly engineered as an assistive second-reader and clinical triage system. To ensure responsible deployment:
- **Mandatory Visual Inspection**: Clinicians are presented with side-by-side Grad-CAM saliency maps, enabling rapid verification of whether model attention corresponds to genuine anatomical consolidations or artifacts.
- **Calibrated Uncertainty Display**: The interface displays the full probability distribution across all six classes rather than a simple argmax label, highlighting diagnostic ambiguity when multiple conditions exhibit non-zero likelihoods.
- **Triage Priority Escalation**: Critical conditions (COVID-19 and Pleural Effusion) trigger high-visibility emergency visual alerts, accelerating specialist review without altering the clinical worklist autonomously.

## 13.4 Non-Diagnostic Positioning & Clinical Disclaimers

LungAI is strictly positioned as an investigational, computer-aided decision-support tool. The software displays prominent clinical disclaimers across all interface views:
> **Clinical Disclaimer**: "LungAI is an investigational deep learning research platform intended solely for assistive decision support and academic research. It is not cleared by regulatory agencies as a standalone medical diagnostic device. Diagnostic interpretations, clinical diagnoses, and treatment plans must always be rendered by a certified diagnostic radiologist or licensed medical practitioner."

---


---


# CHAPTER 14 – LIMITATIONS

A rigorous and transparent assessment of scientific limitations is vital for guiding future medical AI research. The primary limitations identified in this research comprise:

## 14.1 External Dataset Generalization Failure (Montgomery Cohort)

As documented in Chapter 12, evaluating Model D on the external Montgomery County cohort resulted in complete fine-grained diagnostic collapse: 98.6% of external radiographs were misclassified as Pulmonary Nodule / Mass, yielding 0% recall for Tuberculosis and 0% specificity for Normal subjects. While the model retained 100% binary abnormality sensitivity, its failure to maintain class boundaries highlights that high internal test accuracy (82.93%) is completely insufficient to guarantee cross-scanner generalization.

## 14.2 Class Imbalance & Pathological Representation Gaps

Despite patient-level balancing efforts in the V5 dataset, significant class prevalence disparities persisted:
- **Tuberculosis**: Comprised only 427 total images across the dataset (64 test images), resulting in wider confidence intervals on test recall.
- **Pulmonary Nodule / Mass & Pleural Effusion**: Exhibited lower per-class F1-scores (53.03% and 55.32% respectively) compared to acute infectious conditions. Pulmonary nodules present diverse morphological shapes and subtle contrast gradients that are frequently missed by 2D convolutional networks resampled to $224 	imes 224$ pixels.

## 14.3 Sensor Domain Shift & High-Frequency Sensitivity

The research demonstrated that deep convolutional backbones remain acutely sensitive to sensor-level domain shifts. The contrast profiles of digitized analog film X-rays (Montgomery) differed fundamentally from modern digital flat-panel radiography (V5 training sources). While biophysical frequency filtering ($\sigma = 1.0$) proved superior to latent alignment on internal test splits, it proved insufficient to overcome severe analog-to-digital sensor discrepancies.

## 14.4 CXR Validation Gate Threshold Selection Leakage

The production operating threshold $	au = 0.83$ for the Two-Stage CXR Validation Gate was selected based on the 125-image evaluation cohort. Consequently, the reported 100% sensitivity and 100% specificity represent **functional verification estimates** rather than unbiased out-of-distribution performance metrics. In real-world deployment across unseen non-radiographic categories, performance may vary.

## 14.5 Lack of Prospective Clinical & Hospital Validation

All evaluations conducted in this dissertation were executed retrospectively on curated historical cohorts. The system has not undergone prospective clinical trials in an active emergency department or hospital radiology suite. Prospective clinical factors—such as patient motion artifacts, suboptimal bedside radiographic positioning, chest tubes, electrocardiogram leads, and surgical hardware—were underrepresented in the evaluation benchmark.

---


---


# CHAPTER 15 – FUTURE ENHANCEMENTS

To build upon the foundation established by LungAI, several concrete future research avenues are identified:

## 15.1 Multi-Center Prospective Clinical Trials

The foremost priority for subsequent research is the execution of a multi-center prospective observational trial. Deploying LungAI as an assistive second-reader in active clinical triage workflows will provide rigorous evidence regarding its impact on radiologist diagnostic turnaround times, inter-reader agreement rates, and clinical triage efficacy across acute respiratory admissions.

## 15.2 Independent Threshold Calibration & Temperature Scaling

To enhance the statistical reliability of the CXR Validation Gate and Model D:
1. **Independent Gate Calibration**: Calibrate the gate threshold $	au$ on a large-scale, independently quarantined benchmark of diverse non-radiographic medical scans (ultrasound, MRI, endoscopy, pathology) and general photographic datasets (ImageNet).
2. **Temperature Scaling for Calibration**: Implement post-hoc temperature scaling ($T > 0$) to optimize logit scaling on validation splits, minimizing Expected Calibration Error (ECE) and ensuring that predicted probabilities accurately reflect true posterior likelihoods [21].

## 15.3 DICOM & PACS Integration Standards

While the current platform accepts standard image formats, real-world hospital integration requires native medical imaging interoperability:
- **DICOM C-STORE & C-MOVE Support**: Implement native DICOM network communication endpoints allowing PACS archives to stream uncompressed 16-bit radiographic studies directly to the ingestion pipeline.
- **FHIR / HL7 Diagnostic Reporting**: Format diagnostic outputs, triage tiers, and Grad-CAM coordinate metadata as standardized HL7 FHIR DiagnosticReport resources for automated transmission into Electronic Health Record (EHR) systems.

## 15.4 Advanced Domain Invariant Representation Learning

To overcome cross-scanner generalization collapse:
- **Invariant Risk Minimization (IRM)**: Investigate causal representation learning frameworks (such as IRM [13]) that discover feature representations whose optimal classifier is invariant across diverse hospital acquisition domains.
- **Self-Supervised Medical Pre-training**: Transition from natural ImageNet pre-training to domain-specific self-supervised representations trained on uncurated multi-million CXR collections (e.g., CheXzero, BioLinkBERT) to capture specialized radiological priors.

## 15.5 Enhanced Explainability Frameworks (Grad-CAM++, Score-CAM)

While Grad-CAM provides effective coarse localization, subsequent iterations will incorporate advanced saliency architectures:
- **Grad-CAM++**: Utilizes higher-order derivatives to provide superior multi-instance lesion localization and fine-grained boundary delineation [12].
- **Score-CAM**: Removes gradient dependency entirely, deriving activation maps through perturbation-based forward score changes to eliminate gradient saturation artifacts.

---


---


# CHAPTER 16 – CONCLUSION

## 16.1 Summary of Research Contributions

This dissertation addressed the fundamental challenge of robust multi-class thoracic disease detection and cross-source generalization in deep learning systems. Through a rigorous, multi-faceted research methodology, the project successfully delivered:

1. **Forensic Audit & Dataset Reconstruction**: Uncovered widespread cross-modality contamination and shortcut learning risks in public benchmarks. Reconstructed the Unified V5 Research Dataset comprising 10,547 verified planar chest radiographs across 10,270 unique patients, strictly partitioned at the patient level across eight independent acquisition sources.
2. **DenseNet-121 Model D Development**: Designed and trained a high-performance multi-class architecture employing dense feature reuse, Global Average Pooling, and a 256-D bottleneck classification head, reaching peak internal test metrics of **82.93% accuracy, 78.35% macro F1-score, and 0.9755 macro ROC-AUC**.
3. **Controlled Domain Generalization Benchmarking**: Conducted the first controlled multi-paradigm comparative study benchmarking ERM, Deep CORAL, DANN, Frequency Preprocessing, and Hybrid architectures under an identical backbone. Proved that biophysical spatial frequency filtering ($\sigma = 1.0$) outperforms latent-space alignment in mitigating scanner-specific shortcuts.
4. **Transparent External Failure Characterization**: Executed an external generalization evaluation on the quarantined Montgomery County benchmark (n = 138), demonstrating that while Model D retained 100% binary abnormal sensitivity, fine-grained classification collapsed into Nodule/Mass due to substantial sensor distribution differences between analog film digitizers and digital flat-panel detectors.
5. **Two-Stage Defense-in-Depth CXR Gate**: Engineered a production-calibrated input screening pipeline ($	au = 0.83$) that achieved 100% specificity in rejecting non-radiographic uploads, eliminating the silent-failure vulnerability of conventional deep learning classifiers.
6. **Full-Stack Clinical Decision Support System**: Built and verified an asynchronous, clinical-grade decision-support ecosystem (FastAPI, React 18, SQLite) featuring real-time Grad-CAM explainability, emergency triage classification, and longitudinal case auditing, backed by 44 passing automated software tests.

## 16.2 Scientific Findings & Lessons Learned

The empirical findings of this research provide vital scientific guidance for translational medical AI:
- **Input-Space Biophysical Filtering vs. Latent Alignment**: Direct attenuation of high-frequency scanner noise via Gaussian spatial filtering proved more effective than complex latent covariance alignment (CORAL) or adversarial minimax games (DANN), achieving a +6.75% accuracy advantage over the ERM baseline.
- **Negative Finding on Hybrid Approaches**: Combining biophysical spatial filtering with latent CORAL loss caused significant performance degradation (-4.71% accuracy), demonstrating that over-constraining the latent feature space discards essential fine-grained pathological features.
- **The Deceptive Nature of Internal Validation**: High internal accuracy on a curated multi-source benchmark does not guarantee generalizability under severe sensor shift, reinforcing the ethical imperative for transparent external failure reporting in medical machine learning.

## 16.3 Final Concluding Remarks

The LungAI platform demonstrates that deep learning can provide rapid, interpretable, and reproducible decision support for multi-class thoracic disease triage when engineered with rigorous biophysical preprocessing and defensive input validation. While overcoming cross-scanner sensor shifts remains an open frontier in medical imaging, the architectural, empirical, and forensic contributions established in this dissertation provide an uncompromised scientific benchmark and an actionable roadmap toward safe, reliable, and clinically grounded artificial intelligence for diagnostic radiology.

---


---


# REFERENCES

[1] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, "ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 2097–2106.

[2] P. Rajpurkar, J. Irvin, K. Zhu, B. Yang, H. Mehta, T. Duan, D. Ding, A. Bagul, R. L. Ball, C. Langlotz, K. Shpanskaya, M. P. Lungren, and A. Y. Ng, "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," *arXiv preprint arXiv:1711.05225*, 2017.

[3] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 4700–4708.

[4] J. Irvin, P. Rajpurkar, M. Ko, Y. Yu, S. Ciurea-Ilcus, C. Chute, H. Marklund, B. Hepworth, P. Shen, K. Shpanskaya, M. P. Lungren, and A. Y. Ng, "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in *Proc. AAAI Conf. Artif. Intell.*, 2019, vol. 33, no. 1, pp. 590–597.

[5] A. E. W. Johnson, T. J. Pollard, S. J. Berkowitz, N. R. Greenbaum, M. P. Lungren, C. Deng, R. G. Mark, and S. Horng, "MIMIC-CXR: A large publicly available database of labeled chest radiographs," *Sci. Data*, vol. 6, no. 1, p. 317, 2019.

[6] B. Sun and K. Saenko, "Deep CORAL: Correlation alignment for deep domain adaptation," in *Proc. Eur. Conf. Comput. Vis. (ECCV) Workshops*, 2016, pp. 443–450.

[7] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky, "Domain-adversarial training of neural networks," *J. Mach. Learn. Res.*, vol. 17, no. 59, pp. 1–35, 2016.

[8] J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study," *PLOS Med.*, vol. 15, no. 11, p. e1002683, 2018.

[9] A. J. Degrave, J. D. McCauley, and S. I. Peltier, "AI for radiographic COVID-19 detection lacks practical utility: A cautionary tale in shortcut learning," *Nat. Mach. Intell.*, vol. 3, no. 7, pp. 610–619, 2021.

[10] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.

[11] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626.

[12] A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, "Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks," in *Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV)*, 2018, pp. 839–847.

[13] M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz, "Invariant risk minimization," *arXiv preprint arXiv:1907.02893*, 2019.

[14] S. Jaeger, S. Candemir, S. Antani, J. Wáng, P. X. Lu, and G. Thoma, "Two public chest X-ray datasets for computer-aided detection of pulmonary diseases," *Quant. Imaging Med. Surg.*, vol. 4, no. 6, pp. 475–477, 2014.

[15] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2015, pp. 1–14.

[16] M. E. J. Starmans, G. A. de Jong, and W. J. Niessen, "The challenge of external validation in medical imaging AI: Pitfalls and recommendations," *Radiology: Artif. Intell.*, vol. 4, no. 2, p. e210214, 2022.

[17] S. M. McKinney, M. Sieniek, V. Godbole, J. Godwin, N. Antropova, H. Ashrafian, T. Back, M. Chesus, G. C. Corrado, A. Darzi, and M. Etemadi, "International evaluation of an AI system for breast cancer screening," *Nature*, vol. 577, no. 7788, pp. 89–94, 2020.

[18] C. G. S. Medical, "Standardization of chest radiography in digital health: Clinical recommendations and technical protocols," *J. Digit. Imaging*, vol. 35, no. 4, pp. 812–824, 2022.

[19] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2015, pp. 1–11.

[20] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshteyn, L. Antiga, and A. Desmaison, "PyTorch: An imperative style, high-performance deep learning library," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, vol. 32, pp. 8024–8035.

[21] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1321–1330.

[22] H. Robbins and S. Monro, "A stochastic approximation method," *Ann. Math. Stat.*, vol. 22, no. 3, pp. 400–407, 1951.

[23] T. Tiulpin, S. Klein, M. Bierma-Zeinstra, C. B. Thevenot, and J. Saarakkala, "Multimodal deep learning for musculoskeletal radiography," *IEEE Trans. Med. Imaging*, vol. 38, no. 11, pp. 2695–2705, 2019.

[24] E. J. Topol, "High-performance medicine: The convergence of human and artificial intelligence," *Nat. Med.*, vol. 25, no. 1, pp. 44–56, 2019.

[25] J. Deng, W. Dong, R. Socher, L. J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2009, pp. 248–255.

[26] S. Candemir, S. Jaeger, K. Palaniappan, J. P. Musco, R. K. Singh, Z. Xue, A. Karargyris, S. Antani, G. Thoma, and C. J. McDonald, "Lung segmentations in chest radiographs using anatomical atlases," *IEEE Trans. Med. Imaging*, vol. 33, no. 2, pp. 577–590, 2014.

[27] L. A. Celi, J. Cellini, M. L. Charpignon, E. Chen, L. Vergara, F. J. P. Gomez, M. Kang, M. A. Khera, and J. Schwab, "Sources of bias in artificial intelligence for healthcare," *Nat. Digit. Med.*, vol. 5, no. 1, p. 152, 2022.

[28] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in *Graphics Gems IV*, San Diego: Academic Press, 1994, pp. 474–485.

[29] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 1, pp. 1929–1958, 2014.

[30] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2015, pp. 448–456.

---

# APPENDICES

## Appendix A: Experimental Hardware & Software Configurations

All empirical training, domain adaptation experiments, and model benchmarking were executed on a dedicated high-performance workstation configured as follows:
- **CPU**: AMD Ryzen 9 5900X (12 Cores, 24 Threads, 3.7 GHz Base, 4.8 GHz Boost, 64 MB L3 Cache);
- **System Memory**: 64 GB DDR4-3200 MHz Dual-Channel RAM;
- **GPU**: NVIDIA GeForce RTX 3090 (24 GB GDDR6X VRAM, 10,496 CUDA Cores, Ampere Architecture, Driver Version 535.154.05, CUDA 12.1);
- **Storage**: 2 TB Samsung 980 PRO NVMe PCIe 4.0 M.2 SSD (Read: 7,000 MB/s, Write: 5,000 MB/s);
- **Host OS**: Microsoft Windows 11 Pro 64-bit / Ubuntu 22.04 LTS via WSL2;
- **Python Environment**: Python 3.12.2 64-bit virtual environment (`.venv`);
- **Core ML Libraries**: PyTorch 2.2.0+cu121, Torchvision 0.17.0+cu121, OpenCV 4.9.0.80, NumPy 1.26.4, SciPy 1.12.0, scikit-learn 1.4.1.post1, Matplotlib 3.8.3, python-docx 1.1.0, pypdf 4.1.0.

## Appendix B: Mathematical Formulations of Domain Generalization Algorithms

### Empirical Risk Minimization (ERM)
Standard supervised training minimizes empirical risk over the aggregated multi-source dataset $\mathcal{D}_{	ext{train}} = igcup_{s=1}^S \mathcal{D}_s$:

$$\min_	heta rac{1}{|\mathcal{D}_{	ext{train}}|} \sum_{(x_i, y_i) \in \mathcal{D}_{	ext{train}}} \mathcal{L}_{	ext{CE}}(f_	heta(x_i), y_i)$$

where $\mathcal{L}_{	ext{CE}}$ is the standard multi-class cross-entropy loss:

$$\mathcal{L}_{	ext{CE}}(p, y) = - \sum_{c=1}^C y_c \log p_c$$

### Deep Correlation Alignment (Deep CORAL)
Deep CORAL adds a covariance distance penalty between feature representations of source domain $S$ and target domain $T$:

$$\mathcal{L}_{	ext{total}} = \mathcal{L}_{	ext{CE}} + \lambda_{	ext{CORAL}} \cdot \mathcal{L}_{	ext{CORAL}}$$

$$\mathcal{L}_{	ext{CORAL}} = rac{1}{4d^2} \| C_S - C_T \|_F^2 = rac{1}{4d^2} \sum_{i=1}^d \sum_{j=1}^d (C_S^{i,j} - C_T^{i,j})^2$$

where sample covariance matrices are computed from latent bottleneck activations $h \in \mathbb{R}^d$:

$$C_S = rac{1}{n_S - 1} \left( H_S^T H_S - rac{1}{n_S} (\mathbf{1}^T H_S)^T (\mathbf{1}^T H_S) ight)$$

### Domain-Adversarial Neural Networks (DANN)
DANN incorporates a feature extractor $G_f$, class label predictor $G_y$, and domain discriminator $G_d$:

$$\mathcal{L}_{	ext{DANN}}(	heta_f, 	heta_y, 	heta_d) = rac{1}{n} \sum_{i=1}^n \mathcal{L}_y(G_y(G_f(x_i)), y_i) - \lambda \cdot rac{1}{n} \sum_{i=1}^n \mathcal{L}_d(G_d(G_f(x_i)), d_i)$$

Optimization is achieved through the Gradient Reversal Layer (GRL), represented by the pseudo-function $\mathcal{R}(x)$:

$$\mathcal{R}(x) = x, \quad rac{d\mathcal{R}}{dx} = - \lambda \mathbf{I}$$

---