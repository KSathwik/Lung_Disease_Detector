# Problem Definition and Objectives

## Problem statement

Given one frontal chest radiograph, assign it to exactly one of six classes: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass or Tuberculosis. The class order is fixed in the configuration file and is used by the trainer, the evaluator, the backend and the frontend. Alongside the class, the system must return the six class scores, a Grad-CAM overlay for the predicted class, and the evaluation metrics of the model that produced them.

The task is to design and develop a system that does this end to end: from public image collections to a trained classifier, and from the classifier to an application in which a user uploads a radiograph and receives a prediction. The numbers reported for the system must also describe how it will behave on images it has not seen, from sources it has not seen. That requires a dataset whose splits are free of leakage of known patient identities and of duplicates, whose labels are defined by explicit rules, and whose evaluation includes a cohort that played no part in development.

## Single-label formulation and its consequences

Real radiographs often show more than one finding. The classifier treats the six classes as mutually exclusive, so a multi-finding image has to be forced into one class or excluded. The project excludes multi-finding images from the NIH collection (only images with exactly one of the target findings, or with "No Finding", are used). The consequence is that the test set contains clear single-label cases, and the reported metrics do not describe performance on the full clinical mixture. This is a modelling assumption, not a clinical statement, and it is repeated among the limitations.

## Requirements

[[tab:funcreq]] lists the functional requirements of the system and [[tab:nonfuncreq]] the non-functional requirements. Chapter 6 describes how each is implemented and Appendix D how it is tested.

Table: Functional requirements of the LungAI system. {#tab:funcreq}

| No. | Requirement |
|---|---|
| F1 | Accept a chest radiograph upload (JPEG, PNG, BMP, TIFF or WebP, at most 10 MB) from the browser |
| F2 | Reject images that do not look like a chest radiograph, with a reason, before any prediction is made |
| F3 | Apply the same preprocessing that was used in training and return the six class scores and the predicted class |
| F4 | Return a Grad-CAM overlay for the predicted class |
| F5 | Mark a prediction as inconclusive when its highest score is below a configured floor (0.5) |
| F6 | Store every prediction with the model version and checkpoint hash, and let the user browse the history |
| F7 | Register and edit patient records and attach a prediction to a patient |
| F8 | Generate a plain-text report for a stored prediction |
| F9 | Show the evaluation metrics of the model that is loaded, including its external limitation |

Table: Non-functional requirements of the LungAI system. {#tab:nonfuncreq}

| No. | Requirement |
|---|---|
| N1 | Integrity: load a checkpoint only if its SHA-256 matches the configuration; never return a placeholder or mock prediction |
| N2 | Consistency: one configuration and one preprocessing implementation for training, evaluation and serving |
| N3 | Honesty of reporting: every number shown by the application or the report comes from a generated evaluation file |
| N4 | Reproducibility: every step is scripted, and hashes and split fingerprints are recorded |
| N5 | Hardware: training and serving run on a laptop CPU with 8 GB of memory |
| N6 | Security: an API key or loopback-only access, server-generated file names and validated upload types |
| N7 | Safety notice: every response states that the system is a research prototype and not a diagnostic tool |
| N8 | Testability: automated tests for the package, the backend and the frontend services |

## Lessons from the first version

The first version of the system (V5) met most of the functional requirements but not the data requirements behind them. [[tab:defects]] lists the defects found by auditing its pipeline and the design rule adopted in V6 against each. The audit is documented in Appendix E.

Table: Defects found in the V5 pipeline and the V6 countermeasure. {#tab:defects}

| Defect in V5 | Evidence | V6 countermeasure |
|---|---|---|
| Patient leakage between training and test | 154 of 1,570 test images (9.8%) share a patient with training images; 289 patient groups span splits | Whole patient groups are assigned to one split; groups that share an exact or near-duplicate image are merged |
| No usable patient identity | 8,478 rows had synthetic one-per-image identifiers | Identity status is recorded per row (genuine, derived from file name, or unknown); identity is never generated |
| Proxy labels | 1,395 TBX11K "sick but not TB" images labelled Pneumonia (49.7% of the class); 124 latent-TB images labelled Tuberculosis | "Sick but not TB" and latent TB are excluded; each class label comes from the source's own label for that condition |
| Multi-finding images | 1,440 of 1,467 VinDr images carried two or more findings | Only images with a single target finding are used; conflicting findings are excluded |
| Unclear provenance | Aggregated archives with renamed files | Each source is downloaded from its official point with URL, size and SHA-256 logged; aggregated archives are not used |
| Possible contamination of the external cohort | An aggregated TB archive is reported to contain Montgomery images | The archive is not used; every development image is checked by content hash and near-duplicate test against the external cohort |
| Optimistic, unverifiable evaluation | External specificity 0/80; checkpoint lost; no per-image predictions | Per-image predictions are saved; test and external sets are evaluated once; external sensitivity is reported only with specificity |
| Demonstration output without a model | The application returned a fixed prediction when no checkpoint was loaded | A missing or mismatching checkpoint returns an error and nothing is stored |

## Objectives

The project has the following objectives.

1. **System design.** Design an end-to-end system that predicts one of six lung conditions from a frontal chest radiograph, with one configuration and one preprocessing implementation shared by training, evaluation and serving.
2. **Data.** Acquire the public chest X-ray sources that can be obtained without special credentials from their official distribution points, record provenance, and construct a six-class dataset under label rules that are written down before training.
3. **Leakage control.** Split the dataset by patient group with duplicate merging, run leakage checks that stop the build on failure, and record which rows lack patient identity so that the strength of the separation can be stated honestly.
4. **Model.** Train a DenseNet-121 transfer-learning classifier with a protocol that uses only training and validation data for every decision, on commodity hardware (a laptop CPU).
5. **Application.** Develop a FastAPI and React application that meets the requirements of [[tab:funcreq]] and [[tab:nonfuncreq]]: input checking, predictions with Grad-CAM, an inconclusive outcome, stored results, patient records, reports and the metrics of the loaded checkpoint.
6. **Evaluation.** Evaluate on the internal test split and on the external cohort exactly once each, report confidence intervals, per-class, per-source and calibration results, repeat the training with further seeds and one preprocessing ablation, and report external sensitivity only together with specificity.
7. **Testing and reproducibility.** Test the code automatically, make every step scriptable, record hashes and fingerprints, and document how to reproduce each result and what cannot be reproduced without the images.

## Success criteria

The project is judged on whether the system works as specified and whether its numbers can be trusted, with no target for headline accuracy. The criteria are:

- the application meets every requirement of [[tab:funcreq]] and [[tab:nonfuncreq]], checked by automated tests and a manual browser check;
- the build report shows zero patient groups, zero identical files and zero external-cohort files spanning development splits;
- the training run reads only the training and validation splits, and its metadata records the build and split fingerprints;
- the test split and the external cohort are evaluated once, after the checkpoint is chosen;
- the application shows the same numbers as the evaluation reports, from a generated file;
- every claim in the report can be traced to a generated artifact, and every limitation that the artifacts reveal is stated.

## Evaluation questions

The evaluation of the system answers three questions.

1. How accurately does the system classify the six classes on an internal test split that is free of leakage of known patient identities and of duplicates, and how stable is this across training seeds?
2. How does the system behave on radiographs from a hospital that played no part in development?
3. Can the complete system be built, trained and served on a laptop CPU, and what does it cost in time and memory?

In brief: for the final model the internal test accuracy is {{TestAcc}}% (macro-F1 {{TestMacroFOne}}%), and over three seeds it is {{MSTestAccMean}} ± {{MSTestAccSD}}% (question 1). On the external cohort the specificity is {{ExtSpecK}} / {{ExtSpecN}} and no tuberculosis scan is named Tuberculosis ({{ExtTBK}} / {{ExtTBN}}), so the system does not yet transfer to an unseen hospital (question 2). The per-source, probe, external and Grad-CAM evidence is consistent with source cues being available to the classifier, but does not prove that they are used. The full answers are in Chapters 7 and 8; the third question is answered in Chapter 6. Supplementary measurements of the source information in the features are reported in Appendix G.

## Constraints

The project runs under the following constraints, which shape the design.

- **Compute.** The training machine is an 8 GB laptop without a GPU supported by the TensorFlow version used. A training epoch takes roughly 20 to 26 minutes for the final build (run 2), so the number of epochs per run is small.
- **Data access.** Sources behind credentialing (VinDr-CXR, via PhysioNet) or registration (JSRT) were not available.
- **Time.** The overnight build, training and evaluation run had to complete within a fixed time box.
- **Few seeds.** Only three seeds of the final recipe and one ablation run were possible in the time available.
- **No clinical partner.** There is no clinician-reviewed ground truth beyond that of the source datasets, so no clinical claim can be made.
