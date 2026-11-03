# Problem Definition and Objectives

## Problem statement

Given one frontal chest radiograph, assign it to exactly one of six classes: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass or Tuberculosis. The class order is fixed in the configuration file and is used by the trainer, the evaluator, the backend and the frontend. Alongside the class, the system must return the six class scores, a Grad-CAM overlay for the predicted class, and the evaluation metrics of the model that produced them.

The scientific problem is the second half of the task: the numbers reported for such a classifier must describe how it will behave on images it has not seen, from sources it has not seen. That requires a dataset whose splits are free of patient and duplicate leakage, whose labels are defined by explicit rules, and whose evaluation includes a cohort that played no part in development.

## Single-label formulation and its consequences

Real radiographs often show more than one finding. The classifier treats the six classes as mutually exclusive, so a multi-finding image has to be forced into one class or excluded. The project excludes multi-finding images from the NIH collection (only images with exactly one of the target findings, or with "No Finding", are used). The consequence is that the test set contains clear single-label cases, and the reported metrics do not describe performance on the full clinical mixture. This is a modelling assumption, not a clinical statement, and it is repeated among the limitations.

## Defects of the earlier pipeline that must be removed

[[tab:defects]] lists the defects found by auditing the V5 pipeline and the rule adopted in V6 against each. The audit is documented in Appendix E.

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

## Objectives

The project has the following objectives.

1. **Data.** Acquire the public chest X-ray sources that can be obtained without special credentials from their official distribution points, record provenance, and construct a six-class dataset under label rules that are written down before training.
2. **Leakage control.** Split the dataset by patient group with duplicate merging, run leakage checks that stop the build on failure, and record which rows lack patient identity so that the strength of the separation can be stated honestly.
3. **Model.** Train a DenseNet-121 transfer-learning classifier with a protocol that uses only training and validation data for every decision, on commodity hardware (a laptop CPU).
4. **Evaluation.** Evaluate on the internal test split and on the external cohort exactly once each, report confidence intervals, per-class, per-source and calibration results, and report external sensitivity only together with specificity.
5. **Application.** Provide a FastAPI and React application that serves predictions with Grad-CAM, rejects inputs that are not chest radiographs, stores results, and shows the metrics of the checkpoint that is loaded.
6. **Reproducibility.** Make every step scriptable, record hashes and fingerprints, test the code, and document how to reproduce each result and what cannot be reproduced without the images.
7. **Transparency about the past.** Document the audit of the earlier version so that the reasons for the rebuild are on record.

## Success criteria

The project is judged on process rather than on a headline accuracy. The criteria are:

- the build report shows zero patient groups, zero identical files and zero external-cohort files spanning development splits;
- the training run reads only the training and validation splits, and its metadata records the build and split fingerprints;
- the test split and the external cohort are evaluated once, after the checkpoint is chosen;
- the application shows the same numbers as the evaluation reports, from a generated file;
- every claim in the report can be traced to a generated artifact, and every limitation that the artifacts reveal is stated.

## Research questions

The work is organised around three questions.

1. How different are internal and external results when the internal split is free of patient and duplicate leakage?
2. Do per-source and external breakdowns show that any class is recognised by its source rather than by disease?
3. Can a leakage-controlled six-class pipeline be built, trained and served on a laptop CPU, and what does it cost in time and memory?

The answers to the first two questions depend on the final model and are given in Chapters 7 and 8. In brief: for the final model the internal test accuracy is {{TestAcc}}% while the external specificity is {{ExtSpecK}} / {{ExtSpecN}} and no tuberculosis scan is named Tuberculosis ({{ExtTBK}} / {{ExtTBN}}), so the gap is large even with a leakage-controlled split (question 1); the per-source, probe, external and Grad-CAM evidence is consistent with source cues being available to the classifier, but does not prove that they are used (question 2). The third is answered in Chapter 6.

## Constraints

The project runs under the following constraints, which shape the design.

- **Compute.** The training machine is an 8 GB laptop without a GPU supported by the TensorFlow version used. A training epoch takes roughly 13 to 14 minutes, so a run is limited to about ten epochs.
- **Data access.** Sources behind credentialing (VinDr-CXR, via PhysioNet) or registration (JSRT) were not available.
- **Time.** The overnight build, training and evaluation run had to complete within a fixed time box.
- **Single seed.** Repeating the training with several seeds was not possible in the time available.
- **No clinical partner.** There is no clinician-reviewed ground truth beyond that of the source datasets, so no clinical claim can be made.
