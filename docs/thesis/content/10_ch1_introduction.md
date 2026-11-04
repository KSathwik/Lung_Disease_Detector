# Introduction

## Background and motivation

Chest radiography (chest X-ray, CXR) is the most frequently performed diagnostic imaging examination in the world. It is cheap, fast and widely available, and it is usually the first test ordered when a patient presents with cough, breathlessness, fever or chest pain. A single frontal radiograph can show a pleural effusion, a lung nodule or mass, consolidation caused by pneumonia, or the characteristic upper-lobe changes of tuberculosis. Reading it well, however, takes training. In many regions the number of radiologists is small compared with the number of examinations, and the delay between acquisition and report can be long.

Computer-aided interpretation is attractive in exactly this setting. Convolutional neural networks (CNNs) trained on large collections of labelled radiographs can assign a finding to an image in a fraction of a second [@rajpurkar2017chexnet; @irvin2019chexpert]. A dense network such as DenseNet-121 [@huang2017densely], pre-trained on natural images [@deng2009imagenet] and fine-tuned on radiographs, is a standard and efficient starting point. A tool that ranks or flags studies could help a clinician decide which images to read first, and a tool that highlights the image regions behind its output, for example with Grad-CAM [@selvaraju2017gradcam], gives the reader something to check.

Tuberculosis (TB) is a particularly important target. It remains a leading infectious cause of death worldwide, and chest radiography is a recommended screening step in high-burden settings. Pneumonia, pleural effusion and lung nodules are common reasons for imaging in every health system, and since 2020 COVID-19 has added a further class that dominated the literature for several years.

## The problem with reported accuracy

The literature on automatic chest X-ray classification reports very high figures, but some of them cannot be taken at face value [@roberts2021common; @wynants2020prediction]. A model evaluated on a test set that contains other images of the same patients as the training set is partly tested on what it has memorised. A model trained on a class that comes from a single web collection can learn the collection's image style instead of the disease. A model whose labels were extracted from radiology reports by text mining inherits the errors of that process [@oakdenrayner2020exploring]. Systematic reviews of COVID-19 models found that almost none of the published systems was free of high risk of bias [@wynants2020prediction; @roberts2021common], and controlled experiments showed that networks can classify COVID-19 radiographs from regions that contain no lung at all [@maguolo2020critic] or from shortcuts that disappear at a new hospital [@degrave2021ai; @geirhos2020shortcut]. A model that does well on images from the hospitals it was trained on can lose much of its performance at a new one [@zech2018variable].

For a student project the practical lesson is that a high internal accuracy is easy to obtain and says little. The pipeline should instead close off each of these failure modes, or at least measure it, so that the number reported at the end means what it appears to mean.

## Where this project started

The project began with an earlier version of the system, called V5 in this report. That version was a DenseNet-121 trained on 10,547 images pooled from eight public sources. It recorded 82.93% accuracy on its internal test split (1,570 images) and was wrapped in a working web application. When the training and evaluation code was re-implemented from the data up and the dataset was audited, the number could not be trusted:

- 154 of the 1,570 test images (9.8%) belonged to patients who also had images in the training set, because the paediatric Kermany images had been split by image rather than by patient;
- for most sources the manifest held synthetic, one-per-image patient identifiers, so patient separation could not be checked at all;
- almost half of the Pneumonia class (1,395 images) consisted of TBX11K images labelled "sick but not TB", which are not evidence of pneumonia, and 124 Tuberculosis labels were latent TB;
- on an external cohort from a different hospital (the Montgomery County TB screening set), the model called none of the 80 normal scans Normal (specificity 0/80), although it flagged all 58 TB scans as abnormal.

Sensitivity quoted without specificity is not a result, and the V5 checkpoint was later lost. The V5 numbers therefore appear only as history, in Chapter 8 and in Appendix E, and are never compared with the new results as if they had been measured on the same data.

## The proposed system

This report describes V6, a rebuild of the data pipeline and of the model around it. The system, called LungAI in this report, is the DenseNet-121 Six-Class Chest X-Ray Classifier. It has four parts.

1. **A data pipeline** that downloads each source from its official distribution point with logged provenance, applies label rules that were fixed before any training, caps each (source, class) cell, hashes every image, merges duplicate images into patient groups and splits whole groups 70/15/15 into training, validation and test sets. A separate external cohort is excluded from every development decision.
2. **A model and training protocol**: DenseNet-121 with a small classification head, a two-phase schedule, class-balanced loss, checkpoint selection on validation macro-F1 only, trained on a laptop CPU within a time budget.
3. **An evaluation protocol** in which the test split and the external cohort are each evaluated exactly once, with confidence intervals, per-class and per-source breakdowns, calibration, and external sensitivity reported only together with specificity.
4. **A web application**: a FastAPI backend that serves the model, a heuristic input gate that rejects images that are not chest radiographs, Grad-CAM overlays and an SQLite store, and a React frontend that shows the scores and the metrics of the checkpoint actually loaded.

[[fig:pipeline]] shows how these parts fit together.

![fig_pipeline](End-to-end pipeline of the LungAI project, from source acquisition through the patient-grouped build, training, once-only evaluation and served metrics to the web application.) {#fig:pipeline}

## Objectives of the project

The objectives are stated precisely in Chapter 3. In brief, the project aims to (i) obtain a defensible multi-source six-class benchmark from official sources, (ii) train and serve a DenseNet-121 classifier on it, (iii) measure internal and external behaviour honestly, (iv) measure where source information lies in the features and how far it can be removed, and (v) document everything needed to reproduce the result or to find its faults.

## Contributions

The contributions that this report can defend are the following.

1. An auditable, leakage-controlled rebuild of a multi-source six-class chest X-ray benchmark from official sources, with label rules, caps, exclusions and checks recorded in machine-readable build reports.
2. A reproducible end-to-end pipeline: acquisition, patient-grouped splitting, training, once-only test and external evaluation, and serving of the resulting metrics by the same application that serves predictions.
3. Evidence that internal metrics overstate generalisation, from an external cohort and from a per-source analysis that looks for indirect signs of source shortcuts.
4. An audit that documents how the leakage and proxy labels of the earlier pipeline inflated its results.
5. A measurement of source information in frozen features of the benchmark: a band-by-band map of source and disease information, a linear source-erasure analysis, linear-head leave-one-source-out results, and a go/no-go decision on band suppression that came out negative. The results are diagnostic and indirect. To our knowledge, within the searches described, we are not aware of prior work that quantifies source versus disease information band by band in features of chest radiographs.

The report does not claim a new network architecture, a new state-of-the-art accuracy, a method for suppressing source information, or any clinical validity.

## Scope and limits

The classifier solves a single-label problem over six mutually exclusive classes. It sees one frontal image at a time and no clinical information. The final recipe was trained with three seeds on a laptop CPU, which gives only a rough estimate of run-to-run variation, and the preprocessing ablation is one run. Two data sources that would have strengthened the Pleural Effusion and Nodule / Mass classes (VinDr-CXR and JSRT) require credentialing or registration and were not available. The application is a research prototype. These limits are discussed in Chapter 8.

## Organisation of the report

Chapter 2 reviews the datasets, models and evaluation pitfalls relevant to chest X-ray classification. Chapter 3 defines the problem and objectives and lists the defects of the earlier pipeline that the new one must remove. Chapter 4 presents the methodology, including the source-information analyses, and the system architecture. Chapter 5 describes data collection, label rules, the split and preprocessing. Chapter 6 covers model development, training, the backend, the frontend, the input gate and testing. Chapter 7 gives the experimental setup and results. Chapter 8 covers validation, discussion and limitations. Chapter 9 concludes and lists future work. The appendices contain the API reference, the configuration, the reproduction commands, the test summary and the history of the V5 audit.
