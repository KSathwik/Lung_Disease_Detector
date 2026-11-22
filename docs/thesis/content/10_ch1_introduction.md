# Introduction

## Background and motivation

Chest radiography (chest X-ray, CXR) is the most frequently performed diagnostic imaging examination in the world. It is cheap, fast and widely available, and it is usually the first test ordered when a patient presents with cough, breathlessness, fever or chest pain. A single frontal radiograph can show a pleural effusion, a lung nodule or mass, consolidation caused by pneumonia, or the characteristic upper-lobe changes of tuberculosis. Reading it well, however, takes training. In many regions the number of radiologists is small compared with the number of examinations, and the delay between acquisition and report can be long.

Computer-aided interpretation is attractive in exactly this setting. Convolutional neural networks (CNNs) trained on large collections of labelled radiographs can assign a finding to an image in a fraction of a second [@rajpurkar2017chexnet; @irvin2019chexpert]. A dense network such as DenseNet-121 [@huang2017densely], pre-trained on natural images [@deng2009imagenet] and fine-tuned on radiographs, is a standard and efficient starting point. A tool that ranks or flags studies could help a clinician decide which images to read first, and a tool that highlights the image regions behind its output, for example with Grad-CAM [@selvaraju2017gradcam], gives the reader something to check.

Tuberculosis (TB) is a particularly important target. It remains a leading infectious cause of death worldwide, and chest radiography is a recommended screening step in high-burden settings. Pneumonia, pleural effusion and lung nodules are common reasons for imaging in every health system, and since 2020 COVID-19 has added a further class that dominated the literature for several years.

## What a usable prediction system needs

A trained network is only one part of a system that a person can use. The system has to accept an image from a user, refuse input that is not a chest radiograph, apply exactly the preprocessing the network was trained with, present the result with something the reader can check, keep a record of what it predicted and with which model, and state honestly how well that model performs.

The last point is the hardest. The literature on automatic chest X-ray classification reports very high figures, but some of them cannot be taken at face value [@roberts2021common; @wynants2020prediction]. A model tested on images of patients it was trained on is partly tested on what it has memorised, a class that comes from a single collection can be recognised by the collection's image style instead of the disease [@maguolo2020critic; @degrave2021ai; @geirhos2020shortcut], labels mined from radiology reports carry the errors of that process [@oakdenrayner2020exploring], and a model that does well at the hospitals it was trained on can lose much of its performance at a new one [@zech2018variable]. A system that predicts disease therefore has to be designed so that its data are controlled and its reported numbers mean what they appear to mean.

## The proposed system

This report describes the design and development of LungAI, the DenseNet-121 Six-Class Chest X-Ray Classifier. Given one frontal radiograph it predicts one of six classes: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass or Tuberculosis. The system has four parts.

1. **A data pipeline** that downloads each source from its official distribution point with logged provenance, applies label rules that were fixed before any training, caps each (source, class) cell, hashes every image, merges duplicate images into patient groups and splits whole groups 70/15/15 into training, validation and test sets. A separate external cohort is excluded from every development decision.
2. **A model and training protocol**: DenseNet-121 with a small classification head, a two-phase transfer-learning schedule, class-balanced loss and checkpoint selection on validation macro-F1 only, trained on a laptop CPU within a time budget.
3. **A web application**: a FastAPI backend that serves the model, an input gate that rejects images that are not chest radiographs, Grad-CAM overlays, an inconclusive outcome for low-confidence predictions and an SQLite store, and a React frontend that shows the scores, the overlay, the history, patient records and the metrics of the checkpoint actually loaded.
4. **An evaluation protocol** in which the test split and the external cohort are each evaluated exactly once, with confidence intervals, per-class and per-source breakdowns, calibration, and external sensitivity reported only together with specificity.

[[fig:pipeline]] shows how these parts fit together.

![fig_pipeline](End-to-end pipeline of the LungAI project, from source acquisition through the patient-grouped build, training, once-only evaluation and served metrics to the web application.) {#fig:pipeline}

## Development history

LungAI was developed in two versions. The first, called V5 in this report, was a DenseNet-121 trained on 10,547 images pooled from eight public sources and wrapped in a working web application. It recorded 82.93% accuracy on its internal test split. When the code was re-implemented and the dataset was audited, faults were found in the data: some patients had images in both the training and the test split, part of the Pneumonia class consisted of proxy labels, and on an external cohort the model called none of 80 normal scans Normal. The checkpoint was later lost. The present version, V6, keeps the architecture and the application and rebuilds the data pipeline, the training protocol and the evaluation so that these faults cannot recur. The V5 numbers appear only as history, in Chapter 8 and in Appendix E, and are never compared with the new results as if they had been measured on the same data.

## Objectives of the project

The objectives are stated precisely in Chapter 3. In brief, the project aims to (i) design a system that predicts one of six lung conditions from a chest radiograph, (ii) build the dataset it needs from official public sources, (iii) train a DenseNet-121 classifier on commodity hardware, (iv) develop a web application that serves the predictions with an explanation and with safeguards, (v) measure internal and external behaviour honestly, and (vi) document everything needed to reproduce the result or to find its faults.

## Contributions

The contributions that this report can defend are the following.

1. The design of an end-to-end system for six-class chest X-ray prediction, from data acquisition to a served prediction, in which training, evaluation and deployment share one configuration and one preprocessing implementation.
2. A data pipeline that builds a six-class dataset from official sources, with label rules, caps, exclusions, patient-grouped splits and duplicate checks recorded in machine-readable build reports.
3. A working web application (FastAPI and React) that checks its input, returns the class scores with a Grad-CAM overlay, marks low-confidence predictions as inconclusive, records each prediction with the identity of the model, and cannot show a prediction or a metric that the loaded model did not produce.
4. An evaluation of the classifier on an internal test split, over three training seeds, with a preprocessing ablation and on an external cohort, including the limitations that the external evaluation reveals.
5. A record of the audit of the earlier version, which explains the design decisions of the data pipeline.

Supplementary measurements of how far the source of an image can be predicted from the features, and of whether that information can be removed, are reported in Appendix G. They are diagnostic and indirect and are not part of the system.

The report does not claim a new network architecture, a new state-of-the-art accuracy or any clinical validity.

## Scope and limits

The classifier solves a single-label problem over six mutually exclusive classes. It sees one frontal image at a time and no clinical information. The final recipe was trained with three seeds on a laptop CPU, which gives only a rough estimate of run-to-run variation, and the preprocessing ablation is one run. Two data sources that would have strengthened the Pleural Effusion and Nodule / Mass classes (VinDr-CXR and JSRT) require credentialing or registration and were not available. On the one external cohort available the system does not perform well, and this is reported in full. The application is a research prototype. These limits are discussed in Chapter 8.

## Organisation of the report

Chapter 2 reviews the datasets, models and evaluation pitfalls relevant to chest X-ray classification. Chapter 3 defines the problem, the requirements and the objectives. Chapter 4 presents the methodology and the system architecture. Chapter 5 describes data collection, label rules, the split and preprocessing. Chapter 6 covers model development, training, the backend, the frontend, the input gate and testing. Chapter 7 gives the experimental setup and results. Chapter 8 covers validation, discussion and limitations. Chapter 9 concludes and lists future work. The appendices contain the API reference, the configuration, the reproduction commands, the test summary, the history of the V5 audit, the reporting checklist and the supplementary source-information analyses.
