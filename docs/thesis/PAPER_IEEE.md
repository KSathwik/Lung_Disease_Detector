# A Dual-Network Computer-Aided Diagnosis System for Multi-Class Lung Disease Classification from Chest Radiographs

*(IEEE conference format — two-column on submission. Author/affiliation block to be completed before submission.)*

**Sathwik Katkam**
[Department], [Institute/University], [City, Country]
Email: sathwikkatkam@gmail.com

---

## Abstract

Chest radiography is the most accessible imaging test for thoracic disease, yet in many primary-care and rural settings there is no radiologist available to read the films promptly. The result is delayed diagnosis and, for time-critical conditions, avoidable harm. We present a computer-aided diagnosis system that classifies chest radiographs into five categories — normal, pneumonia, COVID-19, tuberculosis, and lung cancer — and embeds the classifier inside a working clinical pipeline that handles patient records, scan storage, urgency triage, and report generation. Two networks are trained in parallel: a compact convolutional network learned from scratch and a ResNet50 model fine-tuned from ImageNet. Rather than committing to one architecture in advance, the system trains both and keeps whichever scores higher on a composite of F1, accuracy, area under the ROC curve, and recall. Images are standardised through contrast-limited adaptive histogram equalization, resizing, and normalisation before inference. To remain practical on modest hardware as the dataset grows, the training routine switches automatically between in-memory arrays and a streaming data pipeline. We describe the dataset of 10,864 radiographs assembled from three public sources, the training and evaluation protocol, and the full-stack implementation. We also discuss, candidly, the limitations that remain — the absence of visual explanation, the use of raw softmax scores for urgency thresholds, and the modality mismatch in the lung-cancer class — and we set out the experiments needed to turn the prototype into a clinically credible tool.

**Index Terms** — Chest radiography, convolutional neural networks, transfer learning, ResNet, computer-aided diagnosis, medical image classification, clinical decision support.

---

## I. Introduction

Pneumonia, tuberculosis, COVID-19, and lung cancer together account for a large share of respiratory illness and death worldwide. For all of them, the chest X-ray is usually the first image a clinician orders: it is cheap, quick, and almost universally available. The bottleneck is not the imaging but the reading. Skilled radiologists are concentrated in tertiary hospitals and large cities, so a film taken at a rural clinic may wait hours or days for an expert opinion, and even then two readers may disagree.

Deep learning has narrowed this gap in the research literature. Convolutional networks now match specialist performance on several radiographic tasks [1], [5], and transfer learning from large natural-image collections lets these models reach high accuracy without enormous medical datasets [4], [16]. What is less common is a system that carries a trained classifier all the way through to something a clinic could actually use. Most published work stops at a model and a table of metrics. A tool that helps in practice has to do more: register the patient, store the scan, return a result a non-specialist can act on, flag the urgent cases, and leave an auditable record.

This paper describes such a system and is honest about how far it is from the clinic. Our contributions are: (i) a dual-network design that trains a custom CNN and a fine-tuned ResNet50 and selects between them using a single composite score, removing the guesswork of picking an architecture up front; (ii) a memory-aware training routine that scales from a laptop to large corpora without code changes; (iii) a complete decision-support workflow built around the classifier, with urgency triage and security hardening; and (iv) a critical analysis that identifies exactly which experiments would make the approach publishable as a clinical contribution rather than a software artifact.

## II. Related Work

Early demonstrations that image-based deep learning could reach expert level in medicine came from work on retinal disease and pneumonia [1] and skin cancer [11], the latter using a single convolutional network trained on a large curated set. For the chest specifically, CheXNet [5] showed radiologist-level pneumonia detection on the NIH ChestX-ray14 collection [6], and CheXpert [7] later added explicit uncertainty labels at much larger scale. The COVID-19 pandemic produced a wave of radiographic classifiers; the COVID-19 Radiography Database and the accompanying study by Chowdhury et al. [2] are among the most cited, and Rahman et al. [3] showed that segmenting the lung fields before classification markedly improved tuberculosis detection. Our datasets for the normal, pneumonia, COVID-19, and tuberculosis classes come from these sources.

On the architecture side, residual learning [4] made very deep networks trainable and remains a default backbone; VGG [13] earlier established that depth with small filters helps. CLAHE [14] is the standard contrast operation for radiographs, and data augmentation is well surveyed by Shorten and Khoshgoftaar [15].

Two threads of the literature shape our discussion of limitations. Guo et al. [9] showed that modern networks are poorly calibrated — their confidence scores do not match their accuracy — which matters here because we use confidence to decide urgency. Selvaraju et al. [8] introduced Grad-CAM, the most common way to show where a network is looking, which our current system does not yet do. Most pointedly, DeGrave et al. [12] demonstrated that COVID-19 radiograph classifiers often key on dataset artifacts rather than disease, a warning we take seriously given that our classes are drawn from different sources.

## III. Methodology

### A. Overview

The system has two parts that share trained weights but otherwise operate independently: an offline training pipeline and an online inference service. The training pipeline reads labelled images, learns the two networks, evaluates them, and writes the better one's identity to a results file. The inference service loads the trained weights once at start-up and answers requests over a REST interface.

### B. Pre-processing

Every image follows the same path before it reaches a network. We decode it, apply contrast-limited adaptive histogram equalization to bring out local structure that global contrast stretching tends to wash away, resize to 224×224 to match the ResNet input, and scale pixel intensities to the unit interval. During training we add light augmentation — small rotations and horizontal flips — to reduce overfitting, taking care to keep the labels consistent with the augmented copies.

### C. Networks

The custom network is deliberately small: four convolutional blocks with batch normalisation, ReLU activations, max-pooling, and dropout, followed by global average pooling and a dense classifier. It is trained from random initialisation with the Adam optimiser at a low learning rate.

The second network starts from a ResNet50 pre-trained on ImageNet. We first train only the new classification head with the base frozen, then unfreeze the upper residual blocks and continue at a reduced learning rate so the pre-trained features adapt gently to radiographs rather than being overwritten.

### D. Model Selection

Instead of arguing which network is better in the abstract, we let the data decide. After both are trained, each is scored on the held-out test set with the composite

> score = 0.4·F1 + 0.3·Accuracy + 0.2·AUC + 0.1·Recall.

The weighting favours F1 and recall because, in a triage setting, missing a sick patient is worse than a false alarm. The higher-scoring network becomes the primary model; the other is still run at inference and reported alongside, which gives the clinician a second opinion and gives us a cheap consistency signal.

### E. Inference and Triage

At inference the chosen model produces a class-probability vector. We report the top class as the diagnosis, its probability as confidence, and the next two classes as alternatives. A small rule base attaches typical radiographic findings and precautions to each condition, and an urgency level is assigned from the condition and its confidence: suspected lung cancer or COVID-19 above a high-confidence threshold is marked as an emergency, tuberculosis, pleural effusion, and pneumonia as urgent, and the rest as routine. Every result is stored against the patient and scan so the history can be reviewed.

### F. Scaling the Training

A practical obstacle to training on tens of thousands of images is memory. Loading everything into RAM is simplest but fails as the dataset grows. Our routine measures the expected footprint and, above a configurable threshold, switches to a streaming data pipeline that reads and augments images on demand; below it, the faster in-memory path is used. The same training script therefore runs unchanged on a laptop and on a larger machine.

## IV. System Architecture and Implementation

The deployed system is a three-tier web application. A React single-page front end lets the user register patients, upload a film, view the result, and generate a report. A FastAPI back end exposes the REST endpoints, validates uploads, runs inference, and persists results through an asynchronous SQLAlchemy layer backed by SQLite in development and ready for PostgreSQL in production. The inference engine is a singleton initialised at start-up so the networks load only once.

Security measures include an allow-list for cross-origin requests, standard response headers against framing and content-type sniffing, file-type and size checks on uploads, and validated request schemas. The whole application is packaged with a multi-stage Docker build that compiles the front end with Node and serves it from the Python image.

## V. Experimental Setup

### A. Dataset

We assembled 10,864 radiographs across five classes. Normal (1,583) and pneumonia (4,273) come from the Kermany et al. collection [1]; COVID-19 (3,616) from the COVID-19 Radiography Database [2]; tuberculosis (700) from the database of Rahman et al. [3]; and lung cancer (692) from a public chest CT collection. The data are split into training, validation, and test partitions in a 70:15:15 ratio with stratification so the class balance is preserved across splits.

### B. Metrics

We report accuracy, weighted precision, recall, and F1, and macro one-vs-rest area under the ROC curve, together with the confusion matrix and a per-class report. These are the standard quantities for multi-class medical classification and are sufficient to compare against the source studies on their respective classes.

### C. Training Configuration

Both networks are trained with Adam, a batch size of 32, early stopping on the validation loss, and learning-rate reduction on plateau. The custom network trains for up to fifty epochs; the ResNet50 follows the two-stage frozen-then-fine-tuned schedule described above.

## VI. Results

Table I reports the performance of the custom CNN, the ResNet50 transfer learning model, and the selected architecture evaluated on the completely held-out test set of 1,630 images. Confusion matrices are rendered in `docs/confusion_matrix_ResNet.png` and `docs/confusion_matrix_CNN.png`, and training convergence curves are available in `docs/training_curves.png`.

**TABLE I. CLASSIFICATION PERFORMANCE ON THE HELD-OUT TEST SET (1,630 IMAGES)**

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Composite Score |
|---|---|---|---|---|---|---|
| Custom CNN | 0.7822 | 0.7850 | 0.7822 | 0.7216 | 0.8850 | 0.7919 |
| ResNet50 | **0.9521** | **0.9518** | **0.9521** | **0.9517** | **0.9939** | **0.9603** |
| **Selected (ResNet50)** | **0.9521** | **0.9518** | **0.9521** | **0.9517** | **0.9939** | **0.9603** |

Per-class test set evaluation demonstrates strong diagnostic sensitivity: COVID-19 (F1: 0.97, Recall: 0.98), Lung Cancer (F1: 1.00, Recall: 1.00), Normal (F1: 0.91, Recall: 0.90), Pneumonia (F1: 0.96, Recall: 0.97), and Tuberculosis (F1: 0.86, Recall: 0.81).

## VII. Discussion

Two issues deserve emphasis rather than a footnote. First, the lung-cancer class is drawn from CT images while the rest are radiographs. A network could separate that class on modality cues alone and still post a high accuracy, which would be meaningless clinically — exactly the shortcut behaviour DeGrave et al. [12] documented. A fair evaluation needs either a radiographic lung-cancer set or a single-modality experiment, and we treat the current cross-modality number as provisional. Second, the urgency rules read raw softmax confidence, but such scores are known to be overconfident [9]; calibrating them with temperature scaling and reporting the expected calibration error would make the triage thresholds defensible.

The class imbalance — pneumonia outnumbers lung cancer roughly six to one — is a further concern that class weighting or focal loss should address, and we will report minority-class recall explicitly rather than letting it hide inside a weighted average.

## VIII. Conclusion and Future Work

We have built and described an end-to-end chest-radiograph diagnosis system that trains two networks, keeps the better one by an explicit composite score, and serves it inside a complete clinical workflow with triage and reporting. The engineering is sound and reproducible; the scientific claims wait on the training run and, more importantly, on three additions that would make the work clinically and academically credible: Grad-CAM explanations so a reader can see the evidence, confidence calibration so the urgency thresholds mean something, and a single-modality, leave-one-source-out study that shows the model has learned disease rather than dataset. Lung-field segmentation before classification, following the gains reported for tuberculosis [3], and privacy-preserving training across institutions are natural longer-term directions.

## References

[1] D. S. Kermany et al., "Identifying medical diagnoses and treatable diseases by image-based deep learning," *Cell*, vol. 172, no. 5, pp. 1122–1131, 2018, doi: 10.1016/j.cell.2018.02.010.
[2] M. E. H. Chowdhury et al., "Can AI help in screening viral and COVID-19 pneumonia?," *IEEE Access*, vol. 8, pp. 132665–132676, 2020, doi: 10.1109/ACCESS.2020.3010287.
[3] T. Rahman et al., "Reliable tuberculosis detection using chest X-ray with deep learning, segmentation and visualization," *IEEE Access*, vol. 8, pp. 191586–191601, 2020, doi: 10.1109/ACCESS.2020.3031384.
[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.
[5] P. Rajpurkar et al., "CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning," arXiv:1711.05225, 2017.
[6] X. Wang et al., "ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases," in *Proc. IEEE CVPR*, 2017, pp. 2097–2106, doi: 10.1109/CVPR.2017.369.
[7] J. Irvin et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in *Proc. AAAI*, vol. 33, no. 1, 2019, pp. 590–597, doi: 10.1609/aaai.v33i01.3301590.
[8] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE ICCV*, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.
[9] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017, pp. 1321–1330.
[10] G. Litjens et al., "A survey on deep learning in medical image analysis," *Medical Image Analysis*, vol. 42, pp. 60–88, 2017, doi: 10.1016/j.media.2017.07.005.
[11] A. Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, vol. 542, pp. 115–118, 2017, doi: 10.1038/nature21056.
[12] A. J. DeGrave, J. D. Janizek, and S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal," *Nature Machine Intelligence*, vol. 3, pp. 610–619, 2021, doi: 10.1038/s42256-021-00338-7.
[13] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. ICLR*, 2015, arXiv:1409.1556.
[14] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in *Graphics Gems IV*, Academic Press, 1994, pp. 474–485.
[15] C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," *Journal of Big Data*, vol. 6, no. 60, 2019, doi: 10.1186/s40537-019-0197-0.
[16] J. Deng et al., "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE CVPR*, 2009, pp. 248–255, doi: 10.1109/CVPR.2009.5206848.
