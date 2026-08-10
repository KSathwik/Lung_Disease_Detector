# Dual-Network Classification of Chest Radiographs for Lung Disease Triage: A Workflow-Integrated Approach

*(Springer LNCS format — to be typeset with the llncs class on submission. Author block and ORCID to be added before submission.)*

**Sathwik Katkam**

[Department], [Institute/University], [City], [Country]
`sathwikkatkam@gmail.com`

---

**Abstract.** The chest radiograph is the first-line image for most thoracic complaints, but expert interpretation is scarce outside large hospitals, and the delay between exposure and report can be clinically costly. We present a computer-aided diagnosis system that sorts chest radiographs into five categories — normal, pneumonia, COVID-19, tuberculosis, and lung cancer — and, unusually, ships the classifier inside a working clinical workflow with patient records, scan storage, urgency triage, and report generation. The method trains two networks together, a small convolutional network from scratch and a ResNet50 fine-tuned from ImageNet, and retains whichever attains the higher value of a composite of F1, accuracy, ROC area, and recall, weighted toward F1 and recall to suit a triage context. Radiographs are standardised with contrast-limited adaptive histogram equalization, resizing, and normalisation, and the training routine scales from a laptop to large corpora by switching automatically between in-memory and streaming data paths. We describe a dataset of 10,864 radiographs from three public sources, the evaluation protocol, and the full-stack implementation, and we set out plainly the three studies — visual explanation, confidence calibration, and single-modality generalisation — required before the system can be considered clinically sound.

**Keywords:** Chest radiography · Convolutional neural networks · Transfer learning · ResNet · Computer-aided diagnosis · Clinical decision support.

---

## 1 Introduction

Respiratory diseases such as pneumonia, tuberculosis, COVID-19, and lung cancer are a major cause of illness and death, and for each of them the chest X-ray is the usual entry point into care. Imaging itself is widely available; the difficulty is reading the films. Radiological expertise clusters in cities and tertiary centres, so a radiograph taken at a small clinic often waits for a report, and inter-reader disagreement is common even when an expert is present.

Deep learning has shown, repeatedly, that convolutional networks can read radiographs at a level comparable to specialists [1,5], and transfer learning makes this feasible without very large labelled sets [4,16]. Yet most studies deliver a model and a metrics table and stop there. Turning that model into something a clinic can use requires registering patients, storing scans, presenting results a non-specialist can act on, flagging urgent cases, and keeping an auditable trail.

We describe a system that does all of this, and we are deliberate about its limits. The contributions are fourfold. First, a dual-network design trains a custom CNN and a fine-tuned ResNet50 and selects between them by a single composite score, so the architecture choice is settled by evidence rather than assumption. Second, a memory-aware training routine scales without code changes. Third, the classifier is wrapped in a complete decision-support workflow with triage and security hardening. Fourth, we give a critical account of what is missing and which experiments would close the gap.

## 2 Related Work

The case that image-based deep learning can reach expert level in medicine was made early for retinal disease and pneumonia [1] and for skin cancer [11]. For the chest, CheXNet reported radiologist-level pneumonia detection on ChestX-ray14 [5,6], and CheXpert added uncertainty labels at large scale [7]. During the pandemic, the COVID-19 Radiography Database and the study of Chowdhury et al. [2] became standard references, and Rahman et al. [3] showed that segmenting the lungs before classification improves tuberculosis detection. The normal, pneumonia, COVID-19, and tuberculosis data used here originate in these works.

Architecturally, residual learning [4] underpins our ResNet50 backbone, building on earlier evidence that depth helps [13]. CLAHE [14] is the conventional contrast operation for radiographs, and augmentation is surveyed in [15]. Two further results frame our discussion: Guo et al. [9] showed that neural networks are poorly calibrated, which bears on our confidence-based triage, and DeGrave et al. [12] showed that radiographic COVID-19 classifiers frequently exploit dataset artifacts rather than pathology — a caution we apply to our own multi-source data. Grad-CAM [8] is the explanation technique our system does not yet include.

## 3 Method

### 3.1 Pipeline

The system separates an offline training pipeline from an online inference service that shares its weights. Training reads labelled images, learns both networks, evaluates them, and records the better one; inference loads the weights once and answers REST requests.

### 3.2 Pre-processing

Each image is decoded, enhanced with contrast-limited adaptive histogram equalization, resized to 224×224, and normalised to the unit interval. Training adds small rotations and horizontal flips, with labels duplicated to match the augmented images.

### 3.3 Networks and Selection

The custom network comprises four convolutional blocks with batch normalisation, ReLU, pooling, and dropout, then global average pooling and a dense classifier, trained from scratch with Adam. The second network fine-tunes an ImageNet ResNet50, first training the new head with the base frozen and then unfreezing the upper blocks at a lower learning rate. After training, each network is scored on the test set as

*score = 0.4 · F1 + 0.3 · Accuracy + 0.2 · AUC + 0.1 · Recall,*

with the weighting tilted toward F1 and recall because a missed diagnosis is the costlier error in triage. The higher-scoring network is primary; the other is still evaluated and reported as a second reading.

### 3.4 Triage and Persistence

The primary model yields a probability vector. The system reports the top class as the diagnosis, its probability as confidence, and the next two as alternatives, attaches rule-based findings and precautions, and assigns urgency from condition and confidence — emergency for high-confidence lung cancer or COVID-19, urgent for tuberculosis, pleural effusion, and pneumonia, routine otherwise. Results are stored against patient and scan.

### 3.5 Scaling

To handle large corpora on modest hardware, the training routine estimates the memory footprint and, above a configurable threshold, streams and augments images on demand; below it, it uses the faster in-memory path. The same script runs unchanged across hardware.

## 4 System Implementation

The application is a three-tier web system: a React front end for patient management, upload, results, and reporting; a FastAPI back end that validates uploads, runs inference, and persists data through asynchronous SQLAlchemy over SQLite (PostgreSQL-ready); and a singleton inference engine initialised at start-up. Security covers cross-origin allow-listing, protective response headers, upload type and size checks, and validated schemas. A multi-stage Docker build compiles the front end and serves it from the Python image.

## 5 Experimental Setup

We use 10,864 radiographs: normal (1,583) and pneumonia (4,273) from [1], COVID-19 (3,616) from [2], tuberculosis (700) from [3], and lung cancer (692) from a public chest CT collection. The data are split 70:15:15 with stratification. We report accuracy, weighted precision, recall and F1, macro one-vs-rest ROC area, the confusion matrix, and a per-class report. Both networks train with Adam, batch size 32, early stopping, and learning-rate reduction on plateau; ResNet50 follows the frozen-then-fine-tuned schedule.

## 6 Results

*Author note: numerical results are pending the training run; the weights had not been produced at the time of writing. The protocol and tables are fixed so that completion is mechanical.*

Table 1 will report test-set performance for the custom CNN, the ResNet50, and the selected model, with confusion matrices and learning curves as figures. We anticipate that the fine-tuned ResNet50 will lead on the larger classes and that the smaller tuberculosis and lung-cancer classes will be the more telling test of generalisation.

**Table 1.** Classification performance on the test set (to be completed).

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Custom CNN | – | – | – | – | – |
| ResNet50 | – | – | – | – | – |
| Selected | – | – | – | – | – |

Per-class results will be compared with [1] for pneumonia, [2] for COVID-19, and [3] for tuberculosis, allowing for the fact that those studies address fewer classes than ours.

## 7 Discussion

Two points carry the most weight. The lung-cancer class is CT-derived while the others are radiographs, so a model might separate it on modality alone and still report high accuracy — the shortcut failure described in [12]. We therefore treat the cross-modality figure as provisional and call for a single-modality evaluation. The urgency rules also rely on raw softmax confidence, which is typically overconfident [9]; temperature scaling and a reported calibration error would make the thresholds trustworthy. Finally, the six-to-one imbalance between pneumonia and lung cancer warrants class weighting or focal loss, with minority-class recall reported explicitly.

## 8 Conclusion

We have presented a chest-radiograph diagnosis system that trains two networks, selects the better by an explicit composite score, and serves it inside a complete clinical workflow with triage and reporting. The implementation is reproducible; the scientific contribution depends on the pending training run and on three additions — Grad-CAM explanation, confidence calibration, and a single-modality leave-one-source-out study. Lung-field segmentation before classification, following the tuberculosis results of [3], and federated training across institutions are promising next steps.

## References

1. Kermany, D.S., et al.: Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell 172(5), 1122–1131 (2018). https://doi.org/10.1016/j.cell.2018.02.010
2. Chowdhury, M.E.H., et al.: Can AI help in screening viral and COVID-19 pneumonia? IEEE Access 8, 132665–132676 (2020). https://doi.org/10.1109/ACCESS.2020.3010287
3. Rahman, T., et al.: Reliable tuberculosis detection using chest X-ray with deep learning, segmentation and visualization. IEEE Access 8, 191586–191601 (2020). https://doi.org/10.1109/ACCESS.2020.3031384
4. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: IEEE CVPR, pp. 770–778 (2016). https://doi.org/10.1109/CVPR.2016.90
5. Rajpurkar, P., et al.: CheXNet: radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv:1711.05225 (2017)
6. Wang, X., et al.: ChestX-ray8: hospital-scale chest X-ray database and benchmarks. In: IEEE CVPR, pp. 2097–2106 (2017). https://doi.org/10.1109/CVPR.2017.369
7. Irvin, J., et al.: CheXpert: a large chest radiograph dataset with uncertainty labels and expert comparison. In: AAAI, vol. 33(1), pp. 590–597 (2019). https://doi.org/10.1609/aaai.v33i01.3301590
8. Selvaraju, R.R., et al.: Grad-CAM: visual explanations from deep networks via gradient-based localization. In: IEEE ICCV, pp. 618–626 (2017). https://doi.org/10.1109/ICCV.2017.74
9. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: ICML, pp. 1321–1330 (2017)
10. Litjens, G., et al.: A survey on deep learning in medical image analysis. Medical Image Analysis 42, 60–88 (2017). https://doi.org/10.1016/j.media.2017.07.005
11. Esteva, A., et al.: Dermatologist-level classification of skin cancer with deep neural networks. Nature 542, 115–118 (2017). https://doi.org/10.1038/nature21056
12. DeGrave, A.J., Janizek, J.D., Lee, S.-I.: AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence 3, 610–619 (2021). https://doi.org/10.1038/s42256-021-00338-7
13. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. In: ICLR (2015). arXiv:1409.1556
14. Zuiderveld, K.: Contrast limited adaptive histogram equalization. In: Graphics Gems IV, pp. 474–485. Academic Press (1994)
15. Shorten, C., Khoshgoftaar, T.M.: A survey on image data augmentation for deep learning. Journal of Big Data 6(60) (2019). https://doi.org/10.1186/s40537-019-0197-0
16. Deng, J., et al.: ImageNet: a large-scale hierarchical image database. In: IEEE CVPR, pp. 248–255 (2009). https://doi.org/10.1109/CVPR.2009.5206848
