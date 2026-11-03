# Literature Review

Statements about cited works rest on their abstracts or titles as recorded in the project's literature notes; works verified by title only are cited only for what the title says. No number from a cited paper is compared with the numbers of this project.

This chapter reviews the work that the project builds on. It is organised around the questions that decide whether a chest X-ray classifier can be trusted: which datasets exist and how their labels were made, what models have been used, how classifiers have been explained, and which evaluation mistakes have been documented in earlier studies. It ends with the gap that this project addresses.

## Public chest radiograph datasets

### Large multi-label collections

The NIH ChestX-ray14 collection, introduced as ChestX-ray8 and later extended [@wang2017chestxray8], contains 112,120 frontal radiographs from 30,805 patients. Its image labels were mined from the radiology reports with natural language processing rather than read from the images. Independent re-reading has shown that the label quality varies by finding, which matters for any project that uses these labels as ground truth [@oakdenrayner2020exploring]. Even so, ChestX-ray14 is one of the few public sources with genuine patient identifiers, which makes patient-grouped splitting possible.

CheXpert [@irvin2019chexpert] provides 224,316 radiographs from Stanford Hospital with labels extracted from reports and with uncertainty labels, and MIMIC-CXR [@johnson2019mimic] provides a large collection from a Boston hospital with free-text reports. Both are valuable for pre-training and for multi-label research, but access is controlled by data use agreements and credentialing. VinDr-CXR [@nguyen2022vindr] is notable because its labels were produced by radiologists reading the images, with several readers per training image and bounding-box annotations, but it is distributed through PhysioNet under credentialed access. The JSRT database [@shiraishi2000development] contains 247 radiographs with CT-confirmed nodule status and is available on registration. These last two sources were identified as useful for the Pleural Effusion and Nodule / Mass classes of this project, but were not accessible and were therefore not used.

### Tuberculosis collections

Two small tuberculosis sets were released by the US National Library of Medicine: the Montgomery County set (138 radiographs, 58 with TB and 80 normal) and the Shenzhen set (662 radiographs) [@jaeger2014two]. Lung segmentation masks for the Montgomery images were described by Candemir et al. [@candemir2014lung]. TBX11K [@liu2020rethinking] is a larger collection of 11,200 images with image-level categories (healthy, sick but not TB, active TB, latent TB) and bounding boxes for TB lesions. A widely circulated Kaggle archive aggregates several TB sources [@rahman2020reliable]; because it merges images from different hospitals under new file names, it is unsuitable as a source when an independent external cohort is needed.

### Pneumonia and COVID-19 collections

Kermany et al. [@kermany2018identifying] released paediatric chest radiographs labelled normal or pneumonia (bacterial or viral) from a children's hospital in Guangzhou. The images come from children aged one to five years, which makes them very different from adult radiographs. The COVID-19 image data collection of Cohen et al. [@cohen2020covid] assembles radiographs and CT images from publications and websites, with metadata including patient identifiers assigned by the collection. The COVID-19 Radiography Database [@chowdhury2020can; @rahman2021exploring] aggregates COVID-19, viral pneumonia and normal images from several upstream sources. Both COVID-19 collections are convenient and widely used, but their images were gathered from heterogeneous places, and COVID-19 images frequently come from a different set of sources than the non-COVID images they are compared with. This is the root of the shortcut problem discussed in Section 2.4.

[[tab:datasets]] summarises the sources used or considered in this project. The licence and access conditions are described in Chapter 5.

Table: Public sources used or considered for the six-class benchmark. Counts are those stated by the cited publications; the numbers used in this project are in Chapter 5. {#tab:datasets}

| Source | Population and content | Label origin | Patient identity | Role here |
|---|---|---|---|---|
| NIH ChestX-ray14 [@wang2017chestxray8] | Adults, 14 findings | Mined from reports | Genuine | Development |
| Kermany [@kermany2018identifying] | Children 1–5 years, normal / pneumonia | Expert-reviewed | Derived from file names | Development |
| Shenzhen [@jaeger2014two] | Adults, normal / TB | Clinical reading | One image per subject (per documentation) | Development |
| TBX11K [@liu2020rethinking] | Adults, healthy / sick / TB | Expert annotation | Not available | Development (run 2) |
| Cohen COVID-19 collection [@cohen2020covid] | Mixed, from publications | Source papers | Assigned by collection | Development |
| COVID-19 Radiography Database [@chowdhury2020can] | Aggregated COVID-19 images | Upstream sources | Not available | Development (run 2) |
| Montgomery [@jaeger2014two] | Adults, normal / TB | Clinical reading | Not needed (external) | External cohort only |
| VinDr-CXR [@nguyen2022vindr] | Adults, radiologist labels | Radiologist consensus | Not available | Not accessed |
| JSRT [@shiraishi2000development] | Adults, nodule / non-nodule | CT-confirmed | Not available | Not accessed |

## Deep learning for chest radiograph classification

Transfer learning from ImageNet [@deng2009imagenet] is the dominant strategy, because radiograph collections are small compared with natural image sets. DenseNet [@huang2017densely] connects each layer to all later layers within a block, which encourages feature reuse and keeps the parameter count low. CheXNet [@rajpurkar2017chexnet] fine-tuned a DenseNet-121 on ChestX-ray14 and reported radiologist-level pneumonia detection on its own test split; it popularised DenseNet-121 as the baseline for chest X-ray work. COVID-Net [@wang2020covidnet] proposed a purpose-built architecture for COVID-19 detection, and Apostolopoulos and Mpesiana [@apostolopoulos2020covid] applied transfer learning to the same task with small collections. Rahman et al. studied image enhancement for COVID-19 detection [@rahman2021exploring] and deep networks for TB detection [@rahman2020reliable].

Some of these studies report very high accuracy. The numbers are valid only for the test set on which they were measured. Whether they transfer to another hospital depends on how the data were collected and split, which is the subject of the following sections. This report does not compare its figures with those of other papers, because the test sets differ.

## Image preprocessing

Radiographs vary in contrast, exposure and scanner characteristics. Histogram equalisation redistributes grey levels to increase contrast [@pizer1987adaptive]. Contrast-limited adaptive histogram equalisation (CLAHE) applies the idea to small tiles and limits the amplification of noise by clipping the histogram [@zuiderveld1994contrast]. CLAHE applied to the luminance channel, with a mild Gaussian filter before and after, is a common recipe for chest X-ray enhancement. The chain used here follows that recipe and is described in Chapter 5. Preprocessing is not a cure for domain shift: a fixed enhancement cannot remove the differences between hospitals, and this is examined in the external evaluation.

## Shortcut learning and generalisation failures

A recurring finding is that classifiers learn to recognise where an image comes from rather than what it shows. Zech et al. [@zech2018variable] trained pneumonia classifiers on data from several hospitals and showed that performance dropped on images from a hospital not seen in training, and that the networks could identify the source hospital from the image. Geirhos et al. [@geirhos2020shortcut] described shortcut learning in general: a network adopts whatever simple cue separates the classes in the training distribution, and fails when the cue is absent.

For COVID-19 the evidence is particularly strong. DeGrave et al. [@degrave2021ai] showed that models for radiographic COVID-19 detection relied on shortcuts and generalised poorly to external data. Maguolo and Nanni [@maguolo2020critic] obtained good classification of COVID-19 images from radiographs in which the lungs had been removed, which cannot reflect disease. Two systematic reviews concluded that nearly all published COVID-19 imaging models were at high or unclear risk of bias and none was ready for clinical use [@roberts2021common; @wynants2020prediction]. Seyyed-Kalantari et al. found that chest X-ray classifiers can under-diagnose patients in under-served groups [@seyyedkalantari2021underdiagnosis], a reminder that aggregate accuracy hides disparities that this project cannot measure because the datasets lack demographic data.

## Data leakage

The most basic protection is to keep every image of a patient on one side of the split. Several public collections contain multiple images per patient, and some distribute images with new file names and without patient identifiers. Splitting such data at the image level places near-identical images of the same patient in training and test sets, which inflates accuracy. Exact and near-duplicate images across collections have the same effect. Oakden-Rayner [@oakdenrayner2020exploring] documents label and duplication problems in large public sets. Good practice, followed here, is to split by patient group, to merge groups that share duplicate images, to record which rows have no patient identity, and to state that separation is proven only where identity is known.

## Explainability and calibration

Grad-CAM [@selvaraju2017gradcam] uses the gradient of a class score with respect to the last convolutional feature maps to produce a coarse heat map of the regions that influenced the score. It is easy to attach to a CNN, and it can reveal when a model attends to text markers, borders or devices in the image. It does not show radiological reasoning, and it can look plausible when the model is wrong.

Modern networks tend to be over-confident [@guo2017calibration]. Calibration can be summarised by the expected calibration error, which compares confidence with accuracy over bins of the top-class score [@naeini2015obtaining], and reliability diagrams plot the same comparison. A softmax output should not be read as a probability of disease unless calibration has been checked and fitted on appropriate data.

## Statistical reporting

For proportions such as accuracy, recall and specificity, the Wilson score interval [@wilson1927probable] has better coverage than the normal approximation for small samples and at extreme values; it also gives a sensible interval when the observed count is 0 or all. Percentile bootstrap intervals [@efron1994introduction] are used for statistics without a simple closed form. Optimisers such as Adam [@kingma2015adam] are the usual choice for fine-tuning. Reporting guidelines for medical imaging AI, the CLAIM checklist [@mongan2020claim] and the TRIPOD+AI statement [@collins2024tripodai], ask authors to state the data sources, the split, the handling of duplicates, the performance measures with intervals and the limitations. Appendix F maps this report to those items.

## External validation, leakage taxonomy and saliency

A systematic review of radiology deep-learning studies with external validation found that most reported at least some drop in performance on external data [@yu2022external]. Tartaglione et al. showed how dataset bias and transfer-learning bias can enter small-data COVID-19 radiograph classifiers, and validated on a new hospital's data [@tartaglione2020unveiling]. Bassi and Attux, who classified COVID-19, normal and pneumonia radiographs with a DenseNet201 behind a lung-segmentation stage, reported lower accuracy on an external dataset than internally [@bassi2022covid]. Hidden stratification describes how an aggregate metric can conceal poor performance on clinically distinct subsets [@oakdenrayner2020hidden]. Kapoor and Narayanan surveyed data leakage across scientific fields that use machine learning and proposed a taxonomy of leakage types [@kapoor2023leakage]; patient overlap and duplicates between splits belong to it. Other work on domain shift in radiograph classification exists [@pooch2020can; @cohen2020limits], and the title-only entries are cited for what their titles say.

The evidence for saliency maps is weaker than their popularity suggests. Arun et al. found that the saliency methods they tested on chest radiograph data failed at least one localisation or sanity criterion [@arun2021assessing], and Ghassemi et al. argued that current explainable-AI methods do not give reliable assurance for individual decisions in health care [@ghassemi2021false]. This project therefore presents Grad-CAM as an illustrative aid and not as localisation evidence.

## Multi-class classifiers on public data

Several studies classify COVID-19 alongside other conditions on public data, among them DarkCovidNet [@ozturk2020automated], CNN ensembles [@rajaraman2020iteratively], and ensemble or comparison studies published in Biomedical Signal Processing and Control [@amin2024automated; @nayak2021application; @iqbal2023tuberculosis; @arias2024analysis]. The latter group is cited by title only. Six-class settings that include pleural effusion and nodules or masses are less common, and no verified six-class comparison on a common protocol was found, so none is claimed.

## Research gap and positioning

The studies above establish three points. Convenient public chest X-ray collections are heterogeneous. Models trained on them learn source cues unless the data are arranged to prevent it. And reported accuracies are usually not comparable because the test sets differ in construction. What is rarely published is a complete and checkable account of how a particular multi-source benchmark was assembled, split and audited, together with an evaluation that includes an untouched external cohort and an honest breakdown by source.

This project contributes such an account on a modest scale. It does not propose a new network. It adopts a standard DenseNet-121 and concentrates on the parts that earlier work often left unchecked: label rules fixed in advance, official sources with logged provenance, a patient-grouped split with duplicate merging and recorded checks, a once-only test and external evaluation, and an application that reports the metrics of the model it serves. It also documents, as a case study, how an earlier version of the same system obtained a misleading result.
