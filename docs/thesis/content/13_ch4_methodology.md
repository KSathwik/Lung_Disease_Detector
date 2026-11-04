# Proposed Methodology and System Architecture

## Overview of the methodology

The methodology has seven stages, each of which produces a file that the next stage reads. Because the stages are separate and a hash or fingerprint is recorded at each boundary, the final numbers can be audited.

1. **Acquisition.** Each source is downloaded from its official distribution point by one script. The script resumes interrupted transfers, compares the final size with the server's stated length, rejects HTML error pages saved as data, and logs URL, time, size and SHA-256 for every file.
2. **Build.** A builder applies the label rules, the exclusions and the per-cell caps, decodes and hashes every image, merges duplicates into groups, assigns whole groups to training, validation and test sets, runs the leakage checks, preprocesses each image once and caches the result. It writes a manifest and a build report.
3. **Training.** The trainer reads the manifest and the cached images, uses only the training and validation splits, and records metadata including the build and split fingerprints and the checkpoint hash.
4. **Selection.** The checkpoint is the epoch with the best validation macro-F1. When more than one run exists, the final model is chosen on a validation set that both runs treat as unseen. The rule is in the run selection protocol and was fixed before the results of run 2 were seen. Run 2 (build {{DatasetBuild}}) was selected: on the {{CmpN}} common validation images its macro-F1 was {{CmpRunTwoMacroFOne}}% against {{CmpRunOneMacroFOne}}% for run 1, and the difference exceeds the margin of the rule. This comparison is reported in Chapters 6 and 7 with a caveat about the sources the two runs share.
5. **Evaluation.** The selected checkpoint is evaluated on the test split and the external cohort once each, with the flag that marks the evaluation as final. Per-image predictions are saved.
6. **Serving of metrics.** A script copies the three evaluation reports and the run metadata into one file, refusing to combine reports of different checkpoints. The application's metrics endpoint serves that file, and the documentation is generated from it.
7. **Application.** The backend loads the checkpoint only if its SHA-256 matches the configuration, and the frontend shows what the backend returns.

The stages correspond to the boxes of [[fig:pipeline]] in Chapter 1.

## Design principles

Five principles were applied throughout.

- **One source of truth for the model definition.** The class names and order, the preprocessing parameters, the training hyper-parameters, the checkpoint file name and its SHA-256, the confidence floor and the input gate parameters are all in one JSON configuration. The trainer, the evaluator and the backend read it, and a test fails if any of them declares its own class list.
- **One implementation of preprocessing.** The builder, the trainer, the evaluator and the backend call the same functions. A test compares them with a frozen copy of the original preprocessing on several image sizes and channel layouts.
- **Rules before data.** The label rules, the caps and the split fractions are written in a proposal file that is hashed into the build report, and that was fixed before the first training run.
- **No silent fallbacks.** A missing or mismatching checkpoint produces a 503 error and nothing is stored. There is no fallback model, no mock output and no random output. A malformed model output produces a 500 error.
- **Test data are touched once.** The test split and the external cohort are evaluated by an explicit final flag, once, after selection. They are never used for thresholds, tuning or model choice.

## Model architecture

The classifier is a DenseNet-121 [@huang2017densely] followed by a small head (see [[fig:model]]). The backbone is the Keras Applications implementation without its top layer, initialised with ImageNet weights. It runs in inference mode, so its batch-normalisation statistics stay frozen even while the last layers are trained. The head consists of global average pooling, batch normalisation, a dense layer with 256 units and ReLU activation, dropout with rate 0.3, and a dense layer with six units and softmax activation. The input is a 224 × 224 × 3 image normalised with the ImageNet mean and standard deviation. The output is a vector of six non-negative numbers that sum to one.

![fig_model_architecture](DenseNet-121 backbone with the classification head and the two-phase training schedule.) {#fig:model}

The layer names of the head (gap, bn, dense_features, dropout, class_predictions) are fixed in the configuration, because the checkpoint format and the Grad-CAM code depend on them.

## Two-phase transfer learning

In phase 1 the backbone is frozen and only the head is trained, for two epochs with Adam [@kingma2015adam] at learning rate 10⁻³. This lets the randomly initialised head adapt to the pre-trained features without distorting them. In phase 2 the last 30 layers of the backbone (17 of which carry weights) are unfrozen and the network is trained with Adam at learning rate 10⁻⁵ until early stopping (three epochs without improvement of validation macro-F1), the epoch limit or the wall-clock budget. The small learning rate protects the pre-trained weights.

The loss is sparse categorical cross-entropy with per-sample weights equal to n / (k · n_c), where n is the number of training images, k = 6 the number of classes and n_c the number of training images of the class. This compensates for class imbalance without resampling.

## System architecture

The application has three tiers (see [[fig:system]]). The browser runs a React single-page application. It calls a REST API under the path /api/v1, with an API key header when a key is configured. The FastAPI backend validates uploads, applies the input gate, runs the inference engine, builds the Grad-CAM overlay and writes the result to an SQLite database. The model store holds the checkpoint and the metrics file.

![fig_system_architecture](Architecture of the LungAI application: frontend, FastAPI backend with input gate, inference engine, Grad-CAM and database, model store and metrics file.) {#fig:system}

The code is organised in two packages with a one-way dependency. The package lungai holds everything that defines and evaluates the model: configuration, preprocessing, the dataset builder logic, the model definition, the trainer, the evaluator and the metric implementations. The backend package imports lungai, and lungai never imports the backend. This keeps the scientific code independent of the web framework and lets the evaluation scripts run without a server.

### Request path

[[fig:dataflow]] shows what happens to one prediction request. The upload is checked for type (JPEG, PNG, BMP, TIFF or WebP), size (at most 10 MB) and emptiness. The stored file name is generated by the server. The input gate then decides whether the image looks like a chest radiograph. Accepted images are preprocessed with the canonical chain, passed through the network, and the output is validated for shape, finiteness and sum. Grad-CAM is computed for the predicted class. The probabilities, the model identity and the checkpoint hash are stored with the prediction, and the response is returned to the browser, which draws the six scores and the overlay.

![fig_dataflow](Data flow of one prediction request from upload to display.) {#fig:dataflow}

### Endpoints and persistence

The backend exposes endpoints for health, model identity, model metrics, prediction, stored predictions, report generation and patient records. They are listed in Appendix A. The database has four tables (patients, lung scans, predictions and reports), created and migrated by the backend at start-up. Predictions store the six probabilities, the model identifier and version, and the checkpoint SHA-256, so any stored result can be traced to the exact weights that produced it.

## Evaluation methodology

Evaluation follows the definitions below.

- **Internal test, six classes.** Accuracy with a Wilson 95% interval [@wilson1927probable] and a percentile bootstrap interval with 2,000 resamples [@efron1994introduction]; per-class precision, recall (with Wilson interval), specificity and F1; macro and support-weighted averages; one-versus-rest ROC-AUC and PR-AUC with macro averaging; the confusion matrix.
- **Calibration.** Expected calibration error with 15 equal-width bins on the top-class score [@naeini2015obtaining; @guo2017calibration] and a reliability diagram. No calibration is fitted.
- **Per source.** Accuracy and per-class recall within each source, because overall accuracy depends on the source mix and because a class with a single source can be recognised by that source.
- **External cohort, binary.** The Montgomery County set contains only normal and TB scans. A scan is called positive if the predicted class is anything other than Normal. Sensitivity and specificity are reported together with Wilson intervals, with the ROC-AUC of P(abnormal) = 1 − P(Normal) and the exact Tuberculosis recall. The prediction breakdown shows which classes the model assigned to each external scan.

The evaluator validates its inputs before computing anything: columns, a single split and checkpoint, unique image identifiers, agreement of the stored class index with the arg-max of the stored probabilities, agreement of the identifier set with the manifest, true labels that agree with the manifest, and well-formed probabilities. It never skips an image silently. The metric code is tested against hand-computed values and against scikit-learn.

## Source-information analyses on frozen features

The source-identifiability probe of Chapter 7 shows that the source of an image can be predicted from the features. Four further analyses ask where that information lies, how much of it a linear projection can remove, how a linear head behaves on a source it was not trained on, and whether removing one frequency band could be a cheap remedy. They are measurements on frozen features. No network was retrained for them, and the served model, the input gate, the backend and the frontend were not changed. They do not propose a clinical method.

### Questions and wording

A probe fitted on frozen features shows that information is available in those features. It does not show that the trained network uses it. Statements about shortcuts therefore stay at the level of indirect evidence and of consistency with a shortcut. The thresholds and procedures described in this section were fixed in advance and recorded in the repository (see the last subsection), and every number in Chapter 7 is read from the generated result files.

### Data and probe samples

All analyses use the cached 224 × 224 images of build {{DatasetBuild}} and the input normalisation of the trained model. Each cache image has already passed the Gaussian pre-blur, CLAHE and the Gaussian low-pass filter at its native resolution and was then resized to 224 × 224. Sources differ in native resolution and field of view, so the band-wise results partly reflect how different pixel grids were resampled and not only properties of the scanners. For this reason the results are described as a map of the cached images. A second cache, built without the three filters (the build of the ablation run), is used as a sensitivity check.

The spectral and band-stop analyses use one fixed sample per split, drawn with sampling seed 0 and the same for every variant, extractor and cache, so that comparisons are paired. A (source, class) cell contributes at most 150 training, 100 validation and 100 test images. This gives {{SpecMainTestNTrain}} training images and {{SpecMainTestN}} test images. Because class and source are confounded (Pleural Effusion and Nodule / Mass occur only in NIH, Pneumonia mostly in Kermany), the primary disease score is computed within each source: the macro-F1 over the classes that a source contains with at least 20 images in the scored sample, averaged over the {{SpecMainTestNWithinSources}} qualifying sources. The six-class macro-F1 over the whole sample is secondary, because it can be raised by source cues. A further score is the accuracy of the source probe among Normal images only, over the four sources that contain enough of them.

### Spectral map of source and disease information

Each cached image is split into six octave bands of radial spatial frequency, labelled A to F ([[tab:bands]]). The band masks have raised-cosine transitions of a quarter octave on either side of each edge, and the six masks add up to one at every frequency. Multi-frequency (Laplacian-pyramid) standardisation has been used to harmonise multi-institutional chest radiographs [@kim2023homogenization]; here the bands serve for measurement only. The mean brightness (the zero-frequency term) is kept in every variant, the result is clipped and rounded to the cache data type, and band-pass images are not rescaled in contrast.

Table: Octave bands of the spectral analysis on the 224 × 224 grid. {#tab:bands}

| Band | Radial frequency (cycles per pixel) | Period (pixels) |
|---|---|---|
| A | 0 to 1/64 | 64 and longer |
| B | 1/64 to 1/32 | 64 to 32 |
| C | 1/32 to 1/16 | 32 to 16 |
| D | 1/16 to 1/8 | 16 to 8 |
| E | 1/8 to 1/4 | 8 to 4 |
| F | 1/4 and above | 4 to 2 |

Thirteen variants of every image are analysed: the full image, six band-pass variants (one band plus the mean brightness) and six band-stop variants (everything except one band). Three extractors or caches are used. The primary extractor is the ImageNet-initialised DenseNet-121 without its top, with global average pooling (1,024 features). The secondary extractor is the backbone of the final model, which was trained on all six sources and therefore measures what the served network can represent, not an untouched model; the band-pass variants only were run for it. The sensitivity check applies the ImageNet extractor to the no-filter cache, again for the band-pass variants only.

For every variant, two multinomial logistic-regression probes with standardised inputs (class-balanced weights, regularisation constant 1) are fitted on the training sample and scored on the validation or test sample: one predicts the source, the other the class. The reported quantities are the source balanced accuracy (chance 1/6, that is 16.7%), the within-source disease macro-F1, the six-class macro-F1 and the within-Normal source accuracy. The band-stop differences are ΔS, the source balanced accuracy of the full image minus that of the band-stop image, and ΔD, the same difference for the within-source disease macro-F1; a positive value means that information was lost by removing the band. Intervals are percentile-bootstrap 95% intervals with 1,000 resamples, the unit being the scored image, with the same resampled indices for all variants. Each probe is fitted once, so the intervals describe variation of the scored sample and not training variance.

### Linear source erasure (INLP and LEACE)

The second analysis removes linearly decodable source information from the features of the full image and measures what remains. It uses the iterative nullspace projection (INLP) [@ravfogel2020null] and the closed-form LEACE estimator [@belrose2023leace], which are published methods; here they serve only as a measurement on frozen features. The features of all training, validation, test and Montgomery images are standardised with training statistics. At each iteration a logistic-regression classifier is fitted to predict the source from the current training features, the row space of its coefficients is removed by projection, and the process repeats for at most 20 iterations. The rule fixed in advance stops earlier if a freshly fitted source classifier reaches a validation balanced accuracy of at most 21.7% (chance plus five points). A disease head (multinomial logistic regression, regularisation chosen once on the validation split from three values) is retrained on the projected training features at every iteration, and the source accuracy is re-measured with a fresh classifier. LEACE, fitted on the training features with the six source labels, is scored as a one-row check. The disease scores use the complete validation and test splits ({{InlpImgNTest}} test images). The change relative to the unprojected features is reported as a paired difference with a bootstrap interval over test images. After the test table was written, the heads on the unprojected features and at the last iteration were scored once on the Montgomery cohort (Normal against Tuberculosis, sensitivity and specificity together, Wilson intervals). Both extractors are used, the ImageNet one as primary and the trained backbone as secondary.

### Linear-head leave-one-source-out

The third analysis, called here linear-head leave-one-source-out on frozen features, trains a multinomial logistic-regression head on the training split of five sources and scores it on the test split of the sixth, following the leave-one-domain-out training of Yao et al. [@yao2019strong] for chest X-ray datasets. The regularisation constant is chosen on the validation images of the five training sources. A held-out source is included if it has at least two evaluated classes, where an evaluated class has at least 20 test images in the held-out source and at least 50 training images in the other sources. Five sources qualify: Cohen (COVID-19, Pneumonia), Kermany (Normal, Pneumonia), NIH (Normal, Pneumonia), Shenzhen (Normal, Tuberculosis) and TBX11K (Normal, Tuberculosis). The COVID-19 Radiography Database has a single class and is excluded; the NIH classes Pleural Effusion and Nodule / Mass cannot be tested because no other source contains them.

The primary metric is the macro-F1 over the evaluated classes, where a prediction of any other class counts as an error. The in-distribution reference uses the same recipe, trained on the training splits of all six sources and scored on the same test images. The gap is the reference minus the held-out value, in percentage points. A binary view, abnormal against normal, uses 1 − P(Normal) as score and reports the ROC-AUC for sources with at least 20 normal and 20 abnormal test images. Intervals are bootstrap 95% intervals over held-out test images. The head trained on all six sources is also scored once on Montgomery. ImageNet features give the primary matrix. The matrix from the trained backbone is secondary, is described only as frozen-feature results from a backbone that saw all six sources, and is never called held-out or external. Every number carries the patient-identity caveat of Chapter 5: patient separation is established only for rows with known identity, 4,130 TBX11K and COVID-19 Radiography rows are grouped per image, and Kermany identities are derived from file names.

### Band-stop screening and the decision rule for suppression

A fourth step examines whether removing a band could suppress source information at little cost to disease information. A band-suppression retraining of the network (analysis C2) was made conditional on a go/no-go rule written down before any spectral result existed. The rule uses validation data, the ImageNet extractor and the paired probe sample, and it calls for a GO only if all four criteria hold:

1. some band has ΔS of at least 5 points with the lower bound of its interval above zero, and ΔD of at most 2 points with the upper bound of its interval below 3 points;
2. a matched-energy random band-stop control, which removes the same share of spectral energy, lowers source accuracy by less than the selected band does;
3. a linear head on ImageNet features of band-stopped images (the exploratory screening step described next) lowers source balanced accuracy by at least 5 points and loses at most 2 points of within-source disease macro-F1 on validation;
4. the earlier deliverables are finished and the student approves a night of CPU time.

The screening step uses the candidate band with the largest ΔS (or, if no band qualifies under criterion 1, the band with the largest ΔS and a NO-GO outcome), a larger training sample of up to 400 images per cell ({{BsLiteValNTrain}} images), and the same two probes as above on the full image and on the band-stop image. Its validation table decides, and its test table is written once and labelled exploratory. If no band satisfies criterion 1, the random control is not run. Otherwise the outcome is NO-GO, the screening is reported as exploratory and the retraining is listed as future work.

### Rules fixed in advance, approval and access log

All settings above (bands, variants, samples, minimum cell sizes, stopping rule, inclusion rules, intervals and the go/no-go thresholds) are written in a protocol file of the repository (docs/publication/NOVELTY_PROTOCOL.md). It was drafted before any result of these analyses existed, was approved by the student in chat on 4 November 2026, and is recorded in the repository by a commit. Any later change would have been entered in its change log, with date and reason, before the affected result was computed. A few implementation details that the protocol left open were logged before the first image-level result. The words "fixed in advance and recorded in the repository" are used deliberately: the protocol was not registered with any external body.

Selection steps (the INLP stopping check, the regularisation constants and the candidate band) use the validation split only. The test split is scored once per analysis, in a logged order, and Montgomery is scored in exactly two analyses ([[tab:accesslog]]). Montgomery is not untouched: it had already been evaluated for runs 1 to 5, and run 2 was started after the external results of run 1 were known. It was never used for training or for selecting the served model. No p-values are computed. With 13 variants, six bands, five folds and up to 20 iterations, each with several intervals, about one interval in twenty can miss the true value by chance. A statement about a single band, fold or iteration is therefore made only if its interval excludes the reference value and the same pattern appears in neighbouring bands or iterations or in the secondary extractor. Otherwise the result is reported as not distinguishable.

Table: Order in which the source-information analyses read the test split and the Montgomery cohort. All analyses ran on 4 November 2026; dates, code commits and output files are in the access log of the protocol. {#tab:accesslog}

| Order | Analysis | Splits scored | Montgomery |
|---|---|---|---|
| 1 | Spectral map, ImageNet features | Validation, then test | Not used |
| 2 | Spectral map, trained backbone | Test | Not used |
| 3 | Spectral map, no-filter cache | Test | Not used |
| 4 | INLP and LEACE | Validation, then test | Once |
| 5 | Leave-one-source-out | Validation (constant), then test | Once |
| 6 | Band-stop screening | Validation (decision), then test (exploratory) | Not used |
| 7 | Go/no-go decision | Validation numbers only | Not used |

## Technology stack

[[tab:stack]] lists the tools used.

Table: Software stack of the project. {#tab:stack}

| Component | Technology |
|---|---|
| Language and runtime | Python 3.12 (3.11 or 3.12 required), Node 18 |
| Deep learning | TensorFlow 2.16.1 with Keras; DenseNet-121 from Keras Applications |
| Image processing | OpenCV 4.10, NumPy 1.26 |
| Metrics | scikit-learn, own implementations of Wilson and bootstrap intervals and ECE |
| Backend | FastAPI, SQLAlchemy, SQLite, Uvicorn |
| Frontend | React 18 (Create React App), Jest for service tests |
| Quality | pytest (backend and package), Jest, production build with warnings as errors |
| Reproducibility | Pinned requirements, SHA-256 fingerprints, run metadata, build reports |
| Hardware | Laptop CPU with 8 GB RAM (no GPU support in the native-Windows TensorFlow build used) |
