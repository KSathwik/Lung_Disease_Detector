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
