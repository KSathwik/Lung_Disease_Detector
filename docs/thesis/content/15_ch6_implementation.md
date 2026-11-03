# Model Development and Implementation

## Code organisation

The repository separates the scientific code from the application. [[tab:modules]] lists the main modules.

Table: Main modules of the implementation. {#tab:modules}

| Module | Responsibility |
|---|---|
| lungai/configs/lungai_densenet121.json | Model identity, class order, preprocessing, training, evaluation and gate parameters, checkpoint name and SHA-256 |
| lungai/preprocessing.py | The single preprocessing chain used by builder, trainer, evaluator and backend |
| lungai/data/v6.py | Patient grouping, duplicate merging, split assignment and leakage checks |
| scripts/data/acquire.py | Downloads with resume, size check, HTML rejection and provenance log |
| scripts/data/build_v6_manifest.py | Applies label rules, caps and exclusions, hashes and caches images, writes the manifest and build report |
| lungai/models/densenet.py | Builds the model; loads a checkpoint only if its SHA-256 matches |
| lungai/training/train_v6.py | Two-phase trainer with per-epoch validation, checkpointing, resume and run metadata |
| lungai/evaluation/ | Metric implementations, evaluator, served-metrics writer and documentation renderer |
| backend/ | FastAPI application: routes, inference engine, input gate, database |
| frontend/ | React application: Analyze, History, Model Performance and Patients pages |
| scripts/pipeline/run_overnight.py | Chains download wait, extraction, build, training and evaluation, with stage state for restart |

## Model construction and checkpoint integrity

The model is built from the configuration in one function. Keras Applications supplies DenseNet-121 with ImageNet weights, which Keras downloads on first use. The head layers are given the fixed names listed in the configuration. A checkpoint is loaded only if its SHA-256 equals the value in the configuration. A mismatch produces a 503 response from the backend with both hashes (shortened) in the message, unless an explicit environment switch allows an unverified checkpoint, in which case every response states that the checkpoint is not verified and the frontend shows a warning. A checkpoint whose input is not 224 × 224 × 3 or whose output is not six-way is refused.

## Training protocol

The trainer implements the protocol of Chapter 4. Its steps are:

1. read the manifest and the cached 224 × 224 images made by the builder; stop if any cached image is missing;
2. compute balanced class weights from the training split;
3. phase 1: two epochs with the backbone frozen, Adam at 10⁻³;
4. phase 2: the last 30 backbone layers trainable, Adam at 10⁻⁵, batch size 32, until early stopping with patience three, the epoch limit or the wall-clock budget;
5. after every epoch, a validation pass; the model is saved if validation macro-F1 is the best so far (ties broken by validation macro ROC-AUC); the last weights and a state file are always saved;
6. write run metadata at the end.

The run metadata records the code commit, the build and manifest hashes, the split fingerprint, library versions, the per-epoch history, the selected epoch, the class weights and the checkpoint SHA-256. The trainer never reads the test split or the external cohort. Re-running the same command resumes after the last completed epoch; the weights are restored but the optimiser moments restart, and this is recorded in the metadata.

Python, NumPy and TensorFlow are seeded with 42. Augmentation draws come from one seeded generator and are made sequentially. Floating-point accumulation across threads is not deterministic, so a repeated run is expected to differ slightly.

### The CPU and memory problem

The training machine has 8 GB of memory and no GPU supported by the native-Windows TensorFlow 2.16 build. The first two overnight attempts stalled at the first validation pass. Calling the model eagerly on the CPU took minutes per batch and exhausted memory, because each call re-traced and re-allocated the full DenseNet graph. The fix was to run inference in the trainer and in the evaluator as one compiled TensorFlow graph with a fixed batch shape. With that change an epoch of the first build (8,054 training images, run 1) took about 13 to 14 minutes including validation. For the final build (run 2, 10,957 training images) the recorded epochs took about 1,190 to 1,560 seconds (roughly 20 to 26 minutes), and the full 16-epoch schedule took about 5.8 hours. The compiled graph does not change the computed values, and the evaluation reports are computed from the saved per-image outputs of the same graph. The measured time per epoch is the practical limit on this project: it is the reason for the single seed, for the cap on epochs, and for the time-boxed stop.

### Two runs

Two runs were trained with the same protocol on two builds. Run 1 used build 001. Run 2 used build 002, which adds TBX11K and the COVID-19 Radiography Database. Both used seed 42. The final model was chosen after run 2 on a common validation set that neither run used for selection. Run 2 was selected by the rule fixed before its results were seen. On the {{CmpN}} validation images that neither build used for training or testing, run 2 reached macro-F1 {{CmpRunTwoMacroFOne}}% against {{CmpRunOneMacroFOne}}% for run 1, accuracy {{CmpRunTwoAcc}}% against {{CmpRunOneAcc}}% and macro ROC-AUC {{CmpRunTwoMacroAUC}} against {{CmpRunOneMacroAUC}}. Restricted to the {{CmpRunOneSharedN}} of those images from the four sources both runs trained on, accuracy is almost equal ({{CmpRunTwoSharedAcc}}% against {{CmpRunOneSharedAcc}}%) and the macro-F1 gap is smaller ({{CmpRunTwoSharedMacroFOne}}% against {{CmpRunOneSharedMacroFOne}}%), so much of the advantage of run 2 is on TBX11K and COVID-19 Radiography images, which are in-distribution for run 2 only. Run 2 was started after the external results of run 1 were known; only the selection rule was fixed before run 2's results. The full comparison is in Chapter 7. Because each run is a single seed, a small difference between runs is not evidence that one is better.

## Backend implementation

The backend is a FastAPI application with a start-up routine that loads the model, the input gate and the database. Its main behaviours are as follows.

- **Authentication.** If an API key is configured, every route except health requires the header X-API-Key. If no key is configured only loopback clients are served. There are no user accounts.
- **Upload handling.** The content type must be JPEG, PNG, BMP, TIFF or WebP, the size at most 10 MB (read with a cap), and the file non-empty. The stored name is generated by the server, which blocks path traversal.
- **Inference engine.** One model instance per process, held by a singleton. The loading rules are summarised in [[tab:loading]]. After the network runs, the output is checked for shape (6), finiteness, non-negativity and a sum of one within a tolerance; otherwise the response is a 500 and nothing is stored.
- **Confidence floor.** If the top score is below 0.5 the result is marked inconclusive. The threshold is a display rule that has not been validated clinically, because the softmax scores are not calibrated.
- **Grad-CAM.** The engine builds a sub-model from the final convolutional feature map of the backbone to the backbone output, computes the gradient of the predicted class score with respect to that map, pools it over the spatial dimensions, weights the maps, applies ReLU, normalises and overlays the heat map on the original image [@selvaraju2017gradcam]. Its caption states that it shows influence on the score and not radiological reasoning.
- **Persistence.** The probabilities, model identity and checkpoint hash are stored with each prediction in SQLite.
- **Metrics endpoint.** The model-metrics endpoint serves the generated metrics file and flags a mismatch if the file belongs to a different checkpoint. The limitation notice returned with every prediction is built at start-up from the same file.

Table: Behaviour of the inference engine for the states of the checkpoint. {#tab:loading}

| Situation | Engine state | Health status | Predict endpoint |
|---|---|---|---|
| File missing | Not ready | Degraded, "not found" | 503, nothing stored |
| SHA-256 differs | Not ready | Degraded, both hashes named | 503 |
| SHA-256 differs and override set | Ready, checkpoint_verified false | Healthy | 200 with an unverified flag |
| Corrupt or incompatible file | Not ready | Degraded | 503 |
| Canonical file | Ready, checkpoint_verified true | Healthy | 200 |

## Frontend implementation

The frontend is a React 18 application with four pages. The Analyze page uploads an image and shows the predicted class (or "Inconclusive"), the top score, six score bars and the Grad-CAM overlay, and a warning when the checkpoint is not verified. The History page lists stored predictions with their six scores, the model and the checkpoint hash. The Model Performance page shows the generated metrics with the evidence level, the selection epoch and per-class ROC-AUC; external sensitivity is always shown with specificity, and the V5 record is shown separately as history. The Patients page manages patient records.

The code enforces several rules. Class names and order come from the API response, so the frontend has no class list of its own. A malformed response (a missing or out-of-range score, or a predicted class that is not in the order) raises an error that is displayed as such, and a missing value is never drawn as zero. Scores are labelled "uncalibrated softmax" and not as probability of disease. Errors are never replaced with mock results, and the errors module maps the 401, 413, 422, 503 and network failures to specific messages.

[[fig:uiresult]] and [[fig:uigradcam]] show the Analyze page of the running application with the final model, for a test-split Shenzhen radiograph whose true label is Tuberculosis. They were captured with `scripts/capture_screenshots.py`, which drives the installed Microsoft Edge against the backend serving the production build. A single correctly classified example illustrates the interface only; it is not evidence of accuracy. Full-page captures of the upload, Model Performance and History pages are in `docs/figures/screenshots/`.

![screenshots/07_result_card](Analyze page: predicted class with its top-class score, the model identity and the six softmax scores for a test-split radiograph (true label Tuberculosis).) {#fig:uiresult}

![screenshots/08_gradcam_card](Analyze page: Grad-CAM overlay for the predicted class, with the disclaimer shown by the application.) {#fig:uigradcam}

## The chest X-ray input gate

### Purpose and design

A classifier that always returns one of six diseases will happily classify a photograph of a car. The input gate tries to reject images that are not chest radiographs before inference. It has two stages. Stage 1 applies simple heuristics on image size, aspect ratio, contrast, colour content and brightness. Stage 2 scores the anatomical plausibility of the image from its intensity profile. A trained gate model was planned but is absent, so stage 2 runs in a heuristic fallback mode, reported by the health endpoint as heuristic_fallback, and an image passes if its score reaches the heuristic threshold. A rejected upload receives HTTP 422 with the error code INVALID_CXR and no prediction. The reported model metrics are computed without the gate.

### The original threshold rejected most real radiographs

The threshold was initially 0.83. It was measured on the radiographs of the V6 test split of build 001 and on the Montgomery set, with the script that applies the gate to each image. At 0.83 the gate rejected {{GateRejTestK}} of {{GateRejTestN}} test radiographs ({{GateRejTestPct}}%) and {{GateRejExtK}} of {{GateRejExtN}} Montgomery radiographs ({{GateRejExtPct}}%). The rejection rate by test source ranged from {{GateRejSrc_shenzhen}}% for Shenzhen to {{GateRejSrc_cohen_covid}}% for the Cohen collection. In practice the Analyze page would have refused most genuine X-rays.

### Recalibration on validation radiographs

The threshold was recalibrated on validation radiographs only, using the validation split of build 002 ({{GateN}} images), so no test image influenced it. The rule is: take the larger of (a) the quantile of the score distribution that passes {{GateTarget}}% of validation radiographs and (b) the highest score among the available non-radiograph images that pass stage 1, plus 0.01. The first value is {{GateQuantile}}. The non-radiograph images are seven synthetic images (a colour and a grayscale car image, a dog, a selfie, a landscape, a screenshot and a CT slice); the highest stage-1-passing negative, the CT slice, scored {{GateHighNeg}}. The second value therefore dominates and sets the threshold to {{GateThr}}. Stage 1 rejected {{GateStage1Rej}} of the validation radiographs. [[tab:gatethr]] shows how the validation pass rate falls with the threshold, and [[tab:gatesrc]] shows the pass rate by source at the new threshold.

Table: Fraction of validation radiographs (build 002) passing the gate at selected thresholds. {#tab:gatethr}

| Threshold | Validation pass rate (%) |
|---|---|
| 0.50 | {{GatePass_05}} |
| 0.60 | {{GatePass_06}} |
| 0.70 | {{GatePass_07}} |
| **{{GateThr}} (adopted)** | **{{GatePassNew}}** |
| 0.78 | {{GatePass_078}} |
| 0.80 | {{GatePass_08}} |
| 0.83 (original) | {{GatePassOld}} |

Table: Gate pass rate by source on the validation split at the adopted threshold. {#tab:gatesrc}

| Source | Validation images | Pass rate (%) |
|---|---|---|
| Cohen COVID-19 collection | {{GateSrcN_cohen_covid}} | {{GateSrc_cohen_covid}} |
| COVID-19 Radiography Database | {{GateSrcN_covid_radiography}} | {{GateSrc_covid_radiography}} |
| Kermany | {{GateSrcN_kermany}} | {{GateSrc_kermany}} |
| NIH ChestX-ray14 | {{GateSrcN_nih_cxr14}} | {{GateSrc_nih_cxr14}} |
| Shenzhen | {{GateSrcN_shenzhen}} | {{GateSrc_shenzhen}} |
| TBX11K | {{GateSrcN_tbx11k}} | {{GateSrc_tbx11k}} |

All {{GateNegRej}} of the {{GateNeg}} synthetic non-radiographs are rejected at the adopted threshold. The validation pass rate rises from {{GatePassOld}}% at 0.83 to {{GatePassNew}}% at {{GateThr}}.

### Re-measurement on unseen radiographs

After the change the gate was applied again to the test radiographs of build 002 and to the Montgomery set. It rejects {{GateNewRejTestK}} of {{GateNewRejTestN}} real test radiographs ({{GateNewRejTestPct}}%) and {{GateNewRejExtK}} of {{GateNewRejExtN}} Montgomery radiographs ({{GateNewRejExtPct}}%), compared with {{CxrGateBTwoOldTestRej}} of {{CxrGateBTwoOldTestN}} ({{CxrGateBTwoOldTestRejPct}}%) and {{CxrGateBTwoOldExtRej}} of {{CxrGateBTwoNewExtN}} before, measured on the same build-2 test split and the same Montgomery set. The first measurement, {{CxrGateBOneOldTestRej}} of {{CxrGateBOneOldTestN}} on the build-1 test split, is how the problem was found and is not paired with the build-2 figure. The validation pass rate, measured on the same validation split before and after, rose from {{CxrGateValPassOld}}% to {{CxrGateValPassNew}}%. Because the test and external images were not used to set the threshold, these figures are an honest measure of how often the gate refuses a genuine radiograph.

### What the gate does not establish

The negatives used for calibration are synthetic and few, so the rejection of real non-radiographs, such as photographs of films, other body regions or CT scans in the wild, is not established. A gate trained or calibrated on both radiographs and a real set of non-radiographs is future work. The gate is also a convenience for the demonstration. It is not a safety control, and the model metrics do not depend on it.

## Testing

Testing is described in Appendix D. In summary, the backend and package suites cover configuration, preprocessing equivalence, model construction and checkpoint refusal, metric definitions, the evaluator's input validation, the data splitting rules, consistency of the documentation with the generated evidence, every API behaviour listed above, and the integration of the backend with a random-weight network of the canonical architecture. The frontend service modules have Jest tests, and the tests are listed in [[tab:tests]] with their coverage in [[tab:coverage]]; and the production build compiles with warnings treated as errors. A manual browser check with the production build served from the backend confirmed that the scores on the page equal the API response and that the missing-checkpoint, unverified-checkpoint and inconclusive states are shown correctly. Tests that use stubs or random weights check mechanics only and are not evidence that the trained model is accurate.
