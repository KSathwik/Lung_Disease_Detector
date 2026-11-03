# Model card — LungAI DenseNet-121 — Six-Class Chest X-Ray Classifier

| | |
|---|---|
| Model ID | `lungai-densenet121-6class` |
| Version, checkpoint, dataset | See [Evaluation](#evaluation) (generated from the evaluation artifacts) |
| Definition | [`lungai/configs/lungai_densenet121.json`](../lungai/configs/lungai_densenet121.json), built by [`lungai/models/densenet.py`](../lungai/models/densenet.py) |
| Status | Research prototype. Single training run on CPU; results reproduced from saved per-image predictions. The checkpoint is not distributed (gitignored); see [inference.md](inference.md) |

## Architecture

- **Backbone:** `tf.keras.applications.DenseNet121`, ImageNet weights, without its top layer. It runs in inference mode (`training=False`), so its BatchNorm statistics stay frozen.
- **Head:** `GlobalAveragePooling2D (gap)` → `BatchNormalization (bn)` → `Dense(256, relu) (dense_features)` → `Dropout(0.3) (dropout)` → `Dense(6, softmax) (class_predictions)`.
- **Output:** a probability vector over the six classes in the canonical order (below). Single-label and mutually exclusive. Images with co-existing findings are forced into one class, which is a modelling assumption.

## Classes

| Index | ID | Name |
|---:|---|---|
| 0 | covid19 | COVID-19 |
| 1 | normal | Normal |
| 2 | pleural_effusion | Pleural Effusion |
| 3 | pneumonia | Pneumonia |
| 4 | nodule_mass | Pulmonary Nodule / Mass |
| 5 | tuberculosis | Tuberculosis |

The order matches the training mapping used by v5.0 and v6.0 (V5 file: `experiments/densenet_frequency_v5/class_mapping.json`). `tests/ml/test_config.py` checks this.

## Preprocessing

Implemented once, in [`lungai/preprocessing.py`](../lungai/preprocessing.py), and used by training, evaluation and the backend:

1. Decode to 3-channel 8-bit. Grayscale is replicated to 3 channels; an alpha channel is dropped.
2. Convert BGR → RGB.
3. Gaussian blur, 3×3, σ = 0.8.
4. CLAHE on the L channel of CIE LAB (clip limit 2.0, 8×8 tiles).
5. Gaussian low-pass filter, σ = 1.0, 7×7 kernel.
6. Lanczos-4 resize to 224×224.
7. Normalise: `(x/255 − mean)/std`, with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].

Steps 3–5 run at native resolution, *before* resizing, as in the original training run. Their effective strength therefore depends on the input size (see [limitations.md](limitations.md)). This order is kept so that training and serving match the checkpoints trained with it (v5.0 and v6.0). `tests/ml/test_preprocessing.py` checks the implementation against a frozen copy of the original training function (tolerance 1e-6).

## Training procedure

Trainer: [`lungai/training/train_v6.py`](../lungai/training/train_v6.py) on a V6 build (images preprocessed once and cached by the builder, using the same `lungai.preprocessing` functions as the backend).

- **Loss:** sparse categorical cross-entropy, with balanced per-sample class weights `n/(k·n_c)` computed on the training split (values in the run's `run_metadata.json`).
- **Phase 1 (head):** 2 epochs, backbone frozen, Adam with lr = 1e-3.
- **Phase 2 (fine-tune):** last 30 backbone layers unfrozen, Adam with lr = 1e-5, up to the epoch limit given to the trainer, with early stopping after 3 epochs without a validation macro-F1 gain, inside a wall-clock time box. Batch size 32.
- **Augmentation (training only):** horizontal flip with p = 0.5; rotation ±10° with reflect border.
- **Seed:** 42. **One run per build**, with no repeat seeds, on CPU (TensorFlow 2.16 has no native-Windows GPU support).
- **Checkpoint selection:** the epoch with the best **validation** macro-F1 (tie-break: validation macro ROC-AUC). The test split and the external cohort are evaluated once, after selection.

The per-epoch curve and the selected epoch are in [results/current_model.md](results/current_model.md#training).

## Why this model

The architecture and preprocessing were fixed before V6 training, carried over from the V5 work below. V6 changed the data (rebuilt from official sources with patient-grouped splits), not the model.

### History: how the V5 variant was chosen

All five recorded V5 runs use the same DenseNet-121 backbone, data and schedule. They differ only in preprocessing or an auxiliary alignment loss. The evidence ([experiment_results.md](experiment_results.md)) supports the following points.

1. **Architecture:** DenseNet-121 is the only architecture with recorded results on the six-class V5 data. The legacy runs of other architectures used different 5- or 7-class datasets. They are archived and not comparable.
2. **Variant:** the Gaussian low-pass variant was chosen for these reasons:
   - **Its preprocessing was selected on validation data.** The baseline checkpoint was evaluated on validation under 10 candidate filters. σ = 1.0 had the best validation macro-F1: 0.8457, against 0.7495 unfiltered.
   - **It is the only run with a recorded checkpoint hash.** If the weights are recovered, they can be identified exactly.
   - **The backend already implements its preprocessing exactly.** No retraining is needed to serve it.
3. **What was *not* used to choose it:** its lead on the internal test split (82.93% against 80.89% for CORAL). Choosing on test results would bias the comparison. Each run is also a single seed, so differences of a few points may be noise. CORAL has the best *recorded validation* accuracy among the runs that saved one (0.815). The selected run's own trained-model validation score was not saved, so validation cannot rank the two. The choice is a defensible continuity decision. It does **not** show that this variant is better.
4. **External behaviour:** every variant fails the same way on the external cohort, with 0 of 80 normal scans called Normal. External results do not favour any of them.

## Evaluation

<!-- BEGIN GENERATED: summary -->
_Generated by `python -m lungai.evaluation.render_docs` from `artifacts/evaluation_reports/model_metrics.json` (2026-11-03T11:05:58+00:00). Do not edit by hand._

**LungAI DenseNet-121 — Six-Class Chest X-Ray Classifier v6.1**, dataset V6 (`v6_build_002`), checkpoint `densenet121_v6r2_seed42.h5` (SHA-256 `7992b8da10ff…`).

| Split | N | Accuracy (95% CI) | Macro F1 | Macro ROC-AUC |
|---|---:|---:|---:|---:|
| Validation (checkpoint selection) | 2348 | 76.9% | 78.5% | 0.953 |
| Test (evaluated once) | 2350 | 77.4% (75.7–79.0%) | 78.8% | 0.955 |

External, Montgomery County (N=138, normal vs TB only, evaluated once): TB scans flagged abnormal 55/58 (94.8%, 95% CI 85.9–98.2%); normal scans called Normal 19/80 (23.8%, 95% CI 15.8–34.1%); TB scans called Tuberculosis 0/58 (0.0%, 95% CI 0.0–6.2%); ROC-AUC of P(abnormal) 0.809.

Evidence: REPRODUCED: computed by lungai.evaluation.evaluate_v6 from per-image predictions of this checkpoint (saved under artifacts/evaluation_reports/v6_r2_*). Single training run (seed 42), CPU-only and time-boxed; no repeated seeds, so run-to-run variation is not measured.

Previous model v5.0 (dataset V5, history only): internal test accuracy 82.9% (N=1570), external sensitivity 58/58 with specificity 0/80. Earlier model, kept for the record only. Its checkpoint is lost and its dataset (V5) had patient overlap between training and test; these numbers are not comparable with the current model's.
<!-- END GENERATED: summary -->

Per-class, per-source and confusion-matrix tables: [results/current_model.md](results/current_model.md).

### History: the V5 model's recorded evaluation (superseded)

Source: [artifacts/evaluation_reports/historical_record_verification.md](../artifacts/evaluation_reports/historical_record_verification.md). All values were recomputed from the stored confusion matrix and match the recorded values. The v5.0 checkpoint is lost.

**Internal test split (N = 1,570; 154 of these images are patient-leaked):**

- Accuracy: 82.93% (Wilson 95% CI 80.99–84.71%).
- Macro precision / recall / F1: 79.31% / 80.73% / 78.35%. Weighted F1: 84.18%.
- Macro ROC-AUC (OvR): 0.9755 and macro PR-AUC: 0.8391. Both are recorded but **unverifiable**, because no probabilities were saved.

| Class | Support | Precision | Recall (95% CI) | Specificity | F1 |
|---|---:|---:|---|---:|---:|
| COVID-19 | 292 | 0.996 | 0.945 (0.913–0.966) | 0.999 | 0.970 |
| Normal | 394 | 0.909 | 0.883 (0.848–0.911) | 0.970 | 0.896 |
| Pleural Effusion | 152 | 0.600 | 0.513 (0.434–0.591) | 0.963 | 0.553 |
| Pneumonia | 423 | 0.949 | 0.794 (0.753–0.830) | 0.984 | 0.865 |
| Pulmonary Nodule / Mass | 108 | 0.385 | 0.852 (0.773–0.907) | 0.900 | 0.530 |
| Tuberculosis | 201 | 0.920 | 0.856 (0.800–0.898) | 0.989 | 0.887 |

**External cohort (Montgomery County, N = 138: 58 TB, 80 normal; binary task):**

- Sensitivity, defined as predicted ≠ Normal among TB scans: 58/58 (95% CI 0.938–1.000).
- Specificity: **0/80** (95% CI 0.000–0.046).
- Exact Tuberculosis recall: 0/58.
- Predictions: TB scans → 50 Nodule/Mass and 8 Pleural Effusion; normal scans → 80 Nodule/Mass.

The v5.0 model did not separate normal from TB scans in this cohort. Its failure modes were a Nodule/Mass "sink" class (147 of 268 internal-test errors) and source confounding (Normal recall 0.00 on held-out JSRT, 0.08 on held-out NIH; `experiments/densenet_frequency_v5/frequency_source_heldout.json`).

## Known failure modes

See [limitations.md](limitations.md#current-model) for the failure modes of the current checkpoint, read from its external-cohort prediction breakdown and per-source results. Class and source are still partly confounded in V6 (for example, COVID-19 images come from web collections and Pneumonia partly from a paediatric source), so the model may learn the source rather than the disease.

## Calibration

The expected calibration error on the test split is reported in [results/current_model.md](results/current_model.md#test-split-per-class). Scores are uncalibrated softmax outputs; no calibration was fitted. A result is shown as inconclusive when the top score is below 0.5, a threshold that is not clinically validated.

## Intended and prohibited use

- **Intended:** academic research, teaching, and demonstrating a reproducible imaging-AI pipeline.
- **Prohibited:**
  - diagnosis, triage, screening or treatment decisions;
  - use on patients;
  - any claim of clinical performance.

## Ethical and medical limitations

The training data is public, collected for other purposes, and labelled by each source (NIH labels are mined from reports by NLP, not read from the images). The Kermany images are paediatric, unlike the other sources. Performance has not been assessed across demographics, devices or institutions. See [limitations.md](limitations.md).
