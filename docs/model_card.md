# Model card — LungAI DenseNet-121 — Six-Class Chest X-Ray Classifier

| | |
|---|---|
| Model ID / version | `lungai-densenet121-6class` / `v5.0` |
| Definition | [`lungai/configs/lungai_densenet121.json`](../lungai/configs/lungai_densenet121.json), built by [`lungai/models/densenet.py`](../lungai/models/densenet.py) |
| Checkpoint | `densenet121_frequency_v5.h5`, Keras HDF5, SHA-256 `e2c95dc5eba588f877280948593f2965442d9950aa891ed5765eaf0117889022`. Not distributed; see [inference.md](inference.md) |
| Status | Research prototype. Recorded results only; not reproducible from this repository (no checkpoint, images or per-image predictions) |

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

The order matches the recorded training mapping (`experiments/densenet_frequency_v5/class_mapping.json`). `tests/ml/test_config.py` checks this.

## Preprocessing

Implemented once, in [`lungai/preprocessing.py`](../lungai/preprocessing.py), and used by training, evaluation and the backend:

1. Decode to 3-channel 8-bit. Grayscale is replicated to 3 channels; an alpha channel is dropped.
2. Convert BGR → RGB.
3. Gaussian blur, 3×3, σ = 0.8.
4. CLAHE on the L channel of CIE LAB (clip limit 2.0, 8×8 tiles).
5. Gaussian low-pass filter, σ = 1.0, 7×7 kernel.
6. Lanczos-4 resize to 224×224.
7. Normalise: `(x/255 − mean)/std`, with ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].

Steps 3–5 run at native resolution, *before* resizing, as in the original training run. Their effective strength therefore depends on the input size (see [limitations.md](limitations.md)). This order is kept so that training and serving match the recorded checkpoint. `tests/ml/test_preprocessing.py` checks the implementation against a frozen copy of the original training function (tolerance 1e-6).

## Training procedure (recorded run)

- **Data:** the V5 training split (7,398 images). Validation: 1,579 images.
- **Loss:** sparse categorical cross-entropy, with balanced per-sample class weights `n/(k·n_c)`: COVID-19 0.907, Normal 0.667, Pleural Effusion 1.633, Pneumonia 0.628, Nodule/Mass 2.331, TB 1.306.
- **Phase 1:** 2 epochs, backbone frozen, Adam with lr = 1e-3.
- **Phase 2:** 2 epochs, last 30 backbone layers unfrozen, Adam with lr = 1e-5. Each epoch is 231 steps of 32 images.
- **Augmentation (training only):** horizontal flip with p = 0.5; rotation ±10° with reflect border.
- **Seed:** 42. **One run only**, with no repeat seeds.
- **Checkpoint selection:** the recorded checkpoint is the **final epoch**. Validation was evaluated once after training and its value was not saved.

The canonical trainer ([training.md](training.md)) instead selects the epoch with the best validation macro-F1 and saves everything listed above.

## Why this model

All five recorded V5 runs use the same DenseNet-121 backbone, data and schedule. They differ only in preprocessing or an auxiliary alignment loss. The evidence ([experiment_results.md](experiment_results.md)) supports the following points.

1. **Architecture:** DenseNet-121 is the only architecture with recorded results on the six-class V5 data. The legacy runs of other architectures used different 5- or 7-class datasets. They are archived and not comparable.
2. **Variant:** the Gaussian low-pass variant was chosen for these reasons:
   - **Its preprocessing was selected on validation data.** The baseline checkpoint was evaluated on validation under 10 candidate filters. σ = 1.0 had the best validation macro-F1: 0.8457, against 0.7495 unfiltered.
   - **It is the only run with a recorded checkpoint hash.** If the weights are recovered, they can be identified exactly.
   - **The backend already implements its preprocessing exactly.** No retraining is needed to serve it.
3. **What was *not* used to choose it:** its lead on the internal test split (82.93% against 80.89% for CORAL). Choosing on test results would bias the comparison. Each run is also a single seed, so differences of a few points may be noise. CORAL has the best *recorded validation* accuracy among the runs that saved one (0.815). The selected run's own trained-model validation score was not saved, so validation cannot rank the two. The choice is a defensible continuity decision. It does **not** show that this variant is better.
4. **External behaviour:** every variant fails the same way on the external cohort, with 0 of 80 normal scans called Normal. External results do not favour any of them.

## Evaluation (recorded)

Source: [artifacts/evaluation_reports/historical_record_verification.md](../artifacts/evaluation_reports/historical_record_verification.md). All values were recomputed from the stored confusion matrix and match the recorded values.

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

The model does not separate normal from TB scans in this cohort.

## Known failure modes

- **Nodule/Mass acts as a sink class.** 147 of 268 internal-test errors are predictions of Nodule/Mass, and 74 of 152 Pleural Effusion cases are called Nodule/Mass. Images from unfamiliar sources are mostly called Nodule/Mass.
- **The model may be detecting the source, not the disease.** Classes are largely confounded with their source datasets. Normal recall on held-out sources is 0.00 for JSRT and 0.08 for NIH (`experiments/densenet_frequency_v5/frequency_source_heldout.json`).

## Calibration

Not evaluated. Only the mean top-class score is recorded: 0.904 when correct and 0.682 when wrong. The UI labels scores as "uncalibrated softmax", and a result is shown as inconclusive when the top score is below 0.5. That threshold is not clinically validated.

## Intended and prohibited use

- **Intended:** academic research, teaching, and demonstrating a reproducible imaging-AI pipeline.
- **Prohibited:**
  - diagnosis, triage, screening or treatment decisions;
  - use on patients;
  - any claim of clinical performance.

## Ethical and medical limitations

The training data is public and was retrospectively relabelled. For example, half of the Pneumonia labels are TBX11K "sick but not TB" images, and the Kermany images are paediatric. Performance has not been assessed across demographics, devices or institutions. See [limitations.md](limitations.md).
