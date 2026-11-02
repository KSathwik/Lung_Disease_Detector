# Methodology (for the thesis)

This text describes only what the code and recorded artifacts support. Where the recorded run differs from the canonical pipeline, both are stated. It contains no citations. Add references to the original dataset and method papers yourself.

## 1. Task

Single-label classification of frontal chest radiographs into six mutually exclusive classes: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule/Mass and Tuberculosis. The class order is fixed in `lungai/configs/lungai_densenet121.json`.

## 2. Dataset construction

- **Composition:** the V5 dataset unifies eight public sources into 10,547 images (TBX11K 3,277; COVID-19 radiography 1,942; VinDr-CXR 1,467; Kermany paediatric pneumonia 1,395 and normal 1,199; Shenzhen 665; NIH ChestX-ray14 457; JSRT 145).
- **Label mapping:** source labels were mapped to the six classes. TBX11K "sick but not TB" images (1,395) were mapped to Pneumonia. Images with several findings were filtered to single findings.
- **Content hashes:** an MD5 content hash and a 64-bit difference hash were recorded for each image.

## 3. Partitioning

- **Proportions:** the dataset was partitioned 70/15/15 into training (7,398), validation (1,579) and internal test (1,570).
- **Method:** partitions were built incrementally across dataset versions (V2 to V5). Each version preserved earlier assignments and assigned new patient IDs with seed 42.
- **Synthetic IDs:** for the Kermany, COVID-19 radiography, Shenzhen and TBX11K sources, the patient ID was a synthetic per-image identifier.
- **Leakage found by audit:** re-deriving Kermany patient identity from filenames shows that 154 internal-test images (9.8%) share a patient with training images. Internal-test results are therefore optimistically biased.
- **External cohort:** the Montgomery County set (138 images: 58 TB, 80 normal) was excluded from all partitions and used only for external assessment. It had informed earlier preprocessing experiments.

## 4. Preprocessing

Each image is decoded to 3-channel 8-bit RGB and then processed as follows:

1. Gaussian blur (3×3, σ = 0.8).
2. CLAHE on the CIE LAB L channel (clip limit 2.0, 8×8 tiles).
3. Gaussian low-pass filter (σ = 1.0).
4. Lanczos-4 resize to 224×224.
5. ImageNet mean/std normalisation.

**How the σ = 1.0 filter was chosen:** an unfiltered baseline checkpoint was evaluated on the validation set under ten candidate filters (Gaussian σ ∈ {1, 2, 3}, ideal and Butterworth low-pass at three cut-offs, and none). The filter with the highest validation macro-F1 was selected.

**Training-only augmentation:** horizontal flip (p = 0.5) and rotation of ±10° with reflected borders.

## 5. Model and training

- **Architecture:** DenseNet-121 initialised with ImageNet weights, followed by global average pooling, batch normalisation, a 256-unit ReLU layer, dropout 0.3 and a six-way softmax.
- **Loss:** sparse categorical cross-entropy with balanced class weights.
- **Phase 1:** two epochs with the backbone frozen (Adam, lr = 10⁻³).
- **Phase 2:** two epochs with the last 30 backbone layers unfrozen (Adam, lr = 10⁻⁵). Each epoch is 231 steps of 32 images. Seed 42.
- **Checkpoint:** the recorded checkpoint is the final epoch. The canonical pipeline selects the epoch with the best validation macro-F1.

Four other variants were trained once each under the same protocol: ERM without the filter, Deep CORAL, DANN, and filter + CORAL. Their internal-test results are reported only as historical comparators.

## 6. Evaluation protocol

**Internal test, six-class:**

- accuracy (Wilson 95% CI);
- per-class precision, recall (Wilson CI), specificity and F1;
- macro and support-weighted averages;
- one-vs-rest ROC-AUC and PR-AUC, with macro averaging.

**External cohort, binary:** sensitivity and specificity of "any non-Normal prediction" against TB confirmation, each with Wilson CIs, plus exact Tuberculosis recall. Thresholds were not tuned on either test set.

## 7. Results to report

Use [experiment_results.md](experiment_results.md) and the model card. Report them as single-run, recorded results:

- **Internal test:** accuracy 82.93% (95% CI 80.99–84.71%) and macro-F1 78.35%. Macro ROC-AUC is 0.9755 as recorded; it is not independently verifiable.
- **External:** sensitivity 58/58 with specificity 0/80. The model did not distinguish normal from TB scans in the external cohort.

## 8. Limitations to state

State the patient leakage, the proxy labels, the confounding between classes and their source datasets, the single-run design, the missing per-image predictions, the uncalibrated scores and the external failure. See [limitations.md](limitations.md).

## Do not claim

- leakage-free splits;
- clinical validation;
- screening ability based on 58/58;
- superiority of one variant on the basis of the test-split ranking;
- the ERM baseline's 76.18% / 0.9600 as the canonical model's results.
