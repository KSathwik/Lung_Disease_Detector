# Methodology (for the thesis)

This text describes only what the code and generated artifacts support. It covers the current V6 model; the earlier V5 work is summarised in the appendix. It contains no citations. Add references to the original dataset and method papers yourself.

## 1. Task

Single-label classification of frontal chest radiographs into six mutually exclusive classes: COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule/Mass and Tuberculosis. The class order is fixed in `lungai/configs/lungai_densenet121.json`.

## 2. Dataset construction (V6)

- **Acquisition:** each source was downloaded from its official distribution point by `scripts/data/acquire.py`, which resumes interrupted transfers, checks the final size against the server's Content-Length, rejects HTML responses, and logs URL, time, size and SHA-256 per file (`data/raw/_provenance/`). Sources behind access controls (VinDr-CXR, JSRT) were not used.
- **Label rules, fixed before training** (`lungai/configs/dataset_v6_proposal.json`): labels come from each source's own labels. NIH ChestX-ray14: single-finding Effusion, Nodule and/or Mass, Pneumonia, and No Finding (one image per patient). Kermany: NORMAL / PNEUMONIA. Shenzhen: normal / TB from the file name. TBX11K: healthy → Normal, active TB only → Tuberculosis; "sick but not TB" and latent TB excluded. Cohen collection: frontal X-rays labelled COVID-19, other pneumonia, TB or no finding. COVID-19 Radiography Database: COVID-19 folder only.
- **Caps:** per (source, class), sampled by whole patient group with a fixed seed, to bound CPU training time and stop one source dominating a class.
- **Content hashes:** MD5, SHA-256 and a 64-bit DCT perceptual hash of every image; near-duplicates are pairs within 6 bits whose 64×64 thumbnails correlate at ≥ 0.97.
- **Composition of the build used:** the generated table in [results/current_model.md](results/current_model.md#dataset-composition).

## 3. Partitioning

- **Proportions:** 70/15/15 into training, validation and internal test, stratified by class and source.
- **Unit:** the patient group. Groups use the genuine patient ID where the source provides one (NIH, Cohen), a filename-derived ID where possible (Kermany), and otherwise one group per image. Groups that share an exact or near-duplicate image are merged, so duplicates cannot cross splits.
- **Checks recorded in the build report:** groups spanning splits, identical files spanning splits, and external-cohort files present in development data (all zero for the builds used). Separation is proven only where patient identity is known.
- **External cohort:** the Montgomery County set (138 images: 58 TB, 80 normal) is never used for training or selection; development images matching it are excluded.

## 4. Preprocessing

Each image is decoded to 3-channel 8-bit RGB and then processed as follows:

1. Gaussian blur (3×3, σ = 0.8).
2. CLAHE on the CIE LAB L channel (clip limit 2.0, 8×8 tiles).
3. Gaussian low-pass filter (σ = 1.0).
4. Lanczos-4 resize to 224×224.
5. ImageNet mean/std normalisation.

The builder applies steps 1–4 once and caches the 224×224 result; training, evaluation and the web service all call the same functions in `lungai/preprocessing.py`.

**How the σ = 1.0 filter was chosen (V5 work):** an unfiltered baseline checkpoint was evaluated on the validation set under ten candidate filters (Gaussian σ ∈ {1, 2, 3}, ideal and Butterworth low-pass at three cut-offs, and none). The filter with the highest validation macro-F1 was selected.

**Training-only augmentation:** horizontal flip (p = 0.5) and rotation of ±10° with reflected borders.

## 5. Model and training

- **Architecture:** DenseNet-121 initialised with ImageNet weights, followed by global average pooling, batch normalisation, a 256-unit ReLU layer, dropout 0.3 and a six-way softmax.
- **Loss:** sparse categorical cross-entropy with balanced class weights.
- **Phase 1:** two epochs with the backbone frozen (Adam, lr = 10⁻³).
- **Phase 2:** the last 30 backbone layers unfrozen (Adam, lr = 10⁻⁵), batch 32, seed 42, until early stopping (3 epochs without validation macro-F1 gain), the epoch limit, or a wall-clock budget.
- **Checkpoint:** the epoch with the best validation macro-F1 (tie: validation macro ROC-AUC).
- **Compute:** a laptop CPU (TensorFlow 2.16 has no GPU support on native Windows); the served model is one run (seed 42); two further seeds and one ablation run are reported in `docs/experiment_results.md`.

## 6. Evaluation protocol

**Internal test, six-class:**

- accuracy (Wilson 95% CI);
- per-class precision, recall (Wilson CI), specificity and F1;
- macro and support-weighted averages;
- one-vs-rest ROC-AUC and PR-AUC, with macro averaging.

**External cohort, binary:** sensitivity and specificity of "any non-Normal prediction" against TB confirmation, each with Wilson CIs, plus exact Tuberculosis recall. Thresholds were not tuned on either test set.

## 7. Results to report

Use the generated tables in [results/current_model.md](results/current_model.md) (regenerated by `python -m lungai.evaluation.render_docs`; never copy numbers by hand). Report them as **single-run** results of the V6 model:

- **Internal test:** accuracy with its 95% CI, macro-F1 and macro ROC-AUC, the per-class table and the confusion matrix. The test split was evaluated once, after the checkpoint was chosen on validation.
- **Per source:** accuracy by source, because it differs strongly between sources and the overall figure depends on the source mix.
- **External:** abnormal-vs-normal sensitivity **together with** specificity, the exact Tuberculosis recall, and the prediction breakdown (which classes the model assigned to Montgomery's normal and TB scans).

Report the V5 model only as history: its checkpoint is lost, its test split was patient-leaked, and its numbers are not comparable with V6.

## 8. Limitations to state

State the single-run, CPU, time-boxed design; the source confounding of classes (for example COVID-19 coming from web collections) and what the external breakdown shows about it; the NLP-mined NIH labels; the paediatric Kermany images; unknown patient identity for some sources; the uncalibrated scores; and the blocked sources (VinDr, JSRT). See [limitations.md](limitations.md).

## Do not claim

- clinical validation, or screening ability from a sensitivity quoted without its specificity;
- TB detection from the "abnormal" rate on Montgomery when exact Tuberculosis recall is low;
- that patient-level separation is proven for sources without patient identifiers;
- that a few points' difference between runs is meaningful (only three seeds of the served recipe and one ablation seed were run; see `docs/experiment_results.md`);
- any V5 number (82.93%, 58/58, 76.18% / 0.9600) as a result of the current model.

## Appendix: the V5 work (superseded)

V5 unified eight public sources into 10,547 images, including Kaggle aggregates and TBX11K "sick but not TB" images mapped to Pneumonia, with partitions built incrementally across dataset versions. An audit found 154 internal-test images (9.8%) sharing a patient with training images. Five variants (filter, ERM, Deep CORAL, DANN, filter + CORAL) were trained once each for 4 epochs; the kept checkpoint was the final epoch and is now lost. Their recorded results are in [experiment_results.md](experiment_results.md#history-v5-runs-recorded-superseded). V6 was built to remove these defects.
