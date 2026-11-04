# Novelty extension protocol (C1-C4): decision rules fixed in advance

Status: **APPROVED by the student on 2026-11-04 (morning) and binding from the approval commit** (hash in section 13). Drafted 2026-11-03 (evening) while runs 4 and 5 were still training and before any number of the novelty extension (spectral map, INLP, leave-one-source-out, band-stop) existed; committed first as an unapproved draft (`e3e1aa1`). Nothing in `artifacts/novelty/` exists yet. Style and purpose follow `docs/data/run_selection_protocol.md`.

Inputs read for this draft: `docs/publication/NOVELTY_ROADMAP.md`, `docs/publication/lit_search/SCOPE_MERGE_REVIEW.md` (flags F1-F6, rule B8), `docs/publication/lit_search/INVENTORY_AND_WORKPLAN.md`, `docs/manuscripts/shared/WRITING_BRIEF.md`, `lungai/evaluation/source_probe.py`, `lungai/training/train_v6.py`, `lungai/config.py`, and the manifest of `data/derived/v6_build_002` (metadata columns only; no image was opened).

## 1. Purpose and wording

Questions (stated neutrally): in which spatial-frequency bands do frozen CNN features carry information about the source dataset and about the disease; how much of the source information can a linear projection remove, and what happens to disease accuracy when it is removed; how does a linear head behave on a source it was not trained on. These are measurements. They do not propose a clinical method.

Wording rules (from `WRITING_BRIEF.md` section 3 and `RESULTS_FACTS.md`):

- Say "fixed in advance and recorded in the repository", never "pre-registered".
- A probe shows that information is **available** in the features. It does not show that the trained model **uses** it. Shortcut statements are "indirect evidence" or "consistent with", never "the model uses the source" or "scanner fingerprint" (F1).
- Every leakage statement and every C4 held-out number carries the patient-identity caveat: separation is proven only for rows with known patient identity; 4,130 TBX11K and COVID-19 Radiography rows are grouped per image, and Kermany identities are derived from file names.
- Novelty wording: "To our knowledge, within the searches described, we are not aware of prior work that quantifies, band by band, source versus disease information in CNN features of chest radiographs." No "first", no "state of the art", no "novel method" for C3 or C4. Halimi et al. 2026 and Bouaziz and Chebouba 2026 are arXiv preprints opened at abstract level only.
- No comparison with other papers' numbers. No clinical claim.
- Every number in any document is a generated macro: `\RSpec...` (C1), `\RInlp...` (C3), `\RLoso...` (C4), `\RBs...` (C2), written by `lungai/evaluation/render_docs.py` from `artifacts/novelty/*.json`. Macro names use letters only (bands are labelled A-F, see section 4). Hand-typed numbers are not allowed.

## 2. Scope

| Part | Status | Rule |
|---|---|---|
| C1 spectral source/disease map | **Confirmed** | Sections 4-6 |
| C3 INLP source erasure | **Confirmed** | Section 7 |
| C4 leave-one-source-out | **Confirmed** | Section 8 |
| C2-lite (linear head on band-stopped ImageNet features) | **Confirmed**, exploratory | Section 9 |
| C2 (band-stop build and one retraining) | **Conditional**: runs only if the go/no-go rule in section 10 gives GO | Section 10 |

Out of scope: more than one C2 run, multi-seed C2, a second backbone, retraining per held-out source, a second external cohort, any change to the served model, gate, backend or frontend, lung masking, a phase-versus-amplitude split per band and an amplitude-mixing (Fourier domain generalisation) baseline beside C2 (both suggested in `lit_search/LIT_C1_C2_frequency.md` section 5 and listed as future work).

## 3. Data

### 3.1 Builds

| Name | Path | Use |
|---|---|---|
| main | `data/derived/v6_build_002` (manifest rows: train 10,957, val 2,348, test 2,350, Montgomery 138 = 15,793; the inventory counts 15,748 cache files, because 81 manifest rows (covid_radiography 53, kermany 28) are byte-identical copies sharing 36 SHA-256 cache files; every duplicate group lies within one split (checked 2026-11-03), so there is no cross-split duplication. Probes sample manifest rows; a duplicated image can appear twice in a sample and this is reported, not removed) | All analyses (primary) |
| no-filter | `data/derived/v6_build_002_no_filters` (same manifest, byte-identical `manifest_v6.csv`; run 5 build) | C1 sensitivity only |

Each cache file is `cache/<raw_sha256>.npy`, uint8, 224 x 224 x 3, RGB. Input to every extractor is `lungai.preprocessing.normalize` (divide by 255, subtract ImageNet mean [0.485, 0.456, 0.406], divide by std [0.229, 0.224, 0.225]), exactly as `source_probe` does through `train_v6.cached_batch`.

**F1 (stated, not hidden).** The main cache has already passed Gaussian pre-blur (sigma 0.8, 3x3), CLAHE (clip 2.0, 8x8 tiles, L channel) and a Gaussian low-pass (sigma 1.0) at native resolution, and was then resized to 224 x 224 with Lanczos-4 (`lungai/configs/lungai_densenet121.json`, `lungai/preprocessing.py`; the filters run before the resize). Sources differ in native resolution and field of view, so per-band source information partly reflects how different pixel grids were resampled and not only scanner physics. The band map is therefore described as a map of the cached 224 x 224 images; the no-filter cache is the sensitivity check; "scanner fingerprint" is not a finding.

### 3.2 Counts in the main build (from `manifest_v6.csv`, split not `excluded`)

Images per (source, class). Dashes mean no image.

| Source | Class | Train | Val | Test |
|---|---|---|---|---|
| cohen_covid | COVID-19 | 341 | 60 | 77 |
| cohen_covid | Normal | 11 | 3 | 4 |
| cohen_covid | Pneumonia | 133 | 29 | 29 |
| cohen_covid | Tuberculosis | 8 | 2 | 1 |
| covid_radiography | COVID-19 | 1,742 | 384 | 374 |
| kermany | Normal | 377 | 81 | 81 |
| kermany | Pneumonia | 1,471 | 316 | 317 |
| nih_cxr14 | Normal | 836 | 181 | 183 |
| nih_cxr14 | Pleural Effusion | 2,099 | 457 | 444 |
| nih_cxr14 | Pneumonia | 225 | 49 | 48 |
| nih_cxr14 | Pulmonary Nodule / Mass | 2,110 | 441 | 449 |
| shenzhen | Normal | 228 | 49 | 49 |
| shenzhen | Tuberculosis | 235 | 51 | 50 |
| tbx11k | Normal | 700 | 150 | 150 |
| tbx11k | Tuberculosis | 441 | 95 | 94 |
| montgomery (external) | Normal / Tuberculosis | - | - | 80 / 58 |

Totals check: train 10,957, val 2,348, test 2,350, external 138. Class-source confounding is built in (F3): Pleural Effusion and Pulmonary Nodule / Mass occur only in NIH; Pneumonia is mostly Kermany.

### 3.3 Probe samples (C1, C2-lite)

One sample per split, drawn with the `source_probe._sample` rule (per (source, class) cell, `min(n, cap)` images, `DataFrame.sample(random_state=seed)`), **sampling seed 0**, the same sample for every band variant, every extractor and both builds (paired, F5). Caps: train 150, val 100, test 100 (train and test equal the `source_probe` defaults). The no-filter build uses the same `raw_sha256` rows; the script asserts that the sampled hash lists are identical.

| Split | Cap per cell | Resulting images | Cells below the cap |
|---|---|---|---|
| train | 150 | 1,952 | cohen Normal 11, cohen Pneumonia 133, cohen TB 8 (all other cells reach the cap of 150) |
| val | 100 | 1,019 | cohen COVID-19 60, Normal 3, Pneumonia 29, TB 2; kermany Normal 81; nih Pneumonia 49; shenzhen Normal 49, TB 51; tbx11k TB 95 |
| test | 100 | 1,033 | cohen COVID-19 77, Normal 4, Pneumonia 29, TB 1; kermany Normal 81; nih Pneumonia 48; shenzhen Normal 49, TB 50; tbx11k TB 94 |

The script prints and stores the realised counts; if they differ from this table, the run stops and the difference is logged under section 13.

**Minimum cell rule.** A (source, class) cell enters a within-source score only if it has at least **20** images in the sample of the split being scored (and at least 20 training-sample images for the source/class pair). Sources with fewer than two qualifying cells are excluded from within-source scores. With the counts above:

- Test and val within-source disease sample: cohen_covid (COVID-19, Pneumonia), kermany (Normal, Pneumonia), nih_cxr14 (Normal, Pneumonia, Pleural Effusion, Pulmonary Nodule / Mass), shenzhen (Normal, Tuberculosis), tbx11k (Normal, Tuberculosis): **5 sources**.
- Excluded and listed in every table: covid_radiography (one class), cohen Normal (4 test / 3 val images), cohen Tuberculosis (1 test / 2 val images).
- Within-Normal source score (differs from the existing probe in `source_probe_v6_build_002.json`, which uses 5 sources and chance 0.20; macros `\RSpecWithinNormal...` versus `\RProbeImgWithinNormal`, never compared directly): Normal images of kermany, nih_cxr14, shenzhen, tbx11k (4 sources, nominal chance 0.25); cohen Normal is excluded (4 test images).

## 4. C1: spectral map of source and disease information

### 4.1 Band definition (224 x 224 grid)

Frequency grid `fx, fy = fftfreq(224)` (cycles per pixel, step 1/224, Nyquist 0.5); radial frequency `r = sqrt(fx^2 + fy^2)` (corner values up to 0.707 belong to the top band). Six octave bands from DC to Nyquist, labelled A-F:

| Band | Radial range r (cycles/pixel) | Period (pixels) |
|---|---|---|
| A | 0 (DC) to 0.015625 (1/64) | 64 and longer |
| B | 0.015625 to 0.03125 | 64 to 32 |
| C | 0.03125 to 0.0625 | 32 to 16 |
| D | 0.0625 to 0.125 | 16 to 8 |
| E | 0.125 to 0.25 | 8 to 4 |
| F | 0.25 to 0.5 (and the corners beyond 0.5) | 4 to 2 |

Edges e1..e5 = 0.015625, 0.03125, 0.0625, 0.125, 0.25.

**Filter shape: smooth, not ideal** (to limit ringing). For an edge e, define `t = clip((log2(r/e) + 0.25) / 0.5, 0, 1)` and the high-side weight `h_e(r) = 0.5 (1 - cos(pi t))`, so each edge has a raised-cosine transition of +/-0.25 octave (0.84 e to 1.19 e). Band masks: `M_A = 1 - h_e1`; `M_k = h_e(k-1) * (1 - h_e(k))` for B-E; `M_F = h_e5`. The six masks sum to 1 at every frequency. Bins with r = 0 belong to band A.

**DC rule.** The DC coefficient (image mean) is kept in every variant, so brightness is not a confounder and uint8 range is respected. The inverse transform is real-valued (conjugate-symmetric masks), applied independently to each of the three channels, clipped to [0, 255] and rounded to uint8 (the cache dtype). Band-pass images are **not** re-scaled in contrast; their low contrast is part of what is measured.

### 4.2 Variants (capped at 13 per extractor and cache)

1. `full` (unfiltered cache image).
2. Six band-pass variants `pass_A ... pass_F` (DC plus band k).
3. Six band-stop variants `stop_A ... stop_F` (all frequencies except band k, DC kept).

Cumulative low-pass and high-pass variants are **not** run (cap; they can be derived by reading the band-pass and band-stop curves). No variant may be added after the first C1 number exists except through section 13.

| Run | Extractor | Cache | Variants | Splits | Forward passes |
|---|---|---|---|---|---|
| C1-main | ImageNet DenseNet-121 (primary) | main | all 13 | train, val, test samples (4,004) | 52,052 |
| C1-served | run-2 backbone (secondary) | main | `full` + 6 `pass` (7) | train, test samples (2,985) | 20,895 |
| C1-nofilter | ImageNet DenseNet-121 | no-filter | `full` + 6 `pass` (7) | train, test samples (2,985) | 20,895 |

Stages run in this order and each runs to completion or not at all; a stage that cannot finish in the session is reported as "not run" (no partial reporting). Before the stages start, speed is timed on 200 images and the measured seconds per image and the projected hours are written to the C1 JSON; the timing does not change any setting.

### 4.3 Extractors

- **ImageNet (primary):** `keras.applications.DenseNet121(include_top=False, weights="imagenet", pooling="avg", input_shape=(224,224,3))`, 1,024-d global-average-pooled features, as in `source_probe`.
- **Served run-2 backbone (secondary):** the `densenet121` layer of `models/densenet121_v6r2_seed42.h5` (SHA-256 7992b8da10ff97f90207dc730c3163ba968a4d6094e1bff263377612ee0597c6) followed by `GlobalAveragePooling2D`, as the `model` extractor of `source_probe`. It was trained on all six sources, so it measures what the served network can represent, not an untouched model (F2).

### 4.4 Probes (identical for every variant)

`StandardScaler` fitted on the train sample of that variant, then `LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=0)` (lbfgs, multinomial), as in `source_probe.probe`. Two probes per variant, both fitted on the train sample and scored on the val or test sample: a **source probe** (6 classes) and a **disease probe** (6 classes).

### 4.5 Metrics

Let `BA_src(v)` be the source balanced accuracy over the six sources (chance 1/6 = 0.1667) on the scored sample.

- **Primary disease score (F3):** `F1_ws(v)`, the within-source disease macro-F1: for each source in the within-source sample (section 3.3), macro-F1 over that source's qualifying classes on that source's images (labels restricted to those classes; a prediction of any other class counts as an error); then the unweighted mean over the 5 sources.
- **Secondary:** `F1_all(v)`, the six-class macro-F1 over the whole sample, reported with the warning that it can be bought with source cues.
- **Within-Normal source accuracy** `BA_norm(v)`: balanced accuracy of the source probe over true-Normal images of the 4 qualifying sources.
- **Source-dominance score (SDS), normalised to chance:** `S*(v) = (BA_src(v) - 1/6) / (1 - 1/6)` and `D*(v) = (F1_ws(v) - c_D) / (1 - c_D)` with `c_D = mean over the 5 sources of 1/k_s = (0.5 + 0.5 + 0.25 + 0.5 + 0.5) / 5 = 0.45` (k_s = number of qualifying classes; a nominal chance level, conservative because the probe has six outputs). `SDS(v) = S*(v) - D*(v)`. Positive: source information exceeds disease information in that band; zero: equal; negative: the reverse. SDS is a descriptive index with no threshold.
- **Band-stop differences:** `dS(b) = BA_src(full) - BA_src(stop_b)` and `dD(b) = F1_ws(full) - F1_ws(stop_b)` (positive means information lost by removing b).
- **Band-pass retention:** `S*(pass_b) / S*(full)` and `D*(pass_b) / D*(full)` (reported only if the denominator is above 0.05, otherwise "not defined").

### 4.6 Uncertainty

Percentile bootstrap, **B = 1,000**, **95% interval**, resampling unit **the scored image** (val or test sample), drawn with bootstrap seed 1. The same image indices are used for every variant, so differences (dS, dD, SDS contrasts between bands) are paired. Probes are fitted once on the fixed train sample, so intervals describe test-sample variation, not training variance or seed variation; this is stated wherever the intervals appear. Only one probe fit per variant, no refits under resampling.

## 5. Sensitivity and robustness in C1

C1-served and C1-nofilter repeat the band-pass analysis for the stated extractor or cache. Agreement or disagreement of the band profile of `BA_src` and `F1_ws` is reported in words ("the profile is similar / differs"); no formal test.

## 6. Hypotheses for C1 (neutral)

H1a: `BA_src` exceeds chance in the full image (expected from the existing probe, 93.8% balanced accuracy over six sources; `\RProbe...`). H1b: the band profile of source information and of disease information are described; no band is predicted to be the carrier. H1c: within-Normal source accuracy is above 0.25 in at least one band-pass variant. Each is reported whether it holds or not, in every extractor and cache that ran.

## 7. C3: linear source erasure (INLP) on frozen features

- **Feature layer:** the 1,024-d global-average-pooled features of the **full** (unfiltered) cache image, for all train, val, test and Montgomery images of the main build (15,793 images per extractor), cached to `artifacts/novelty/features_<extractor>.npy`. Primary extractor: ImageNet (F2: no project training). Secondary: run-2 backbone, labelled "trained backbone, not a held-out test" (F2). The model's own BatchNorm and Dense(256) head are not used; the disease head is retrained from scratch.
- **Standardisation:** `StandardScaler` fitted on all train images; INLP iterates in the standardised space.
- **INLP:** at iteration k, fit `LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=0)` to predict the six sources from the current train features; take the row space of the coefficient matrix (SVD, singular values above 1e-3 times the largest), project the features (train, val, test, Montgomery) onto its orthogonal complement, and repeat. **Iteration cap 20.** **Stopping rule:** stop at the first iteration k* at which a freshly fitted source classifier on projected train features has **validation** source balanced accuracy at most 1/6 + 0.05 = 0.2167, or at k* = 20. Validation here means all val images of the main build (class_weight balanced, same settings).
- **Disease head:** multinomial `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)` on projected standardised train features (all 10,957 train images). C is chosen once from {0.01, 0.1, 1.0} by validation six-class macro-F1 on **unprojected** features and then kept for every k.
- **LEACE check:** closed-form LEACE (Belrose et al. 2023) with the six source labels one-hot, fitted on train features, applied to val and test, with the same disease head; reported as a one-row check against INLP at k*.
- **Metrics per iteration k = 0..k*:** validation and test source balanced accuracy (fresh source classifier, chance 0.1667); disease `F1_ws` and `F1_all` computed on the **full** val and test splits (all images of the main build, not the probe sample) with the minimum cell rule of section 3.3 applied to the full-split counts; overall accuracy; macro ROC-AUC. Disease-head change is reported as the paired difference to k = 0 with a bootstrap 95% interval (B = 1,000, unit = test image, seed 1).
- **Montgomery, once:** after the test table is written, the k = 0 and k = k* heads (both extractors) are scored on Montgomery in one run: Normal/Tuberculosis, sensitivity and specificity reported together, with Wilson 95% intervals. Not used for any choice.
- INLP may collapse disease accuracy; that is reported as a finding about entanglement of source and disease, not as a failure of the run.

Hypotheses (neutral): H3a source balanced accuracy falls toward chance as k grows; H3b the within-source disease macro-F1 may rise, fall or stay within its interval; both outcomes and any mixture are reported.

## 8. C4: leave-one-source-out (LOSO) linear heads

- **Features:** the cached features of section 7. **Primary matrix: ImageNet features (F2).** **Secondary matrix: run-2 backbone features**, named "linear-head leave-one-source-out on frozen features" and always stated to come from a backbone that saw all six sources in training; never called held-out or external.
- **Head:** multinomial `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)` on `StandardScaler`-ed features. For each fold C is chosen from {0.01, 0.1, 1.0} by macro-F1 on the **validation** images of the sources that are *not* held out, computed over the classes that occur in those validation images only (a fixed `labels` list per fold; a class with no training or validation image in the other sources, for example Pleural Effusion and Pulmonary Nodule / Mass when NIH is held out, is neither scored nor a selection criterion; ties go to the smaller C); the held-out source's train, val and test images are never used for fitting or selection.
- **Folds and training data:** one fold per held-out source; training uses the train split of the other five sources.
- **Held-out evaluation set:** the held-out source's **test split**.
- **Inclusion rule for held-out sources:** an *evaluated class* of a fold has at least 20 test images in the held-out source **and** at least 50 train images in the other sources. A source is a held-out row only if it has at least 2 evaluated classes. From the counts in section 3.2: included are **cohen_covid** (COVID-19, Pneumonia), **kermany** (Normal, Pneumonia), **nih_cxr14** (Normal, Pneumonia; Pleural Effusion and Pulmonary Nodule / Mass are untestable because no other source contains them), **shenzhen** (Normal, Tuberculosis), **tbx11k** (Normal, Tuberculosis); excluded and listed: covid_radiography (one class), and cohen Normal and cohen Tuberculosis as classes.
- **Metrics per fold (evaluated classes only):** macro-F1 (a prediction of a non-evaluated class is an error), balanced accuracy, per-class recall; the **in-distribution reference** (same head recipe trained on all six sources' train split, scored on the same test images) and the gap reference minus LOSO.
- **Binary abnormal-vs-normal view:** score = 1 - P(Normal) from the same head; abnormal = any non-Normal label in the held-out source, including classes absent from training (so NIH Effusion and Nodule/Mass count as abnormal); reported as ROC-AUC and balanced accuracy at 0.5, for held-out sources with at least 20 Normal and 20 abnormal test images (kermany, nih_cxr14, shenzhen, tbx11k).
- **Uncertainty:** bootstrap, B = 1,000, 95%, unit = held-out test image, seed 1.
- **Montgomery, once:** the head trained on all six sources' train split (both extractors) is scored on Montgomery in one run, sensitivity and specificity together, Wilson 95% intervals.
- Each LOSO number is stated with the patient-identity caveat; in addition, held-out numbers from the trained backbone never carry the word "external".

Hypothesis (neutral): H4 LOSO macro-F1 may differ from the in-distribution reference; the sign and size are reported for every included source, including sources where LOSO is equal to or better than the reference.

## 9. C2-lite (exploratory, validation decision, no training)

Run only after C1-main (ImageNet) has finished; all selection uses validation data. The **candidate band** is `b*` from section 10 criterion 1 (largest `dS` among bands that satisfy it); if no band satisfies it, C2-lite is run for the band with the largest `dS(b)` and the decision is NO-GO.

- Features: ImageNet GAP features of `full` and `stop_b*` for a larger train sample (**cap 400 per cell**, sampling seed 0; 4,358 images, no more than 8,716 new passes) and the val and test samples already extracted in C1.
- Linear heads as in section 4.4, fitted on the cap-400 train features: a source head and a disease head, once on `full` and once on `stop_b*`.
- Metrics (val sample): `dS_lite = BA_src(full) - BA_src(stop_b*)`, `dD_lite = F1_ws(full) - F1_ws(stop_b*)`, with paired bootstrap intervals (B = 1,000, 95%, unit = val image). The same table on the test sample is written once, in the order of section 11, labelled exploratory.
- **Matched-energy random control** (needed for criterion 2): for `b*`, `q` is the mean over val-sample images of the fraction of non-DC spectral energy `sum |F|^2` that `M_b*` weights. The control removes each conjugate-symmetric frequency pair independently with probability `q` (DC kept). **Three** control masks (mask seeds 1, 2, 3), each applied to train and val samples (cap 150 and cap 100, ImageNet), giving three `dS` values; the control value is their mean (and min and max are reported). The control is run only if some band satisfies criterion 1.

## 10. C2 go/no-go rule (copied from SCOPE_MERGE_REVIEW B8; thresholds fixed here, before any C1 number)

All quantities use **validation data only**, the paired probe sample of C1, ImageNet extractor (the served extractor is a sensitivity check only), and 95% bootstrap intervals over validation images. For band b the band-stop image is "all frequencies except b".

`dS(b)` = source balanced accuracy on the full image minus on the band-stop image. `dD(b)` = within-source disease macro-F1 on the full image minus on the band-stop image.

**GO to full C2 only if all hold:**

1. There is a band b with `dS(b) >= 0.05` and the lower CI bound of `dS(b)` above 0, **and** `dD(b) <= 0.02` with the upper CI bound of `dD(b)` below 0.03.
2. The matched-energy random band-stop control (same energy removed) shows `dS` smaller than for b (control mean below the CI lower bound of `dS(b)`), so the selected band matters.
3. C2-lite (linear head on ImageNet features of band-stopped images) shows a drop in source balanced accuracy of at least 0.05 and no fall in within-source disease macro-F1 greater than 0.02 on validation.
4. Run 5 and all M.Tech deliverables (handoff steps 1-5) are finished, and the student approves one night of CPU time before 22:00.

If several bands satisfy criterion 1, b* is the one with the largest `dS(b)`. Otherwise **NO-GO**: report C1, C3, C4 and C2-lite (exploratory), and list C2 as future work. The decision and the values behind it are written to `artifacts/novelty/c2_decision.json` and logged in section 13 before any C2 build exists.

**If GO** (one exploratory seed): `scripts/data/make_bandstop_build.py` applies `stop_b*` to every cached image of `v6_build_002` (train, val, test, external; manifest and split fingerprint copied unchanged); train seed 42 with the run-2 recipe (2 head epochs plus 14 fine-tune epochs, patience 3, as run 2 completed 16 epochs); evaluate val, test and Montgomery once with the same code as runs 1-5; run `source_probe --build <bandstop build> --checkpoint <new run>` with the settings of section 3.3 (sampling seed 0, caps 150/100) for the post-suppression source probe. Report next to the seed spread of runs 2-4 (mean, sample SD, min, max, from `aggregate_seeds.py`) and run 5. A difference is described as inside or outside that spread; the run is labelled "exploratory, single seed"; no improvement claim from one run. Montgomery (138 images, Normal/Tuberculosis) is a weak judge and is reported with specificity next to sensitivity.

## 11. Evaluation order and access log (F4)

"Access" means computing a metric on that split. Extracting features from images does not count until a metric is computed. Each analysis below runs its test and Montgomery computation **once**, from one script invocation, in this order; the log (date, commit hash of the code, JSON file) is appended in section 13 as each step completes.

| # | Analysis | Splits scored | Montgomery |
|---|---|---|---|
| 0 | Timing on 200 images; feature cache for sections 7-8 | none | features only, no metric |
| 1 | C1-main (ImageNet, main cache): val results written first, then test | train (fit), val, test | none |
| 2 | C1-served (main cache) | train (fit), test | none |
| 3 | C1-nofilter (ImageNet, no-filter cache) | train (fit), test | none |
| 4 | C3 INLP: val curve first, then test table; LEACE check | val, test | **once** (k = 0 and k*, both extractors) |
| 5 | C4 LOSO matrices (ImageNet, then trained-backbone) | val (C selection), test | **once** (both extractors) |
| 6 | C2-lite, val decision then test table | val, test | none |
| 7 | C2 go/no-go (validation numbers only, no new data access) | val | none |
| 8 | C2 retraining and evaluation, **only if GO** | val, test | **once** |

Montgomery was already used for runs 1 to 3 (evaluated once each on val, test and external, and run 2 was started after run 1's external results were seen) and for runs 4 and 5 (the follow-up queue evaluates each run on Montgomery once, automatically, when its training ends; both finish before any novelty analysis starts). It was never used for training or for selecting the served model. The novelty extension therefore adds exactly two Montgomery accesses (steps 4 and 5), none in C1 or C2-lite, and one in C2 if GO. Any statement about Montgomery must say it is not untouched.

The validation split is used for selection in steps 1, 4, 5, 6 and 7 (INLP stopping, C choices, band choice); it may be read repeatedly.

## 12. Reporting, negative results, multiple comparisons

- Every band, every variant, every fold and every iteration that ran is reported in the tables (supplementary if long). No variant, extractor or fold is omitted because of its result. Stages that did not run are listed as not run.
- Negative, null and mixed results are reported as such: for example "source information is not concentrated in one band", "INLP removed source information and also removed disease information", "LOSO macro-F1 equals the in-distribution reference". Sentences state what the numbers show and name the alternative explanations (F1 resampling, F3 confounding, F2 backbone exposure, patient-identity caveat).
- **Multiple comparisons:** there are 13 variants, 6 bands, 5 folds and up to 20 iterations, each with several 95% intervals. No p-values are computed and no band is picked after the fact. Intervals are descriptive; roughly one in twenty may exclude the true value by chance. A statement about a band, fold or iteration is made only (a) if its interval excludes the reference value and (b) if the same pattern holds across neighbouring bands, iterations or the secondary extractor; otherwise it is reported as "not distinguishable". The only formal thresholds are those of section 10, fixed before any result.
- Source BA, disease scores and SDS are properties of the probe sample (150/100 per cell) and of fitted linear probes; they are not population estimates and not clinical accuracy.
- The C2 single run, the small Montgomery cohort and the patient-identity caveat are limitations stated in the manuscript and the thesis threats chapter.

## 13. Change control and log

**Start condition (all must hold before any novelty analysis, including feature extraction, is launched).** (1) `followups.out` shows runs 4 and 5 finished (training and their val, test and Montgomery evaluations) and the M.Tech handoff steps 1 to 5 are done or the student explicitly allows the overlap; (2) the student has approved this protocol in chat; (3) this file is committed to `research/lungai-clean-baseline` and the commit hash and date are entered in the table below. Until then every statement in section 11 about runs 4 and 5 being finished is a requirement, not a fact. The student approves; nobody else fills the approval or the hash.

After this file is committed, any change to a setting, threshold, variant list, sample size, seed, stopping rule, scope or order is entered in the table below with the date and the reason, **before** the affected result is computed. A change made after an affected result exists is not allowed to alter which result is reported; the original rule is applied and the deviation is described. The student approves each entry.

| Date | Change | Reason | Affected results computed before the change? | Approved |
|---|---|---|---|---|
| 2026-11-03 | Draft committed as `e3e1aa1` (2026-11-04 ~05:50), labelled not approved | - | none (no novelty result existed) | not approved |
| 2026-11-04 | Protocol approved unchanged; scope C1 + C3 + C4 + C2-lite, C2 only by the section 10 rule; journal choice left open | Student's reply in chat to the request for approval: "Proceed with your findings" | none (no novelty result existed; `artifacts/novelty/` absent) | student, in chat |
| 2026-11-04 | Approval commit: `f427c04` (pushed to `research/lungai-clean-baseline`); hash entered in the following commit because a commit cannot contain its own hash | - | none | - |
| 2026-11-04 | Implementation clarifications where this protocol is silent (no setting changed): (1) feature cache stored as one file per variant and set, same rows as section 7; (2) F1_ws scores only images of qualifying cells, so cohen Normal/TB images are not scored; (3) C3: ties in C go to the smaller C; (4) C4: the in-distribution reference head chooses C on validation images of all six sources; (5) C4: a class present in validation with no training images stops the run (cannot occur with the section 3.2 counts); (6) C4 binary view: P(abnormal) of exactly 0.5 counts as abnormal; (7) section 10 criterion 3 states no interval condition, so point estimates are used. Also: matched-energy q pools the three channels weighted by the smooth mask; the control mask is 0/1; the INLP stopping check uses the same fit whose row space is removed | Code written in `lungai/novelty/` (174 tests pass) before any real-image result | none (no novelty result existed; `artifacts/novelty/` absent) | logged before results; student acknowledgement requested in chat |

Access log (appended as steps complete): step number, date and time, code commit hash, output file, remarks.

| Step | Date/time | Code commit | Output | Remarks |
|---|---|---|---|---|
| (none yet) | | | | |
