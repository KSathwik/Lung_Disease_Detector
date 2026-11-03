# Research brief: what we did, what is wrong, what we propose, and how to defend it

Written 2026-11-03 (night) by the supervisor agent for the student (Katkam Sathwik), to be read in the morning and used for the viva, the defence and writing. It explains the work in plain terms, keeps every statement tied to a file, and ends with a strict consistency review of all existing planning documents.

Conventions. Numbers are quoted only from the files named in square brackets; none is new. `[PENDING: seed runs]` and `[PENDING: ablation]` mark results that do not exist yet (runs 4 and 5 are still training). Nothing in the novelty extension (C1 to C4) has been run: `artifacts/novelty/` does not exist and `NOVELTY_PROTOCOL.md` is an uncommitted draft. Wording rules used throughout (from `docs/manuscripts/shared/WRITING_BRIEF.md` section 3 and `RESULTS_FACTS.md`): decision rules are "fixed in advance and recorded in the repository" (the word that starts with "pre-regist" is never used, because nothing was registered with an outside body); shortcut evidence is "indirect" or "consistent with", never proven; every leakage statement carries the patient-identity caveat (4,130 TBX11K and COVID-19 Radiography rows are grouped per image, Kermany identities come from file names); no state-of-the-art claim, no clinical claim, no comparison with other papers' numbers.

---

## 1. One-paragraph summary

We rebuilt a six-class chest X-ray classifier (COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, Tuberculosis) from official public sources, after auditing an earlier version (V5) that had patient leakage, proxy labels and an external specificity of 0/80. With patient-grouped splits, leakage checks and once-only evaluation, a standard DenseNet-121 reaches 77.4% internal test accuracy, but on an unseen hospital (Montgomery, 138 images) it calls only 19 of 80 normal scans Normal and none of 58 tuberculosis scans Tuberculosis. A linear probe shows that the source dataset of an image can be read from the network's features with 93.8% balanced accuracy (chance 16.7%), so a source shortcut is possible, and our evidence for it is indirect. The proposed extension turns this into a measurement study: in which spatial-frequency bands does source information live compared with disease information (C1), how much of it can a linear projection remove and what does that cost the disease signal (C3), how does a linear head behave on a source it never saw (C4), and, only if validation rules allow it, whether suppressing source-heavy bands helps (C2). Results, positive or negative, are reported under rules fixed in advance and recorded in the repository.

Sources: `PROJECT_EXECUTION_STATUS.md` (HANDOFF), `docs/manuscripts/shared/results.json`, `artifacts/evaluation_reports/model_metrics.json`, `artifacts/data_quality/source_probe_v6_build_002.json`, `docs/publication/NOVELTY_ROADMAP.md`.

---

## 2. Background: what the project did so far

### 2.1 Why the project exists
The earlier pipeline V5 recorded 82.93% internal accuracy. The audit found that 154 of 1,570 test images shared a patient with training images, that 1,395 TBX11K "sick but not TB" images had been labelled Pneumonia (49.7% of that class), and that on Montgomery the model called 0 of 80 normal scans Normal; the checkpoint was lost [`docs/thesis/content/12_ch3_problem_objectives.md` table of defects; `docs/manuscripts/shared/WRITING_BRIEF.md` section 1]. V6 is the rebuild.

### 2.2 Data (V6 build 2)
- Six development sources: NIH ChestX-ray14, Kermany (paediatric), COVID-19 Radiography Database, TBX11K, Cohen COVID, Shenzhen. External cohort: Montgomery County (80 Normal, 58 TB).
- Sizes: train 10,957, validation 2,348, test 2,350, external 138 [`docs/manuscripts/shared/results.json`; `NOVELTY_PROTOCOL.md` section 3.1]. Every file is logged with URL, size and SHA-256 (`data/raw/_provenance/`).
- Label rules, per-source caps and the split rule are written in a versioned config before training. Splits are grouped by patient where identity is known, and by merged exact and near-duplicate images otherwise. Leakage checks stop the build on failure and all read 0.
- Identity caveat: patient separation is proven only for rows with known identity (genuine IDs from NIH and Cohen, derived from file names for Kermany); 4,130 TBX11K and COVID-19 Radiography rows are grouped per image [`RESULTS_FACTS.md` wording rule (c)(e)].
- Built-in confounding: Pleural Effusion and Nodule / Mass exist only in NIH; Pneumonia is mostly Kermany; COVID-19 comes from two sources [`NOVELTY_PROTOCOL.md` section 3.2].

### 2.3 Model and training
DenseNet-121 (ImageNet weights) + global average pooling + BatchNorm + Dense(256) + Dropout(0.3) + Dense(6, softmax). Two phases: 2 head epochs, then 14 fine-tune epochs on the last 17 weighted backbone layers, so the features stay close to ImageNet's (checked by weight comparison) [`RESULTS_FACTS.md` F2, F5]. Preprocessing: Gaussian pre-blur (sigma 0.8), CLAHE (clip 2.0, 8x8) on the lightness channel, Gaussian low-pass (sigma 1.0), Lanczos resize to 224 x 224 [`NOVELTY_PROTOCOL.md` section 3.1]. CPU only, 8 GB RAM: one epoch takes about 20 minutes.

### 2.4 Once-only evaluation
Test and Montgomery are each evaluated once per trained run, after the checkpoint is fixed; per-image predictions are saved. Selection between runs uses validation data only, by a rule written down before run 2's results (`docs/data/run_selection_protocol.md`). Honest disclosure: run 2 was started after run 1's Montgomery results were seen, so Montgomery is "not used for training or selection of the served model" but is not untouched [`RESULTS_FACTS.md` wording rule (d)].

### 2.5 Runs 1 to 5
| Run | What | Result (source) |
|---|---|---|
| 1 (build 1, seed 42) | First V6 run | test acc 65.7%, macro-F1 67.7%, AUC 0.921 (N = 1,724); Montgomery called 71 of 138 scans COVID-19 [`results.json` RunOne keys; HANDOFF] |
| 2 (build 2, seed 42) | Served model v6.1 | test acc 77.4% (Wilson 95% CI 75.7 to 79.0), macro-F1 78.8%, AUC 0.955, ECE 0.020; Montgomery sensitivity 55/58, specificity 19/80, TB 0/58, COVID-19 calls 3 [`model_metrics.json`; `results.json`] |
| 3 (build 2, seed 43) | Seed check | test acc 77.1%, macro-F1 78.5%, AUC 0.954; Montgomery 54/58, 25/80, TB 0/58; early-stopped at epoch 9, best epoch 6 [`artifacts/evaluation_reports/v6_r3_test/metrics_test.json`, `v6_r3_external/summary_external.md`; `runs/v6r3_densenet121_seed43/state.json`] |
| 4 (build 2, seed 44) | Seed check | training; epoch 8 of at most 16 done at 22:40 [`docs/TRACKER.md` status board] [PENDING: seed runs] |
| 5 (build 2 without filters, seed 42) | Preprocessing ablation | queued [PENDING: ablation] |

Run 2 was chosen over run 1 on 2,018 common validation images: macro-F1 79.6% vs 59.4%. Caveat that must always be stated: on the 1,392 images from sources both runs trained on, accuracy is about equal (67.8% vs 67.6%) and macro-F1 is 68.5% vs 62.2%, so much of run 2's lead is on sources that are in-distribution for run 2 only [`results.json` Cmp keys; `RESULTS_FACTS.md` F1].

### 2.6 Source probe (the key diagnostic)
A logistic regression on frozen 1,024-d DenseNet features predicts the source dataset: balanced accuracy 93.8% over 6 sources (chance 16.7%) with ImageNet features and 93.8% with the trained backbone; within one class it stays high: Normal 97.2% (5 sources, chance 20%), Tuberculosis 98.9%, Pneumonia 89.1%, COVID-19 89.0% [`artifacts/data_quality/source_probe_v6_build_002.json`; 1,952 train and 1,033 test images]. Meaning: source information is linearly available in the features. It does not prove the classifier uses it.

Per-source test accuracy differs a lot: COVID-19 Radiography 99.5%, Kermany 94.7%, TBX11K 94.7%, Shenzhen 80.8%, NIH 62.5%, Cohen 50.5% [`results.json` SrcAcc keys]. Do not rank sources by data quality.

### 2.7 Grad-CAM and the CXR gate
Grad-CAM on one pre-specified correct, median-confidence test image per class: COVID-19 and Normal heatmaps concentrate on neck, shoulders and borders; Nodule / Mass and TB fall inside the lung fields; the Pneumonia (Kermany) map is diffuse. One image per class: illustrative only [`RESULTS_FACTS.md` F6]. The heuristic CXR input gate (rejects non-radiographs) was recalibrated from threshold 0.83 to 0.703 on validation: validation pass rate 38.7% to 91.0%; on the build-2 test split rejections fell from 1,494 to 218 of 2,350 and on Montgomery from 83 to 18 of 138; no real non-radiograph set exists to test it [`results.json` CxrGate keys; `RESULTS_FACTS.md` F7].

---

## 3. Current problems we face

### 3.1 Scientific
1. **Source shortcut (indirect evidence).** Source is predictable (93.8%), class and source are confounded, Grad-CAM falls outside the lungs for two classes, and Montgomery behaviour is poor. Together this is consistent with a shortcut; none of it proves one.
2. **External gap.** Internal accuracy 77.4% against Montgomery specificity 19/80 (23.8%) [`results.json`].
3. **TB 0/58 on Montgomery in every run so far** (runs 1 to 3: 1/58, 0/58, 0/58). The model recognises TB inside the data (TB F1 85.7%, AUC 0.994 on test) yet never names TB on a new hospital's TB scans; most are called Pneumonia or Nodule / Mass [`results.json`; `RESULTS_FACTS.md` F4]. Plausible explanations (source-specific TB appearance from Shenzhen/TBX11K, no TB-like class mix at Montgomery) are untested.
4. **Limited method novelty.** Standard DenseNet-121, standard preprocessing. `docs/manuscripts/journal_bspc/REVIEWER_RISK.md` expects major revision at BSPC mainly for items 4 (no causal shortcut test), 5 (coarse ablation), 6 (small external cohort), 1 (few seeds) and 14 (one backbone).
5. **Single external cohort**, 138 images, binary, wide intervals.
6. **Source and class are confounded by design**, so even a disease probe can use source cues. Fix in the protocol: within-source disease scoring.
7. **Montgomery is no longer untouched** (runs 1 to 5 all scored on it); the novelty plan adds exactly two more accesses by fixed rule.
8. **Only one seed behind the served model**; three seeds will give a rough spread only [PENDING: seed runs].

### 3.2 Practical
- CPU only, 8 GB RAM: one heavy job at a time; no novelty compute while training runs.
- Run 4 ends about 01:35 on 4 Nov, run 5 about 07:00 to 08:00 [`docs/TRACKER.md`, HANDOFF; the two files give different run-5 estimates]. Then handoff steps 1 to 5 (aggregate, fill placeholders, rebuild, test, audit) before any novelty compute.
- Pending placeholders: 24 `\pending` occurrences in the BSPC paper (23 in `06_results.tex`, 1 in `08_conclusion.tex`), 10 `[PENDING` occurrences in the thesis (6 seed, 4 ablation), slide 23.
- Decisions waiting for the student (section 9). Docker and CI blocked. VinDr-CXR and JSRT blocked on credentials.

---

## 4. Problem statement and research questions

**Problem statement.** Multi-source chest X-ray benchmarks pool images from datasets that differ in acquisition and population, so the source of an image is predictable from the features of a standard classifier and is confounded with the disease label. Internal accuracy under a leakage-controlled split can therefore remain high while behaviour on an unseen hospital is poor. We ask where in the spatial-frequency spectrum source information and disease information are carried by frozen convolutional features, how far the two can be separated by linear operations, and how a linear head generalises across held-out sources, on a patient-grouped, six-class benchmark whose evaluation rules are fixed in advance and recorded in the repository. The study is a measurement and diagnostic study; it proposes no clinical method and makes no clinical claim.

Research questions (numbering used in this brief; the thesis `12_ch3_problem_objectives.md` currently has three different questions, see review item R16):
- **RQ1 (done).** How large is the gap between internal results under a leakage-controlled split and results on an external hospital? Answer so far: 77.4% accuracy against 19/80 specificity and 0/58 exact TB.
- **RQ2 (C1).** In which radial spatial-frequency bands of the cached 224 x 224 images do frozen CNN features carry source information, and in which disease information (within source)?
- **RQ3 (C3, C2-lite, conditional C2).** How much source information can a linear projection (INLP) remove from the features, what happens to disease performance when it is removed, and does suppressing source-heavy bands change source information and disease information (and, only if the rule says GO, held-out performance)?
- **RQ4 (C4).** How does a linear head trained on five sources perform on the sixth, relative to training on all six, and how does this relate to RQ2 and RQ3?

---

## 5. Novelty: what is new and what is not

Read this section before any claim is written. Source: `lit_search/LIT_C1_C2_frequency.md`, `lit_search/LIT_C3_C4_and_VENUES.md`, `lit_search/SCOPE_MERGE_REVIEW.md`, `NOVELTY_ROADMAP.md`. The search was by web agents without Scopus or Google Scholar, so every statement is "within the searches described"; a pre-submission query is mandatory.

| Part | Honest verdict | Closest prior work (as opened by the helpers) | Safe wording |
|---|---|---|---|
| C1 per-band source vs disease probe of frozen CNN features, chest X-ray | The only part that looks new in its narrow form. Not new: source identifiability itself (Zech 2018, Compton 2023, Dack and Dai 2025), frequency shortcuts in general (Wang ICCV 2023, Lu 2024), frequency-filtered dataset classification in general (an unverified natural-image lead, arXiv 2412.01876). | Halimi et al., arXiv 2610.29830 (radial spectrum of chest X-rays, unsupervised failure clusters, no source or disease probes; preprint, abstract level); Lu et al. (CT/MRI, injected artefacts; arXiv 2407.11379, Proc. SPIE 13406); Zhang et al., Sci Rep 13:12690 (2023) (contrast and sharpness at image level) | "To our knowledge, within the searches described, we are not aware of prior work that quantifies, band by band, source versus disease information in CNN features of chest radiographs." No "first". |
| C3 INLP erasure on frozen features, then retrain a linear disease head | **Not a new method.** INLP (ACL 2020), RLACE (ICML 2022), LEACE (NeurIPS 2023) exist. Overlap is real: Bouaziz and Chebouba arXiv 2604.00263 (frozen CXR features, linear site probe, linear head, adversarial), Robinson et al. PLOS ONE 2022 (training-time source disentanglement), Rahbar arXiv 2608.10295 v1 (INLP/ComBat site removal in brain MRI, cite v1 only). The chest X-ray, six-source, disease-head-retrain version was not found. | as left | "...we did not find prior work applying post-hoc linear concept erasure (INLP) to remove source information from CNN features in multi-source chest radiograph classification. We evaluate it as a diagnostic of source-disease entanglement, not as a new method." |
| C4 leave-one-source-out | **Not new as a protocol.** Yao et al. 2019 (arXiv 1904.01638) did leave-one-domain-out on chest X-rays. New only as a leakage-controlled six-class matrix tied to the probe and to C1/C3, and only the ImageNet-feature matrix is a clean held-out test (the trained backbone saw all six sources). | Yao 2019; Compton et al. MLHC 2023 | "Following the leave-one-domain-out design of Yao et al. (2019), we report a held-out-source matrix ... " Not a contribution claim. |
| C2 band-stop then retrain | **Weakest claim.** Overlaps: per-band Laplacian-pyramid harmonisation of chest X-rays (Kim et al., J Med Imaging 10(6):061103, 2023), Fourier-based domain generalisation (Xu CVPR 2021, PASTA, He NeurIPS 2024), spectral editing (Lu 2024). Remaining gap: probe-driven band choice on chest X-rays. | as left | "We evaluate whether band-selective suppression, with bands chosen in advance from the per-band probe on validation data, reduces source information and changes held-out-source and external performance." Optional, one exploratory seed. |

What the overall contribution honestly is: a leakage-controlled benchmark and audit (existing), plus a measurement of where source information lives and how entangled it is with disease (C1, C3), plus a source-by-source generalisation matrix (C4). It is an analysis-first paper. A reviewer wanting a new algorithm will still say "limited method novelty"; the defence is rigour and honesty, not a headline method.

---

## 6. Proposed methodology (as in `NOVELTY_PROTOCOL.md`, draft)

### 6.1 Pipeline

```
 cached images (v6_build_002, 224x224x3 uint8; already blur+CLAHE+low-pass+resize)
        |
        +--[C1] radial FFT band filters (6 octave bands A-F, DC kept)
        |         variants: full, pass_A..F, stop_A..F (13)
        |         |-> frozen DenseNet-121 (ImageNet; secondary: served run-2 backbone)
        |         |-> 1,024-d GAP features
        |         |-> linear probes (source, disease) on fixed samples (1,952 / 1,019 / 1,033)
        |         |-> BA_src, F1_ws, SDS per band, bootstrap CIs (B=1,000)
        |
        +--[features, full image, all 15,793 images] -> artifacts/novelty/features_*.npy
        |         |--[C3] INLP (<=20 iterations, stop at val source BA <= 0.2167)
        |         |        + LEACE check -> retrain linear disease head -> curves; Montgomery once
        |         |--[C4] leave-one-source-out linear heads (5 held-out sources) -> matrix;
        |                  Montgomery once
        |
        +--[C2-lite] linear head on ImageNet features of stop_b* images (cap 400/cell)
                  |-> validation-only GO / NO-GO rule (section 10 of the protocol)
                  +--[C2, only if GO + student approves a night] band-stop build,
                       retrain seed 42 (2+14 epochs), evaluate once, post-suppression probe
```

### 6.2 Each part in plain words
- **C1 (main contribution, measurement).** Split every image into six frequency bands (octaves from coarse structure to fine detail) with smooth filters. Feed band-only or band-removed images to a frozen network, and train two simple classifiers on the resulting features: one predicts the source, one predicts the disease. If source is predicted well from a band where disease is not, that band carries "hospital style" rather than pathology. Metrics: source balanced accuracy `BA_src` (chance 1/6); within-source disease macro-F1 `F1_ws` (primary, because disease is confounded with source); six-class `F1_all` (secondary); within-Normal source accuracy; a source-dominance score `SDS = S* - D*` with both parts normalised to chance (nominal chance 0.45 for disease). Sensitivity checks: served backbone, and the no-filter cache (because the main cache is already low-passed and resized, so per-band source information partly reflects resampling, flag F1). No claim of "scanner fingerprint".
- **C3 (second contribution, diagnostic).** INLP repeatedly trains a linear source classifier on the features and deletes the directions it used. After k rounds the source cannot be read linearly. Then a fresh linear disease head is trained. If disease accuracy survives, source and disease are separable; if it collapses, they are entangled. LEACE is a one-step closed-form check. A collapse is a finding, not a failure.
- **C4 (supporting evaluation).** Train a linear head on five sources and test on the sixth. Primary matrix uses ImageNet features (no project training, so truly held-out); the trained-backbone matrix is called "linear-head leave-one-source-out on frozen features" because that backbone saw all six sources. Included held-out sources are those with at least two evaluated classes of at least 20 test images (cohen_covid, kermany, nih_cxr14, shenzhen, tbx11k). Binary abnormal-vs-normal view for four of them.
- **C2-lite (exploratory, minutes).** Linear head on ImageNet features of images with the candidate band removed, on validation first. It is the cheap test of whether suppression could help.
- **C2 (conditional).** One exploratory training seed on band-stopped images, only if all four GO conditions in `NOVELTY_PROTOCOL.md` section 10 hold: a band with source drop at least 0.05 and disease loss at most 0.02 (CI conditions), a matched-energy random band-stop control that removes less source information, C2-lite agreeing, and run 5 plus all M.Tech deliverables finished with the student approving the night. A single seed is compared with the seed spread of runs 2 to 4 and run 5, labelled exploratory, no improvement claim.

### 6.3 Integrity design
1. **Fixed in advance and recorded in the repository.** Band edges, variant list, sample sizes and seeds, minimum cell size (20 images), INLP stopping rule, LOSO inclusion rule, GO/NO-GO thresholds and evaluation order are in the protocol, which becomes binding only after the student approves it and it is committed; the hash is then entered in its section 13. Changes after that go in the change log before the affected result exists.
2. **Validation-only selection.** INLP stopping, regularisation choices and the C2 band decision use validation data only.
3. **Test / Montgomery access log.** Each analysis scores test and Montgomery once, in the logged order (protocol section 11): C1 (no Montgomery), C3 (once), C4 (once), C2-lite (none), C2 (once, only if GO). Montgomery is described as "used before, not untouched".
4. **Generated macros.** Numbers reach documents only through `\RSpec...`, `\RInlp...`, `\RLoso...`, `\RBs...` written by `render_docs.py` from `artifacts/novelty/*.json`; no hand-typed numbers.
5. **Reporting rules.** Every band, fold and iteration that ran is reported; no p-values; a statement about a band needs the interval to exclude the reference and a consistent neighbouring pattern; stages that did not finish are "not run"; negative results are written as negative; the patient-identity caveat accompanies every C4 number.

---

## 7. Expected outcomes

| Possible result | What it would mean | Publishable? |
|---|---|---|
| Source information concentrated in a band with little disease information | Supports C2-lite and possibly a GO; first quantitative localisation of a source cue in chest X-ray features | Yes, as the main result; C2 still exploratory |
| Source and disease information spread over the same bands | Simple band filtering cannot separate them; suppression is unlikely to help | Yes: a clear negative result for band-based remedies, with the map as evidence |
| INLP removes source information and disease F1 stays | The disease signal is not carried by the source direction; the probe does not imply dependence | Yes |
| INLP removes source information and disease collapses | Source and disease are entangled in these features (also partly by confounded labels, F3) | Yes: a quantitative entanglement measure |
| LOSO near the in-distribution reference | Held-out source generalises better than the Montgomery result suggests; the gap may be cohort-specific | Yes |
| LOSO far below the reference | Within-benchmark held-out sources also show the gap; supports RQ1 with several cohorts instead of one | Yes |
| C2-lite or C2 no better, or worse | Suppression is not a fix on this benchmark | Yes, as future work or a short negative result |

Why a negative result is still publishable: the claim is about measurement and honest evaluation; a null on suppression tells the community that "just filter the hospital style" does not work and that the probe numbers should not be over-read. Caution: BSPC reviewers may still ask for a method; the safest framing is the analysis-first one in section 5.

---

## 8. Publication plan

Current venue analysis: `docs/publication/JOURNAL_SCOPE_MATCH.md` (ScienceDirect Insights and Guide pages opened 2026-11-03; ratings are the helper's reading, not publisher statements). It supersedes the rankings in `JOURNALS.md`, `LIT_C3_C4_and_VENUES.md` and `SUBMISSION_TRACKER.md`. Unverified: live fees and word limits, Scopus status (the journal pages only; check each ISSN in Scopus), the student's country price under Geographical Pricing.

One ranking per scope:

| Scope | First choice | Fallback | Why (from `JOURNAL_SCOPE_MATCH.md`) |
|---|---|---|---|
| **S1**: C1 + C3 + C4, no training | **Informatics in Medicine Unlocked (IMU)**, CiteScore 10.5; gold OA only, USD 2,710 (Geographical Pricing; amount for the student not shown); 2 / 36 / 114 days | **BSPC**, CiteScore 13.0, IF 5.7; no fee on subscription route (OA USD 3,470); 10 / 66 / 203 days | IMU: imaging in scope, no word limit found, an evaluation-style chest X-ray paper without a new network is already there, no method-novelty line. BSPC: a full paper is defined around "a novel technique" at "normally about 5,000 words"; S1 has no novel technique and the manuscript (about 13.2k tokens) is about 2.6 times the figure |
| **S2**: S1 + C2 (one exploratory seed) | **BSPC** | **IMU** | C2 supplies a technique, but with one seed the evidence is weak; the length problem remains |

Low fit in both scopes: CMPB (about 3,500 words, new methods), CBM (Web of Science delisting and integrity notice), PRL (7 pages), CMIG and AIIM (method novelty required). Medium: CMPB Update, ISWA, Array (generic, "must be novel"). Costs: IMU needs an APC decision before submitting (ask the department about waivers or agreements); BSPC needs cutting to about 5,000 to 6,500 words of main text with the rest as supplementary. For IMU: cut about 20 to 30 percent, abstract under 250 words, highlights, ethics statement saying why none is needed. R1 (conflicting venue rankings) is RESOLVED by this table; the student still confirms the scope (section 9) and the live pages.

**Relation to the EMBC paper (overlap statement).** EMBC 2027 (Singapore, 11 to 15 July 2027): submission 24 January 2027, 11:59 PM AoE, no extension; accept/reject 16 April 2027; final 30 April 2027; 4 pages of content plus a 5th page of references (maximum 20 references); a named author must agree to review for EMBC; AI use must be disclosed. The EMBC paper (19 references, complete) is not changed by the novelty work and contains no probe, spectral map, INLP or LOSO. The journal paper must: cite the EMBC paper, say it is substantially extended, list only completed additions (three seeds, ablation, source probe, per-source analysis, run comparison, gate calibration, V5 audit, C1/C3/C4), upload the EMBC PDF for the editors, write the text afresh and redraw figures, check IEEE copyright terms at acceptance, and not have both under review with substantially the same content. An M.Tech thesis is allowed prior material at Elsevier; cite it. The 30 percent "substantial new content" rule is common practice and not stated in the sources read; ask the editor.

**Timeline for publication.** EMBC portal opens December 2026; submit by 24 Jan 2027. Journal: after the thesis is submitted and the student has reviewed the new sections; earliest realistic submission is February to March 2027 [supervisor estimate; the student's thesis deadline is unknown]. Publisher-stated review times (BSPC 66 days to decision after review, 203 days to acceptance) are averages, not guarantees.

---

## 9. Information required from the student

**Decisions**
1. Approve `NOVELTY_PROTOCOL.md` (read it fully; once committed it is binding). Then it is committed and its hash entered in section 13 before any novelty number exists.
2. Scope: C1 + C3 + C4 only (no training, about one day of compute) or the full plan including the conditional C2 (2 to 3 nights).
3. C2 night: if the GO rule is met, will you approve one night of CPU before 22:00? (Needs run 5 and all thesis deliverables finished.)
4. Journal: per `JOURNAL_SCOPE_MATCH.md`, S1 means IMU first and BSPC fallback, S2 means BSPC first and IMU fallback; whether you can pay an APC (IMU USD 2,710, India price not shown) or want the no-fee route; whether to condense to about 5,000 words.
5. Commit and push approval (branch `research/lungai-clean-baseline` only; never create another branch). Latest commit `871ce5b` is already on origin; uncommitted: `NOVELTY_PROTOCOL.md`, `NOVELTY_ROADMAP.md`, `NOVELTY_PLAN.md` (modified), `lit_search/`, `TRACKER.md`, the HANDOFF file, this brief.

**Personal and administrative items** (never invented by agents)
- Contacts (student, Prof. Ramana, Dr. Tirupathi, Dr. Srinivas): `[TO BE PROVIDED]`. Corresponding author: `[TO BE CONFIRMED]`. EMBC reciprocal reviewer (a qualified author who agrees to review): `[TO BE CONFIRMED]`. Author order is confirmed (Sathwik, Ramana, Tirupathi, Srinivas); CRediT roles and every author's approval remain `[AUTHORS TO CONFIRM]`.
- Your M.Tech thesis submission date and any department deadlines for the report and viva (not in any file).
- Ethics statement (public de-identified data: obtain or confirm an exemption wording), licence terms of each source (record upstream terms; NLM Montgomery is "research use, no redistribution"), data and code availability (public repository URL; BSPC "Option C"), generative-AI declaration text (the authors fill in the tool and reason), CLAIM and TRIPOD+AI checklists (`REVIEWER_RISK.md` items 17 to 21).
- Credentials for VinDr-CXR (PhysioNet) and JSRT registration, and a Kaggle token, if a second external cohort is wanted.

**What the student must check personally before submission**
- Run a Scopus and Google Scholar query combining chest X-ray, frequency/spectral/band, source/site/dataset identification, probing, and nullspace/INLP erasure; record the date and the hits. The helpers had only general web search.
- Open the unverified items: DeGrave 2021 full text, arXiv 2412.01876, Ben Ahmed IEEE Access 2022, Rafferty and Rajan journal details; re-check the Halimi preprint listing on the day of writing; confirm Dack and Dai year.
- Confirm live journal pages: BSPC word limit and fee, IMU APC for India, Scopus status by ISSN (BSPC 1746-8094, IMU 2352-9148, JIIM eISSN 2948-2933).
- Read and edit every disclosure, cover letter and overlap statement; no signature or approval is implied anywhere.

---

## 10. Timeline: now to submission (M.Tech deliverables first)

Estimates are supervisor estimates from `TRACKER.md`, `NOVELTY_ROADMAP.md` section 3 and `SCOPE_MERGE_REVIEW.md` B3; compute times are unmeasured.

| When | Work | Owner |
|---|---|---|
| 3 Nov night | Run 4 trains; run 5 queued; no novelty compute | auto |
| About 01:35 4 Nov | Run 4 evaluated; run 5 starts | auto |
| About 07:00 to 08:00 4 Nov | Run 5 evaluated | auto |
| 4 Nov morning | Handoff 1 to 5: `aggregate_seeds`, `render_docs`, F-items in `RESULTS_FACTS.md`, fill 24 `\pending` + 10 thesis + slide 23 placeholders, rebuild .docx/.pptx, pytest, Jest, build, integrity audit; student approves protocol; commit protocol, then deliverables, push | auto + student |
| 4 Nov afternoon | Time 200 images; feature cache; C1-main (ImageNet). Realistic compute is longer than the roadmap says (review item R5): plan C1 to spill to the evening or 5 Nov | auto |
| 4 to 5 Nov | C1-served, C1-nofilter, C3, C4, C2-lite, macros, figures; C2 go/no-go (validation only) | auto, student approves a C2 night |
| 4 to 5 Nov night, only if GO | C2 one seed + evaluation + post-suppression probe | auto |
| After thesis deliverables are final | Add C1/C3/C4 to thesis chapters 3, 4, 7, 8, 9 and BSPC sections; update `REVIEWER_RISK.md`; add verified references to `references.bib`; condense BSPC | auto drafts, student reviews |
| Before 24 Jan 2027 | EMBC submission (portal opens December 2026); name the reciprocal reviewer | student |
| After thesis submission and review | Scopus / Google Scholar query; choose journal; submit with overlap statement | student |
| 16 Apr 2027 | EMBC decision | - |

---

## 11. Glossary

- **Source (dataset) shortcut.** A model predicts the label using cues that identify the dataset (hospital style, border, markers) because source and label are correlated. Our evidence is indirect; the probe shows availability, not use.
- **Leakage.** Information from test patients reaching training, for example the same patient in both splits or the same image via two archives. We group by patient where known; the caveat is that 4,130 rows are grouped per image.
- **Patient-identity caveat.** Separation by patient is proven only for rows with known identity.
- **Frozen features.** Output of the pretrained network (here 1,024-d global-average-pooled DenseNet-121) used without further training.
- **Linear probe.** A logistic regression trained on frozen features to read out some attribute (source or disease). High accuracy means the information is linearly available.
- **Fourier band / radial band.** The 2-D Fourier transform splits an image by spatial frequency; a radial band keeps frequencies whose distance from the centre lies in a range. Octave bands A to F run from coarse (period 64 pixels and longer) to fine (4 to 2 pixels) on the 224 grid.
- **Band-pass / band-stop.** Keep only a band / remove only a band (the image mean is always kept).
- **Balanced accuracy.** Mean of per-class recalls; chance is 1 / number of classes (1/6 for six sources).
- **Macro-F1.** Unweighted mean of per-class F1, which punishes ignoring small classes.
- **ROC-AUC.** Probability that a random positive scores above a random negative (macro: averaged one-vs-rest).
- **ECE (expected calibration error).** Average gap between confidence and accuracy across confidence bins.
- **Within-source disease macro-F1 (`F1_ws`).** Disease macro-F1 computed inside each source and averaged, so source cues cannot inflate it.
- **SDS (source-dominance score).** Chance-normalised source score minus chance-normalised disease score for a band; positive means source information exceeds disease information there. Descriptive, no threshold.
- **INLP (iterative nullspace projection).** Repeatedly fit a linear classifier for an attribute and project the features onto the space it ignores, until the attribute is no longer linearly predictable.
- **LEACE.** Closed-form linear erasure that makes a concept unreadable by any linear classifier.
- **LOSO (leave-one-source-out).** Train on all sources but one; test on the held-out source.
- **Bootstrap CI.** Resample test images with replacement many times (B = 1,000 here) and read the 2.5th and 97.5th percentiles; it describes test-sampling variation only.
- **Wilson interval.** Confidence interval for a proportion that behaves well for small counts.
- **McNemar test.** Compares two classifiers on the same images using the discordant pairs; used descriptively here.
- **Grad-CAM.** Heat map of which image regions raised the score of the predicted class; illustrative only.
- **Once-only evaluation.** Test and external sets are scored once per final configuration; choices use validation only.
- **Fixed in advance and recorded in the repository.** Our wording for rules written into a committed file before the results they govern exist; not an external registration.
- **GO / NO-GO.** The validation-only rule deciding whether the C2 retraining night is run.
- **Matched-energy random control.** A random band-stop that removes the same spectral energy as the chosen band, to show that the band matters and not just removing content.

---

## 12. Ten likely viva questions with short model answers

1. **What is your contribution if you did not propose a new network?** A leakage-controlled, auditable six-class benchmark and pipeline, evidence that internal metrics overstate external behaviour, and an analysis of where source information lives in the features and how entangled it is with disease (planned C1, C3, C4). I make no state-of-the-art or clinical claim.
2. **Your internal accuracy is 77.4%. Others report above 90%. Why?** Because the split is leakage-controlled and labels are rule-based; I never compare against other papers' numbers on different test sets. The earlier V5 pipeline reported 82.93% and was found to leak (154 of 1,570 test images shared a patient with training) and have 0/80 external specificity.
3. **Does your model use the source as a shortcut?** Not proven. A probe shows source is predictable from the features (93.8% balanced accuracy, chance 16.7%), class and source are confounded, Grad-CAM for two classes falls outside the lungs and Montgomery is poor. That is indirect evidence, consistent with a shortcut.
4. **What is a probe and why does it not prove use?** A linear classifier on frozen features shows information is available; the trained classifier may ignore it. Showing use needs interventions, which is why C3 (erasure) and C2 (suppression) exist.
5. **Why did Montgomery give TB 0/58 and specificity 19/80?** I do not know the cause. Internal TB F1 is 85.7%, so the model can recognise TB inside the data; on a new hospital its scans are mostly called other abnormal classes. Plausible causes (source-specific appearance, domain shift) are untested; Montgomery is small (138) and not untouched.
6. **How do you guarantee no test-set tuning?** Selection used validation only, by a rule written to the repository before run 2's results; test and Montgomery are scored once per run; the novelty protocol fixes thresholds before any result, logs each test and Montgomery access, and every number comes from generated macros. Run 2 was started after run 1's Montgomery results were seen, and I state that.
7. **What does a spectral band tell you if the images were already filtered and resized?** The map describes the cached 224 x 224 images; source differences in resolution and field of view enter through resampling, so I do not claim scanner physics. The no-filter cache is the sensitivity check.
8. **Why INLP and what could go wrong?** It is a known linear erasure method; I use it as a diagnostic, not a new method. If disease accuracy collapses, that quantifies entanglement (partly caused by confounded labels). Linear erasure does not remove non-linear source information.
9. **Why leave-one-source-out with ImageNet features as primary?** The trained backbone saw all six sources, so its features of a held-out source are not independent. ImageNet features have no project training, so the matrix is a clean held-out test; the trained-backbone matrix is labelled accordingly. Held-out numbers carry the patient-identity caveat.
10. **What if C2 does not help?** Then it is reported as a negative result and future work. The GO rule is fixed in advance on validation data (source drop at least 0.05, disease loss at most 0.02, a matched-energy control, C2-lite agreement), a single seed is exploratory, and one run is compared with the seed spread of runs 2 to 4, so no improvement is claimed.

---

# Supervisor review (2026-11-03 night)

Scope: `NOVELTY_ROADMAP.md`, `NOVELTY_PROTOCOL.md`, `lit_search/*` (including `SCOPE_MERGE_REVIEW.md`), `NOVELTY_PLAN.md`, `JOURNALS.md`, `CONFERENCE_VS_JOURNAL.md`, `SUBMISSION_TRACKER.md`, `docs/TRACKER.md`, the HANDOFF, `WRITING_BRIEF.md`, `RESULTS_FACTS.md`, `REVIEWER_RISK.md`, the thesis chapters, the BSPC and EMBC sources, and the generated results. Read-only; no other file was edited and no computation was run. Severity: H = fix before the item is used or committed; M = fix this week; L = cosmetic or note.

## A. What I verified as consistent
- Probe sample sizes in `NOVELTY_PROTOCOL.md` 3.3 (train 1,952, val 1,019, test 1,033) match my sum of the cell counts in 3.2 and `source_probe_v6_build_002.json` (1,952 / 1,033). Forward-pass totals (52,052; 20,895) and the C2-lite sample (4,358) recompute correctly. `c_D = 0.45` recomputes. Duplicate cache arithmetic (15,793 rows, 81 duplicate rows over 36 hashes, 15,748 files) is consistent with the inventory.
- Run 2, run 3 and probe numbers quoted in the HANDOFF match `model_metrics.json`, `results.json`, the run 3 reports (77.1%, 78.5%, 0.954; 54/58, 25/80, 0/58) and the probe JSON.
- The C2 GO rule is identical in `NOVELTY_ROADMAP.md` section 3, `NOVELTY_PROTOCOL.md` section 10 and `SCOPE_MERGE_REVIEW.md` B8 (thresholds 0.05, 0.02, 0.03).
- `NOVELTY_PROTOCOL.md` is uncommitted, section 13 still holds the two `[TO CONFIRM]` entries, and `artifacts/novelty/` does not exist, so nothing is yet "fixed in advance" and nothing violates it. Protocol and roadmap both use the correct wording ("fixed in advance and recorded in the repository"; no "pre-registered" outside prohibitions).
- Run 4 is alive (epoch 9 in progress at 22:45 in `runs/v6r4_densenet121_seed44/train.log`).

## B. Findings (inconsistencies, integrity risks, unverified claims, missing items)

| ID | Sev | Where (file:line) | Problem | Exact fix |
|---|---|---|---|---|
| R1 | H | resolved | Four conflicting venue rankings. | RESOLVED 2026-11-03: `JOURNAL_SCOPE_MATCH.md` is the current analysis (S1: IMU then BSPC; S2: BSPC then IMU), carried in section 8 of this brief and `NOVELTY_ROADMAP.md` section 4; pointer notes added to `JOURNALS.md` and `SUBMISSION_TRACKER.md`. |
| R2 | H | `NOVELTY_ROADMAP.md` (C1 compute) | C1 compute understated (1-3 h). | FIXED in the roadmap 2026-11-03: 93,842 forward passes for C1 (about 2.6 to 7.8 h at 0.1 to 0.3 s per image, unmeasured), time 200 images first. Word-limit item: see `JOURNAL_SCOPE_MATCH.md` section 1 (BSPC about 5,000 words; IMU none found). |
| R3 | H | `docs/TRACKER.md:35` ("30 `\pending` lines (24 in 06_results.tex, 5 in main.tex, 1 in 08_conclusion.tex)") and `:18`; actual: 23 occurrences in `06_results.tex`, 1 in `08_conclusion.tex`, and `main.tex` has only the `\newcommand` and comments (the "30" and "24 in 06_results" are wrong); `slides.md:5` is a comment, only lines 279-280 are placeholders (TRACKER says 3) | Wrong counts will make the fill-in step look incomplete or finished falsely. | Replace `TRACKER.md:35` with: "24 `\pending` occurrences (23 in `06_results.tex`, 1 in `08_conclusion.tex`); 10 `[PENDING` occurrences in thesis (6 seed, 4 ablation; 7 lines); slide 23 has 2". Re-count by occurrence (`grep -o`) after filling: expected 0. |
| R4 | H | `WRITING_BRIEF.md:10` ("final model is chosen ... after run 2 finishes (~17:00)"), `:15` ("today they hold run 1's values"), `:22` ("Single training run per model (seed 42)") | The writers' contract is stale: the final model is v6.1 = run 2, macros hold run 2 values, and runs 3 (done) and 4 to 5 exist. Novelty-extension writers will read it and copy "single run". | Edit lines 10, 15 and 22: final model v6.1 (run 2); macros regenerated; seed spread from runs 2 to 4 and the run 5 ablation are reported, `[PENDING: seed runs]` until aggregated. |
| R5 | H | `NOVELTY_ROADMAP.md:38-39` ("likely 1-3 h"; C1 afternoon, C3/C4/C2-lite evening) against `NOVELTY_PROTOCOL.md:117-121` | Protocol totals 93,842 forward passes for C1 (52,052 + 20,895 + 20,895), plus 31,586 for the feature cache and 8,716 for C2-lite. At the 0.1 to 0.3 s per image of `SCOPE_MERGE_REVIEW.md:90` this is about 2.6 to 7.8 h for C1 and 0.9 to 2.6 h for the cache, not "1-3 h"; the evening plan is optimistic. | Change roadmap row to "measure first; C1-main likely 1.4 to 4.3 h, all C1 stages 2.6 to 7.8 h, feature cache 0.9 to 2.6 h (unmeasured); C1-served and C1-nofilter may move to 5 Nov, 'not run' rule applies". **FIXED in the roadmap 2026-11-03 (see R2).** |
| R6 | M | `NOVELTY_PROTOCOL.md:173` (C chosen by six-class macro-F1 on validation images of the non-held-out sources) | When NIH is held out, Pleural Effusion and Nodule / Mass have no training or validation images, so six-class macro-F1 is undefined or silently zero for them. | Add: "macro-F1 is computed over the labels present in the non-held-out validation images (`labels=` argument fixed)". Must be edited before the commit. **FIXED in the draft protocol 2026-11-03 (section 8, C selection defined on classes present).** |
| R7 | M | `NOVELTY_PROTOCOL.md:86` and `:155` (within-Normal: 4 sources, chance 0.25, cohen excluded) against `RESULTS_FACTS.md:30` and `source_probe_v6_build_002.json` (within-Normal 97.2%, 5 sources, chance 0.20) | Two different "within-Normal" numbers will exist; a reader may think they should match. | Add to protocol 3.3: "this differs from the existing probe (5 sources, chance 0.20); macros `\RSpecWithinNormal...` versus `\RProbeImgWithinNormal`; do not compare directly". **FIXED in the draft protocol (section 3.3 note).** |
| R8 | M | `lit_search/LIT_C1_C2_frequency.md:50-55` (phase versus amplitude split; amplitude-mixing baseline beside C2) against `NOVELTY_ROADMAP.md:20` and `NOVELTY_PROTOCOL.md:30` | Two suggested additions were dropped without a statement; a reviewer may ask. | Add both to the protocol's "Out of scope" line (`:30`) and to the roadmap, "future work". **FIXED in the draft protocol (section 2 out of scope).** |
| R9 | M | `lit_search/LIT_C1_C2_frequency.md:90-132` (BibTeX not added); `docs/manuscripts/shared/references.bib` (52 entries) lacks Halimi, Lu, Zhang 2023, Xu 2021, McIntosh, Kim 2023, Kim 2026, Dack and Dai, Rahbar v1, Bouaziz, Robinson, Yao, Ravfogel 2020/2022, Belrose, Compton; the lit BibTeX key `degrave2021shortcuts` duplicates existing `degrave2021ai` | The paper cannot cite its own closest work; duplicate keys would break the bibliography. | Add only verified entries (opened pages) with notes for abstract-level ones (Halimi, Bouaziz, Rahbar v1); use key `degrave2021ai`; update `REFERENCES_STATUS.md`; do not add arXiv 2412.01876, Ben Ahmed or Rafferty and Rajan until verified. |
| R10 | M | `lit_search/LIT_C1_C2_frequency.md:140` and `LIT_C3_C4_and_VENUES.md:12-13` (not re-opened items) | Unverified or abstract-level items: DeGrave body not read; arXiv 2412.01876 (search snippet); Ben Ahmed 2022; Rafferty and Rajan journal; Dack and Dai year inferred; Kim 2026 first name "Hyoyi" as shown in Europe PMC; Ravfogel ACL 2020 and LEACE authors not opened; Zech year (arXiv page said 2019, PLoS Med 2018). | Keep the "do not cite" list in the writers' brief; the student verifies as in section 9. State Halimi and Bouaziz content "at abstract level" wherever cited. |
| R11 | M | `NOVELTY_PLAN.md:1-3` (superseded) but `:69-71` (working title promising "localisation and suppression"), `:45` (about 30k passes, 1-2 h), `:18` ("exact citation to be verified"), `:57-67` (C3 as "method contribution") | The superseded file still holds claims the later files corrected (C2/C3 as method contributions, time estimate, unresolved citation). It is also modified but not listed as uncommitted in HANDOFF line 9 / `TRACKER.md:20`. | Add a banner at its top: "Do not use for claims or titles; see roadmap and protocol", or move it to `docs/archive/`; add it to the list of uncommitted files in the HANDOFF and TRACKER. |
| R12 | M | `JOURNALS.md:129` (JNTUH R22 requires a published paper; "ask the department") against `TRACKER.md:87` (resolved: no format, no paper required) and HANDOFF line 10 | Contradiction on whether a paper is required for the thesis. | Replace `JOURNALS.md:129` with "Resolved 2026-11-03 by the student: no paper required before dissertation submission". |
| R13 | M | `JOURNALS.md:4` ("Single seed (42) ... model chosen after run 2"); `CONFERENCE_VS_JOURNAL.md:5,14,17,18` ("planned" probe, ablation, 3 seeds); `SUBMISSION_TRACKER.md:7` ("Final model selected after run 2 ... about 17:00") | Stale status. The probe, per-source analysis and run comparison are done; seeds and ablation are running; C1/C3/C4 are not mentioned as journal extensions. | Update statuses: probe done; seeds running (runs 3 to 4); ablation running (run 5); add a row "C1 spectral map, C3 INLP, C4 LOSO: planned, protocol draft" to the comparison table. Keep "planned" for anything not run. |
| R14 | M | `NOVELTY_PROTOCOL.md:205` ("before 22:00") against `NOVELTY_ROADMAP.md:47` and `SCOPE_MERGE_REVIEW.md:147` (same text) and `NOVELTY_ROADMAP.md:36-40` | Minor wording drift; roadmap also says F1 to F6 (`:5`, `:58`) while the review lists F1 to F7 (F7 prior art). | Pick the protocol text as authoritative; note F7 is covered by the lit files. |
| R15 | H | `NOVELTY_PROTOCOL.md:227` ("runs 4 and 5 ... both finish before any novelty analysis starts") | This is a precondition stated as a fact; it is false tonight. | Add an explicit start condition to protocol section 13 or 11: "novelty analysis starts only after `followups.out` shows runs 4 and 5 complete and the protocol commit hash is in section 13". **FIXED in the draft protocol 2026-11-03 (section 13, start condition).** |
| R16 | M | `docs/thesis/content/12_ch3_problem_objectives.md:51-57` (three research questions) against the RQ1 to RQ4 of the novelty plan; `:68` ("Single seed ... not possible in the time available"); `docs/model_card.md:8,51`; `docs/limitations.md:7`; `model_metrics.json:13` (`evidence_level` "no repeated seeds"); `journal_bspc/sections/00_abstract.tex:8`; `conference/sections/abstract.tex:16` ("single run") | After aggregation these statements become partly wrong: seed runs exist. The novelty RQ numbering differs from the thesis. | In the placeholder-fill step reword each to "the served model is one run (seed 42); the seed spread of three runs of the same recipe is [PENDING: seed runs]"; regenerate `evidence_level` through `served_metrics`; EMBC may keep "single run" for its reported result but must not say seeds do not exist. Map thesis RQ numbering to this brief when chapters are updated. |
| R17 | L | `REVIEWER_RISK.md:14`; `docs/presentation/slides.md:244` ("largely disappeared") | `RESULTS_FACTS.md:3` rule (b) says "fell from ... to ...", never "removed" or implied cause; "disappeared with build 2" edges toward cause. | Use "the COVID-19 calls fell from `\RRunOneExtCovidCalls` to `\RExtCovidCalls` of `\RNExternal`, consistent with, not proof of, a removed source cue". |
| R18 | L | `TRACKER.md:78` (EMBC "20 refs") against HANDOFF line 34 ("19 refs") and my count (19 unique cite keys); `TRACKER.md:74` ("53 entries") against 52 `@` entries in `references.bib` | Count drift. | Use 19 (limit 20) and recount the bib entries before quoting. |
| R19 | L | HANDOFF line 9 ("confirm with `git log` whether `871ce5b` is pushed"); `TRACKER.md:20` | Verified: `origin/research/lungai-clean-baseline` is at `871ce5b`. | State "pushed" in both. |
| R20 | L | `TRACKER.md:16` (run 5 ETA 07:00-07:30) against HANDOFF line 23 (07:30-08:00) and `SCOPE_MERGE_REVIEW.md:11` (about 08:00) | Three ETAs. | Use "about 08:00, 4 Nov (upper bound)" until run 4's real end is known. |
| R21 | M | `NOVELTY_ROADMAP.md:22` against `lit_search/LIT_C1_C2_frequency.md:36` | Two slightly different "safe wording" sentences for C1 (the second adds "patient-grouped multi-source benchmark" without the caveat). | Use the roadmap sentence; if "patient-grouped" appears, attach the patient-identity caveat. |
| R22 | M | `JOURNALS.md:46` and `REVIEWER_RISK.md:25-26` | Elsevier "academic thesis" prior-material rule and the overlap statement need the thesis cited in the journal paper; no cover-letter text exists for C1/C3/C4. | Add the thesis citation line and the extended cover-letter list once results exist. |
| R23 | M (missing) | Not in any file | Missing: thesis submission date and viva date; APC funding source; ethics exemption wording; licence record per source; data and code repository plan; named EMBC reviewer; novelty literature query record; Scopus ISSN checks. | See section 9; add each to `TRACKER.md` section 3 as an explicit blocker with owner "student". |
| R24 | M | `NOVELTY_PROTOCOL.md:164,180` (Montgomery scored for C3 and C4) and `:227` | Montgomery accesses total (runs 1 to 5 plus two novelty analyses, plus one if GO) are correct, but the thesis and paper must list them all; `REVIEWER_RISK.md:17` lists only the run 1/2 timeline. | Add one sentence listing all Montgomery accesses to the Methods sections, with numbers from the access log in protocol section 13. |

## C. Integrity risks (summary)
1. Highest: starting any novelty analysis before the protocol is approved and committed (R15), or reading the stale "single seed" and "chosen after run 2" lines as current (R4, R16).
2. C4 model selection is undefined for held-out NIH (R6); fix before the commit.
3. Wording: "disappeared" (R17); shortcut language must stay "indirect / consistent with"; every C4 held-out number needs the patient-identity caveat; trained-backbone C3/C4 numbers never "held-out" or "external".
4. Montgomery is not untouched; list every access (R24).
5. References: only opened pages may be cited; abstract-level preprints (Halimi, Bouaziz, Rahbar v1) are described only at abstract level (R9, R10).
6. Tests: `tests/test_consistency.py` scans tracked active docs but excludes `docs/publication`, `docs/manuscripts`, `docs/thesis`, `docs/presentation`; run the full pytest after documents and macros change, not tonight.

## D. What I need at the next milestone (queue finished)
- The exact end time and `stop_reason` of runs 4 and 5 (from `runs/*/state.json` and `followups.out`), and the evaluation report paths.
- Output of `aggregate_seeds` (per-seed table, mean, sample SD, min, max; run 5 versus run 2) so I can check that ablation wording is non-causal and that "three runs" is called a rough spread.
- Confirmation that the placeholder grep gives 0 occurrences, and the list of new F-items added to `RESULTS_FACTS.md` before any text is filled.
- Whether the student approved the protocol, and the commit hash entered in `NOVELTY_PROTOCOL.md` section 13, before any novelty compute is launched.

## E. Checklist for the seed-placeholder step (after runs 4 and 5 finish)
Do these then, not before (the files are not to be edited tonight):
1. `docs/manuscripts/shared/WRITING_BRIEF.md` lines 10, 15, 22 (R4): final model v6.1 = run 2; macros regenerated; seed spread and ablation reported.
2. Reword "single run / no repeated seeds" (R16): `docs/thesis/content/12_ch3_problem_objectives.md:68`, `17_ch8_validation_discussion.md:58,70`, `docs/model_card.md:8,51`, `docs/limitations.md:7`, `journal_bspc/sections/00_abstract.tex:8`, `07_discussion.tex:53`, `conference/sections/abstract.tex:16`; regenerate `model_metrics.json` `evidence_level` through `served_metrics` (do not hand-edit).
3. Replace "largely disappeared" (R17) in `journal_bspc/REVIEWER_RISK.md:14` and `docs/presentation/slides.md:244`.
4. Recount placeholders by occurrence (R3) expecting 0 afterwards; fix `docs/TRACKER.md` counts (R18) 19 conference references, bib entry count.
5. Add F-items to `RESULTS_FACTS.md` before filling text; ablation wording non-causal (one seed).

## F. BibTeX entries to add later to `docs/manuscripts/shared/references.bib` (R9; not added yet)
Add only entries whose pages were opened, with a `note` for abstract-level ones. Entries and fields are in `lit_search/LIT_C1_C2_frequency.md` (end section): kim2023homogenization, kim2026sfda, rahbar2026site (v1, abstract level), dack2025bias (year inferred, confirm). Use existing key `degrave2021ai`, not `degrave2021shortcuts`. Still to write from the opened pages in `LIT_C1_C2_frequency.md` section 1 and `LIT_C3_C4_and_VENUES.md` section 1: Halimi 2026 (arXiv 2610.29830, abstract level), Lu 2024/2025 (arXiv 2407.11379), Zhang 2023 (Sci Rep 13:12690), Xu 2021 (arXiv 2105.11120), McIntosh 2021 (arXiv 2205.03898), Yao 2019 (arXiv 1904.01638), Compton 2023 (arXiv 2308.04431), Bouaziz and Chebouba 2026 (arXiv 2604.00263, abstract level), Robinson 2022 (PLOS ONE, doi 10.1371/journal.pone.0274098), Ravfogel 2020 (ACL, not re-opened), Belrose 2023 (arXiv 2306.03819, not opened). Do not add arXiv 2412.01876, Ben Ahmed 2022, Rafferty and Rajan until verified.
