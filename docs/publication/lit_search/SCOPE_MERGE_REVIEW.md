# Scope merge review: novelty extension (C1-C4) versus the current repository

Independent verifier, 2026-11-03 (late evening). Read-only review: nothing was run and no other file was edited. Statements about code and data come from opening the files named (manifest counts were computed from `manifest_v6.csv`). Time figures are estimates that must be measured. Part A is Phase 1 (scope merge); Part B ("Verification and decision") checks `NOVELTY_ROADMAP.md` against the literature files, the repository and Part A.

# Part A. Scope merge

## A1. Constraints that decide everything

| Constraint | Consequence |
|---|---|
| CPU only, 8 GB RAM; runs 4 and 5 train until about 08:00 on 4 Nov (`scripts/pipeline/run_followups.py`) | Until then no full-speed TensorFlow extraction, no pytest, no npm. Only writing protocol, code and text. |
| M.Tech deliverables first (handoff steps 1-5: aggregate seeds, fill placeholders, rebuild report and slides, verification, audit, push) | Novelty compute cannot start before about the afternoon of 4 Nov. |
| Integrity rules (`WRITING_BRIEF.md` section 3, HANDOFF) | Decision rules recorded in the repository before results; test and Montgomery once per final configuration; numbers only through generated macros; "fixed in advance and recorded in the repository", never "pre-registered"; shortcut evidence "indirect / consistent with". |

## A2. What the repository can already do (verified)

Data facts. `data/derived/v6_build_002/manifest_v6.csv`: train 10,957 / val 2,348 / test 2,350 / external 138; one `cache/<raw_sha256>.npy` (224x224x3 uint8) per image; columns `source_id`, `lungai_label`, `split`, `patient_group`. The cache is already after blur, CLAHE, Gaussian low-pass (sigma 1.0) and resize (`lungai/preprocessing.py`, header order; `clean_image`, `gaussian_lowpass`, `resize_display`). `data/derived/v6_build_002_no_filters/` (decode, RGB, Lanczos-4 resize only) exists and is the only less-filtered cache.

Test split, source by class (computed from the manifest): Normal from 5 sources (cohen 4, kermany 81, nih 183, shenzhen 49, tbx11k 150); Pneumonia from 3 (cohen 29, kermany 317, nih 48); TB from 3 (cohen 1, shenzhen 50, tbx11k 94); COVID-19 from 2 (cohen 77, covid_radiography 374); Effusion and Nodule/Mass from NIH only. Montgomery: Normal 80, TB 58.

| Part | Already available (path, function) | Missing | Verdict |
|---|---|---|---|
| C1 spectral map | `lungai/evaluation/source_probe.py`: `_sample` (per source and class), `_features` (compiled DenseNet GAP extractor, padded batches via `train_v6.cached_batch`), `probe` (StandardScaler + multinomial LogisticRegression, balanced accuracy, within-class source accuracy); ImageNet and checkpoint backbones built in `main`. `train_v6.load_split`, `metrics.bootstrap_ci`. | Radial FFT band masks; a cached-batch variant that filters the uint8 array before normalisation; a disease probe (existing `probe` predicts source only); within-source disease macro-F1; SDS index; bootstrap CIs for probes; JSON, figure, macros. About 200-300 lines. | Feasible, no training, about 20-40k forward passes. |
| C2 band-stop + retrain | `scripts/data/make_ablation_build.py` (new build dir, manifest copied unchanged, `VARIANTS` dict); `train_v6.train`; `run_followups.py`; `aggregate_seeds.py`; `compare_runs.py`; `source_probe --checkpoint` for the post-training probe. | Band-selection rule; band-stop variant (works on the existing cache, minutes); queue entry. | Code easy; cost is the problem: one run 6-8 h, cannot start before run 5 ends, 2-3 seeds = 2-3 nights. |
| C3 INLP | `source_probe._features/probe`, `densenet.load_checkpoint`, `train_v6.predict_cached`. Head layout (`densenet.py::build_model`): backbone, GlobalAveragePooling2D, BatchNormalization, Dense(features, relu), Dropout, Dense(softmax). | INLP loop, projection, head retrain, feature cache, evaluation. Layer to erase (GAP 1024-d or the Dense features layer) must be fixed in the protocol. | Feasible, minutes once features exist. |
| C4 leave-one-source-out | Manifest columns, sklearn heads, `metrics.py` (`confusion`, `metrics_from_confusion`, `bootstrap_ci`). | Everything else. | Feasible, minutes; see flag F2. |
| Macros | `render_docs.py::flat_results` already reads `artifacts/data_quality/source_probe_<build>.json` (about lines 220-234, `\RProbe...`); `write_shared` writes `results.json` and `results_macros.tex` (escapes `_`). | A `novelty_flat()` helper called from `flat_results` that reads `artifacts/novelty/*.json` and emits `\RSpec...`, `\RInlp...`, `\RLoso...`, `\RBs...`. | Easy; follow the probe pattern. |
| Figures | `lungai/evaluation/make_figures.py` (one function per figure, png and pdf into `docs/figures/`). | `fig_spectral_map`, `fig_inlp`, `fig_loso_matrix`. | Easy. |

There is no `lungai/novelty/` yet. New code needs synthetic-array tests, run after training ends.

### Flags found while verifying (several are not in NOVELTY_PLAN.md or the roadmap)

- **F1. Resampling and resolution confound C1.** Sources differ in native resolution and field of view (paediatric Kermany JPEGs, NIH 1024 px, Shenzhen large); all are low-passed and resized to 224x224, so per-band source information partly reflects how different pixel grids were resampled. The band map is not a map of scanner physics. Do not write "scanner fingerprint" as a finding; use the no-filters cache as sensitivity check.
- **F2. C3/C4 on the trained backbone are not true held-out-source tests.** The run-2 backbone was trained on images of all six sources, so features of a "held-out" source were already shaped by it. Valid leave-one-source-out needs ImageNet features (no project training) or retraining per source (unaffordable). Call the trained-backbone version "linear-head leave-one-source-out on frozen features"; make the ImageNet matrix the primary one.
- **F3. Disease probes are confounded by source.** Effusion and Nodule/Mass are NIH-only; Pneumonia is mostly Kermany. A disease macro-F1 per band can be bought with source cues. Primary disease measure: within-source (sources with two or more classes: kermany Normal/Pneumonia; shenzhen and tbx11k Normal/TB; nih several); overall score secondary. Exclude cells below a fixed minimum count (cohen has 1 test TB, 4 Normal).
- **F4. "Test once" versus C1/C3/C4.** All three read the test split. Define the final analysis configuration once on validation, compute each test table once, log the order in the protocol. Montgomery was already seen for runs 1-3; C3/C4 must say so. C1 does not need Montgomery.
- **F5. Small probe samples.** `source_probe.py` defaults: 150 train and 100 test per (source, class), 1,952 train / 1,033 test images. Use the same sample for every band (paired), bootstrap over test images, fix size and seed in the protocol.
- **F6. C2 is one seed.** Runs 2-4 give the seed spread; one C2 run is exploratory. Montgomery (138, Normal/TB) is a weak judge.
- **F7. Prior art.** Web search only (no Scopus); Halimi et al. (arXiv, Oct 2026) is the nearest and must be cited; C4 is not novel as a protocol (Yao et al. 2019); C2 reads as an extension of filtering and Fourier augmentation.

## A3. Where each part merges

The EMBC conference paper (`docs/manuscripts/conference/`) stays untouched: no novelty content, figure, macro or reference there.

| Part | Thesis (`docs/thesis/content/`, `build_report.py`) | BSPC (`docs/manuscripts/journal_bspc/sections/`) | Figure | Macros |
|---|---|---|---|---|
| Protocol | `13_ch4_methodology.md`: new section after "Evaluation methodology"; `12_ch3_problem_objectives.md`: one research question | `04_methods.tex`: subsection after "Source-identifiability probe" | none | none |
| C1 | `16_ch7_experiments_results.md`: after "Source-identifiability probe" | `06_results.tex`: after Source-identifiability probe | `fig_spectral_map` | `\RSpec...` |
| C3 | `16_ch7`, same new section | `06_results.tex`, after C1 | `fig_inlp` | `\RInlp...` |
| C4 | `16_ch7`; `17_ch8` Threats to validity (F2) | `06_results.tex`; `07_discussion.tex` (benchmark design, Limitations) | `fig_loso_matrix` | `\RLoso...` |
| C2 (if run) | `16_ch7` after "Preprocessing ablation" | `06_results.tex` after Preprocessing ablation | extra table column, post-suppression probe | `\RBs...` |
| Framing | `18_ch9_conclusion.md` | `01_introduction.tex` (currently "We do not propose a new network architecture": add the analysis contribution), `00_abstract.tex`, `08_conclusion.tex`, `highlights.tex`, `REVIEWER_RISK.md` items 2, 4, 5, 6 | | |
| Related work, facts | `11_ch2_literature_review.md` | `02_related_work.tex`; entries in `references.bib` and `REFERENCES_STATUS.md`; F-items in `RESULTS_FACTS.md` first | | |
| New files | `docs/publication/NOVELTY_PROTOCOL.md`, `lungai/novelty/`, `artifacts/novelty/*.json` | | | `render_docs.py` hook |

The BSPC text is already about 13.2k words (tracker: "likely above guidance"). Three new subsections require compressing or moving material (per-class table, run comparison) to supplementary.

## A4. Out of scope

More than one C2 run; a second backbone; retraining per held-out source; multi-seed C2; a second external cohort (VinDr-CXR and JSRT are blocked on the student); any state-of-the-art or clinical claim or comparison with other papers' numbers; gate, backend or frontend changes; segmentation or lung-masked inference; serving any new model.

## A5. Minimum viable novelty set

C1 (within-source disease probing, within-Normal source probe, no-filters sensitivity) + C3 (INLP) + C4 (ImageNet-feature matrix primary). All on cached features, no training: about 1-1.5 days of code writing (can start now) and half a day of compute after 08:00 on 4 Nov. It addresses REVIEWER_RISK items 2, 4 (partly) and 6 (partly). C2 stays optional and exploratory.

---

# Part B. Verification and decision

Inputs checked: `NOVELTY_ROADMAP.md`, `LIT_C1_C2_frequency.md`, `LIT_C3_C4_and_VENUES.md`, `INVENTORY_AND_WORKPLAN.md`, Part A. I did not re-open the cited papers or journal pages myself (the verifier had no web step in this run); statements below about papers are checked against what the literature files say they opened, and each is marked as such.

## B1. Verdict on the roadmap

Sound overall: roles of C1-C4 are honest (C1 main, C3 second, C4 supporting, C2 optional), the claim wording is safe, the schedule respects the training queue and the M.Tech priority. It needs the corrections in B6 before the protocol is written; none changes the plan's direction.

## B2. Novelty claims against the cited papers

| Claim in roadmap | Check | Result |
|---|---|---|
| C1 "no prior chest X-ray study" | The lit file opened Halimi 2026 (radial spectrum, unsupervised clusters, no source or disease probes), Lu 2024 (CT/MRI, injected artefacts), Zhang 2023 (contrast and sharpness at image level). Search was general web, not Scopus; two Laplacian-pyramid chest X-ray leads and DeGrave 2021 are unopened. | Acceptable only as "to our knowledge, within the searches described". Halimi is two weeks old and under review: re-check arXiv listing the day of writing. Pre-submission Scopus and Google Scholar query is mandatory (student or an agent with access). |
| C3 not a new method; application only | INLP, RLACE, LEACE are published; Bouaziz and Chebouba 2026 (frozen features, linear site probe, linear head, CXR, adversarial) and Robinson 2022 (feature disentanglement) are closest; one unidentified lead remains. | Roadmap is correct to call C3 a diagnostic. Bouaziz is an arXiv preprint (abstract only opened): do not describe its content beyond the abstract. |
| C4 protocol not new | Yao et al. 2019 does leave-one-domain-out on CXR. | Correct. Roadmap's "answers only one external cohort" is only partly true because of F2. |
| C2 probe-driven band choice not found | FACT/AmpMix, PASTA, He 2024 and Lu 2024 are adjacent. | Correct and the weakest claim. A BSPC reviewer will read it as filtering plus Fourier augmentation. |

Two additions the roadmap lacks: Halimi's spectral clusters may overlap with C1's bands, so cite them and state the difference (supervised per-band probes of source and disease versus unsupervised failure slices); and a source-identifiable result by itself is established (Zech 2018, Compton 2023), so the new content is only the band-wise localisation.

## B3. Feasibility and time on CPU and 8 GB

- C1: 20-40k passes (about 1,500 train and 1,000 test samples times 8 band variants times 1-2 extractors). On this CPU DenseNet-121 at 224 px is likely 0.1-0.3 s per image single-stream, so 1-3 h alone; the roadmap's 1-2 h is optimistic. Measure on 200 images first and write the measured figure into the protocol.
- Cumulative low-pass, high-pass, band-pass and (for C2) band-stop doubles or triples the variant count. Fix the variant list in the protocol and cap it (about 6 octave band-pass plus 6 band-stop = 12 variants is already 2.5x the plan's 8).
- Feature cache for all 15.7k images: 1-2 h, fine. The roadmap puts feature cache, C1, C3 and C4 into one afternoon after the heavy morning (pytest, Jest, npm build, audit, rebuilds): tight. Expect C1 to spill to the evening.
- C2: one run 6-8 h plus 30 min evaluation; cannot overlap another heavy job. The roadmap's "4-5 Nov night" is feasible only if C1 val scores are ready by about 18:00 on 4 Nov. Three seeds (needed for any claim) are 2-3 nights: out of scope.
- Writing 5-6 Nov: three new subsections plus compressing a 13k-word paper is more than two days of student review; plan the BSPC submission for the following week, not 6 Nov.

## B4. Integrity compliance

Compliant on: rules recorded before results, validation-only selection, macros, "consistent with" wording, negative results reported, verified references only. Gaps to close in the protocol: F2 (trained-backbone LOSO wording), F3 (within-source disease score as the primary measure), F4 (list each test and Montgomery access and its order; C1 has no Montgomery access), F5 (fixed sample and seed), minimum cell size, multiple-comparison handling (CIs, no p-value hunting), and that thresholds in section B5 are fixed before any C1 number exists. Also run `pytest` after adding docs and code because repository tests scan for banned wording and macro use (the tracker notes a literature-names exclusion in the model-name check); do it after 08:00.

## B5. Venue

The literature files (publisher "Insights" pages as reported by their authors) give: BSPC, subscription route no fee, 10 / 66 / 203 days, CiteScore 13.0; Informatics in Medicine Unlocked, gold OA about USD 2,710, 2 / 36 / 114 days; ISWA and CMPB as alternatives. Assessment:
- BSPC remains the best topical fit for a spectral-band analysis and costs nothing on the subscription route. Risk: reviewers want method novelty; the analysis-first framing of C1/C3/C4 plus the existing audit is only a moderate fit, and `REVIEWER_RISK.md` already predicts major revision. Realistic time to a decision is 2-3 months, to acceptance 6+ months.
- IMU is the realistic fast option but has an APC the student must be able to pay (India amount not shown): student decision.
- The roadmap's word limits for BSPC (about 5,000 words) were carried from `JOURNALS.md` and not re-checked today; check the live Guide for Authors before condensing, because the manuscript is 13k words.
- Recommended decision for the student: submit to BSPC first (no fee), name IMU as the fallback if the time or fee situation changes. Do not submit anywhere before the thesis is final. The EMBC paper is unchanged; the journal version must disclose overlap.

## B6. Corrections the roadmap needs (NOVELTY_ROADMAP.md is not edited)

1. Add flags F1-F6 to the roadmap's risks, especially F2 (C4 on the trained backbone is not held-out) and F3 (disease probe confounded by source; primary disease score within-source).
2. Section 1, C4 row: state that the ImageNet-feature matrix is the clean one and the trained-backbone matrix is "linear-head leave-one-source-out on frozen features".
3. Section 3, C1 time: "1-2 h" is unmeasured; say "measure first; likely 1-3 h" and fix the variant count.
4. Section 3: add the C2-lite step (linear head on ImageNet features of band-stopped images, minutes) before the C2 go decision; the roadmap has no cheap test of whether suppression helps.
5. Section 3 C2 rule: replace "clearly above ... threshold set in the protocol" by the concrete rule below (thresholds recorded before C1 results).
6. Section 3 last row: "5-6 Nov write sections" is not realistic for the review load; set a BSPC submission target after the thesis is submitted and the student has reviewed the text.
7. Section 2: the thesis chapter placement is "ch. 4 (method) and ch. 7 (results), ch. 8 (threats)" rather than "future-work/analysis chapter"; add the thesis only after the M.Tech deliverables are final and keep its numbers from the same macros.
8. Section 4: say the venue data were not re-verified by the verifier; check live pages and the BSPC word limit; note the APC is the student's decision.
9. Section 1: add that Halimi 2026 and Bouaziz 2026 are arXiv preprints, opened by the helper at abstract level only.
10. Section 5: add "patient-identity caveat applies to all C4 held-out numbers" and "Montgomery: no access in C1; C3 and C4 once each, in the order logged".

## B7. Ordered action list

Owner: A = automatic (agent does it without the student), S = needs the student's decision or hands.

| # | When | Action | Owner | Inputs | Output |
|---|---|---|---|---|---|
| 1 | Now | Decide scope: MVNS (C1+C3+C4) confirmed; C2 conditional on the rule in B8 | S | This file, `NOVELTY_PLAN.md` section 9 | One-line decision in `PROJECT_EXECUTION_STATUS.md` |
| 2 | Now (no CPU) | Close the unverified literature leads (two Laplacian-pyramid chest X-ray papers, DeGrave 2021, the unidentified INLP/ComBat lead); add only verified entries to `references.bib` and `REFERENCES_STATUS.md` | A | `LIT_C1_C2_frequency.md` section 2, `LIT_C3_C4_and_VENUES.md` | Updated lit files and bib |
| 3 | Now (no CPU) | Write `docs/publication/NOVELTY_PROTOCOL.md` with every fixed item from A2 flags and B4 (band edges on the 224 grid, variant list, SDS definition on within-source disease score, sample sizes and seed, minimum cell count, INLP layer and iteration cap, LOSO inclusion rule, evaluation order including test and Montgomery access log, C2 rule from B8, hypotheses, how negative results are reported); commit it before any result | A writes, S approves commit and push (branch `research/lungai-clean-baseline` only) | A2 flags | Committed protocol |
| 4 | Now (no CPU) | Write `lungai/novelty/` (`spectral.py`, `spectral_map.py`, `features.py`, `inlp.py`, `inlp_eval.py`, `loso.py`) and synthetic-array tests; do not run TensorFlow or pytest | A | Protocol, `source_probe.py` | Code, tests (unrun) |
| 5 | After 08:00 on 4 Nov | M.Tech handoff steps 1-5 (seeds, ablation, placeholders, rebuild, pytest, Jest, build, audit, commit, push) | A, S confirms | `PROJECT_EXECUTION_STATUS.md` | Final deliverables |
| 6 | 4 Nov afternoon | Measure speed on 200 images; run feature cache; run C1 (ImageNet extractor first, then served model, then no-filters sensitivity); write `artifacts/novelty/c1_spectral_map.json` including validation scores | A | Protocol, code | C1 JSON |
| 7 | 4 Nov evening | Run C3 and C4 once each in the logged order; C2-lite on ImageNet features; render macros and figures (`render_docs.py` hook, `make_figures.py`) | A | Feature cache | `c3_inlp.json`, `c4_loso.json`, figures, macros |
| 8 | 4 Nov evening | C2 go/no-go by the rule in B8 | A computes, S approves a night of CPU | C1 validation JSON, C2-lite | Decision logged in the protocol |
| 9 | 4-5 Nov night, only if go | Build band-stop cache, train one seed, evaluate (test and Montgomery once), post-suppression source probe | A | `make_bandstop_build.py`, `run_followups.py` | `v6_r6bs_*` reports |
| 10 | After 4 Nov | Add F-items to `RESULTS_FACTS.md`, then write subsections (thesis A3, BSPC A3), update `REVIEWER_RISK.md`, compress BSPC | A drafts, S reviews | Macros, figures | Revised manuscript |
| 11 | Before submission | Scopus and Google Scholar novelty query; confirm BSPC limits and fee; choose venue; authors approve; contacts, corresponding author, ethics, licences | S | Lit files | Submission package |

## B8. Go/no-go rule for C2 (recommended text for the protocol; thresholds recorded before any C1 number exists)

All quantities use validation data only, the paired probe sample of C1, and 95% bootstrap CIs over validation images. For each band b let the band-stop image be "all frequencies except b". Define ΔS(b) = source balanced accuracy on the full image minus on the band-stop image, and ΔD(b) = within-source disease macro-F1 on the full image minus on the band-stop image (ImageNet extractor; the served extractor is a sensitivity check only).

GO to full C2 only if all hold:
1. There is a band b with ΔS(b) >= 0.05 and the lower CI bound of ΔS(b) above 0, and ΔD(b) <= 0.02 with the upper CI bound of ΔD(b) below 0.03.
2. The matched-energy random band-stop control (same energy removed) shows ΔS smaller than for b (control mean below the CI lower bound of ΔS(b)), so the selected band matters.
3. C2-lite (linear head on ImageNet features of band-stopped images) shows a drop in source balanced accuracy of at least 0.05 and no fall in within-source disease macro-F1 greater than 0.02 on validation.
4. Run 5 and all M.Tech deliverables (steps 5 above) are finished, and the student approves one night of CPU time before 22:00.

Otherwise NO-GO: report C1, C3, C4 (and C2-lite as an exploratory analysis) and list C2 as future work. A GO yields one exploratory seed: compare with the seed spread of runs 2-4 and with run 5, say "exploratory, single seed", and never claim improvement from one run. Test and Montgomery are touched once for the C2 model. The numeric thresholds above are recommendations; the student or main session may change them before the protocol is committed, not after.
