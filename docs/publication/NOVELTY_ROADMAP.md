# Novelty roadmap (fast track)

Status: **verified draft** (2026-11-03). Built from `NOVELTY_PLAN.md` and the reports in `lit_search/`
(`LIT_C1_C2_frequency.md`, `LIT_C3_C4_and_VENUES.md`, `INVENTORY_AND_WORKPLAN.md`), then corrected after the independent
review in `lit_search/SCOPE_MERGE_REVIEW.md` (flags F1-F6, corrections B6, action list B7, C2 rule B8). Nothing below has been run.
M.Tech deliverables (seed/ablation placeholders, report, slides, tests) stay first.

## 1. Contribution after the literature search

| Part | What it is | Literature verdict (searches so far) | Role in the paper |
|---|---|---|---|
| C1 | Per-band probe of source vs disease information in frozen CNN features (radial Fourier octave bands on the 224x224 grid) | No prior chest X-ray study found that does this. Closest: Halimi et al. arXiv 2610.29830 (preprint, abstract-level; spectral failure slices, no source/disease probe); Lu et al. SPIE 2025 (CT/MRI, injected artefacts); Zhang et al. Sci Rep 2023 (contrast/sharpness shortcuts, no bands). Two unopened chest X-ray leads must be checked. | **Main contribution** (measurement) |
| C3 | Linear source erasure (INLP; LEACE as a check) on frozen features, then a retrained linear disease head | Erasure methods are published (ACL 2020, NeurIPS 2023); no chest X-ray application with a disease-head retrain found. Closest: Bouaziz & Chebouba arXiv 2604.00263 (preprint, abstract-level); Trivedi et al. PLOS ONE 2022. One unverified lead. | **Second contribution** (diagnostic: how entangled source and disease are) |
| C4 | Leave-one-source-out matrix with linear heads. **Primary: ImageNet features** (clean held-out). Secondary: the trained backbone, described as "linear-head leave-one-source-out on frozen features" (F2) | Protocol not new (Yao et al. 2019; Compton et al. MLHC 2023). New only as a six-class, leakage-controlled matrix tied to the source probe and to C1/C3. | **Supporting evaluation** (partly addresses "one external cohort") |
| C2-lite | Linear head on ImageNet features of band-stopped images (minutes, no training) | — | Cheap test of whether suppression could help; input to the C2 decision |
| C2 | Band-stop of source-heavy bands chosen from C1, then retraining (one exploratory seed) | Frequency-based domain generalisation exists (FACT/AmpMix, PASTA). Probe-driven band selection on chest X-ray not found. | **Optional**, rule in section 3 |

Lead resolution (2026-11-03, details at the end of both LIT files): the C1 gap still stands, but C1 must not claim frequency-filtered *source classification* as new in general (a natural-image dataset-classification lead, arXiv 2412.01876, is unverified). C2 overlaps with multi-frequency (Laplacian-pyramid) standardisation for multi-institutional chest X-rays (Kim et al., J Med Imaging 10(6):061103, 2023): cite it and frame C2 as *probe-driven* band suppression. C3 is weakened modestly by Rahbar, arXiv 2608.10295 v1 (INLP/ComBat site removal on frozen brain-MRI foundation-model features; cite v1); the chest X-ray, six-source, disease-head-retrain version stands. Background to cite: DeGrave et al., Nat Mach Intell 2021; Dack & Dai, arXiv 2507.07722.

Cheap additions that raise novelty: C1 within the Normal class only; C1 repeated on the no-filter cache (run 5 build) because the main cache is already low-passed; a matched-energy random band-stop control.

Safe claim wording: "To our knowledge, within the searches described, we are not aware of prior work that quantifies, band by band, source versus disease information in CNN features of chest radiographs." No "first", no "state of the art", no "novel method" for C3/C4, no "scanner fingerprint" as a finding (F1).

## 2. Outputs we will have

- `docs/publication/NOVELTY_PROTOCOL.md`: all decision rules, committed before any result.
- `lungai/novelty/`: `spectral.py`, `spectral_map.py`, `features.py`, `inlp.py`, `inlp_eval.py`, `loso.py` (+ `select_bands.py`, `scripts/data/make_bandstop_build.py` only if C2 goes ahead), with synthetic-array unit tests.
- `artifacts/novelty/`: `c1_spectral_map.json` (incl. validation scores), `features_*.npy`, `c3_inlp.json`, `c4_loso.json`, C2-lite results.
- Figures `fig_spectral_map`, `fig_inlp`, `fig_loso_matrix`; macros `\RSpec…`, `\RInlp…`, `\RLoso…` (`\RBs…` if C2) via `render_docs.py`.
- Journal paper: `04_methods`, `06_results`, `07_discussion`; `REVIEWER_RISK.md` updated. Thesis: ch. 4 (method), ch. 7 (results), ch. 8 (threats), added only after the M.Tech deliverables are final, numbers from the same macros. The EMBC paper is not changed.

## 3. Schedule

| When | Work | CPU |
|---|---|---|
| Now (training busy until ~08:00, 4 Nov) | Close the unverified leads (two chest X-ray Laplacian-pyramid papers, DeGrave 2021, the INLP/ComBat lead); write `NOVELTY_PROTOCOL.md` and commit it after the student approves; write `lungai/novelty/*` and tests (run later) | None |
| 4 Nov morning | Handoff steps 1–5: aggregate seeds and ablation, fill placeholders, rebuild .docx/.pptx, pytest + Jest + build, integrity audit, commit, push | Heavy |
| 4 Nov afternoon | Time 200 images first and write the measured seconds per image into the C1 JSON. The protocol's C1 needs 93,842 forward passes (52,052 + 20,895 + 20,895); at the unmeasured 0.1–0.3 s per image that is about 2.6–7.8 h, plus about 0.9–2.6 h for the 15.7k-image feature cache of two extractors, so C1 may spill into the evening or 5 Nov and unfinished stages are reported as "not run"; feature cache → C1 (ImageNet extractor, then served model, then no-filter sensitivity) | Heavy |
| 4 Nov evening | C3 and C4 once each in the logged order; C2-lite; macros and figures; C2 go/no-go | Light |
| 4–5 Nov night (only if GO) | C2 one exploratory seed + evaluation + post-suppression source probe | Heavy |
| After the thesis is submitted | Write the journal subsections, student reviews, then submit | None |

**C2 go/no-go** (validation only; thresholds recorded in the protocol before any C1 number exists). GO only if all hold:
1. a band b with ΔS(b) ≥ 0.05 (CI lower bound > 0) and ΔD(b) ≤ 0.02 (CI upper bound < 0.03), where ΔS = drop in source balanced accuracy and ΔD = drop in within-source disease macro-F1 when b is removed (ImageNet extractor);
2. a matched-energy random band-stop gives a smaller ΔS (control mean below the CI lower bound of ΔS(b));
3. C2-lite: source balanced accuracy falls by ≥ 0.05 with within-source disease macro-F1 loss ≤ 0.02;
4. run 5 and the M.Tech deliverables are finished, and the student approves one night of CPU.
Otherwise NO-GO: C2-lite is reported as exploratory and C2 as future work. A GO gives one exploratory seed, compared with the seed spread of runs 2–4 and with run 5; no improvement claim from one run.

## 4. Venue (student decision)

**Decision 2026-11-04 (student):** target Informatics in Medicine Unlocked, BSPC fallback; reason: diagnostic study after the C2 NO-GO.

**Updated by `JOURNAL_SCOPE_MATCH.md` (journal pages opened 2026-11-03):** for scope S1 (C1+C3+C4, no training) the best fit is **Informatics in Medicine Unlocked** (no stated length cap; gold OA, APC about USD 2,710 before geographical pricing), fallback BSPC. BSPC full papers are "normally about 5,000 words" and expect a new technique, so BSPC fits **S2** (with C2) better, with the manuscript cut to about 5–6.5k words plus highlights and a graphical abstract; fallback IMU. CMPB, PRL, CMIG, AIIM and CBM fit poorly.

Summary of the choice: scope S1 → IMU first, BSPC fallback; scope S2 → BSPC first (subscription route has no fee; slower, ~203 days to acceptance), IMU fallback (fastest, ~114 days; APC is the student's call). Figures are from the Elsevier Insights pages opened on 2026-11-03; re-check live pages before submission. Next options: Intelligent Systems with Applications, Array.
The EMBC 2027 paper stays as is; the journal version must state the overlap and add C1/C3/C4 as substantial new content.

## 5. Risks (from the review, flags F1–F6)

- **F1** Sources differ in resolution and field of view and are all resized to 224; per-band source information partly reflects resampling, not scanner physics. Use the no-filter cache as a sensitivity check.
- **F2** The trained backbone saw all sources, so trained-backbone C3/C4 are not true held-out tests; the ImageNet matrix is primary.
- **F3** Disease and source are confounded (Effusion, Nodule/Mass NIH-only; Pneumonia mostly Kermany); the primary disease score is within-source, with a fixed minimum cell count (Cohen has very few test images).
- **F4** C1/C3/C4 read the test split: fix the configuration on validation, compute each test table once, log the order. C1 does not touch Montgomery; C3 and C4 touch it once each; its earlier use for runs 1–3 is stated.
- **F5** Small paired probe samples (about 150 train / 100 test per source-class cell); fix size and seed; bootstrap over test images; CIs, no p-value hunting.
- **F6** C2 is one seed; Montgomery (138 images, Normal/TB) is a weak judge.
- Also: band-stop may lower internal accuracy without helping externally; INLP may collapse disease accuracy (a finding, not a failure); 8 GB RAM, one heavy job at a time.

## 6. Integrity

Decision rules fixed in advance and recorded in the repository (never "pre-registered"); validation-only selection; internal test and Montgomery once per final configuration in the logged order; numbers only via generated macros; source-probe results show information is *available*, shortcut use stays "indirect / consistent with"; the patient-identity caveat applies to all leakage statements and all C4 held-out numbers; negative results reported as such; only verified references; repository tests that scan wording and macros are run after the docs and code are added.
