# Audit findings: thesis, slides, conference paper, journal paper, README and model docs
> **Resolution (2026-11-03 ~19:30):** all 8 MUST-FIX items are resolved. M8 was resolved by a new like-for-like measurement (old threshold 0.83 on the same build-2 test split: 1,494/2,350 rejected vs 218/2,350 at 0.703; `artifacts/app_checks/cxr_gate_rejection_v6_build_002_heuristic083.json`). M1 is resolved with separate Wilson (`TestAccLo/Hi`) and bootstrap (`TestAccBootLo/Hi`) macros. M2 was a real control-character corruption (a carriage return in place of `` in `ef`), repaired at byte level; a scan of all LaTeX, Markdown, BibTeX and Python files found no other control characters. SHOULD-FIX S1–S12 are resolved except where noted in the agents' reports (certificate HoD line left to the department). Seed/ablation placeholders remain until runs 3–5 finish.


Audit date 2026-11-03 (read-only). Scope: (A) `docs/thesis/content/*.md`, (B) `docs/presentation/slides.md`, (C) `docs/manuscripts/conference/`, (D) `docs/manuscripts/journal_bspc/`, (E) `README.md`, `docs/model_card.md`, `docs/limitations.md`, `docs/results/current_model.md`. Ground truth: `results.json`/`results_macros.tex`, `model_metrics.json`, `run_comparison.json`, `source_probe_v6_build_002.json`, `artifacts/app_checks/*.json`, both `v6_build_report.json` files, `v6_external/metrics_external.json`, `runs/*/run_metadata.json`.

Line numbers are per file. Quotes are at most 15 words.

## What was checked and found clean

- Every macro and token resolves: no undefined `\R...` macro in C or D outside comments (`\RMS...` and `\RAbl...` appear only in comments or inside `\pending`). All `{{Token}}` in A and B resolve (the `Gate*` tokens are computed by `docs/thesis/build_report.py` from `artifacts/app_checks`; `{{Key}}` at slides.md:4 is documentation text).
- Every `\cite` and `[@key]` exists in `references.bib` (C 20 keys, D 47 keys, A and B all present). EMBC has exactly 20 distinct keys, so it is at the limit with no slack.
- LaTeX: all `\begin`/`\end` and braces balance in C and D; all `\ref` labels resolve except the typo in M2; all `\includegraphics` files exist as PDF in `docs/figures/` (D finds them through `\graphicspath{{../../figures/}}`, C through `../../figures/`). Thesis `[[fig:..]]`/`[[tab:..]]` references all resolve.
- Numbers: every hard-coded result I could check against the ground-truth files agrees (V5 history 154/1,570/9.8%, 8,478, 1,395/49.7%; build-1 and build-2 class, source and exclusion counts; build-2 patient identity 8,882/2,643/4,130; 545 near-duplicate pairs; caps total 34,993; run-1 external 45 TB flagged, 43/80 normal, 71 COVID-19 calls, AUC 0.730; gate 1,096/1,724, 83/138, 218/2,350, 18/138, 38.7% to 91.0% validation pass, 7/7 synthetic rejected; run 1 phase-2 limit 8 and run 2 limit 14).
- Author order, affiliations, `[TO BE PROVIDED]`/`[TO BE CONFIRMED]` placeholders and `[AUTHORS TO CONFIRM]` on CRediT and approval are correct and visible in A, B, C, D. Model version v6.1 and checkpoint (`densenet121_v6r2_seed42.h5`, short hash 7992b8da10ff) agree in README, model card, current_model.md, thesis and both manuscripts. EMBC contains no probe, seed, ablation, Grad-CAM or gate results.

## MUST-FIX (integrity and correctness)

**M1. The internal-accuracy interval is mislabelled as a bootstrap interval (C and B).** `\RTestAccLo`/`\RTestAccHi` (75.7 to 79.0) are the Wilson interval (`internal_test.metrics.accuracy_ci95` = 0.7567 to 0.7905). The bootstrap interval is 75.8 to 79.1 (`bootstrap_ci95.accuracy`).
- `conference/sections/experiments.tex:3-4`: "accuracy with a 2000-resample bootstrap confidence interval".
- `conference/sections/results.tex:13-14`: "accuracy (bootstrap interval \RTestAccLo--\RTestAccHi\%)".
- `presentation/slides.md:206`: "given with the bootstrap interval for accuracy" (also says macro-F1 is given with an interval; it is not).
- Fix: change to "Wilson 95% interval" (as journal 06_results.tex:23 and thesis ch7:36 already say). In experiments.tex say the bootstrap is computed but not reported, or drop it to save space. Slide note: "95% Wilson interval for accuracy; no interval is shown for macro-F1". Also correct `RESULTS_FACTS.md` F3 ("95% bootstrap CI"), the source of the error.

**M2. Broken cross-reference breaks compilation (D).** `journal_bspc/sections/02_related_work.tex:14`: "reported in Section~ef{sec:res-ablation}". The backslash and "r" are missing. Fix: `Section~\ref{sec:res-ablation}`.

**M3. The external cohort is described as never influencing design, which contradicts the run-2 timeline (D, C, A).**
- `journal_bspc/sections/03_materials.tex:135`: "never used for training, checkpoint selection or any design decision". Run 2 was started after run 1's external results were known (own text at 05_experimental_design.tex:10 and 07_discussion.tex:57; run-1 external evaluated 2026-11-03 03:58 UTC, run-2 training started 04:37 UTC). Fix: "never used for training or checkpoint selection; see Section~\ref{sec:design} for the caveat that run 2 was started after run 1's external results were seen".
- `journal_bspc/sections/04_methods.tex:95`: "Neither the internal test split nor the external cohort influences training or selection." Add the same pointer.
- `conference/sections/data.tex:26`: "never used for training or selection". Add "(see Run selection for a caveat)".
- `thesis/content/06_abstract.md:5`: "never used for training or model selection". The abstract also omits the run-2 timeline (the body has it in ch6:48, ch7:152, ch8:80). Add one clause.

**M4. A source shortcut is stated as established (must be "indirect evidence").** The wording "changed which shortcut appeared" and "removed the COVID-19 shortcut" asserts that a shortcut exists. The probe, per-source and Grad-CAM results only make one plausible.
- `conference/sections/abstract.tex:14-15`: "Adding sources changed which shortcut appeared"; the abstract never says the evidence is indirect.
- `conference/sections/discussion.tex:12-13` (repeats it, although line 11-12 hedges); `conference/sections/conclusion.tex:4`.
- `presentation/slides.md:244` "removed that shortcut", `:248`, `:313`, `:317`.
- `limitations.md:9`: heading "The run-1 COVID-19 shortcut was removed".
- `thesis/content/10_ch1_introduction.md:51`: "a per-source analysis that exposes source shortcuts".
- Fix wording (one consistent form): "run 1's COVID-19 calls on Montgomery largely disappeared when build 2 was used, but the external errors moved to other abnormal classes. This is consistent with, but does not prove, a source shortcut (indirect evidence)". The same sentence in `RESULTS_FACTS.md` F4/F8 should be reworded, as the docs copied it from there.
- Also note that build 2 changed three things at once (COVID-19 Radiography, TBX11K, added NIH images). Attributing the change to "a second COVID-19 source" alone (`conference/sections/abstract.tex:13`, `results.tex:59-60`, `journal 00_abstract.tex` line 6 "A second COVID-19 source largely removed...", `06_results.tex:159`, thesis ch7:183, slides:243-248) is a causal claim without an ablation. Say "build 2, which adds ...", or "coincided with".

**M5. Leakage-free implications for sources with unknown patient identity.** `groups_spanning_splits = 0` proves separation only for rows with known identity (4,130 of 15,655 build-2 rows are unknown).
- `README.md:40`: "so no patient or duplicate crosses splits". Add "for rows with known patient identity" (README:43 states the caveat only later).
- `presentation/slides.md:42`: "patient-grouped splits, no leakage". Fix: "patient-grouped splits; leakage checks pass (identity unknown for two sources)".
- `journal_bspc/highlights.tex:14`: "all leakage checks are zero". Fix (within 85 characters): "Patient- and duplicate-grouped splits; leakage checks zero where identity is known".
- `thesis/content/06_abstract.md:5`: "Leakage checks ... all evaluate to zero." Add the caveat sentence (ch5:98 has it).
- `thesis/content/12_ch3_problem_objectives.md:7` and `:55`: "free of patient and duplicate leakage". Add "known".
- `journal_bspc/sections/04_methods.tex:148`: probe train and test images "are patient- and duplicate-separated". Add "for sources with known patient identity".
- `journal_bspc/sections/07_discussion.tex:5`: "even though the usual leakage paths were closed". Add "for sources with patient identifiers".

**M6. Factual contradiction on patient identifiers (A).** `thesis/content/14_ch5_dataset.md:11`: NIH "is the only source with genuine patient identifiers". Line 88 (and journal Table 1, limitations.md:19) says genuine identifiers exist for NIH and Cohen, with Kermany derived from file names. Fix: "one of two sources with genuine patient identifiers (with Cohen)".

**M7. "Pre-registered" overstates the process.** The selection rule was written in a repository file (`docs/data/run_selection_protocol.md`) while run 2 trained; nothing was registered anywhere.
- `limitations.md:5`; `journal_bspc/sections/06_results.tex:6`, `:222`; `07_discussion.tex:36` ("Pre-register selection rules"); `journal_bspc/REVIEWER_RISK.md:39`.
- Fix: "pre-specified" or "a rule written down before run 2's results existed (not externally registered)". Keep the "run 2 was started after run 1's external results" sentence next to every use.

**M8. Gate before/after rates compared as like-for-like although they come from different test splits.** 63.6% (1,096/1,724) is the build-1 test split; 9.3% (218/2,350) is the build-2 test split; Montgomery (60.1% to 13.0%) is the same set. The journal says so (`07_discussion.tex:47-48`, good) but the following do not:
- `presentation/slides.md:184-185` and notes `:188-189` ("Rejection of real test radiographs fell from 63.6 to 9.3 percent").
- `thesis/content/15_ch6_implementation.md:125` ("compared with ...% before ... honest measure"); `17_ch8_validation_discussion.md:93` ("down from").
- Fix: add "(build-1 test split before, build-2 test split after; Montgomery is the same set)", or report the like-for-like validation pass rate (38.7% to 91.0%, same build-2 validation split, `GatePassOld`/`GatePassNew`) as the main comparison. Note that `RESULTS_FACTS.md` F7 and the audit brief label 63.6% as "build 2 test", which is not true.

## SHOULD-FIX (consistency and clarity)

**S1. Hard-coded gate numbers; add macros/tokens.** Values 63.6, 9.3, 60.1, 13.0, 0.703, 0.83 are typed in `presentation/slides.md:184-189`, `limitations.md:33`, `journal_bspc/sections/07_discussion.tex:41,43,45` (1,096/1,724, 83/138, 218/2,350, 18/138, 38.7%, 91.0%). They are correct today. The thesis already uses `Gate*` tokens (computed in `build_report.py`, not in `results.json`). Fix: have `render_docs` emit `Gate*` keys into `results.json` and `\RGate...` macros, then use them in B and D. Related: `thesis/content/30_appendices.md:44` types 0.83 and 0.703.

**S2. EMBC carries journal material (CONFERENCE_VS_JOURNAL.md section 1).** `conference/sections/results.tex:2-10` ("Run selection") gives the full run 1 versus run 2 comparison with macro-F1, AUC and exact McNemar p. The differentiation note assigns the run comparison to the journal and "single selected run" to EMBC. The V5 audit is "at most a brief mention" but takes a paragraph (`introduction.tex:9-12`) plus abstract lines 7-8. Fix: cut Run selection to two sentences (rule, shared-sources caveat, run 2 started after run 1's external results), move McNemar and the common-set table to the journal, and shorten the V5 paragraph. This also helps the 4-page limit.

**S3. Fill the `\confirm` on the run-2 timeline (D).** `journal_bspc/sections/05_experimental_design.tex:10` ("state the actual timeline of the decision to train run~2"). The timeline is in the project record: run-1 external evaluation 2026-11-03 03:58 UTC (`metrics_external.json`), run-2 training 04:37 UTC (`current_model.md`), selection protocol written about 12:00 (`run_selection_protocol.md`). Fix: write it in and remove the `\confirm` once the authors agree.

**S4. Slide statement about "best separated" classes is wrong for ROC (B).** `presentation/slides.md:214` and note `:215`: "Best separated: COVID-19, Tuberculosis, Pneumonia" on the ROC slide. ROC-AUC order is COVID-19 0.994, TB 0.994, Pleural Effusion 0.954, Pneumonia 0.949 (`\RAUCEffusion` greater than `\RAUCPneumonia`). Fix: "Highest ROC-AUC: COVID-19 and Tuberculosis; lowest: Nodule / Mass (0.904) then Normal (0.937)". The F1-based ordering on slide 229 is correct.

**S5. Run-selection description missing in the model docs (E).** `limitations.md` omits that run 2 was started after run 1's external results (add to line 9-11). `docs/model_card.md` and `README.md` describe v6.1 but never mention run 1, the run-selection rule, the shared-sources caveat or the source probe. `model_card.md:41` and `:58` still say checkpoints "v5.0 and v6.0" and "fixed before V6 training" (v6.1 is served). Fix: add a short "Run selection" paragraph with the shared-sources and timeline caveats, and mention v6.1.

**S6. Journal Table 1 states licences flatly; the thesis marks them unverified (D versus A).** `journal_bspc/sections/03_materials.tex:21,24` "CC BY 4.0" for Kermany and TBX11K; thesis ch5 Table "tab:licences" marks both "Secondary; confirm on the publisher page" (TBX11K: "according to third-party listings"). Only the Radiography row has a `\confirm`. Fix: add `\confirm` to Kermany, TBX11K and the NIH wording, or cite the terms page.

**S7. Abstract and front-matter length.**
- `journal_bspc/sections/00_abstract.tex`: about 258 whitespace-separated words including the section labels; BSPC allows about 250 and the brief says "under 250". Count after compiling, then trim (for example the Methods sentence on the probe).
- `thesis/content/06_abstract.md`: 738 words; the university template notes (REPORT_TEMPLATE_FINDINGS.md line 20) give at most 500. Shorten paragraph 4 (result list), which duplicates chapter 7.

**S8. Uncited generalisations about other papers' results.**
- `thesis/content/11_ch2_literature_review.md:43`: "Many of these studies report very high accuracy, often above 95%." (no citation; it is a statement about other papers' numbers).
- `thesis/content/10_ch1_introduction.md:13`: "a large part of them cannot be taken at face value".
- `presentation/slides.md:69` ("External evaluation is rare"), `:71` ("Few projects release an auditable pipeline"), `:70` ("seldom tested").
- Fix: cite `yu2022external`, `roberts2021common`, `kapoor2023leakage` where they support it, or soften to "some".

**S9. Unreferenced tables in the thesis (A).** `thesis/content/30_appendices.md:5,21,93,106,143` define `tab:api`, `tab:errors`, `tab:tests`, `tab:coverage`, `tab:claim`, but no `[[tab:..]]` points to them. Fix: add one sentence before each (or from the chapters: ch6 testing paragraph for `tab:tests`/`tab:coverage`, ch4 or ch6 for `tab:api`).

**S10. Appendix and chapter text that reads as final for build 1 (A).** `thesis/content/15_ch6_implementation.md:44` ("an epoch on 8,054 training images takes about 13 to 14 minutes ... ten-epoch run fits within one night") is run 1/build 1. Run 2 epochs took about 1,190 to 1,560 s (`current_model.md`). Fix: name the build and give the run-2 figure.

**S11. Title-only references (D, A).** No result is attributed to a title-only key, but statements go slightly beyond the titles: `journal_bspc/sections/02_related_work.tex:8` ("recurrent topic in this journal ... 'Clever Hans' behaviour in COVID-19 detection", keys `nayak2021application`, `iqbal2023tuberculosis`, `amin2024automated`, `arias2024analysis`) and `:11` (`pooch2020can`, `cohen2020limits`). Thesis ch2:71 and :77 admit "cited by title only", which is acceptable. Fix: read the abstracts before submission, or shorten line 8 to "Related studies have appeared in this journal [..]".

**S12. Conference is exactly at the 20-reference limit** (`conference/main.tex` cites 20 keys), so any added citation breaks the rule. Page count (4+1) cannot be checked without a compiler; the estimated length is the checklist's own estimate.

## MINOR (style)

- Missing full stops: `thesis/content/16_ch7_experiments_results.md:48` ("this was not tested"); `14_ch5_dataset.md:102` ("common validation set The final model"); `17_ch8_validation_discussion.md:80` ("fixed beforehand"). Garbled sentence: `18_ch9_conclusion.md:26` ("evidence about internal versus external behaviour internal metrics of ...").
- `thesis/content/02_certificate.md:10-11`: the supervisor is already "Principal and Head of the Department (CSE & IT)", but a second "Head of the Department" signature line follows. Confirm with the department whether both lines are wanted.
- Abbreviations (`07_abbreviations.md`): ERM, DCT, SPA, OvR, LAB, WHO, CSV, AP/PA, IEEE, UI, SQL, ML are listed but never used in the body; MD5, ReLU, GAP, SQLite, CORAL are used but missing. The abstract uses CLAHE, ROC-AUC and Grad-CAM before the list; CXR is only expanded in the list (not at first use in ch1).
- "30 layers" versus "17 weighted layers": methods say the last 30 backbone layers are unfrozen (config), while the probe discussion says the last 17 weighted layers changed (journal 06_results.tex:119, thesis ch7:131, slides 273). Add "(30 layers, 17 of which carry weights)" once.
- `artifacts/app_checks/cxr_gate_rejection_v6_build_002_heuristic0703.json` records `"threshold": 0.83` although the counts (218 and 18) correspond to 0.703. Metadata only; I could not edit it. Consider regenerating or noting it.
- `journal_bspc/sections/07_discussion.tex:41`: "from about 34% of Shenzhen images" is 33.3% (0.333). Say "about 33%".
- `presentation/slides.md:5` header comment says demo-screenshot placeholders remain; slide 285 now uses real screenshots. `slides.md:31` "screening aids feasible" (mild clinical framing). Suggest "decision-support research".
- `limitations.md:9`: run-1 numbers (71 of 138; 3) are typed; they match the ground truth but could cite `metrics_external.json`.
- Names: the run is `v6r2_densenet121_seed42`, the checkpoint file is `densenet121_v6r2_seed42.h5`. Both are correct but easy to confuse; state once which is which (`\RCmpSelected` prints the run name).
- `journal_bspc/main.tex:81`: `\ead{[TO BE PROVIDED]}` may produce a malformed mail link in elsarticle; consider a plain `\cortext` note until the address is known.

## Counts

| Severity | Findings |
|---|---|
| MUST-FIX | 8 (M1 to M8) |
| SHOULD-FIX | 12 (S1 to S12) |
| MINOR | 10 bullet groups |

Not checked (no compiler, not allowed to run heavy tools): actual page count of the EMBC PDF, final abstract word counts after macro expansion, LaTeX warnings beyond the static checks above, and `.docx`/`.pptx` rendering.
