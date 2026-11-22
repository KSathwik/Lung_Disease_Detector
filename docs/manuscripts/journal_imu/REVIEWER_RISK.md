# Reviewer risk assessment (IMU submission)

Likely objections from an *Informatics in Medicine Unlocked* reviewer, with status. Severity: High = likely major revision or rejection on its own; Medium = likely revision point; Low = minor. Updated 2026-11-04 after the novelty results (C1, C3, C4) and the C2 NO-GO decision. IMU states no method-novelty requirement (`docs/publication/JOURNAL_SCOPE_MATCH.md`), which removes the main BSPC risk; the items below are what remains.

| # | Objection | Severity | Status / response |
|---|---|---|---|
| 1 | **No new method; analyses are of frozen features with linear methods.** | Medium | Framed as a diagnostic, measurement study. INLP, LEACE and leave-one-source-out are credited to their authors and not claimed. Limitation stated: no nonlinear erasure, no retraining per source, no statement about the served network. |
| 2 | **"Novelty" overclaimed.** | Medium | Only the per-band source-versus-disease localisation is claimed, in the exact hedged wording ("to our knowledge, within the searches described"), and its outcome is a null result. Two near neighbours are preprints read at abstract level only (Halimi et al.; Bouaziz and Chebouba); A reviewer may know work we missed: the search was not exhaustive. |
| 3 | **Indirect shortcut evidence.** Probes show availability, not use. | High to Medium | Stated everywhere as indirect and "consistent with". Not done: leave-one-source-out training of the network, lung-masked inference. Listed as future work. |
| 4 | **Resampling and pre-filter confound for the band map.** | Medium | Stated in Methods and Discussion; the no-filter cache is a sensitivity check (high-band source accuracy lower, so part of the high-band signal is consistent with resampling and filters). A reviewer may ask for a native-resolution or crop-based analysis (not done). |
| 5 | **Trained-backbone results are not held-out** (the backbone saw all sources). | Medium | Labelled "secondary, not held-out" in text, tables and figure captions; ImageNet features are primary. |
| 6 | **Small probe samples; intervals cover test images only; one probe fit.** | Medium | Stated. No p-values; statements require an interval excluding the reference and a consistent pattern. A reviewer may ask for refits under resampling or several seeds. |
| 7 | **INLP stopped by the iteration cap, not the threshold.** "Why only 20?" | Medium | Fixed in the protocol before results; reported plainly. LEACE gives the one-step comparison. More iterations are not run (changing the rule after results is not allowed). |
| 8 | **C4 NIH gap is negative.** | Medium | Explained as a metric artefact (six-class reference predicts NIH-only classes); binary AUC view added. Balanced accuracy below 50% for two-class sources explained. |
| 9 | **C2 not run.** "Why not try it anyway?" | Low to Medium | The rule fixed in advance returned NO-GO on validation numbers; running it would contradict the protocol. Reported as a strength of the design; C2-lite is exploratory. The student could run an exploratory negative-control later, labelled as such (not recommended). |
| 10 | **Patient identity unknown** for TBX11K and COVID-19 Radiography rows; Kermany identities derived. | Medium | Caveat attached to every leakage statement and every held-out number. |
| 11 | **External cohort small, binary and not untouched.** | High to Medium | Stated; access log lists each use. A second external cohort is not available. |
| 12 | **Seeds: three runs; single ablation seed.** | Medium | Reported as rough; no causal ablation claim; C1, C3, C4 are single runs. |
| 13 | **Low absolute internal performance versus papers reporting above 90%.** | Medium | The point of the paper is that such numbers can be inflated; no cross-paper comparison. |
| 14 | **Single architecture.** | Medium | Not done; listed as future work. |
| 15 | **Label noise (NIH), single-label restriction, paediatric Kermany.** | Medium | Stated in Results and limitations. |
| 16 | **Reproducibility: code not public; images not redistributable.** | Medium | Provenance logs, rules, manifests and the access log are in the repository; public release is an author decision (`\confirm`). |
| 17 | **Licences not verified; ethics statement is a draft; AI declaration is a draft.** | Low to Medium | Marked `[AUTHORS TO CONFIRM]` or `[STUDENT TO CONFIRM]`. Fix before submission. |
| 18 | **Conference-version overlap.** | Medium if undisclosed | Disclose and cite; the journal version adds C1, C3, C4, the C2 decision, seeds and ablation. |
| 19 | **Cost: gold open access (APC about USD 2,710 before geographical pricing).** | Not a review risk | Student decision; confirm the price for the student's country. |

## Overall assessment

IMU is the better fit than BSPC for an evaluation-style study without a new method (`JOURNAL_SCOPE_MATCH.md`); acceptance is not assured and the revision requests above are likely. The null and negative results are reported openly. The most likely revision requests are the indirect nature of the shortcut evidence (3), the resampling confound (4), the small external cohort (11) and the single architecture (14). The honest headline (source information spread over all bands and entangled with disease information; linear erasure costs disease information; held-out-source scores drop sharply) is supported by the numbers, but each statement is limited to frozen features, linear methods, one benchmark and one run.
