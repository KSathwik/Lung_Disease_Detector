# Archive — historical documents (do not cite as current)

Everything under `docs/archive/` predates branch `research/lungai-clean-baseline`. It is kept for traceability only. The current documentation is in `docs/*.md` and the README.

| Folder | Contents | Known problems |
|---|---|---|
| `thesis_drafts/` | M.Tech report (MD/DOCX/PDF), presentations, viva guide, BSPC/IEEE paper drafts (`papers/`) | Claims "leak-free / zero leakage" splits (false: 154 test images leaked). Presents 58/58 Montgomery sensitivity as screening ability (specificity was 0/80). Has 13 benchmark-table cells that differ from the result files. Says "44/44 tests passed" and gives a 30 MB upload limit (the code uses 10 MB). Uses "Model A/D" naming |
| `thesis_drafts/papers/figures`, `submission_package/figures` | Paper figures | Figures 4, 6 and 7 show the **ERM baseline** (76.18%, AUC 0.9600) but are captioned as the proposed model |
| `submission_package/` | Journal submission files | Same claims as the thesis drafts |
| `historical_audits/` | Earlier self-audit and test reports | Declare "verified / submission-ready" states that later audits contradicted. `END_TO_END_APPLICATION_TEST_REPORT.md` is the source of the recorded checkpoint SHA-256 |
| `figures_and_screenshots/` | Diagrams, figures and UI screenshots of earlier versions | Show "Model D", CNN/ResNet comparisons and earlier UI text |
| `thesis_tooling/` | Scripts that generated the above | Hardcode old figures and claims |

The binary documents (DOCX, PDF, PPTX) were not edited. Correct them from [../experiment_results.md](../experiment_results.md), [../model_card.md](../model_card.md) and [../thesis_methodology.md](../thesis_methodology.md) before any submission.

See also [DUPLICATES.md](DUPLICATES.md): byte-identical copies removed on 2026-11-04 and the copy that was kept for each.
