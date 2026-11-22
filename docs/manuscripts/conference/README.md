# Conference manuscript (IEEEtran) for IEEE EMBC 2027

Target: IEEE EMBC 2027 (49th Annual International Conference of the IEEE EMBS, Singapore, 11 to 15 July 2027).
Full contributed paper deadline: 24 January 2027, 11:59 PM AoE (source: `docs/publication/CONFERENCES.md`; re-check https://embc.embs.org/2027/important-dates/ before submitting).
Format: 5 pages in total: up to 4 pages of technical content including figures and tables, 5th page references only, at most 20 references. Use the IEEE template offered by the paper management system if it differs from `IEEEtran`.

Title: Deep Learning-Based Lung Disease Detection Using Chest X-Ray Images (framing changed 2026-11-16: a designed-and-developed prediction system; the evaluation and its limits are reported in full).

## Status

- Text is filled from `docs/manuscripts/shared/RESULTS_FACTS.md` (final model: run 2, `v6_build_002`). No `\pending` blocks remain and the macro was removed from `main.tex`.
- Length: about 2,490 words in the sections (estimate, not a compiled count). **Page count must be confirmed after compiling on Overleaf**; if over 4 pages, shorten Related Work and the source-variation paragraph first, then move the system figure into a smaller one-column version.
- 20 references (the maximum). Figures: `fig_system_architecture`, `fig_confusion_matrix`, `fig_external_breakdown`. Table: per-class test results (accuracy by source is now given in the text).
- Included: system design (input check, Grad-CAM, inconclusive outcome, checkpoint hash check), three-seed summary and the one-seed preprocessing ablation (non-causal wording). Excluded: source-identifiability probe, run-1 comparison and the frozen-feature source analyses. All numbers are macros.

## Layout

```
conference/main.tex            entry point (\documentclass[conference]{IEEEtran})
conference/sections/*.tex      abstract, introduction, related, system, data, method, experiments, results, discussion, conclusion
shared/results_macros.tex      all result numbers (generated; do not edit)
shared/references.bib          bibliography (only cited keys are printed)
figures/*.pdf                  fig_system_architecture, fig_confusion_matrix, fig_external_breakdown
```

Relative paths used: `../shared/results_macros.tex`, `../shared/references`, `../../figures/<stem>.pdf`
(relative to `docs/manuscripts/conference/`; the figures are in `docs/figures/`).

## Compile on Overleaf

Option A (keep paths): create a project whose root mirrors the repository: upload
`docs/manuscripts/conference/`, `docs/manuscripts/shared/` and `docs/figures/` so that the folder tree
`docs/manuscripts/conference`, `docs/manuscripts/shared`, `docs/figures` exists, and set `main.tex` in
`docs/manuscripts/conference/` as the main document. Compiler: pdfLaTeX, TeX Live 2022 or newer.

Option B (flatten): put everything in one folder and change three things: `\input{results_macros.tex}`,
`\bibliography{references}`, and the `\includegraphics` paths to `<stem>.pdf`.

`IEEEtran.cls` and `IEEEtran.bst` are provided by Overleaf's TeX Live. Build order: pdfLaTeX, BibTeX, pdfLaTeX twice.

## Packages used

inputenc, fontenc, cite, amsmath, amssymb, graphicx, booktabs, array, xcolor, url, hyperref (all standard).

## Notes

- Author order, roles and emails follow the authors' own draft of 2026-11-16; the corresponding author is `[TO BE CONFIRMED]`. CRediT roles and approval by all authors remain `[AUTHORS TO CONFIRM]`.
- EMBC requires a named qualified author who agrees to review for EMBC 2027 (reciprocal review); not yet decided.
- AI use must be disclosed per IEEE policy; the disclosure paragraph in `main.tex` is the authors' own wording from their draft.
