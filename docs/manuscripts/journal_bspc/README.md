# BSPC journal manuscript (Elsevier `elsarticle`)

Target: *Biomedical Signal Processing and Control* (Elsevier), full-length original research article, review (preprint) format.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Manuscript root. Top comment block lists the **planned extensions required for submission** (not done) |
| `sections/00_abstract.tex` … `08_conclusion.tex` | Abstract and sections 1–8 |
| `declarations.tex` | CRediT, competing interest, funding, ethics, data and code availability, generative-AI declaration (included at the end of `main.tex`) |
| `highlights.tex` | Highlights (separate file for the submission system; compiles on its own) |
| `graphical_abstract.md` | Description of the intended graphical abstract (not drawn yet) |
| `cover_letter.md` | Draft cover letter |
| `REVIEWER_RISK.md` | Likely reviewer objections and how to address them |
| `../shared/results_macros.tex` | All model-result numbers (generated; never edit by hand) |
| `../shared/references.bib` | Shared bibliography |
| `../../figures/<stem>.pdf` | Figures (`docs/figures/`) |

## Compile on Overleaf

The `\input` and `\bibliography` paths point outside this folder, so upload the folder structure, not only this directory:

```
docs/
  figures/                       (fig_*.pdf)
  manuscripts/
    shared/results_macros.tex
    shared/references.bib
    journal_bspc/                (this folder; main.tex is the main document)
```

1. Zip `docs/` keeping only `docs/figures/`, `docs/manuscripts/shared/` and `docs/manuscripts/journal_bspc/` and upload it as a new Overleaf project.
2. Menu → Main document: `docs/manuscripts/journal_bspc/main.tex`. Compiler: **pdfLaTeX**. TeX Live 2022 or later (needs `elsarticle` ≥ 3.3 for the key–value `\affiliation` syntax).
3. Recompile; BibTeX runs automatically (`elsarticle-num` style).
4. For the highlights file, set `highlights.tex` as main document temporarily, or paste the bullets into the submission system.

For submission, Elsevier's system usually wants a flat folder: copy `results_macros.tex`, `references.bib` and the figure PDFs next to `main.tex`, change the three paths (`\input{results_macros.tex}`, `\bibliography{references}`, `\graphicspath{{./}}`), and upload the generated `.bbl`.

## Packages used

`elsarticle` (class, with `elsarticle-num.bst`), `fontenc` (T1), `inputenc` (utf8), `amsmath`, `amssymb`, `graphicx`, `booktabs`, `xcolor`, `url`, `hyperref` (`hidelinks`), `lineno` (line numbers for review). All are in TeX Live.

## Placeholders

- `\pending{...}` (red): result-dependent interpretation to write after the final model is chosen (run 1 vs run 2 on the common selection set).
- `\confirm{...}` (blue): decisions or facts the authors must confirm (author order, CRediT roles, source licences (Table 1 says "see source terms"), ethics exemption, repository URL, AI declaration).
- `[TO BE PROVIDED]`: contact details.

Before submission: `grep -n "pending\|confirm\|TO BE PROVIDED" -r .` must return nothing outside comments, and the macro file must have been regenerated from the final model (`python -m lungai.evaluation.render_docs`).

## Number policy

Model results appear only as macros from `results_macros.tex`. The only hard-coded numbers are: dataset counts copied from `data/derived/v6_build_00{1,2}/v6_build_report.json` (Tables 2–3, cited with split fingerprints), configuration constants (filter parameters, caps, seeds, learning rates), V5 audit counts from `docs/limitations.md` / `docs/experiment_results.md`, and the CXR-gate measurements from `artifacts/app_checks/*.json` (application checks, not model results). If the final model is from build 2, the Section 3 tables need no change (they already show both builds); check the text that names build 1 as the model's build.

## Title

Working title from the brief: "Internal accuracy is not generalisation: a leakage-controlled, multi-source six-class chest radiograph benchmark with external validation and shortcut analysis". Used title: "Internal accuracy is not generalisation: an auditable, leakage-controlled six-class chest radiograph benchmark from public sources with external and per-source evaluation". Reason: "shortcut analysis" suggests a formal shortcut test (source-prediction probe, leave-one-source-out), which was planned but not done; the paper does per-source and external-breakdown analysis. Update 2026-11-03: the source-identifiability probe is now done, but it is indirect evidence (source information is available in the features; no leave-one-source-out or masking test), so the used title is kept. "Source-confounding analysis" would be defensible; "shortcut analysis" still suggests a causal test.
