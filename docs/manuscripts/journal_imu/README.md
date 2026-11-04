# IMU journal manuscript (Elsevier `elsarticle`)

Target: *Informatics in Medicine Unlocked* (IMU, Elsevier), original research article, review (preprint) format. Copy of `../journal_bspc/` (which stays unchanged as the fallback), reframed as a diagnostic, leakage-controlled study of source shortcuts in multi-source chest radiograph classification, with the novelty results C1, C3, C4 and the C2 go/no-go decision (`../shared/RESULTS_FACTS.md` F11 to F14). IMU requirements used (from `docs/publication/JOURNAL_SCOPE_MATCH.md`, pages opened 2026-11-03; re-check before submission): abstract up to 250 words, 1 to 7 keywords, 3 to 5 highlights of at most 85 characters in a separate file, an ethics statement (if none is needed, say why), generative-AI declaration if AI tools were used, no word limit found.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Manuscript root |
| `sections/00_abstract.tex` | Structured abstract (233 words after macro expansion by `check_imu.py`; a plain whitespace count is similar) |
| `sections/01_introduction.tex` | Introduction with the related work merged in, and the contributions |
| `sections/03_materials.tex` to `08_conclusion.tex` | Data and benchmark, methods (including C1, C3, C4, C2-lite and the decision rules), experimental design, results (with the three novelty figures), discussion and limitations, conclusion. There is no `02_`: related work is merged into the introduction |
| `declarations.tex` | CRediT, competing interest, funding, ethics statement (draft, `[STUDENT TO CONFIRM]`), data and code availability, generative-AI declaration (draft, `[AUTHORS TO CONFIRM]`) |
| `highlights.tex` | Five highlights, each at most 85 characters (separate file; compiles on its own) |
| `supplementary.tex` | Tables moved from the main text: builds (S1, S2), per-class (S3), run comparison (S4), follow-up runs (S5), ablation (S6), C1 secondary profiles (S7). Compiles on its own |
| `check_imu.py` | Consistency check (macros, citations, labels, characters, word counts); run with `D:\LungAI\.venv-audit\Scripts\python.exe` |
| `cover_letter.md`, `graphical_abstract.md`, `REVIEWER_RISK.md` | Drafts, updated for IMU and the new results |
| `../shared/results_macros.tex`, `../shared/references.bib` | Generated macros and shared bibliography |
| `../../figures/` | Figures. Used: `fig_external_breakdown`, `fig_spectral_map`, `fig_inlp`, `fig_loso_matrix` |

## Build on Overleaf

The `\input`, `\bibliography` and `\graphicspath` entries point outside this folder, so upload the folder structure:

```
docs/
  figures/                       (fig_*.pdf)
  manuscripts/
    shared/results_macros.tex
    shared/references.bib
    journal_imu/                 (this folder; main.tex is the main document)
```

1. Zip `docs/` keeping only `docs/figures/`, `docs/manuscripts/shared/` and `docs/manuscripts/journal_imu/`; upload as a new Overleaf project.
2. Main document: `docs/manuscripts/journal_imu/main.tex`. Compiler: pdfLaTeX. TeX Live 2022 or later (`elsarticle` 3.3 or later for the key-value `\affiliation` syntax).
3. Recompile; BibTeX runs automatically (`elsarticle-num`, single bibliography: `../shared/references`).
4. `highlights.tex` and `supplementary.tex` each compile on their own (set as the main document temporarily, or upload as separate files).

For submission, Elsevier's system usually wants a flat folder: copy `results_macros.tex`, both `.bib` files and the figure PDFs next to `main.tex`, change the paths, and upload the `.bbl`. Merge the two `.bib` files or keep both names in `\bibliography{...}`.

## Placeholders

- `\confirm{...}` (blue): facts the authors must confirm (CRediT roles, approval, funding, repository URL, licence, weights, AI declaration).
- `\student{...}` (blue, `[STUDENT TO CONFIRM]`): the ethics-statement wording.
- `[TO BE PROVIDED]`: contact details; corresponding author `[TO BE CONFIRMED]`.
- Before submission, `grep -n "confirm\|student\|TO BE PROVIDED\|pending" -r .` must return nothing outside comments.

## Bibliographic records

All entries cited here are in `../shared/references.bib` and were checked against Crossref, arXiv, the NeurIPS and PMLR proceedings pages and the PDF of Yao et al. on 2026-11-04 (see `../shared/REFERENCES_STATUS.md`). Notes:

- Robinson et al. (key `robinson2022shortcut`): Crossref lists Trivedi as first author, so the text says "Trivedi et al.".
- Rahbar (arXiv 2608.10295): cite v1 only; a later version has a different title.

## Number policy and wording

Model results appear only as `\R...` macros. Hard-coded numbers are configuration constants (band edges, caps, thresholds fixed in the protocol, bootstrap size, chance level 1/6) and dataset counts copied from the build reports (supplement). No p-values, no "significant", no "pre-registered" (write "fixed in advance and recorded in the repository"), no "first", no "state of the art", no causal claim about shortcut use. The McNemar p-value of the BSPC version is not reported here (discordant-pair counts only).
