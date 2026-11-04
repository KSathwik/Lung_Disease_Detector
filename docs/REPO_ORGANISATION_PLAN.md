# Repository organisation plan (inventory and proposal, nothing executed)

Prepared 2026-11-04 from a read-only inventory. No file was moved, deleted or edited except this one. Every action below needs the student's approval before it is run, and each step should be its own commit on `research/lungai-clean-baseline` (no new branches, no history rewrite).

## A. Local folder D:\LungAI (outside git)

| Item | Size | Contents | Used by repo? | Proposal |
|---|---|---|---|---|
| `.venv-audit/` | not measured (large) | Python 3.12 venv (TensorFlow, python-docx, python-pptx, playwright) | Yes: named in PROJECT_EXECUTION_STATUS.md, docs/presentation/README.md, build_report.py, build_presentation.py, check_imu.py | Keep. Do not touch. |
| `Lung_Disease_Detector/` | .git 148 MB; tracked files about 119 MB | The repo | n/a | See section B |
| `_e2e_run/` | 469 MB | Dataset acquisition logs (`acq_*.log`), follow-up and gate-calibration outputs (`followups.out`, `gate_*.out`), scratch scripts (`addrows.py`, `builder_middle.py`), `app_shots` (12 MB), `app_v6`, `app_v61` (5 to 6 MB each), `models_surrogate` (29 MB, surrogate .h5), `smoke_build` (14 MB), small `.db` files, and `quarantine/` (403 MB: an incomplete `covid19-radiography-database.zip.incomplete` and `densenet121_frequency_v5.h5`) | Referenced by name in PROJECT_EXECUTION_STATUS.md (download logs `D:\LungAI\_e2e_run\acq_*.log`). No code reads it | Keep the logs (integrity record: provenance of downloads). Moving or deleting anything needs approval. Candidates: `quarantine/` (403 MB, partial zip and an old V5 checkpoint), `models_surrogate`, `smoke_*`, `*.db`. Suggested: leave in place until submission, then zip or move the whole folder to an external drive. If moved, update the one path in PROJECT_EXECUTION_STATUS.md |
| `_audit_run/` | 44 MB | Output of the earlier audit's runtime checks (`app/`, `badckpt/`, `demo/`, `parity/`, `pytest/`, `surrogate/`, `uploads/`, `runtime_results_*.json`, small .db files) | No references in the repo | Keep as audit evidence; optional zip into a separate archive folder (approval) |
| `_audit_tools/` | 66 KB | Audit helper scripts (`docx_claims.py`, `manifest_audit.py`, `leaked_test.py`, `overlap_probe.py`, `recompute_cm.py`, `run_backend.py`, `run_frontend.py`, `runtime_checks.py`, install logs) | No references | Keep; they reproduce the audit findings. Consider copying into the repo (for example `scripts/audit/`) (approval, adds files) |
| `LUNGAI_AUDIT_REPORT.md` | 65 KB | The first full audit report | No code references. `docs/AUDIT_FINDINGS.md` (50 KB) is the in-repo audit | Keep. Consider copying into `docs/archive/historical_audits/` so the record lives in git (approval) |

No old repo copies, zips or extra venvs were found at the top level other than the above. Raw data and models sit inside the repo folder (`data/`, `models/`, `runs/`), are git-ignored and are not to be touched.

Repo references to machine paths: only documentation and script usage lines (the `.venv-audit` interpreter, the `_e2e_run` log path, `Lung_Disease_Detector` in WRITING_BRIEF). `tests/test_consistency.py::test_no_local_absolute_paths_in_active_code` scans code only (not docs), so these are allowed.

## B. Git repository

### B1. Tracked file counts (top two levels)
docs 332 (archive 173, manuscripts 52, figures 43, thesis 20, publication 15, presentation 4, data 5), experiments 235 (results 104, per-model V5 folders about 100, archive 13, data 5), artifacts 32, lungai 38, backend 28, frontend 17, tests 14 plus backend/tests 9, scripts 13.

Sizes of tracked content: docs/archive 53 MB (thesis_drafts 25 MB, figures_and_screenshots 23 MB, submission_package 3.5 MB, historical_audits 0.2 MB), docs/figures 12.4 MB, experiments/results 11 MB, experiments/data 9.4 MB (four manifest CSVs v1 to v4 plus the base one, 2.3 to 2.7 MB each), docs/thesis 6.6 MB, docs/presentation 5.8 MB.

### B2. Large tracked files
Over 5 MB:
- `docs/archive/thesis_drafts/LungAI_MTech_Project_Documentation.docx` 7.5 MB
- `docs/thesis/LungAI_MTech_Project_Report.docx` 6.7 MB (current, generated)
- `docs/archive/thesis_drafts/LungAI_MTech_Project_Report_Final_Corrected.docx` 6.4 MB
- `docs/presentation/LungAI_MTech_Presentation.pptx` 6.1 MB (current, generated)

Next largest: two archived PDFs (4.9 and 3.3 MB); five manifest CSVs of 2.3 to 2.7 MB (`data/manifests/unified_manifest_v5.csv` is under `data/`; the others are in `experiments/data/`); `docs/figures/fig_gradcam_examples.{png,pdf}` (2.3 and 2.1 MB); `fig_preprocessing.{png,pdf}` (1.5 MB each).

### B3. Exact duplicates (55 groups of identical blobs)
- Paper figures exist in three places: `docs/archive/thesis_drafts/papers/figures/`, `docs/archive/submission_package/figures/`, and `docs/archive/figures_and_screenshots/{diagrams/rendered,figures/*}`. Several also exist in `experiments/results/` and `experiments/densenet_*_v5/` (ROC curves, confusion matrices, t-SNE, phase4a plots, cropping and film-normalisation panels).
- `docs/archive/thesis_drafts/papers/*` is an identical copy of `docs/archive/submission_package/*` (cover letters, declarations, CRediT, highlights, suggested reviewers, Elsevier manuscript .docx).
- Screenshots: `figures_and_screenshots/screenshots/{application,e2e,analysis,explanation}` and `images/` repeat the same shots (up to 3 copies each); `old_screenshots/` duplicates `images/`.
- `thesis_drafts/LungAI_Updated_Professional_Presentation.pptx` equals `..._root_copy.pptx`.
- `docs/archive/THESIS.md` equals `thesis_drafts/LungAI_MTech_Project_Report.md`; `MANUSCRIPT_BSPC_LUNGAI.md` equals `papers/JOURNAL_PAPER_BSPC.md`.
- `historical_audits/CXR_GATE_FINAL_FORENSIC_AUDIT.md` equals `CXR_INPUT_VALIDATION_AUDIT.md`.
- `experiments/densenet_v3/class_mapping.json` equals `densenet_v3_full/class_mapping.json`; `experiments/archive/legacy_reports` PNGs equal archive figures.
- `uploads/.gitkeep` and `backend/utils/__init__.py` share the empty blob (harmless).

Estimated working-tree saving if duplicate archive copies were dropped: about 10 to 15 MB of the 53 MB (history keeps them regardless).

### B4. Overlapping documents

| Group | Files | Finding |
|---|---|---|
| Status and trackers | `PROJECT_EXECUTION_STATUS.md` (14.7 KB, HANDOFF), `docs/TRACKER.md` (10.5 KB), `docs/publication/SUBMISSION_TRACKER.md` (7 KB), `artifacts/execution_state.json` (6.5 KB, machine state used by `scripts/pipeline/run_overnight.py`) | Four places hold status. The JSON is code-referenced (keep). The two top-level trackers overlap on run status. SUBMISSION_TRACKER is venue-specific and its header note says the table's fallback order is outdated |
| Novelty planning | `NOVELTY_PLAN.md` (11.6 KB), `NOVELTY_ROADMAP.md` (9.8 KB), `NOVELTY_PROTOCOL.md` (34.9 KB), `PROTOCOL_REVIEW_FOR_STUDENT.md`, `REPORT_TEMPLATE_FINDINGS.md`, `RESEARCH_BRIEF.md` (59.9 KB), `lit_search/*` (4 files, 71 KB) | PROTOCOL is the approved, code-referenced document (cited in lungai/novelty/*.py, last commit recorded): must stay at its path. PLAN and ROADMAP are earlier planning stages that PROTOCOL supersedes. RESEARCH_BRIEF is a large background document |
| Venue analysis | `JOURNALS.md`, `JOURNAL_SCOPE_MATCH.md`, `CONFERENCES.md`, `CONFERENCE_VS_JOURNAL.md`, `lit_search/SCOPE_MERGE_REVIEW.md` | Overlapping venue comparisons; JOURNAL_SCOPE_MATCH is declared current |
| Methodology | `docs/thesis_methodology.md` (7.5 KB, V6 summary), `docs/thesis/content/*` (18 chapter files built into the DOCX by `build_report.py`), `docs/training.md`, `docs/dataset.md`, `docs/data_splits.md`, `docs/data/*` (5 plans) | thesis_methodology.md is the plain methodology source that thesis ch4/ch5 and the manuscripts restate. It is linked from `docs/archive/README.md` |
| Results and model description | `README.md`, `docs/model_card.md`, `docs/experiment_results.md` (generated blocks via `render_docs.py` TARGETS), `docs/results/current_model.md` (compared by `tests/test_docs_metrics.py`), `docs/evaluation.md`, `docs/limitations.md`, `docs/manuscripts/shared/{RESULTS_FACTS.md,results.json,results_macros.tex}` | Intentional generated repetition, protected by tests. Do not merge |
| Audits | `docs/AUDIT_FINDINGS.md` (50 KB), `docs/archive/historical_audits/*` (12 files), `D:\LungAI\LUNGAI_AUDIT_REPORT.md` | Integrity record; keep |
| Journal manuscripts | `journal_imu` (target; 8 files with uncommitted edits) and `journal_bspc` (fallback), plus `conference`, `shared` | Both needed per the submission decision; shared macros serve both |
| Old packages | `docs/archive/{PAPER_IEEE,PAPER_LNCS,THESIS}.md`, `thesis_drafts`, `submission_package`, `old_screenshots`, `thesis_tooling` | Documented in `docs/archive/README.md` as containing false or superseded claims; kept for traceability |

Other observations:
- `reports/` is an empty directory on disk (not tracked).
- `experiments/data/unified_manifest*.csv` (v1 to v4) are superseded by `data/manifests/unified_manifest_v5.csv` and the V6 manifest, but `experiments/rebuild_unified_dataset_v*.py` write them.
- `docs/figures/` (12 MB, 43 files) holds current thesis and manuscript figures.
- `docs/thesis/*.docx` and `docs/presentation/*.pptx` are generated outputs committed next to their sources (about 13 MB).
- `.gitignore` excludes `*.h5`, `runs/`, `artifacts/evaluation_reports/*/`, `data/raw` and similar, `*.log`, `*.db`. Ignored items present locally include `runs/v6*` (5 run folders), model checkpoints, data/raw, frontend/node_modules and `__pycache__`. The complete listing of ignored files was not produced because the data tree is large and was not walked.

### B5. Reference constraints (must not move without editing the referrer)
- `docs/publication/NOVELTY_PROTOCOL.md`: cited by `lungai/novelty/{__init__,common,c2_lite,inlp_eval,loso,spectral_map}.py`; `common.py` also runs `git log` on that path. Path and history must stay.
- `artifacts/execution_state.json` and `artifacts/pipeline/`: used by `scripts/pipeline/run_overnight.py`.
- `README.md`, `docs/model_card.md`, `docs/experiment_results.md`: `lungai/evaluation/render_docs.py` TARGETS and `tests/test_docs_metrics.py`. `docs/results/current_model.md` and `docs/data_splits.md` are also read by `tests/test_docs_metrics.py`.
- `tests/test_consistency.py` scans `README.md` and everything under `docs/` except paths with a component named `archive`, `build` or `node_modules`. Files outside `archive/` are subject to the ambiguous-model-names rule (see the first test in that file) (except manuscripts, thesis, presentation, publication) and the unsupported-claims rule (all .md). So moving a document into a folder named `archive` exempts it from these checks, and moving one out of `archive` may fail the tests.
- `docs/thesis/build_report.py` reads `docs/thesis/content/`, `docs/manuscripts/shared/references.bib` and figures; `docs/presentation/build_presentation.py` reads `slides.md`. Both document the `.venv-audit` interpreter.
- `docs/manuscripts/journal_imu/check_imu.py` and `shared/results_macros.tex` (generated); both journal main.tex files use the shared macros.
- CI (`.github/workflows/ci.yml`) runs `lungai.data.validate`, `lungai.evaluation.historical`, `pytest`, and the frontend tests. `historical` may read `experiments/` result JSON, so do not relocate `experiments/*_v5` or `experiments/results` JSON without running it.
- `docs/archive/README.md` links to `../experiment_results.md`, `../model_card.md`, `../thesis_methodology.md`.
- No code imports anything under `docs/archive` or `experiments/archive`, apart from the archived scripts themselves.

## C. Target structure

```
README.md                      keep (generated block)
PROJECT_EXECUTION_STATUS.md    keep (HANDOFF; memory points to it)
docs/
  README.md                    NEW small index: which file is current, which is archived
  TRACKER.md                   keep (single human tracker; links to SUBMISSION_TRACKER)
  current topic docs           api, architecture, data_splits, dataset, deployment, evaluation,
                               experiment_results, frontend, inference, limitations, model_card,
                               reproducibility, testing, training, troubleshooting,
                               thesis_methodology, AUDIT_FINDINGS, results/current_model
  data/  figures/  thesis/  presentation/   keep
  manuscripts/{conference,journal_imu,journal_bspc,shared}   keep
  publication/                 NOVELTY_PROTOCOL, PROTOCOL_REVIEW_FOR_STUDENT, REPORT_TEMPLATE_FINDINGS,
                               SUBMISSION_TRACKER, JOURNALS, JOURNAL_SCOPE_MATCH, CONFERENCES,
                               RESEARCH_BRIEF, lit_search/
    archive/ (NEW)             NOVELTY_PLAN, NOVELTY_ROADMAP, CONFERENCE_VS_JOURNAL (superseded)
  archive/                     keep README; drop byte-identical copies (step 5)
experiments/                   keep; optionally experiments/data v1 to v4 manifests into experiments/archive/
```

## D. Step-by-step plan

Order is lowest risk first. "Approval" means the student must say yes before it is run. After each step run `pytest -q` and `python -m lungai.evaluation.historical --out <tmp dir>`; commit each step separately.

| # | Item | Action | Reason | Risk | References to update | Approval |
|---|---|---|---|---|---|---|
| 1 | Pending IMU manuscript edits (8 files) | keep; commit as the student's work | uncommitted | none | none | Student decides when |
| 2 | `docs/README.md` index | add | one entry point | none | none | No |
| 3 | `NOVELTY_PLAN.md`, `NOVELTY_ROADMAP.md`, `CONFERENCE_VS_JOURNAL.md` | move (git mv) to `docs/publication/archive/` with a one-line "superseded by" header | superseded planning; history kept | Low; links must be fixed (`git grep` each name) | links in README, TRACKER, RESEARCH_BRIEF, lit_search | Yes (moves documents) |
| 4 | `SUBMISSION_TRACKER.md` vs `docs/TRACKER.md` | keep both; fix the stale table-order note; cross-link | different scope; merging risks losing the venue table | low | none | Yes (edits the student's tracker) |
| 5 | Duplicate archive copies: `thesis_drafts/papers/*` (same as `submission_package/*`), `..._root_copy.pptx`, `old_screenshots/`, `screenshots/e2e` duplicates, `CXR_INPUT_VALIDATION_AUDIT.md`, `archive/THESIS.md` | delete from git, keep one canonical copy; write `docs/archive/DUPLICATES.md` mapping removed paths to kept ones (or move removed copies to an untracked folder outside the repo) | about 10 to 15 MB; history unchanged | Low, but deletes archived evidence copies | `docs/archive/README.md` | Yes |
| 6 | Archived `thesis_drafts` binaries (about 25 MB) | keep; do not delete. Optionally offer a GitHub Release asset plus pointer | largest blobs; removal from history needs a rewrite, out of scope | n/a | n/a | Yes only if wanted |
| 7 | `experiments/results/{cropping_samples,film_normalization}` and per-model V5 PNGs that duplicate archive figures | keep (run outputs); only the archive copies are dropped in step 5 | experiments copies are the source | low | none | covered by step 5 |
| 8 | `experiments/data/unified_manifest*.csv` (v1 to v4) | move to `experiments/archive/manifests/` or keep | superseded but written by `rebuild_unified_dataset_v*.py` | Medium; scripts, `lungai.evaluation.historical` and tests may read them | the four rebuild scripts and any `git grep unified_manifest` hit | Yes; skip unless grep shows no readers |
| 9 | Generated `docs/thesis/*.docx`, `docs/presentation/*.pptx` (13 MB) | keep for now; decide at submission whether to stop tracking generated binaries | convenient for the student; regeneration needs the venv | low | `.gitignore` if untracked | Yes |
| 10 | `thesis_methodology.md` vs `docs/thesis/content` | keep both; add a header note on which is the source | merging would break archive README links and tests | low | none | No |
| 11 | `docs/manuscripts/journal_bspc` | keep; README line "fallback, not updated since the IMU decision" | submission decision 2026-11-04 | none | none | No |
| 12 | `D:\LungAI\_e2e_run` (469 MB, `quarantine` 403 MB) | report only; propose moving the whole folder to an external drive after submission | not referenced by code; logs are provenance evidence | Medium (status file cites the log path) | PROJECT_EXECUTION_STATUS.md, "Download logs" line | Yes (outside repo) |
| 13 | `D:\LungAI\_audit_run`, `_audit_tools`, `LUNGAI_AUDIT_REPORT.md` | keep; optionally copy tools and report into the repo | integrity record | low | none | Yes (adds files) |
| 14 | Empty local `reports/` | delete locally or leave | untracked, empty | none | none | Yes (trivial) |
| 15 | `.venv-audit`, `data/`, `models/`, `runs/` | do not touch | per instructions | n/a | n/a | n/a |

Nothing here rewrites git history, force-pushes, or removes the integrity record (NOVELTY_PROTOCOL, PROTOCOL_REVIEW_FOR_STUDENT, AUDIT_FINDINGS, historical_audits, `artifacts/*`, access logs, run reports).

## E. Not checked

- Size of `.venv-audit`, `data/` and `models/` (deliberately not walked).
- Full list of ignored files inside the repo.
- Whether `lungai.evaluation.historical` or tests read `experiments/data/unified_manifest*.csv` (run `git grep unified_manifest` before step 8).
- `frontend/public` and `backend/tests` were not reviewed for duplicates.

## Done (2026-11-04)

Student approved steps 1 to 5 only (plan table items 3, 4 cross-link, 5, 8 conditional, 9 decision). Applied to the working tree, not committed.

- Step 1 (superseded planning): `git mv` of `NOVELTY_PLAN.md`, `NOVELTY_ROADMAP.md`, `CONFERENCE_VS_JOURNAL.md` to `docs/publication/archive/`, each with a "superseded by" header, plus `docs/publication/archive/README.md`. All references (13 files: PROJECT_EXECUTION_STATUS, TRACKER, AUDIT_FINDINGS, conference README and checklist, RESULTS_FACTS, JOURNAL_SCOPE_MATCH, NOVELTY_PROTOCOL (link text only), RESEARCH_BRIEF, four lit_search files) now point to `docs/publication/archive/...`. `NOVELTY_PROTOCOL.md` was not moved.
- Step 2 (duplicates): 70 byte-identical files removed with `git rm` (about 17.5 MB in the working tree), one canonical copy kept per group (the `experiments/` copy where one exists, else `submission_package`, `screenshots/application`, `thesis_drafts/...Report.md`, `thesis_drafts/MANUSCRIPT_BSPC_LUNGAI.md`, the non-`_root_copy` pptx, `CXR_GATE_FINAL_FORENSIC_AUDIT.md`). Mapping in `docs/archive/DUPLICATES.md`. `docs/archive/old_screenshots/` is not byte-identical to `images/` (different blobs), so it was kept. `experiments/densenet_v3/class_mapping.json` duplicate and the empty-blob pair were left (code paths).
- Step 3 (single status page): pointers added at the top of `PROJECT_EXECUTION_STATUS.md`, `docs/TRACKER.md` and `docs/publication/SUBMISSION_TRACKER.md`; new `docs/README.md` index. `artifacts/execution_state.json` untouched. The stale table-order note in SUBMISSION_TRACKER was left as is.
- Step 4 (manifests v1 to v4): NOT moved. `git grep` shows readers: `experiments/audit_*`, `evaluate_source_holdout.py`, `phase2d/2e/2f`, `rebuild_unified_dataset_v*.py`, `inspect_montgomery.py`, `dataset_reconstruction_analysis.py`, and `artifacts/data_quality/v5_provenance.json`.
- Step 5 (generated binaries): decision recorded: `docs/thesis/*.docx` and `docs/presentation/*.pptx` stay tracked; revisit at submission.
- Checks after the change: pytest, `build_report.py` (missing from bib and tokens empty), `build_presentation.py` (0 overflow), `check_imu.py` (no problems), and a relative-link check of docs/ and root .md files (0 broken outside `docs/archive`; remaining broken links are pre-existing stale paths inside archived documents).
