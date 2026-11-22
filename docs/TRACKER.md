# Project and submission tracker

> This is the single status page. Related: `PROJECT_EXECUTION_STATUS.md` (handoff context), `docs/publication/SUBMISSION_TRACKER.md` (venue table), `docs/README.md` (catalogue of every document), `artifacts/execution_state.json` (machine state for the pipeline scripts, not edited by hand).

Updated: 2026-11-22. Student: Katkam Sathwik (24567T7604), M.Tech CSE, KUCE&T, Kakatiya University. Supervisor: Prof. Nagavelli Ramana.

Legend: DONE · OPEN (work left) · STUDENT (needs the student or the authors) · BLOCKED (needs something outside the project).

## Framing

Since 2026-11-16 the thesis, the EMBC conference paper, the slides and the viva guide present LungAI as a **system designed and developed to predict lung disease** (data pipeline, DenseNet-121 classifier, web application, honest evaluation). The source-information analyses (C1-C4) are supplementary: thesis Appendix G and backup slides. The IMU journal draft keeps the earlier source-shortcut framing; it was not changed.

## 1. Engineering

| Item | Status | Notes |
|---|---|---|
| Data acquisition from official sources | DONE | NIH (12/12), Montgomery, Shenzhen, Kermany, Cohen, TBX11K, COVID-19 Radiography DB, Rahman TB (overlap check only). Provenance and SHA-256 in `data/raw/_provenance/` |
| VinDr-CXR, JSRT | BLOCKED | PhysioNet credentialing / JSRT registration |
| V6 builds (rules, caps, patient-grouped split, leakage checks) | DONE | `v6_build_001`, `v6_build_002`; all leakage checks 0; identity unknown for 4,130 TBX11K and COVID-19 Radiography rows (grouped per image) |
| Runs 1-5 | DONE | Run 1 (v6.0, superseded); run 2 = served v6.1 (seed 42); runs 3-4 seeds 43 and 44; run 5 no-filter ablation |
| Run selection (common validation set, rule fixed in advance) | DONE | 2,018 images; run 2 selected |
| Source-identifiability probe | DONE | 93.8% balanced accuracy (chance 16.7%) |
| Source-information analyses C1, C3, C4, C2-lite | DONE | Protocol approved at `f427c04`; C2 NO-GO; reported as supplementary (thesis Appendix G) |
| CXR gate recalibration | DONE | Heuristic threshold 0.83 → 0.703 on validation; no real non-radiograph set |
| Backend and frontend serving v6.1 | DONE | Browser check done; metrics page equals the API |
| Tests | DONE | pytest 173 passed, 1 skipped (2026-11-22); Jest 19/19; frontend production build OK |
| CI (GitHub Actions) | DONE | Green on `research/lungai-clean-baseline` and `main` for `9b9a1fd` (2026-11-22) |
| Docker image build | BLOCKED | Docker not installed on the laptop; Dockerfile unbuilt |
| Commit and push | DONE | Work goes on `research/lungai-clean-baseline`; `main` is fast-forwarded to it. GitHub's default branch is still `develop` (only the owner can change it in Settings) |

## 2. Academic deliverables

| Deliverable | Folder | Status | Left to do |
|---|---|---|---|
| M.Tech report (.docx) | `docs/thesis/` | DONE (system framing, rebuilt 2026-11-22) | Contacts and acknowledgement personalisation (13 `[TO BE PROVIDED]` markers); update fields in Word (Ctrl+A, F9) before printing |
| Presentation (.pptx) | `docs/presentation/` | DONE (system framing, 30 slides + 3 backup, rebuilt 2026-11-22) | Corresponding author; check visually in PowerPoint |
| Viva guide (.md and .docx) | `docs/viva/` | DONE (story and Q3/Q4 aligned, rebuilt 2026-11-22) | Read it before the viva |
| Conference paper, IEEE EMBC 2027 | `docs/manuscripts/conference/` | COMPLETE DRAFT (system framing) | Compile on Overleaf and confirm 4 pages; corresponding author; reciprocal reviewer; CRediT and approvals |
| Journal paper, Informatics in Medicine Unlocked | `docs/manuscripts/journal_imu/` | COMPLETE DRAFT (source-shortcut framing) | Student decisions below; disclose overlap with the EMBC paper |
| Journal paper, BSPC (fallback venue) | none | Earlier BSPC draft superseded by the IMU draft and removed (in git history) | Adapt the IMU draft only if IMU is dropped |
| Shared results macros, facts, bibliography | `docs/manuscripts/shared/` | DONE | 64 verified references; regenerate macros only if a model changes |
| Figures | `docs/figures/` | DONE | Regenerate only if a model changes |

## 3. Decisions and inputs waiting for the student

1. Corresponding author for every paper and the EMBC reciprocal reviewer.
2. Contact details in the thesis (13 `[TO BE PROVIDED]` markers) and the acknowledgement text.
3. Author emails are now in the conference paper and the repository is public: confirm every author agrees.
4. Ethics statement, AI-use declaration and CRediT roles approved by all authors.
5. Written confirmation of the novelty-protocol approval and its seven clarifications (audit items N7/W9), needed for the journal paper.
6. Journal paper: keep the IMU draft in its source-shortcut framing, or align it with the system framing. IMU APC and a Scopus / Google Scholar novelty check before submission.
7. A real set of non-radiograph images, if the input gate is to be validated.

## 4. Risks

- Run 2 (served) is the highest of three seeds within a spread of under one point: never call it "representative". The ablation is one seed and selected its last epoch: keep its wording non-causal.
- The conference paper is about 2,490 words with three figures and one table and may exceed 4 pages; the conference README lists what to cut first.
- Never commit the Word lock file (`~$*.docx`) or a `_NEW` copy of the report.
- Only one Claude session should write to the repository at a time.
- Laptop has 8 GB RAM: one heavy job at a time.
