# Project and submission tracker

Owner: main session. Updated: 2026-11-04 ~05:40. Student: Katkam Sathwik (24567T7604), M.Tech CSE, KUCE&T, Kakatiya University. Supervisor: Prof. Nagavelli Ramana.

Legend: DONE · IN PROGRESS · PENDING · BLOCKED (needs someone outside this session).

## Status board (monitor)

Checked 2026-11-04 ~05:40 from files and logs (refresh at each milestone: re-read `followups.out`, `runs/*/state.json`, `artifacts/evaluation_reports/`, the placeholder greps, `git status`). Owner: auto = this session or its agents; student = needs the student.

| Task | Owner | State | Evidence | Next action | ETA |
|---|---|---|---|---|---|
| Run 2 (seed 42) train + eval | auto | done | `artifacts/evaluation_reports/v6_r2_{val,test,external}` | none; served as v6.1 | done |
| Run 3 (seed 43) train + eval | auto | done | `followups.out` "end v6r3_densenet121_seed43 rc=0" 19:55; `v6_r3_{val,test,external}` exist; state.json: 9 epochs, best epoch 6, bad_epochs 3 (early stop) | none | done |
| Run 4 (seed 44) train + eval | auto | done | `followups.out` "end v6r4_densenet121_seed44 rc=0" 23:31; `v6_r4_{val,test,external}` exist; early stop after 10 epochs (best 7) | none | done |
| Run 5 (no-filter ablation, seed 42) | auto | done | `followups.out` "end v6r5_densenet121_seed42_nofilters rc=0" 04:52 on 4 Nov; `v6_r5abl_{val,test,external}`; 16 of 16 epochs; same split fingerprint as run 2 | none | done |
| Aggregate seeds + ablation, render docs | auto | done | `artifacts/evaluation_reports/multi_seed_summary.json`; macros regenerated; F9/F10 in `RESULTS_FACTS.md` | none | done |
| Fill result placeholders | auto | done | 0 `[PENDING` in thesis content and `slides.md`; 0 active `\pending` in the BSPC sections | none | done |
| Rebuild report/slides, pytest, Jest, build, final audit | auto | done (docx blocked) | pptx rebuilt; pytest 150 passed 1 skipped; Jest 19/19; frontend build OK; audit in `docs/AUDIT_FINDINGS.md` (2026-11-04); the .docx could not be overwritten while Word had it open, so it was built to `LungAI_MTech_Project_Report_NEW.docx` (not committed) | student closes Word, then `python docs/thesis/build_report.py` and deletes the `_NEW` file | student |
| Commit and push (branch `research/lungai-clean-baseline`) | auto | see git log | committed and pushed after the 2026-11-04 audit fixes | none | done |
| Novelty literature search | auto | done | `docs/publication/lit_search/` (4 files, 22:28-22:37) | none until student decides scope | done |
| `NOVELTY_ROADMAP.md` | auto | done (verified draft) | `docs/publication/NOVELTY_ROADMAP.md`; nothing run | student review | done |
| `NOVELTY_PROTOCOL.md` | student | waiting (draft committed, NOT approved) | section 13 holds the approval and the hash of the approval commit | student approves; then record the approval commit hash | student |
| C1/C3/C4 novelty runs | auto | blocked | protocol not approved or committed; `artifacts/novelty/` absent | start only after protocol commit | after approval; roadmap places C3/C4/C2-lite on 4 Nov evening |
| C2 (band-selective suppression) | student | waiting | roadmap section 3 go/no-go rule on validation only | decide after C2-lite | after C2-lite |
| `JOURNAL_SCOPE_MATCH.md` | auto | done | S1 (C1+C3+C4) → Informatics in Medicine Unlocked, fallback BSPC; S2 (with C2) → BSPC, fallback IMU | student chooses scope and journal | student |
| Docker build, CI | student | blocked | Docker not installed; branch CI unrun | outside this session | - |
| Contacts, corresponding author, EMBC reciprocal reviewer | student | blocked | placeholders `[TO BE PROVIDED]` / `[TO BE CONFIRMED]` | student supplies | - |
| VinDr-CXR, JSRT, Kaggle token (Covid-Radiography, Rahman TB) | student | blocked | `artifacts/execution_state.json` blockers | student credentials | - |

### Risks and inconsistencies
- `NOVELTY_PROTOCOL.md` is committed as a draft but is binding only after the student's approval; running any novelty analysis before that would break "fixed in advance".
- Run 2 (served) is the highest of the three seeds on the internal test (spread under one point): never call it "representative" or "not a lucky seed". The ablation is one seed and run 5 selected its last epoch: keep its wording non-causal.
- Word had the report open on 4 Nov (lock file `docs/thesis/~$ngAI_MTech_Project_Report.docx`, untracked): the committed .docx is the 3 Nov build; rebuild after closing Word. Do not commit the lock file or the `_NEW` file.
- A second Claude session worked in parallel on 4 Nov 04:55 (thesis content, `build_presentation.py`); it handed ownership to this session and stopped. Only one session should write to the repository at a time.
- Laptop has 8 GB RAM: one heavy job at a time.

### Decisions waiting for the student
1. Approve `NOVELTY_PROTOCOL.md` so it can be committed before any novelty number exists.
2. Journal choice (BSPC first per `JOURNALS.md`; `JOURNAL_SCOPE_MATCH.md` still in progress).
3. Scope: C1+C3+C4 only (no training, about one day) or the full plan with C2 (2-3 nights); C2 go/no-go after C2-lite.
4. Approve commit and push of the uncommitted work.
5. Contact details, corresponding author, EMBC reciprocal reviewer.

## 1. Engineering

| Item | Status | Notes |
|---|---|---|
| Data acquisition from official sources | DONE | NIH (12/12), Montgomery, Shenzhen, Kermany, Cohen, TBX11K, COVID-19 Radiography DB, Rahman TB (unused). Provenance + SHA-256 in `data/raw/_provenance/` |
| VinDr-CXR, JSRT | BLOCKED | PhysioNet credentialing / JSRT registration (user) |
| V6 builds (rules, caps, patient-grouped split, leakage checks) | DONE | `v6_build_001`, `v6_build_002`; all leakage checks 0 |
| Run 1 training + evaluation | DONE | Integrated in the app as v6.0 |
| Run 2 training + evaluation | DONE | 16 epochs (schedule completed); test acc 77.4%, macro-F1 78.8%, ROC-AUC 0.955; Montgomery sens 55/58, spec 19/80, exact TB 0/58, COVID-19 calls 3 (run 1: 71) |
| Fair comparison of runs on a common unseen validation set | DONE | 2,018 images; macro-F1 run 1 59.4% vs run 2 79.6% → run 2 selected (rule fixed in advance in the repository); shared sources only: acc 67.6% vs 67.8%, macro-F1 62.2% vs 68.5% |
| Integrate final model, regenerate metrics/docs/macros | DONE | v6.1 = run 2 (`densenet121_v6r2_seed42.h5`, sha 7992b8da…); figures + Grad-CAM regenerated; run 1 = v6.0 history |
| Run 3 (seed 43) | DONE | Early-stopped at epoch 9 (best 6); test acc 77.1%, macro-F1 78.5%, AUC 0.954; Montgomery 54/58, 25/80, TB 0/58 |
| Runs 4 (seed 44), 5 (no-filter ablation) | IN PROGRESS | Sequential queue (`scripts/pipeline/run_followups.py`); see the status board for current epoch and ETA |
| Source-identifiability probe | DONE | Source predictable from frozen features: 93.8% balanced acc (6 sources, chance 16.7%); within Normal 97.2% (5 sources); within TB 98.9% |
| Backend/frontend integration | DONE (run 1) | Metrics endpoint, generated limitation notice, Metrics page |
| CXR gate recalibration (heuristic threshold 0.83 → 0.703) | DONE | Validation pass 38.7% → 91.0%; test rejection 9.3% (build 2), Montgomery 13.0%; all synthetic negatives rejected; no real negative set |
| Tests: pytest / Jest / build | pytest green for v6.1 (3 fixes: version source, root route with static build, literature names excluded from the model-name check); Jest/build pending re-run | Final full run at the end |
| Browser end-to-end check | DONE for v6.1 | Metrics page = API; TB test image → Tuberculosis 96.8%, v6.1, hash verified, Grad-CAM rendered, generated notice; History version label added afterwards (API-tested) |
| Docker image build | BLOCKED | Docker not installed on this laptop; Dockerfile fixed but unbuilt |
| CI | BLOCKED | Branch not pushed (push needs user approval) |
| Commit | PENDING | Needs user approval; nothing committed after `c1c82f7` |

## 2. Academic deliverables

| Deliverable | Folder | Status | Result-dependent parts |
|---|---|---|---|
| Shared results macros / JSON | `docs/manuscripts/shared/` | DONE (run 1 values) | Regenerate after final model |
| Verified bibliography | `docs/manuscripts/shared/references.bib` | DONE | 53 entries; 37 requested all verified; title-only extras flagged in REFERENCES_STATUS.md |
| Figures and diagrams | `docs/figures/` | DONE (run 1 data) | `python -m lungai.evaluation.make_figures`; re-run on final model; Grad-CAM figure still to implement; pipeline box must list build-2 sources if run 2 wins |
| M.Tech report (.docx) | `docs/thesis/` | FILLED for v6.1 | ~22.5k words (~98 pp. est.); seed and ablation sections filled 2026-11-04 (rebuild after closing Word); remaining: contacts, personalisation; update fields in Word |
| Presentation (.pptx) | `docs/presentation/` | FILLED for v6.1 | 30 slides; seeds/ablation slide filled 2026-11-04; remaining: demo screenshots (manual), author confirmations; check visually in PowerPoint |
| Conference manuscript (IEEE LaTeX) | `docs/manuscripts/conference/` | COMPLETE DRAFT (EMBC) | ~2.4k words, 20 refs, 2 figs, 2 tables, no pending; page count to confirm on Overleaf; name the EMBC reciprocal reviewer |
| Journal manuscript (BSPC, elsarticle) | `docs/manuscripts/journal_bspc/` | FILLED | seed and ablation tables filled 2026-11-04 (0 `\pending`); probe + per-source + run comparison done; length likely above BSPC guidance — condense; REVIEWER_RISK: major revision likely |
| Venue comparison + submission tracker | `docs/publication/` | DONE | BSPC first (then JIIM, Sci. Rep.); EMBC 2027 first (CONECCT 2027 backup); no official KU template found |
| Thesis binaries in `docs/archive/` (old V5 DOCX/PDF/PPTX) | `docs/archive/` | PENDING | Kept as history; superseded by the new report |

## 3. Blockers and decisions for the user

1. Contact details (emails, mobiles): placeholders `[TO BE PROVIDED]`.
2. Author order and corresponding author for manuscripts: `[AUTHORS TO CONFIRM]`.
3. RESOLVED: no prescribed KUCE&T report format and no paper required before dissertation submission (student, 2026-11-03); the report uses the documented Osmania-based standard (docs/publication/REPORT_TEMPLATE_FINDINGS.md).
8. EMBC 2027 requires one author to act as a reciprocal reviewer (named at submission; reviews a few EMBC papers): `[TO BE CONFIRMED]` in docs/manuscripts/conference/SUBMISSION_CHECKLIST.md.
4. No LaTeX compiler here: manuscripts compile on Overleaf.
5. Journal strength: three seeds of the final recipe and one ablation seed (done 2026-11-04); more ablation seeds (≈5 h CPU each) would strengthen a journal submission.
6. Commit/push approval.
7. Real non-radiograph image set to validate the CXR gate properly.
