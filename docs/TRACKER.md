# Project and submission tracker

Owner: main session. Updated: 2026-11-03 ~17:35. Student: Katkam Sathwik (24567T7604), M.Tech CSE, KUCE&T, Kakatiya University. Supervisor: Prof. Nagavelli Ramana.

Legend: DONE · IN PROGRESS · PENDING · BLOCKED (needs someone outside this session).

## 1. Engineering

| Item | Status | Notes |
|---|---|---|
| Data acquisition from official sources | DONE | NIH (12/12), Montgomery, Shenzhen, Kermany, Cohen, TBX11K, COVID-19 Radiography DB, Rahman TB (unused). Provenance + SHA-256 in `data/raw/_provenance/` |
| VinDr-CXR, JSRT | BLOCKED | PhysioNet credentialing / JSRT registration (user) |
| V6 builds (rules, caps, patient-grouped split, leakage checks) | DONE | `v6_build_001`, `v6_build_002`; all leakage checks 0 |
| Run 1 training + evaluation | DONE | Integrated in the app as v6.0 |
| Run 2 training + evaluation | DONE | 16 epochs (schedule completed); test acc 77.4%, macro-F1 78.8%, ROC-AUC 0.955; Montgomery sens 55/58, spec 19/80, exact TB 0/58, COVID-19 calls 3 (run 1: 71) |
| Fair comparison of runs on a common unseen validation set | DONE | 2,018 images; macro-F1 run 1 59.4% vs run 2 79.6% → run 2 selected (pre-registered rule); shared sources only: acc 67.6% vs 67.8%, macro-F1 62.2% vs 68.5% |
| Integrate final model, regenerate metrics/docs/macros | DONE | v6.1 = run 2 (`densenet121_v6r2_seed42.h5`, sha 7992b8da…); figures + Grad-CAM regenerated; run 1 = v6.0 history |
| Runs 3 (seed 43), 4 (seed 44), 5 (no-filter ablation) | IN PROGRESS | Sequential queue started 16:14 (`scripts/pipeline/run_followups.py`); ~5 h each, done ~tomorrow 07:30 |
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
| M.Tech report (.docx) | `docs/thesis/` | FILLED for v6.1 | ~22.5k words (~98 pp. est.); remaining: seed (6) and ablation (4) placeholders, contacts, personalisation; update fields in Word |
| Presentation (.pptx) | `docs/presentation/` | FILLED for v6.1 | 30 slides; remaining: seeds/ablation slide, demo screenshots (manual), author confirmations; check visually in PowerPoint |
| Conference manuscript (IEEE LaTeX) | `docs/manuscripts/conference/` | COMPLETE DRAFT (EMBC) | ~2.4k words, 20 refs, 2 figs, 2 tables, no pending; page count to confirm on Overleaf; name the EMBC reciprocal reviewer |
| Journal manuscript (BSPC, elsarticle) | `docs/manuscripts/journal_bspc/` | FILLED except seeds/ablation | 24 `\pending` (seed table, ablation table, 1 conclusion sentence); probe + per-source + run comparison done; length likely above BSPC guidance — condense; REVIEWER_RISK: major revision likely |
| Venue comparison + submission tracker | `docs/publication/` | DONE | BSPC first (then JIIM, Sci. Rep.); EMBC 2027 first (CONECCT 2027 backup); no official KU template found |
| Thesis binaries in `docs/archive/` (old V5 DOCX/PDF/PPTX) | `docs/archive/` | PENDING | Kept as history; superseded by the new report |

## 3. Blockers and decisions for the user

1. Contact details (emails, mobiles): placeholders `[TO BE PROVIDED]`.
2. Author order and corresponding author for manuscripts: `[AUTHORS TO CONFIRM]`.
3. No official KUCE&T / Kakatiya University template found (kucet.ac.in unreachable); report uses an Osmania-based standard — confirm with the department, and ask whether a published paper is required before dissertation submission.
4. No LaTeX compiler here: manuscripts compile on Overleaf.
5. Journal strength: one run per model (single seed). Additional seeds (≈5 h CPU each) would strengthen a BSPC submission.
6. Commit/push approval.
7. Real non-radiograph image set to validate the CXR gate properly.
