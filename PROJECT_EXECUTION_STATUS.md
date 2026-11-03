# Project execution status and handoff

Read this file first in any new chat. It is the complete context needed to continue.

- **Repository:** `D:\LungAI\Lung_Disease_Detector`, branch `research/lungai-clean-baseline`. Not pushed. Nothing after `c1c82f7` is committed (commit only with the user's permission; never push).
- **Python:** `D:\LungAI\.venv-audit\Scripts\python.exe` (3.12, TensorFlow 2.16.1, Keras 3.15.1). **CPU only** (no native-Windows GPU support). Laptop: 8 GB RAM, so run only one heavy job (training, the backend, a frontend build) next to training.
- **Machine-readable state:** `artifacts/execution_state.json`. **Pipeline log:** `artifacts/pipeline/pipeline.log`. Download logs: `D:\LungAI\_e2e_run\acq_*.log`. Provenance: `data/raw/_provenance/*.jsonl`.

## State on 2026-11-03 (~17:40) — read this section first

- **Served model: v6.1 = run 2** (`runs/v6r2_densenet121_seed42`, build `v6_build_002`, `models/densenet121_v6r2_seed42.h5`, sha 7992b8da…), selected by `docs/data/run_selection_protocol.md` (`artifacts/evaluation_reports/run_comparison.json`). Run 1 = v6.0 history.
- **Overnight queue running:** `scripts/pipeline/run_followups.py --build v6_build_002 --phase2-epochs 14 --hours-per-run 8` (started 16:14): run 3 seed 43 → run 4 seed 44 → run 5 no-filter ablation (`data/derived/v6_build_002_no_filters`), stage tags `_r3`, `_r4`, `_r5abl`, evaluations in `artifacts/evaluation_reports/v6_r3_*`, `v6_r4_*`, `v6_r5abl_*`. If it stopped, re-run the same command: finished jobs are skipped, an interrupted one resumes. If a run's `stop_reason` says "time budget", resume it with `python -m lungai.training.train_v6 --build … --run-name … --phase2-epochs 14 --seed … --max-hours 3` so it gets the full 16 epochs like run 2, then re-run its evaluation, and record this.
- **Tomorrow, in order:** (1) `python -m lungai.evaluation.aggregate_seeds --seeds v6_r2:runs/v6r2_densenet121_seed42 v6_r3:runs/v6r3_densenet121_seed43 v6_r4:runs/v6r4_densenet121_seed44 --ablation v6_r5abl:runs/v6r5_densenet121_seed42_nofilters`; (2) `python -m lungai.evaluation.render_docs`; (3) fill the seed/ablation placeholders (BSPC `\pending` with `\RMSRun…`/`\RMS…`/`\RAbl…` names already in comments; thesis `[PENDING: seed runs]`/`[PENDING: ablation]`; slide 23), rebuild .docx/.pptx; (4) full pytest, Jest, build; (5) final integrity/citation/consistency audit; (6) ask the user about commit.
- Deliverables: see `docs/TRACKER.md`. Facts for writers: `docs/manuscripts/shared/RESULTS_FACTS.md`.

## Earlier state on 2026-11-03 (~10:50, kept for history)

| Item | State |
|---|---|
| Run 1 (`v6_build_001`, run `v6_densenet121_seed42`) | **Done and integrated.** 10 epochs (stopped by schedule), val macro-F1 0.678; test acc 0.657, macro-F1 0.677, macro ROC-AUC 0.921 (N=1,724); Montgomery sens 45/58, spec 43/80, exact TB 1/58 (71/138 called COVID-19: likely Cohen-style shortcut). Checkpoint `models/densenet121_v6_seed42.h5`, SHA-256 `c51ed228…` in the config as v6.0 |
| Run 2 (`v6_build_002`, run `v6r2_densenet121_seed42`, tag `_r2`) | **Training since 10:07**, time box until 16:30 then evaluation to `artifacts/evaluation_reports/v6_r2_{val,test,external}`. Build 2 adds TBX11K, COVID-19 Radiography DB and NIH 009/012: train 10,957 (COVID-19 2,083, TB 684), val 2,348, test 2,350; leakage checks 0; 4,130 rows with unknown patient identity (grouped per image) |
| Downloads | All 12 NIH archives, Montgomery, Shenzhen, Kermany, Cohen, TBX11K (Drive quota cleared 09:24), COVID-19 Radiography DB, Rahman TB (not used). Blocked: VinDr, JSRT |
| R1 (Montgomery in Rahman TB) | Check inconclusive: 0 matches, but the Shenzhen positive control also gave 0. V6 does not use Rahman TB. `artifacts/data_quality/external_overlap_rahman_tb/README.md` |

Restart the run-2 pipeline if it is not running and its stages are not done (same command, it resumes):

```text
python scripts/pipeline/run_overnight.py --build-deadline 2026-11-03T10:30 --train-until 2026-11-03T16:30 --build-name v6_build_002 --run-name v6r2_densenet121_seed42 --tag _r2 --skip-overlap --phase2-epochs 14
```

## Fixes made today (uncommitted)

- `lungai/training/train_v6.py::predict_cached`: inference as one compiled graph with a fixed (padded) batch. Eager calls took minutes per batch and filled RAM, which crashed (04:10) and stalled (04:30–06:42) the first two runs at their first validation.
- `scripts/data/acquire.py`: finds the `kaggle` CLI in the venv's Scripts folder.
- `scripts/pipeline/run_overnight.py`: `--build-name`, `--run-name`, `--tag`, `--skip-overlap`, `--phase2-epochs` (a second build/run keeps the first one's state and reports).

## Integration done (run 1)

- Config `lungai/configs/lungai_densenet121.json`: v6.0, new checkpoint name/hash, `dataset` = V6, V5 moved to `dataset_v5_historical` (`DATASET_V5` in `lungai/config.py`, used only by the V5 tools), `historical_checkpoints`.
- `lungai/evaluation/served_metrics.py` → `artifacts/evaluation_reports/model_metrics.json` (served by `/api/v1/model-metrics`, includes V5 under `historical`); `lungai/evaluation/render_docs.py` fills `<!-- BEGIN/END GENERATED -->` blocks in README, model card, experiment results and writes `docs/results/current_model.md`.
- Backend: `/model-metrics` reads `model_metrics.json` and reports `matches_configured_checkpoint`; the per-prediction limitation notice is built from it. Frontend Metrics page: evidence banner from payload, ROC-AUC column, validation line, V5 history card.
- Docs rewritten for V6 (README, model card, experiment results, limitations, training, evaluation, inference, API, architecture, reproducibility, thesis methodology, dataset/data_splits marked as V5 history). Dockerfile copies `model_metrics.json`.
- Verified: pytest 143 passed / 1 expected skip (before the last doc-test edits; docs+predictions tests 23 passed / 1 skip after); frontend Jest 19/19; production build compiles; browser E2E on 127.0.0.1:8001 passed (Metrics page = API; a test-split TB image predicted Tuberculosis 97.0%, v6.0, verified hash, generated notice; stored record has v6.0 + hash).
- **Found:** the heuristic CXR gate rejects 1,096/1,724 real test X-rays and 83/138 Montgomery (`artifacts/app_checks/cxr_gate_rejection_v6_build_001.json`). Documented; not changed (user decision).

## Remaining

1. When run 2's evaluation is done: compare runs **on validation macro-F1 only**. If run 2 is better: install its checkpoint (new filename, e.g. `densenet121_v6r2_seed42.h5`, version v6.1, `dataset.build` = v6_build_002, split fingerprint from its build report), `python -m lungai.evaluation.served_metrics --eval-prefix v6_r2 --run runs/v6r2_densenet121_seed42`, `python -m lungai.evaluation.render_docs`, update the run-specific statements in `docs/limitations.md` (COVID-19 source, TB transfer), re-run the gate measurement on build 2. Keep run 1 documented as a superseded V6 run.
2. Full verification: `python -m pytest`; frontend Jest + build; browser E2E (backend on port 8001 serving `backend/static`, gitignored; port 8000 is the old audit server PID 21036); stale-value grep.
3. Rewrite `docs/testing.md` with the final results.
4. Report honestly; ask the user about commit, the CXR gate, and the thesis binaries in `docs/archive/` (text corrected; DOCX/PDF/PPTX still need regenerating).
