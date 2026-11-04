# Testing

```bash
python -m pytest                                              # from the repository root
cd frontend && CI=true npx react-scripts test --watchAll=false
cd frontend && CI=true npm run build
```

CI ([.github/workflows/ci.yml](../.github/workflows/ci.yml)) runs the same commands. **CI has not run yet**, because the branch has not been pushed.

## Results on branch `research/lungai-clean-baseline` / `main` (2026-11-04, Windows 11, Python 3.12, TF 2.16.1; CI on Node 18)

| Suite | Result |
|---|---|
| `python -m pytest` (tests/ + backend/tests, incl. `tests/ml/test_novelty*.py`) | **173 passed, 1 skipped** |
| Frontend Jest (`errors`, `prediction`, `metrics`) | **19 passed** |
| Frontend production build (`CI=true`, warnings are errors) | **Compiled successfully** |
| `python -m lungai.data.validate` | Runs; overall **FAIL**. This is the expected result: it detects the known V5 patient leakage. Image checks are BLOCKED |
| `python -m lungai.evaluation.historical` | Runs; 5/5 recorded runs consistent |
| `python -m lungai.training.train --dry-run` | Exits 2: "7398 of 7398 train images are missing" (expected without data) |
| `python -m lungai.evaluation.evaluate --split val …` | Exits 2: "1579 of 1579 images are missing" (expected without data) |

### Skipped (blocked)

| Test | Missing requirement |
|---|---|
| `backend/tests/test_model_integration.py::test_canonical_checkpoint_loads_and_verifies` | The canonical checkpoint in `models/` |
| `…::test_canonical_checkpoint_predicts_six_class_distribution` | The canonical checkpoint in `models/` |

Not automated (blocked): image existence, md5 and decodability checks; training; evaluation of the real checkpoint on val/test/external. All of these need the images and/or the checkpoint.

## What the suites cover

| Area | File(s) | Covers |
|---|---|---|
| Config / class mapping | `tests/ml/test_config.py` | Fixed class order; agreement with the recorded training mapping, metric files and preprocessing/training configs; invalid configs rejected; path-free identity |
| Preprocessing | `tests/ml/test_preprocessing.py` | Equals a frozen copy of the training preprocessing (5 sizes/channel layouts, 16-bit PNG, tolerance 1e-6); backend bytes path equals file path; missing, corrupt and tiny inputs raise; seeded augmentation; the evaluation loader never augments |
| Model | `tests/ml/test_model.py` | 6-way softmax output; layer names match the checkpoint; save/load round trip; SHA-256 refusal; missing checkpoint |
| Metrics | `tests/ml/test_metrics.py` | Hand-computed confusion-matrix values; agreement with sklearn; Wilson CIs (58/58, 0/80); degenerate binary classifier; ECE; invalid shapes, indices, dtypes, sums and NaN rejected; absent class → AUC undefined |
| Evaluation | `tests/ml/test_evaluate.py` | Recompute from a predictions CSV; missing sample, wrong label, tampered argmax and mixed checkpoints detected; external binary task; CLI writes reports; `--final` required |
| Data | `tests/data/test_manifest.py`, `test_splits.py` | Checksum; schema; split and class counts; **known leakage pinned (154)**; synthetic IDs (8,478); duplicates; external isolation; path relocation; missing/corrupt file detection; deterministic grouped split with no patient group spanning partitions |
| Consistency | `tests/test_consistency.py`, `tests/test_docs_metrics.py` | No competing model names, no re-declared class lists, no machine paths, no mock prediction paths, no unsupported claims in active docs; README/docs numbers match the generated evidence files |
| Backend API | `backend/tests/test_predictions.py`, `test_health.py`, `test_cxr_gate.py`, `test_patients.py`, `test_migrations.py`, `test_preprocessing.py` | Valid upload persists model output and identity; non-probability model output → 500 with nothing stored; 503 with nothing stored when no model is loaded; unsupported type 400; oversized 413; empty 400; corrupt 422; path traversal blocked; inconclusive state; auth; gate rejections; `/model`, `/model-metrics` and `/health` content; DB migration |
| Backend ↔ model | `backend/tests/test_model_integration.py` | With a random-weight network of the canonical architecture: hash refusal → 503; missing, corrupt and wrong-shape checkpoints refused; explicit override flagged `checkpoint_verified=false`; **API probabilities equal the raw model softmax**; Grad-CAM wiring; determinism |

Stubs and random-weight models check the mechanics only. They are **not** evidence that the trained model works or is accurate.

## Manual browser check (2026-11-02)

The backend served the production frontend build at `127.0.0.1:8010`, using a temporary database and uploads folder outside the repository. A synthetic thoracic image was uploaded through the Analyze page.

| Configuration | Observed |
|---|---|
| No checkpoint | `/health` = degraded. The Analyze page shows "Model Not Loaded" with the path-free reason; no result is rendered and nothing is stored. The Model Performance page shows the recorded results labelled "recorded, not reproduced", with 58/58 next to 0/80 |
| Random-weight stand-in + `LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT=true` | All six scores displayed equal the API response values. The page shows the warning that the checkpoint does not match the canonical model, and "Inconclusive" (top score 17.1% < 50%). Grad-CAM is returned. The stored prediction has `model_used=lungai-densenet121-6class`, `model_version=v5.0` and the same six probabilities |

The browser check found one display bug: macro ROC-AUC was shown as "97.5%". It was fixed, so the value now shows as 0.9755.
