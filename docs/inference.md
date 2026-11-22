# Inference

## Installing the checkpoint

The checkpoint is not in the repository (`*.h5` is gitignored) and has no public release. It is produced by `lungai.training.train_v6`; install it from the run folder:

```bash
python scripts/install_checkpoint.py --from runs/<run>/model.h5
python scripts/install_checkpoint.py --check
```

The script copies the file to `models/` under the configured name (`model.checkpoint.filename`) only if its SHA-256 equals `model.checkpoint.sha256` in [the config](../lungai/configs/lungai_densenet121.json). `MODELS_DIR` can point elsewhere. Docker expects the file at `/app/models/`. The previous v5.0 checkpoint is lost (`historical_checkpoints` in the config).

The notice returned with every prediction (`external_generalization_limitation`) is built at start-up from `artifacts/evaluation_reports/model_metrics.json` when that file belongs to the configured checkpoint; otherwise a generic notice without numbers is returned.

## Loading rules (`backend/ml/inference.py`)

| Situation | Engine state | `/health` | `/predict` |
|---|---|---|---|
| File missing | not ready | `degraded`, `load_error` says "not found" | 503 `MODEL_NOT_LOADED`; nothing stored |
| SHA-256 differs | not ready | `degraded`, error names both hashes (first 12 characters) | 503 |
| SHA-256 differs and `LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT=true` | ready, `checkpoint_verified=false` | `healthy` | 200; every response says `checkpoint_verified: false` and the UI shows a warning |
| Corrupt or incompatible file (output ≠ 6 or input ≠ 224×224×3) | not ready | `degraded` | 503 |
| Canonical file | ready, `checkpoint_verified=true` | `healthy` | 200 |

There is no fallback model, no mock output and no random output. A model output that is not a valid probability vector produces a 500 response, and nothing is stored.

## Request path

1. **Upload checks (`predictions.py`):**
   - content type must be JPEG, PNG, BMP, TIFF or WebP;
   - at most 10 MB, read with a cap (413 if larger);
   - empty files are rejected (400);
   - the stored filename is server-generated (`{uuid}{ext}`).
2. **CXR gate (`cxr_gate.py`):** heuristic checks (size, aspect ratio, contrast, colour, brightness, anatomical profile). Failure returns 422 `INVALID_CXR`.
3. **Plausibility checks and canonical preprocessing:** `ml/preprocessing.py` → `lungai.preprocessing`.
4. **Softmax:** checked for shape (6,), finite values, non-negativity and a sum of 1 ± 1e-3.
5. **Response:**
   - `predicted_class` and `predicted_index`;
   - `confidence` (top score, 0–1);
   - `probabilities` (6 values to 4 d.p., keyed by class name);
   - `class_order`;
   - `model_id`, `model_version`, `selected_model` (display name), `model_sha256` and `checkpoint_verified`;
   - `inconclusive` (true when the top score is below `confidence_floor`, 0.5);
   - Grad-CAM overlay;
   - the limitation text.
6. **Persistence:** the probabilities, model ID and version, and checkpoint SHA-256 are stored with the prediction.

## Concurrency

There is one model instance per process, held by a singleton. Keras `predict` is called from the async route without a lock. That is acceptable for a single-worker demo. For concurrent use, run several workers, each loading its own copy.
