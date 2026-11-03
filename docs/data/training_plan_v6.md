# Retraining plan (V6) — prepared, not executed

> **Status (2026-11-03): executed.** This page is the plan as written before training and is kept unchanged as a record. What was implemented differs in places: training reads V6 builds through the new `lungai/training/train_v6.py` and `lungai/evaluation/evaluate_v6.py` instead of adapting `train.py`/`evaluate.py`; the config's `dataset` block now describes V6 and V5 moved to `dataset_v5_historical`; the preprocessing-order change was not made. Results: [../results/current_model.md](../results/current_model.md).

No new run has been made, and no new metric exists. This plan reuses the canonical pipeline (`lungai/`) unchanged where possible and lists the changes needed.

## Fixed before training

| Item | Plan | Where |
|---|---|---|
| Architecture | DenseNet-121 (Keras Applications), ImageNet initialisation → GAP → BN → Dense 256 ReLU → Dropout 0.3 → Dense 6 softmax | `lungai/models/densenet.py` |
| Input | 224×224×3, ImageNet mean/std | config `preprocessing` |
| Preprocessing | **Proposed change:** resize to a canonical size first (e.g. 512 px short side), then apply blur, CLAHE and the σ = 1.0 low-pass, then resize to 224. This removes the resolution dependence (limitations.md). Choose between this and the historical order **on validation** only. Implemented as a new config version, so the V5 protocol stays documented | `lungai/preprocessing.py` (new variant behind a config flag) |
| Augmentation | Training only: rotation ±10°, reflect border. Proposed: drop horizontal flip (it reverses cardiac laterality), decided on validation. Optional mild brightness/contrast jitter | config `augmentation` |
| Class imbalance | Balanced per-sample weights `n/(k·n_c)` computed on V6 train | `lungai/training/train.py` |
| Splits | V6 patient- and duplicate-grouped split ([dataset_v6_proposal.md](dataset_v6_proposal.md)) | V6 manifest |
| Seeds | 5 seeds: 42, 43, 44, 45, 46. Report mean ± SD and per-seed values | `--run-name …_seed<k>` |
| Optimiser and schedule | Phase 1: 2 epochs, frozen backbone, Adam 1e-3. Phase 2: up to 20 epochs, last 30 layers trainable, Adam 1e-5. **Early stopping** with patience 4 on validation macro-F1. Batch 32 | config `training` (epochs and patience become config values) |
| Model selection | Best validation macro-F1 per run (tie: validation macro ROC-AUC). The paradigm or preprocessing variant is chosen on the **mean validation macro-F1 across seeds** | `train.py` (implemented) |
| Test | One evaluation of the selected configuration (all seeds) with `--final`. No further changes afterwards | `lungai.evaluation.evaluate` |
| External | Montgomery, binary: sensitivity **and** specificity with Wilson CIs, ROC-AUC of P(abnormal), exact TB recall | `evaluate --split external --final` |
| Metrics | Per-class precision, recall (Wilson CI), specificity, F1; confusion matrix; OvR ROC-AUC and PR-AUC where both classes are present; bootstrap CIs (2,000 resamples); ECE; per-source results | `lungai/evaluation/metrics.py` |
| Controls | (a) Source-prediction probe on frozen features. (b) Per-source class recall. (c) Leave-one-source-out for multi-source classes | New script (to write after V6 exists) |

## Recorded for every run

`run_metadata.json` (implemented) contains the following, and each evaluation report repeats the checkpoint SHA-256, manifest SHA-256 and split fingerprint:

- git commit and full config snapshot;
- manifest SHA-256 and **split fingerprint**;
- seed and library versions (Python, TensorFlow, NumPy), platform and GPUs;
- per-epoch history and the selected epoch;
- class weights;
- checkpoint SHA-256 and validation predictions.

## Code changes still needed before V6 training

1. **Dataset version as configuration:** the canonical config pins the V5 manifest checksum. Add a `dataset` block for V6, with its own manifest path, checksum and data root, behind a new model version (for example `v6.0`), and keep V5 for the recorded checkpoint.
2. **Manifest columns:** `train.py` and `evaluate.py` read V5 column names (`clinical_label`, `image_path`). Map them from V6's `lungai_label` and `raw_relpath`.
3. **Configurable training:** make the phase-2 epoch count and early-stopping patience config values, and add the preprocessing-order flag.

## Compute estimate (assumptions stated, not measured)

| Item | Estimate | Assumption |
|---|---|---|
| Training images | ~10–15k | Similar size to V5 |
| One run on a single 8–16 GB GPU | ~0.5–1.5 h | ≤ 22 epochs |
| One run on CPU only | ~10–20 h | Not recommended |
| 5 seeds × 2 preprocessing variants on GPU | ~5–15 GPU-hours | 10 runs |
| Storage per run | ~30 MB checkpoint + predictions | |

Training starts only after you approve the V6 manifest and this plan.
