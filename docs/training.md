# Training

Canonical entry point: [`lungai/training/train_v6.py`](../lungai/training/train_v6.py), on a V6 build. The settings come from `training`, `augmentation` and `preprocessing` in [`lungai/configs/lungai_densenet121.json`](../lungai/configs/lungai_densenet121.json).

## V6 trainer

```bash
python scripts/data/build_v6_manifest.py --raw-root data/raw --out data/derived/<build> --sources <ids...>
python -m lungai.training.train_v6 --build data/derived/<build> --run-name <run> [--max-hours H] [--phase2-epochs N] [--patience 3]
```

1. Reads `manifest_v6.csv` from the build and the cached 224×224 images the builder made with `lungai.preprocessing` (the same code the backend uses, so training and serving see identical pixels). Fails if any cached image is missing.
2. Computes balanced class weights from the training split.
3. **Phase 1 (head):** 2 epochs, backbone frozen, Adam lr 1e-3. **Phase 2 (fine-tune):** last 30 backbone layers trainable, Adam lr 1e-5, up to `--phase2-epochs` (default 8). Batch 32, sample-weighted sparse cross-entropy, flip and ±10° rotation augmentation.
4. After every epoch: validation pass, then save `model.h5` if validation macro-F1 is the best so far (tie: macro ROC-AUC); always save `last.weights.h5` and `state.json`.
5. Stops on early stopping (3 fine-tune epochs without gain), at the end of the schedule, or when `--max-hours` is used up.
6. Writes `runs/<run>/run_metadata.json`: git commit, build and manifest hash, split fingerprint, library versions, per-epoch history, selected epoch, class weights, checkpoint SHA-256.

Never used by training: `test` or the external cohort. Re-running the same command resumes after the last completed epoch (weights restored; optimiser moments restart, which is recorded in `resumed_from`).

Inference inside the trainer and evaluator runs as one compiled TensorFlow graph with a fixed batch shape. Eager model calls on CPU took minutes per batch and exhausted memory on the 8 GB training laptop; that stalled the first two overnight attempts at their first validation pass.

**Measured cost** (8,054 training images, laptop CPU): about 13–14 minutes per epoch including validation. The per-epoch times of the current checkpoint are in [results/current_model.md](results/current_model.md#training).

`scripts/pipeline/run_overnight.py` chains download wait, extraction, build, training and evaluation, records each stage in `artifacts/execution_state.json`, and continues from the last completed stage when restarted.

## V5 trainer (historical)

[`lungai/training/train.py`](../lungai/training/train.py) trains on the V5 manifest (`dataset_v5_historical` in the config). It is kept to document the V5 protocol; its images are not available here.

### Commands

```bash
export LUNGAI_DATA_ROOT=/path/to/folder-with-data   # contains data/raw and data/downloads
python -m lungai.training.train --dry-run           # checks config, manifest checksum, class weights and every train/val image
python -m lungai.training.train --run-name densenet121_v5_seed42
```

`--dry-run` and training both exit with code 2 and list the problem if any train or val image is missing.

### What it does

1. Loads the manifest and verifies its checksum.
2. Resolves the train and val image paths, and fails on any missing file.
3. Computes balanced class weights from the training labels.
4. Builds the canonical model (DenseNet-121 with ImageNet weights, which Keras downloads on first use).
5. **Phase 1:** 2 epochs, backbone frozen, Adam lr 1e-3. **Phase 2:** 2 epochs, last 30 backbone layers trainable, Adam lr 1e-5. Each epoch is 231 steps × 32 images with sample-weighted sparse cross-entropy.
6. After each epoch, evaluates on `val` and saves the checkpoint if validation macro-F1 is the best so far (ties go to validation macro ROC-AUC).
7. Writes `runs/<run-name>/`:
   - `model.h5`
   - `val_predictions.csv` for the selected epoch
   - `run_metadata.json`: config snapshot, git commit, manifest hash, library versions, GPUs, per-epoch history, selected epoch, class weights and checkpoint SHA-256

Never used by training: `test`, the external cohort, or an existing checkpoint. The run folder must not already exist.

### Reproducibility

- **Seeds:** Python, NumPy and TensorFlow are seeded with 42, and `PYTHONHASHSEED` is set.
- **Augmentation:** draws come from one seeded `numpy.random.Generator` and are made sequentially, so they are reproducible.
- **Not deterministic:**
  - cuDNN/GPU kernels;
  - the order of float accumulation across threads;
  - DenseNet weights downloaded by Keras, if the upstream file changes.

  Expect small run-to-run differences. Record several seeds before comparing variants.

### Compute and cost (estimate, not measured in this branch)

| Item | Estimate |
|---|---|
| Work per epoch | ≈7,400 image decodes + preprocesses, plus a full validation pass (≈1,600 images) |
| Total | 4 epochs |
| GPU (≥8 GB) | Tens of minutes per run |
| CPU only | Several hours per run |
| Storage | ≈30 MB per checkpoint, plus a few MB of metadata |

The V5 images are not available here, so this trainer has not been run on this branch.

### Relationship to the recorded checkpoint

The recorded v5.0 checkpoint came from `experiments/train_densenet_frequency_v5.py`. It used the same architecture, preprocessing, loss, weights, schedule and seed. It differed from the canonical trainer in four ways:

- It kept the **final** epoch rather than the best validation epoch.
- It drew augmentation from the global NumPy RNG.
- It saved no run metadata.
- It reused an existing checkpoint file if one was present.

A new run is therefore expected to be close to it, but not bit-identical.
