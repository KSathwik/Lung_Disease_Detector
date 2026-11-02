# Training

Canonical entry point: [`lungai/training/train.py`](../lungai/training/train.py). The settings come from `training`, `augmentation` and `preprocessing` in [`lungai/configs/lungai_densenet121.json`](../lungai/configs/lungai_densenet121.json).

## Commands

```bash
export LUNGAI_DATA_ROOT=/path/to/folder-with-data   # contains data/raw and data/downloads
python -m lungai.training.train --dry-run           # checks config, manifest checksum, class weights and every train/val image
python -m lungai.training.train --run-name densenet121_v5_seed42
```

`--dry-run` and training both exit with code 2 and list the problem if any train or val image is missing.

## What it does

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

## Reproducibility

- **Seeds:** Python, NumPy and TensorFlow are seeded with 42, and `PYTHONHASHSEED` is set.
- **Augmentation:** draws come from one seeded `numpy.random.Generator` and are made sequentially, so they are reproducible.
- **Not deterministic:**
  - cuDNN/GPU kernels;
  - the order of float accumulation across threads;
  - DenseNet weights downloaded by Keras, if the upstream file changes.

  Expect small run-to-run differences. Record several seeds before comparing variants.

## Compute and cost (estimate, not measured in this branch)

| Item | Estimate |
|---|---|
| Work per epoch | ≈7,400 image decodes + preprocesses, plus a full validation pass (≈1,600 images) |
| Total | 4 epochs |
| GPU (≥8 GB) | Tens of minutes per run |
| CPU only | Several hours per run |
| Storage | ≈30 MB per checkpoint, plus a few MB of metadata |

**Retraining has not been run on this branch.** The images are not available here, and retraining needs explicit approval.

## Relationship to the recorded checkpoint

The recorded v5.0 checkpoint came from `experiments/train_densenet_frequency_v5.py`. It used the same architecture, preprocessing, loss, weights, schedule and seed. It differed from the canonical trainer in four ways:

- It kept the **final** epoch rather than the best validation epoch.
- It drew augmentation from the global NumPy RNG.
- It saved no run metadata.
- It reused an existing checkpoint file if one was present.

A new run is therefore expected to be close to it, but not bit-identical.
