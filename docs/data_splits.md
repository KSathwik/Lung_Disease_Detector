# Data splits — V5 (history)

> This page documents the **V5** splits of the previous model. The current V6 splits are grouped by patient and checked in each build's `v6_build_report.json`; their counts and leakage checks are in [results/current_model.md](results/current_model.md#dataset-composition).

All numbers on this page come from `python -m lungai.data.validate` ([report](../artifacts/data_validation/data_validation_report.md), [JSON](../artifacts/data_validation/data_validation_report.json)), run on the manifest. Image-level checks are **blocked** because the images are not in the repository.

| Item | Value |
|---|---|
| Dataset version | V5 |
| Manifest | `data/manifests/unified_manifest_v5.csv` (moved from `experiments/data/` and byte-identical) |
| Manifest SHA-256 | `8117a441d5d200e781df0af33685cde32c6c7a24a5e30d73f5744ec592702e26` (LF line endings; `load_manifest()` checks it). With CRLF line endings it is `6730495a…eda2c4`, which matches the hash recorded in 2026 |
| Split column | `split` ∈ {train, val, test} |

## Partitions

| Partition | Purpose | Images | Manifest patient IDs | Derived patient groups | Permitted use |
|---|---|---:|---:|---:|---|
| train | Learn model parameters | 7,398 | 7,189 | 6,722 | Gradient updates |
| val | Select settings and checkpoints | 1,579 | 1,540 | 1,513 | Model selection |
| test | Final internal evaluation | 1,570 | 1,541 | 1,521 | Report once; never tune |
| **total** | | **10,547** | **10,270** | **9,432** | |
| external (Montgomery) | Independent external check | 138 (recorded) | — | — | External assessment only |

**Patient IDs.** 8,478 rows have a synthetic, one-per-image `patient_id` (`existing_pt_*`, `tbx11k_pt_*`). "Manifest patient IDs" therefore counts ID strings. "Derived patient groups" also merges Kermany images that share a patient token in the filename (`person123_…`, `IM-0115-…`). Patient identity cannot be recovered from the manifest for COVID-QU (`Existing_COVID-19`), Shenzhen (`Existing_Tuberculosis`) or TBX11K.

## Class distribution

| Class | train | val | test | total |
|---|---:|---:|---:|---:|
| COVID-19 | 1,359 | 291 | 292 | 1,942 |
| Normal | 1,848 | 394 | 394 | 2,636 |
| Pleural Effusion | 755 | 155 | 152 | 1,062 |
| Pneumonia | 1,963 | 419 | 423 | 2,805 |
| Pulmonary Nodule / Mass | 529 | 117 | 108 | 754 |
| Tuberculosis | 944 | 203 | 201 | 1,348 |

## How the split was generated

V5 was built in stages by `experiments/rebuild_unified_dataset_v2.py` through `…_v5.py`, which are kept for history. Each version kept the split of every patient ID it inherited. New patient IDs were then assigned 70/15/15, stratified, with seed 42. Because the IDs were synthetic per image for most sources, this produced an image-level split for those sources. **The split is preserved as recorded.** Regenerating it would change the population the checkpoint was evaluated on.

For the next dataset version, [`lungai/data/splits.py`](../lungai/data/splits.py) provides a deterministic split that is grouped by patient and stratified by class. `tests/data/test_splits.py` shows that it produces zero groups spanning splits on the V5 manifest.

## Integrity checks

| Check | Result | Evidence |
|---|---|---|
| Schema, labels, splits, duplicate IDs/paths | PASS | `schema_errors()` is empty |
| Patient overlap by manifest `patient_id` | 0, but this is **uninformative** because the IDs are synthetic | — |
| Patient overlap by derived group | **FAIL.** 289 groups span splits. **154 test images** (Kermany Pneumonia 110, Normal 44) and 191 val images share a patient with training | `check_patient_leakage` |
| Exact duplicates (manifest md5) | PASS. 0 duplicate hashes | `check_exact_duplicates` |
| Identical perceptual hash (dHash) | WARN. 3 pairs; 1 spans val/test with conflicting labels (Existing_Tuberculosis vs NIH Normal) | `check_perceptual_duplicates` |
| External cohort excluded (name/path) | PASS. 0 Montgomery rows | `check_external_isolation` |
| Label/source confounding | WARN. COVID-19 has a single source; Pleural Effusion is 88% VinDr | `check_label_source_confounding` |
| Proxy labels | WARN. 1,395 TBX11K "sick but not TB" images are labelled Pneumonia (49.7% of the class) | `check_proxy_labels` |
| Metadata columns | WARN. 10,090 rows have default width/height/intensity values, not measurements | `check_placeholder_metadata` |
| Absolute paths | WARN. 211 rows; relocated under the data root on load | `check_absolute_paths` |
| Files exist, md5 matches, decodable | **BLOCKED.** No images in the repository | `check_files` |

## Leakage prevention in the pipeline

- **Training** reads only `train` (gradients) and `val` (checkpoint selection). `lungai/training/train.py` never loads `test` or the external cohort.
- **Augmentation** is applied only in the training loader. Evaluation uses `load_batch()` without an RNG, which is tested.
- **Normalisation** uses fixed ImageNet statistics. Nothing is fitted on data.
- **Test and external evaluation** require `--final`.
- **Montgomery isolation:** Montgomery is not in the manifest. It was, however, used during the historical method development (the frequency filter was motivated by it, and border-cropping and film-normalisation experiments used it), so it is no longer a strictly untouched cohort.

## Inclusion and exclusion criteria (as recorded)

- **Included:** frontal chest radiographs whose source label maps to one of the six classes.
- **Excluded:** NIH and VinDr images with more than one finding, which were filtered to single findings upstream (not re-verified, because the images are absent), and every Montgomery image.
- **Lineage:** see `experiments/results/dataset_reconstruction_v5_report.md`.
