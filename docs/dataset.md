# Dataset

The canonical dataset is **V5**, defined by `data/manifests/unified_manifest_v5.csv`. Partitions, counts and integrity checks are in [data_splits.md](data_splits.md).

## Sources (rows in the manifest)

| `source_dataset` | Rows | Contributes |
|---|---:|---|
| TBX11K | 3,277 | Normal, Tuberculosis, and Pneumonia (proxy: "sick but not TB") |
| Existing_COVID-19 (COVID-QU / COVID-19 Radiography, renamed files) | 1,942 | COVID-19 (all of it) |
| VinBigData_VinDr_CXR | 1,467 | Pleural Effusion, Nodule/Mass |
| Existing_Pneumonia (Kermany, paediatric) | 1,395 | Pneumonia |
| Existing_Normal (Kermany, paediatric) | 1,199 | Normal |
| Existing_Tuberculosis (Shenzhen, renamed files) | 665 | Tuberculosis |
| NIH_ChestX-ray14 | 457 | Normal, Effusion, Nodule/Mass, Pneumonia |
| JSRT | 145 | Normal, Nodule/Mass |

## Manifest schema

These columns are required and validated by `lungai/data/manifest.py`: `image_id`, `image_path`, `patient_id`, `source_dataset`, `clinical_label`, `split`, `md5_hash`.

Informational columns: `dhash`, `width`, `height`, `aspect_ratio`, `mean_intensity`, `std_intensity`, `original_finding`, `mapping_confidence`, `view_position`. For 10,090 rows, `width`, `height` and the intensity columns hold defaults, not measurements.

## Locating images

`image_path` is relative to a **data root**, the folder that contains `data/raw/` and `data/downloads/`. Set the root with `LUNGAI_DATA_ROOT` or `--data-root`. The 211 historical absolute paths are relocated under the root automatically.

The images are not in this repository and must be obtained from the original public sources under their own licences. The external cohort is expected at `data/downloads/montgomery/` (`montgomery_metadata.csv` with `study_id` and `findings`, and images in `images/images/`).

## Versions

V1 to V4 manifests are kept in `experiments/data/` for history only. Only V5 is used.
