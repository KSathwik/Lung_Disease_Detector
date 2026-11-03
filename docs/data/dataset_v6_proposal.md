# Dataset V6 — proposal

**Status: PROPOSED.** No V6 manifest exists yet, and no metadata has been invented.

| Part | Location |
|---|---|
| Labelling rules | [`lungai/configs/dataset_v6_proposal.json`](../../lungai/configs/dataset_v6_proposal.json) |
| Schema and split logic | [`lungai/data/v6.py`](../../lungai/data/v6.py) |
| Builder | [`scripts/data/build_v6_manifest.py`](../../scripts/data/build_v6_manifest.py) |
| Tests | [`tests/data/test_data_scripts.py`](../../tests/data/test_data_scripts.py) |

## Why V6 is needed

V5 cannot support a clean evaluation, for the reasons evidenced in [v5_provenance.md](../../artifacts/data_quality/v5_provenance.md) and the [inventory](dataset_source_inventory.md):

- **Patient leakage:** 154 test images share a patient with training.
- **Unknown identity:** 8,478 images have no real patient identity.
- **Proxy labels:** 1,395 Pneumonia and 124 latent-TB labels are proxies.
- **Multi-finding images:** most VinDr images carry several findings.
- **External cohort:** probably contaminated (R1).

## Class inclusion and exclusion rules

These must be fixed **before** any V6 training. Every included image records the rule ID that labelled it. Every excluded image records why.

| Source | Included as | Excluded |
|---|---|---|
| NIH ChestX-ray14 | Exactly one finding: Effusion → Pleural Effusion; Nodule and/or Mass → Nodule/Mass; Pneumonia → Pneumonia; No Finding → Normal (one image per patient) | Any other or additional finding |
| VinDr-CXR (official) | Majority (≥2 of 3) radiologists: Pleural effusion without a majority for nodule/mass, lung tumour, pneumonia, TB or consolidation → Pleural Effusion. Nodule/Mass without a majority for effusion, pneumonia, TB or consolidation → Nodule/Mass. All radiologists "No finding" → Normal | No majority, or conflicting findings. Every co-finding is kept in `source_label_set` |
| TBX11K (official) | healthy → Normal; **active TB only** → Tuberculosis | **sick but non-TB** (cause unknown; not Pneumonia); **any latent-TB** annotation; unlabelled official test images |
| Shenzhen (NLM) | `_0` → Normal; `_1` → Tuberculosis | — |
| Rahman TB archive | — | **Whole source**: it aggregates NLM Montgomery, Shenzhen and Belarus images without identity. Replaced by official Shenzhen |
| Kermany (Mendeley v2) | NORMAL → Normal; PNEUMONIA → Pneumonia (decision D4) | — |
| COVID-19 Radiography | COVID → COVID-19 (decision D5) | Other classes in that archive |
| JSRT | Nodule (malignant **and** benign, CT-confirmed) → Nodule/Mass; non-nodule → Normal | — |
| Montgomery | — (role `external`) | Never in development splits |

Tuberculosis comes only from a source label of TB: active TB in TBX11K, or a TB manifestation in Shenzhen. An image is never labelled TB just because it comes from a TB-related dataset.

## Manifest schema (`manifest_v6.csv`)

| Column | Content | Rule |
|---|---|---|
| `image_id` | `<source_id>:<source_image_id>` | Unique |
| `source_id`, `source_version` | Registry ID and publisher version | Required |
| `source_image_id` | Publisher filename or ID, verbatim | Required |
| `source_patient_id` | Publisher patient ID, verbatim | Empty if none; **never generated** |
| `patient_id_status` | `genuine`, `derived_from_filename` or `unknown` | `genuine` and `derived_from_filename` require `source_patient_id` |
| `source_label_raw`, `source_label_set` | Publisher label(s) and all co-findings, verbatim | Required |
| `lungai_label`, `label_rule_id` | Canonical class and the rule that produced it | Empty when excluded |
| `exclusion_reason` | Why the image is not used | Empty when included |
| `role` | `development` or `external` | — |
| `view_position`, `age`, `sex` | As published | **Never imputed** |
| `raw_relpath` | Path under the raw root | — |
| `raw_md5`, `raw_sha256`, `phash` | Content hashes of the raw file | Measured |
| `width`, `height` | Measured from the file | Measured |
| `patient_group` | Grouping key used for splitting | Derived |
| `split` | `train`, `val`, `test`, `external` or `excluded` | Derived |

`validate_rows()` enforces these rules. The builder stops on any violation.

## Split methodology

1. **Groups.** The base group is the patient ID where one is known (genuine or derived from the filename). Otherwise each image is its own group.
2. **Duplicate merging.** Groups are merged with union-find across identical MD5. They are also merged across confirmed near-duplicates: pHash within 10 bits and 64×64 thumbnail correlation ≥ 0.97, run across all sources and including the external cohort. Duplicates therefore cannot cross splits.
3. **Assignment.** Whole groups are assigned to train/val/test at 70/15/15. Assignment is stratified by each group's majority (label, source), uses a fixed seed (20261102), and fills greedily toward the target image counts.
4. **External cohort.** Montgomery is `external`. Any development image matching it by MD5 or near-duplicate is excluded, not just kept out of test.
5. **Fingerprints.** `split_fingerprint` is the SHA-256 of the sorted `image_id<TAB>split` lines. It is recorded in the build report, in every training `run_metadata.json` and in every evaluation report (implemented for the current pipeline too).

## Test-set isolation

The test split is used once, by `lungai.evaluation.evaluate --final`. Preprocessing choices, thresholds, augmentation and model selection use train and val only. Per-source and per-class results on test are reported but never acted on.

## What V6 can and cannot claim

| | Claim |
|---|---|
| **Can** | Zero patient overlap for rows with `genuine` or `derived_from_filename` identity (NIH, Shenzhen, Montgomery, Kermany). Zero exact duplicates and zero confirmed near-duplicates across splits. No external-cohort image in development |
| **Cannot** | Patient separation for `unknown` rows (VinDr, TBX11K, COVID-19 Radiography). The build report states the share of such rows. Leakage there is reduced by near-duplicate merging but **not eliminated** |
| **Remaining confounding** | COVID-19 still has a single source. Pneumonia is paediatric-heavy if Kermany is kept. Report per-source metrics and a source-prediction control |

## Data-quality report (current evidence, manifest-level)

From `artifacts/data_quality/v5_provenance.json` and `artifacts/data_validation/`:

| Issue | V5 | V6 rule |
|---|---|---|
| Kermany patients spanning splits | 289 groups; 154 test images | Grouped by filename token → 0 by construction |
| Synthetic per-image patient IDs | 8,478 | `unknown`, stated as such |
| TBX11K sick-non-TB as Pneumonia | 1,395 | Excluded |
| TBX11K latent TB as TB | 124 | Excluded |
| VinDr multi-finding | 1,440 of 1,467 with ≥ 2 findings; 181 effusion+nodule | Majority vote; conflicting findings excluded |
| JSRT benign nodules | 54 excluded | Included as Nodule/Mass |
| Exact duplicates across splits | 0 (manifest MD5) | Merged into one group |
| Near-duplicates | Manifest dHash not informative (6,650 pairs ≤ 2 bits) | pHash + correlation on the images |
| External cohort in development | Name check only; R1 open | Content check; mandatory exclusion |

Image-level numbers for V6 (counts per class, near-duplicate clusters, measured image sizes) can only be produced after acquisition. None are claimed here.
