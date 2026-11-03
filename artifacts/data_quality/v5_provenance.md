# V5 provenance reconstruction

Generated 2026-11-02T16:44:33+00:00 by `python scripts/data/analyze_v5_provenance.py` from the manifests only. Do not edit by hand.

| Source (manifest) | Registry id | Images | Labels | Manifest patient IDs | Derived groups | Patient identity |
|---|---|---:|---|---:|---:|---|
| Existing_COVID-19 | covid_radiography | 1,942 | COVID-19 1942 | 1,942 | 1,942 | unknown (archive renamed files COVID-<n>.png; upstream IDs only via source URLs in the archive metadata) |
| Existing_Normal | kermany | 1,199 | Normal 1199 | 1,199 | 976 | recoverable from filename token (Kermany personNNN / IM-NNNN); manifest uses synthetic per-image IDs |
| Existing_Pneumonia | kermany | 1,395 | Pneumonia 1395 | 1,395 | 780 | recoverable from filename token (Kermany personNNN); manifest uses synthetic per-image IDs |
| Existing_Tuberculosis | rahman_tb | 665 | Tuberculosis 665 | 665 | 665 | unknown (archive renamed files Tuberculosis-<n>.png) |
| JSRT | jsrt | 145 | Pulmonary Nodule / Mass 78, Normal 67 | 145 | 145 | one image per case (per JSRT documentation, to confirm); manifest uses per-case IDs |
| NIH_ChestX-ray14 | nih_cxr14 | 457 | Normal 171, Pulmonary Nodule / Mass 140, Pleural Effusion 131, Pneumonia 15 | 302 | 302 | genuine (NIH Patient ID, from filename prefix) |
| TBX11K | tbx11k | 3,277 | Pneumonia 1395, Normal 1199, Tuberculosis 683 | 3,277 | 3,277 | unknown (dataset publishes no patient IDs); manifest uses synthetic per-image IDs |
| VinBigData_VinDr_CXR | vindr_cxr | 1,467 | Pleural Effusion 931, Pulmonary Nodule / Mass 536 | 1,345 | 1,345 | unknown (VinDr-CXR is de-identified per image); manifest 'vinbigdata_pt_N' grouping has undocumented origin |

## VinDr: findings per image (V1 record)

- 1,440 of 1,467 V5 VinDr images carry 2 or more distinct findings.
- 181 images labelled Pleural Effusion also carry Nodule/Mass.
- Distinct findings per image: {1: 27, 2: 101, 3: 237, 4: 344, 5: 302, 6: 227, 7: 142, 8: 64, 9: 20, 10: 3}
- Rule used: experiments/rebuild_unified_dataset_v2.py: 'Pleural_effusion' in cats and len(cats) <= 3 (WBF COCO categories); effusion takes priority over Nodule/Mass. Class-id mapping: assumed VinBigData competition order (to verify).

## TBX11K: V5 label vs original category

- Normal: {'not_in_v1_record': 871, 'no_tb_healthy_none': 328}
- Pneumonia: {'not_in_v1_record': 969, 'no_tb_sick_but_no_tb_none': 426}
- Tuberculosis: {'tb_tb_active_tb': 559, 'tb_tb_latent_tb': 124}

## JSRT: V5 label vs original category

- Normal: {'non-nodule': 67}
- Pulmonary Nodule / Mass: {'malignant': 78}

## Near-duplicate candidates (manifest dHash)

| Hamming distance | Pairs | Across splits | Across sources | Label conflicts |
|---:|---:|---:|---:|---:|
| 0 | 3 | 1 | 3 | 2 |
| 1 | 41 | 16 | 41 | 36 |
| 2 | 6,650 | 3,074 | 3,184 | 3,633 |

The ingestion dHash algorithm is undocumented and a 64-bit dHash discriminates chest X-rays poorly (their coarse structure is similar), so pair counts grow quickly with distance. These are review candidates only; V6 needs a stronger near-duplicate method computed from the images.
