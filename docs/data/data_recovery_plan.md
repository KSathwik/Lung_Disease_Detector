# Data recovery and acquisition plan

**Status: awaiting approval.** Nothing in this plan has been executed. Every command below is dry-run by default (or needs data that is not present).

## Principles

- **Publishers first.** Download from the publisher's official channel. Kaggle is used only where the dataset authors themselves publish there (COVID-19 Radiography, Rahman TB), or to recover the exact historical files of a third-party derivative. Recovering a derivative requires separate approval because it is a mirror.
- **Respect access controls.** Logins, registrations and PhysioNet credentialing are done by you. The scripts never handle credentials. Kaggle downloads use the official `kaggle` CLI with your own configured token.
- **Layout.** Raw files go in `<root>/raw/<source_id>/` exactly as published. Derived files go in `<root>/derived/…`. Every download is logged in `<root>/raw/_provenance/<source_id>.jsonl` (URL, UTC time, bytes, SHA-256, and whether a publisher checksum was checked).
- **Nothing is overwritten.** Existing files and output folders are never overwritten. The V5 manifest and historical results are read-only.
- **Images stay out of git.** `data/raw/`, `data/downloads/` and `data/derived/` are gitignored.

## Step 0 — decisions needed from you

| # | Decision | Default proposed |
|---|---|---|
| D1 | Data root location: a disk with ≥ 300 GB free if official VinDr DICOMs are included, otherwise ≥ 60 GB | `LUNGAI_RAW_ROOT=<your path>` |
| D2 | Recover V5 exactly (needs the same Kaggle derivatives) or build V6 from official sources only | Both: recover what the official sources reproduce byte-for-byte, and build V6 from official sources |
| D3 | VinDr-CXR via PhysioNet (credentialed; ~200 GB of DICOM, estimate) | Yes, if you can complete credentialing; otherwise drop VinDr and use NIH for effusion and nodule/mass |
| D4 | Keep Kermany (paediatric) in V6? | Yes, with per-source reporting |
| D5 | Keep the COVID-19 Radiography archive as the COVID source? | Yes; record that patient identity is unknown |
| D6 | Read and accept each source's licence (R7 in the inventory) | Required before any download |

## Step 1 — open sources (no login)

```bash
python scripts/data/acquire.py --list
```
```bash
python scripts/data/acquire.py nih_cxr14
```

- **Approve:** run `python scripts/data/acquire.py nih_cxr14 --execute`. This downloads `images_001.tar.gz` (the archive V5 used; about 2 GB, estimate) with resume. The NIH metadata `Data_Entry_2017_v2020.csv` must be saved manually from the Box folder.
- **Montgomery and Shenzhen:** download manually from the NLM directory listing (`CXR_png/`, `ClinicalReadings/`, the ReadMe PDF) into `<root>/raw/montgomery/` and `<root>/raw/shenzhen/`. `acquire.py` deliberately does not crawl directory listings.

## Step 2 — account or registration sources (manual)

| Source | What you do | Target folder |
|---|---|---|
| COVID-19 Radiography | `kaggle datasets download -d tawsifurrahman/covid19-radiography-database`, or `acquire.py covid_radiography --execute` with your Kaggle token | `raw/covid_radiography/` |
| Rahman TB | `acquire.py rahman_tb --execute`. **Only to test R1:** this archive is not proposed for V6 training | `raw/rahman_tb/` |
| Kermany | Download Mendeley v2 (`ChestXRay2017.zip`) from the DOI page | `raw/kermany/` |
| TBX11K | Official link from mmcheng.net/tb (licence to confirm) | `raw/tbx11k/` |
| JSRT | Register on the JSRT site and download the 247 images and clinical data | `raw/jsrt/` |
| VinDr-CXR | Complete PhysioNet credentialing, sign the DUA, then download v1.0.0 and verify `SHA256SUMS.txt` | `raw/vindr_cxr/` |

## Step 3 — verify

```bash
python scripts/data/verify_recovery.py --search <root>/raw
```

This matches every recovered file to the V5 manifest by MD5. It shows, per source, how much of V5 is recoverable byte-for-byte. Expected result: Kermany, NIH and the Rahman archives may match exactly. The VinDr and TBX11K derivatives will not match official files.

```bash
python scripts/data/check_external_overlap.py --external <root>/raw/montgomery --candidates <root>/raw/rahman_tb <root>/raw/shenzhen <root>/raw/tbx11k
```

This settles R1. Any match means V5's external evaluation was contaminated, and the image must be excluded from every development split.

```bash
python -m lungai.data.validate --data-root <root>
```

This runs the V5 file checks (existence, MD5, decodability) for whatever was recovered.

## Step 4 — adapters, then build V6 (derived)

| Adapter | Status |
|---|---|
| NIH, Montgomery, Shenzhen, Kermany | Written and tested on synthetic fixtures with the publishers' documented layouts |
| VinDr | Checks for the documented PhysioNet label file |
| TBX11K, COVID-19 Radiography, JSRT | Stop with an error until their official files have been inspected |

```bash
python scripts/data/build_v6_manifest.py --raw-root <root>/raw --out <root>/derived/v6_build_001
```

Outputs:
- `manifest_v6.csv`: every candidate image with its rule ID or exclusion reason.
- `v6_build_report.json`: counts, leakage summary, split fingerprint, and hashes of the rules file and the manifest.

**Review gate:** the V6 manifest is copied into `data/manifests/` and the config is updated only after you approve this report.

## Effort and storage

All figures are estimates, not measurements:

| Item | Estimate | Assumption |
|---|---|---|
| NIH `images_001` + metadata | ~2 GB | One of 12 archives (whole set ~42 GB) |
| Montgomery + Shenzhen | ~1–4 GB | Full-resolution PNG |
| COVID-19 Radiography | ~1 GB | Kaggle archive |
| Rahman TB (for the R1 check) | ~1 GB | Kaggle archive |
| Kermany (Mendeley v2) | ~1.2 GB | The chest X-ray archive only (the record also holds OCT data, which is not needed) |
| TBX11K | ~3.3 GB | Third-party listing |
| JSRT | ~2 GB | 247 × 2048² 12-bit RAW |
| VinDr-CXR (PhysioNet) | ~200 GB | DICOM; plus conversion to PNG (~20 GB) |
| **Total without VinDr** | **~12–15 GB** raw + similar derived | |
| Download time | ~1–2 h without VinDr at 50 Mbit/s; +10 h for VinDr | |
| Hashing and V6 build | ~30–60 min CPU without VinDr | |
