# Reproducibility

What another researcher can reproduce, and what they need to do it:

| Step | Needs | Command | Reproducible from this repository alone? |
|---|---|---|---|
| Download V6 sources | Internet (Kaggle token for the COVID-19 Radiography Database) | `python scripts/data/acquire.py <source> --execute` | No: images are downloaded, not distributed; provenance and SHA-256 are logged |
| Build V6 (labels, caps, patient-grouped split, leakage checks) | Downloaded sources | `python scripts/data/build_v6_manifest.py …` | No: needs the images. The split fingerprint in the build report identifies an identical split |
| Training | V6 build; CPU works (hours) | `python -m lungai.training.train_v6 …` | No: needs the build. Single seed; expect small run-to-run differences |
| Evaluation of the current checkpoint | Checkpoint + build | `python -m lungai.evaluation.evaluate_v6 …` | No: needs both. Per-image predictions are saved under `artifacts/evaluation_reports/` |
| Served metrics and docs from saved reports | Saved reports + run metadata | `python -m lungai.evaluation.served_metrics …`; `python -m lungai.evaluation.render_docs` | Yes, once those files are present |
| V5 manifest checks and recorded V5 results | Repository | `python -m lungai.data.validate --skip-files`; `python -m lungai.evaluation.historical` | **Yes** (history) |
| Backend + frontend integration | Repository (+ checkpoint for real predictions) | see README quick start | Partly: without the checkpoint, `/predict` returns 503 |
| Test suites | Repository | `python -m pytest`; `CI=true npx react-scripts test --watchAll=false` | **Yes** (tests needing the checkpoint, run metadata or build report are skipped when those are absent) |

## Environment

- Python 3.12.4 was used on this branch (3.11 or 3.12 required). Dependencies are pinned in `backend/requirements.txt`, including TensorFlow 2.16.1, Keras 3.15.1, h5py 3.16.0, NumPy 1.26.4 and OpenCV 4.10.0.84.
- Node 18+, with `frontend/package-lock.json`.
- Transitive Python dependencies are not locked.

## Obtaining data

The images are public but are not redistributed here. `scripts/data/acquire.py` downloads each V6 source from its official location into `data/raw/<source>/` (registry: `scripts/data/sources.json`) and logs every file's URL, size and SHA-256 in `data/raw/_provenance/`. A rebuild with the same sources, rules and seed reproduces the split; compare its `split_fingerprint` with the one in the config. (For V5: place the images so the manifest paths resolve and run `python -m lungai.data.validate --data-root <root>`.)

## Fingerprints

| Artifact | SHA-256 |
|---|---|
| Current checkpoint, V6 split fingerprint, V6 manifest | In [the config](../lungai/configs/lungai_densenet121.json) (`model.checkpoint.sha256`, `dataset.split_fingerprint`) and in the generated [results](results/current_model.md) |
| V5 manifest (LF) | `8117a441d5d200e781df0af33685cde32c6c7a24a5e30d73f5744ec592702e26` |
| V5 manifest (CRLF, historical record) | `6730495a1e689ea8633595daa59b691d9798793a75b304f63d4f52c9cbeda2c4` |
| V5 checkpoint | `e2c95dc5eba588f877280948593f2965442d9950aa891ed5765eaf0117889022` (recorded; file lost) |

Each training run writes `run_metadata.json` with its git commit, configuration, library versions and checkpoint hash. Each evaluation writes the checkpoint and manifest hashes into its report.
