# Reproducibility

What another researcher can reproduce, and what they need to do it:

| Step | Needs | Command | Reproducible from this repository alone? |
|---|---|---|---|
| Manifest integrity, counts, leakage, duplicates | Repository | `python -m lungai.data.validate --skip-files` | **Yes** |
| Recorded results consistency | Repository | `python -m lungai.evaluation.historical` | **Yes** |
| Image existence, md5, decodability | Images under `LUNGAI_DATA_ROOT` | `python -m lungai.data.validate` | No: needs the images |
| Training | Images + GPU (recommended) | `python -m lungai.training.train` | No: needs the images |
| Evaluation of the recorded checkpoint | Checkpoint + images | `python -m lungai.evaluation.evaluate …` | No: needs both |
| Backend + frontend integration | Repository (+ checkpoint for real predictions) | see README quick start | Partly: without the checkpoint, `/predict` returns 503 |
| Test suites | Repository | `python -m pytest`; `CI=true npx react-scripts test --watchAll=false` | **Yes** (checkpoint tests are skipped) |

## Environment

- Python 3.12.4 was used on this branch (3.11 or 3.12 required). Dependencies are pinned in `backend/requirements.txt`, including TensorFlow 2.16.1, Keras 3.15.1, h5py 3.16.0, NumPy 1.26.4 and OpenCV 4.10.0.84.
- Node 18+, with `frontend/package-lock.json`.
- Transitive Python dependencies are not locked.

## Obtaining data

The images are public but are not redistributed here. Download each source (see [dataset.md](dataset.md)) into `<root>/data/raw/…` and `<root>/data/downloads/…` so that the manifest paths resolve. Then run `python -m lungai.data.validate --data-root <root>`. It reports every missing file and every md5 mismatch. The image-level work is reproducible only if this report shows 0 missing and 0 mismatched files.

## Fingerprints

| Artifact | SHA-256 |
|---|---|
| Manifest (LF) | `8117a441d5d200e781df0af33685cde32c6c7a24a5e30d73f5744ec592702e26` |
| Manifest (CRLF, historical record) | `6730495a1e689ea8633595daa59b691d9798793a75b304f63d4f52c9cbeda2c4` |
| Canonical checkpoint | `e2c95dc5eba588f877280948593f2965442d9950aa891ed5765eaf0117889022` (recorded; file not available) |

Each training run writes `run_metadata.json` with its git commit, configuration, library versions and checkpoint hash. Each evaluation writes the checkpoint and manifest hashes into its report.
