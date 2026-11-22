# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/predict` returns 503 `MODEL_NOT_LOADED`; health says `degraded` | No checkpoint in `models/` | `python scripts/install_checkpoint.py --from <file>` |
| `load_error` mentions SHA-256 | The file is not the canonical checkpoint | Use the canonical file. `LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT=true` is for testing only |
| `load_error` says "could not be loaded" | Corrupt file, or a Keras/h5py version mismatch | Reinstall the pinned `backend/requirements.txt` in a clean venv |
| 401 from every API route | Remote client without an API key | Set `LUNGAI_API_KEY` on the server and send `X-API-Key`, or `REACT_APP_API_KEY` at frontend build time |
| UI shows "Server Unreachable" | Backend not running on :8000, or the proxy cannot reach it | Start the backend (`cd backend && python main.py`) |
| UI shows "Not a Chest X-Ray" for a real radiograph | Heuristic gate rejection (e.g. very bright or inverted export) | Known limitation; see [limitations.md](limitations.md) |
| `ManifestError: checksum mismatch` | The manifest was edited | Do not edit V5. Create a new dataset version and update the config |
| Training or evaluation exits with code 2 and "images are missing" | `LUNGAI_DATA_ROOT` is wrong, or the data is incomplete | Point it at the folder that contains `data/raw` and `data/downloads`. Run `python -m lungai.data.validate` |
| `--split test` refused | Final splits need `--final` | Add `--final` only for the one-off final report |
| `ImportError: lungai` when running the backend from another folder | The repository root is not on `sys.path` | Run from `backend/`, or `pip install -e .` from the repository root |
| TensorFlow fails to install | Python 3.13+ | Use Python 3.11 or 3.12 |
