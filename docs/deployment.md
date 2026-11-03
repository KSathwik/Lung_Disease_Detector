# Deployment

How to run the LungAI backend and frontend on one machine or in a container. This is a research prototype (see [limitations](limitations.md)); nothing here describes a clinical or multi-tenant deployment. Every statement below is taken from the files cited; no infrastructure beyond them is assumed.

**Docker status.** The `Dockerfile` has **not been built or tested** in this project: Docker is not installed on the development laptop. Section 4 documents what the file says it does, not what has been observed.

## 1. System requirements

| Item | Requirement | Source |
|---|---|---|
| Python | 3.11 or 3.12 (`>=3.11,<3.13`). TensorFlow 2.16.1 does not support 3.13+. | `pyproject.toml`; header of `backend/requirements.txt`; [troubleshooting](troubleshooting.md) |
| Node.js | 18 or later. The Dockerfile builds the frontend with `node:18-alpine`; `frontend/package.json` declares no `engines` field. | `Dockerfile`, `frontend/package.json` |
| TensorFlow / Keras | `tensorflow==2.16.1`, `keras==3.15.1`, `h5py==3.16.0`, `numpy==1.26.4`, all pinned. Match the Keras version to the one that saved the checkpoint. | `backend/requirements.txt` |
| Hardware | CPU only. The project was developed and trained on an 8 GB CPU laptop running Windows (`PROJECT_EXECUTION_STATUS.md`, [training](training.md)); the pinned TensorFlow is used without a GPU. | as listed |
| RAM and latency | Not measured or documented in the repository. Do not quote a figure. | n/a |
| Disk | Checkpoint 28.7 MB (`model.checkpoint.availability` in `lungai/configs/lungai_densenet121.json`), plus the SQLite file and uploaded images. | config |

Install the backend dependencies in a clean virtual environment:

```bash
python -m venv .venv                     # Python 3.11 or 3.12
.venv\Scripts\activate                   # Windows; on other systems: source .venv/bin/activate
pip install -r backend/requirements.txt
```

## 2. Local development run

Two processes: the backend on loopback and the React dev server, which proxies API calls to it.

```bash
# terminal 1: backend
cd backend
python main.py                           # uvicorn on API_HOST:API_PORT, default 127.0.0.1:8000

# terminal 2: frontend
cd frontend
npm install                              # or npm ci
npm start                                # react-scripts start, dev server on port 3000
```

- `python main.py` calls `uvicorn.run("main:app", host=API_HOST, port=API_PORT)` with defaults `127.0.0.1` and `8000` (end of `backend/main.py`). Run it from `backend/` so that `main`, `api`, `ml` and `database` import correctly; the `lungai` package is found when the repository root is on `sys.path` (see [troubleshooting](troubleshooting.md): run from `backend/` or `pip install -e .`).
- `frontend/package.json` sets `"proxy": "http://localhost:8000"`, and `frontend/src/services/api.js` uses the relative base `/api/v1` (override with `REACT_APP_API_URL`). The browser talks only to the dev server, which forwards `/api/...` to the backend.
- Behind that proxy the backend sees a loopback client, so no API key is needed (section 10).
- Interactive API docs are at `http://127.0.0.1:8000/docs`.
- Without a checkpoint the backend still starts, but `/api/v1/predict` returns 503 (section 5).

## 3. Production-style single-origin run (no Docker)

`backend/main.py` defines `STATIC_DIR = Path(__file__).resolve().parent / "static"`, i.e. `backend/static`. **If that directory exists when the module is imported**, the app mounts it at `/` with `SPAStaticFiles`, a `StaticFiles` subclass with `html=True`. A request that produces 404 and whose path does not start with `/api/` is answered with `static/index.html`, so client-side routes work on reload. If the directory does not exist, `GET /` returns a small JSON status message instead. The API routers are registered before the mount, so `/api/v1/...` is unaffected.

```bash
cd frontend
npm ci
npm run build                            # writes frontend/build
# copy the build to backend/static, for example in PowerShell:
#   Copy-Item -Recurse frontend\build backend\static
cd ../backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

The UI and API are then served from one origin, so CORS is not involved. To listen on other interfaces, pass `--host 0.0.0.0` **and set `LUNGAI_API_KEY`**; without a key every `/api/` request from a non-loopback client gets 401 (section 10). To send the key from the browser, build the frontend with `REACT_APP_API_KEY` set to the same value; it is embedded in the JavaScript bundle (`frontend/src/services/api.js`), so anyone who can load the page can read it. Treat it as a shared demo secret, not as user authentication. This command runs a single worker; the inference engine is a per-process singleton without a lock ([inference](inference.md), "Concurrency").

## 4. Docker (untested)

The `Dockerfile` has two stages:

1. `node:18-alpine`: `npm ci`, then `npm run build` of `frontend/`. Optional build argument `REACT_APP_API_KEY`.
2. `python:3.11-slim`: installs `libgl1` and `libglib2.0-0` (needed by OpenCV), then `pip install -r requirements.txt`; copies `backend/` to `/app`, `lungai/` to `/app/lungai`, `artifacts/evaluation_reports/model_metrics.json` to `/app/artifacts/evaluation_reports/`, and the frontend build to `/app/static`; creates `models/` and `uploads/`; `EXPOSE 8000`; `CMD uvicorn main:app --host 0.0.0.0 --port 8000`.

Commands implied by the file (**not run in this project**):

```bash
docker build -t lungai .
# optional: --build-arg REACT_APP_API_KEY=<same value as LUNGAI_API_KEY>
docker run --rm -p 8000:8000 \
  -e LUNGAI_API_KEY=<secret> \
  -v /path/to/densenet121_v6_seed42.h5:/app/models/densenet121_v6_seed42.h5:ro \
  lungai
```

Facts from the files:

- `.dockerignore` excludes `models/*.h5`, `models/*.keras`, `data/`, `docs/`, `.env`, `*.db`, `logs/` and `frontend/node_modules/`. The checkpoint is therefore never in the image and must be mounted (the Dockerfile comment gives the same `-v` line). The optional gate model `cxr_gate_model.keras` is excluded too, so the container uses the heuristic gate unless you mount that file as well (section 9).
- The container listens on `0.0.0.0`, and the Dockerfile comment says requests from outside the container are accepted only when `LUNGAI_API_KEY` is set. Whether a key-less request published through `-p` would be treated as local was not tested. Always set the key.
- The Dockerfile declares no volume. The SQLite file (`/app/lung_disease.db`, from the relative default URL and `WORKDIR /app`) and `/app/uploads` live in the container's writable layer and are lost when the container is removed. If you need them kept, mount storage for them (for example a volume on `/app/uploads` and a `DATABASE_URL` that points into a mounted folder). This is a suggestion, not a tested configuration.
- The image uses Python 3.11; the local setup allows 3.11 or 3.12.

## 5. Model installation and verification

The checkpoint `densenet121_v6_seed42.h5` is not in the repository (`*.h5` is gitignored) and has no public release (`scripts/install_checkpoint.py` docstring; `model.checkpoint.availability` in the config).

```bash
python scripts/install_checkpoint.py --from runs/<run>/model.h5    # verify, then copy into models/
python scripts/install_checkpoint.py --url https://<location you control>/densenet121_v6_seed42.h5
python scripts/install_checkpoint.py --check                        # verify models/densenet121_v6_seed42.h5
```

The script stages the file in a temporary folder, computes its SHA-256 and moves it to `models/` only if the digest equals `model.checkpoint.sha256` in `lungai/configs/lungai_densenet121.json` (currently `c51ed228455c2aab93b4b0220ee721443c3c0486eea961a242829ce68ee6c0e6`). A mismatch prints "Not installed." and exits with code 1. `--url` is a plain `urllib` download; the hash is checked after the download, so use a location you control.

At start-up the backend (`backend/ml/inference.py`) looks for the file in the folder returned by `resolve_models_dir()` (`backend/ml/paths.py`): `MODELS_DIR` if set (absolute, or relative to the repository root or `backend/`), otherwise `<repo>/models`, otherwise `backend/models`. It verifies the SHA-256 again. The loading rules (missing, mismatch, corrupt, canonical) are tabulated in [inference](inference.md). In short: without a verified checkpoint `/api/v1/health` reports `degraded` and `/api/v1/predict` returns 503 `MODEL_NOT_LOADED`; there is no fallback or mock model.

`LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT` (`1`, `true` or `yes`, case-insensitive) makes the backend load a file whose hash differs. Every prediction response then carries `checkpoint_verified: false` and the UI shows a warning. Use it for testing only, never for results you report.

## 6. Served metrics file

`GET /api/v1/model-metrics` and the limitation notice attached to every prediction are built from `artifacts/evaluation_reports/model_metrics.json` (`METRICS_FILE` in `backend/ml/inference.py`; the Dockerfile copies this one file into the image).

- The file is generated, not typed: `python -m lungai.evaluation.served_metrics --eval-prefix v6 --run runs/<run>` copies numbers from the per-split evaluation reports and the run's `run_metadata.json` (`lungai/evaluation/served_metrics.py`).
- It must **match the checkpoint**: its `canonical_model.checkpoint_sha256` must equal the configured `model.checkpoint.sha256`. `/api/v1/model-metrics` reports this comparison as `matches_configured_checkpoint`. `external_limitation()` returns a generic notice without numbers when the hashes differ, when the file is missing, or when an expected key is absent.
- The notice is computed once, when `ml.inference` is imported, so restart the backend after replacing the file.
- If the file is missing, `/api/v1/model-metrics` returns `available: false` with a message.

## 7. Configuration (environment variables)

The backend loads a `.env` file with `python-dotenv` (`load_dotenv()` at the top of `backend/main.py`); `backend/.env.example` is the template. All variables read by backend code:

| Variable | Default | Meaning | Read in |
|---|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./lung_disease.db` | SQLAlchemy async URL. The relative SQLite path resolves against the working directory (`backend/lung_disease.db` when started from `backend/`). `.env.example` shows a PostgreSQL URL, but `asyncpg` is not in `requirements.txt` and PostgreSQL was not tested. | `backend/database/connection.py` |
| `SQL_ECHO` | `false` | `true`, `1` or `yes` logs every SQL statement. | same |
| `API_HOST` | `127.0.0.1` | Bind address for `python main.py` only (not for a direct `uvicorn` command). | `backend/main.py` |
| `API_PORT` | `8000` | Port for `python main.py` only. | same |
| `LUNGAI_API_KEY` | unset | If set, every `/api/` route except `/api/v1/health` requires a matching `X-API-Key` header (compared with `hmac.compare_digest`). If unset, only loopback clients are served and a warning is logged at start-up. | same |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:3001` | Comma-separated allowed origins (whitespace is not trimmed). Credentials, all methods and all headers are allowed for those origins. | same |
| `MODELS_DIR` | `<repo>/models` | Folder holding the checkpoint (and the optional gate model). | `backend/ml/paths.py` |
| `LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT` | unset (false) | See section 5. | `backend/ml/inference.py` |
| `CONFIDENCE_FLOOR` | `0.5` (config `inference.confidence_floor`) | A top-class softmax below this is reported as inconclusive. Not clinically validated. | `backend/ml/inference.py` |
| `UPLOADS_DIR` | `uploads` | Folder for uploaded images, created at import time, relative to the working directory. | `backend/api/routes/predictions.py` |

Variables used only by the research pipeline: `LUNGAI_DATA_ROOT` (`lungai/config.py`; folder containing `data/raw` and `data/downloads`), and `LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT`, which `lungai/evaluation/evaluate.py` also reads. Frontend build-time variables: `REACT_APP_API_KEY` and `REACT_APP_API_URL` (`frontend/src/services/api.js`).

## 8. Health check

`GET /api/v1/health` needs no API key (`ApiKeyMiddleware` skips it) and answers 200 while the process is up (`backend/api/routes/health.py`):

| Field | Meaning |
|---|---|
| `status` | `healthy` when the model is loaded, otherwise `degraded` |
| `service_available` | always `true` |
| `model_ready` | the model is loaded (verified, or unverified with the override) |
| `timestamp`, `version` | UTC time; API version `1.0.0` |
| `model.model_id`, `model_name`, `model_version` | configured identity |
| `model.model_loaded`, `checkpoint_file`, `checkpoint_sha256` | load state and file |
| `model.checkpoint_verified` | `true`, `false`, or `null` when not loaded |
| `model.load_error` | path-free cause when not loaded |
| `model.gradcam_available` | the Grad-CAM model was built |
| `cxr_gate_mode` | `trained_model` or `heuristic_fallback` |

Use `status == "healthy"` (or `model_ready`) as a readiness signal and any 200 as liveness. The Dockerfile defines no `HEALTHCHECK`.

## 9. CXR input gate

`backend/ml/cxr_gate.py` rejects non-radiographs before classification (422 `INVALID_CXR`). Stage 1 checks size (at least 32x32), aspect ratio (at most 2.2:1), intensity standard deviation (at least 10.0), mean brightness (at most 200.0) and colour saturation. Stage 2 uses a trained 128x128 classifier `cxr_gate_model.keras` if one is found (in the models folder, else `experiments/cxr_gate/models/`); otherwise a hand-written anatomical-profile score is used and `cxr_gate_mode` is `heuristic_fallback`.

Thresholds come from `inference.cxr_gate` in `lungai/configs/lungai_densenet121.json`:

| Threshold | Value | Used for | Calibration source in the config |
|---|---|---|---|
| `model_threshold` | 0.83 | output of the trained gate model | none recorded in the config |
| `heuristic_threshold` | 0.703 | fallback score | `scripts/data/calibrate_cxr_gate.py` on `v6_build_002` validation radiographs (`artifacts/app_checks/cxr_gate_calibration_v6_build_002.json`): the lowest threshold that still rejects every available non-radiograph test image (highest stage-1-passing negative 0.692, plus 0.01). It passes 91.0% of validation radiographs (38.7% at 0.83). The negatives are synthetic, so rejection of real non-radiographs is not established. |

The gate is a heuristic, not a validated detector. Genuine radiographs can be rejected (for example very bright or inverted exports); see [limitations](limitations.md).

## 10. Security

- **API key.** See section 7. Failures return 401. `/api/v1/health` is the only unauthenticated API route. `/docs` and the static UI are not under `/api/` and are not key-protected.
- **Loopback-only default.** Without `LUNGAI_API_KEY`, `_is_local_client()` accepts a request only if the peer is `127.0.0.1`, `::1`, `::ffff:127.0.0.1` or `localhost`, or if uvicorn reports no client host (the same-machine proxy case explained in a comment in `backend/main.py`). `API_HOST` defaults to `127.0.0.1`.
- **CORS.** Only the `CORS_ORIGINS` list. The single-origin setup needs no CORS configuration.
- **Response headers.** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`, `Referrer-Policy: strict-origin-when-cross-origin`, `Cache-Control: no-store`. The app has no TLS; if it is exposed beyond one machine, a TLS-terminating reverse proxy would be needed (none is provided or tested here).
- **Upload limits.** Content type must be JPEG, PNG, BMP, TIFF or WebP; at most `inference.max_upload_bytes` = 10,485,760 bytes (413 beyond that; the body is read with a cap); empty files are rejected (400); stored filenames are server-generated `{uuid}{ext}` (`backend/api/routes/predictions.py`).
- **Retention.** The backend contains no code that deletes uploaded images or database rows (no `unlink`, `os.remove` or `rmtree` in `backend/` outside the tests). Uploads and records stay until an operator removes them.
- **No PHI.** The system is a research demo built on public, de-identified data. Do not enter real patient data: images are stored unencrypted on disk, and patient records and free-text notes are stored in plain text in SQLite. There is no access control beyond the shared API key.

## 11. Data storage, logging, backup

- **Database.** SQLite at `DATABASE_URL` (default `./lung_disease.db`, relative to the working directory). Tables are created at start-up with `create_all`, and columns added since the first release are added with `ALTER TABLE` (`init_db()` in `backend/database/connection.py`). No other migration tool is used by the app.
- **Uploads.** `UPLOADS_DIR` (default `uploads/`), one file per accepted scan.
- **Logging.** `backend/utils/logger.py` attaches a stream handler (standard error) at INFO level with the format `time | level | name | message`; uvicorn adds its own access log. There is no log file, rotation or remote sink. Start-up messages state whether the model loaded and whether `LUNGAI_API_KEY` is set.
- **Backup.** Nothing is automated. With the service stopped, copy the database file and the uploads folder together, because predictions reference stored images. The checkpoint has no public copy, so keep your own copy and its SHA-256; `model_metrics.json` can be regenerated (section 6).

## 12. Upgrading to a new checkpoint

Run from the repository root after training and the once-only evaluation:

1. **Install.** Update the `model.checkpoint` block (filename, `sha256`) in `lungai/configs/lungai_densenet121.json` from the new run's `run_metadata.json` (`checkpoint_sha256`), then `python scripts/install_checkpoint.py --from runs/<run>/model.h5` and `python scripts/install_checkpoint.py --check`.
2. **Served metrics.** `python -m lungai.evaluation.served_metrics --eval-prefix <prefix> --run runs/<run>` regenerates `artifacts/evaluation_reports/model_metrics.json`.
3. **Render docs.** `python -m lungai.evaluation.render_docs` refreshes `docs/results/current_model.md` and the generated blocks in `README.md`, `docs/model_card.md` and `docs/experiment_results.md`; text outside the markers is not touched ([evaluation](evaluation.md)).
4. **Tests.** `python -m pytest` from the repository root ([testing](testing.md)). `tests/test_docs_metrics.py` and `tests/test_consistency.py` exist to catch disagreement between served metrics, configuration and documentation.
5. **Restart** the backend (or rebuild the image and mount the new checkpoint). Confirm that `GET /api/v1/health` shows `healthy`, `checkpoint_verified: true` and the new hash, and that `/api/v1/model-metrics` shows `matches_configured_checkpoint: true`.

Other configuration fields may also need editing for a new model; rely on the tests in step 4 rather than on this list being complete.

## 13. Troubleshooting

[troubleshooting](troubleshooting.md) has the full table. Deployment-relevant rows: 503 `MODEL_NOT_LOADED` (no checkpoint: run the installer); a `load_error` naming SHA-256 (wrong file); 401 on every API route (remote client without `LUNGAI_API_KEY` / `X-API-Key`); "Server Unreachable" in the UI (backend not on port 8000); `ImportError: lungai` (run from `backend/` or `pip install -e .`); TensorFlow install failure (Python 3.13+). In addition, a generic limitation notice without numbers means `model_metrics.json` is missing or belongs to another checkpoint (section 6), and `cxr_gate_mode: heuristic_fallback` means no gate model file was found (section 9).
