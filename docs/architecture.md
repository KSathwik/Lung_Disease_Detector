# Architecture

```mermaid
flowchart TB
  subgraph PKG["lungai/ — canonical package"]
    CFG["configs/lungai_densenet121.json<br/>identity · classes · preprocessing · training · evaluation"]
    CONF["config.py"]
    PRE["preprocessing.py"]
    MAN["data/v6.py (grouping, splits, leakage checks)"]
    MOD["models/densenet.py<br/>build_model · load_checkpoint (SHA-256)"]
    TRN["training/train_v6.py"]
    EVA["evaluation/metrics.py · evaluate_v6.py · served_metrics.py · render_docs.py"]
    HIST["V5 tools (historical): data/manifest.py · validate.py · training/train.py · evaluation/evaluate.py · historical.py"]
    CFG --> CONF --> PRE & MAN & MOD & TRN & EVA
  end
  RAW[("data/raw/: official downloads<br/>(scripts/data/acquire.py, provenance logged)")] --> BLD["scripts/data/build_v6_manifest.py"]
  PRE --> BLD
  MAN --> BLD --> DATA[("data/derived/v6_build_*<br/>manifest_v6.csv · cache/*.npy · v6_build_report.json")]
  DATA --> TRN --> RUNS[("runs/&lt;name&gt;/model.h5 + run_metadata.json")]
  DATA --> EVA
  RUNS --> EVA
  EVA --> ART[("artifacts/evaluation_reports/<br/>model_metrics.json")]
  EXP[("experiments/*/…metrics.json")] --> HIST

  subgraph BE["backend/ — FastAPI"]
    MAIN["main.py<br/>lifespan · API-key middleware · static SPA"]
    R1["routes/predictions.py<br/>/predict · /predictions · /model · /model-metrics"]
    R2["routes/patients.py · reports.py · health.py"]
    GATE["ml/cxr_gate.py (heuristic input gate)"]
    ENG["ml/inference.py InferenceEngine"]
    BPRE["ml/preprocessing.py (upload checks)"]
    DB[("SQLite: patients · lung_scans · predictions · reports")]
    MAIN --> R1 & R2
    R1 --> GATE --> ENG --> BPRE
    R1 --> DB
  end
  BPRE --> PRE
  ENG --> MOD
  ENG --> CK[("models/densenet121_v6_seed42.h5")]
  R1 --> ART
  FE["frontend/ — React<br/>AnalyzePage · HistoryPage · MetricsPage · PatientsPage<br/>services/api.js · prediction.js · metrics.js · errors.js"] -- "HTTP /api/v1 (X-API-Key)" --> MAIN
```

## Data to prediction

| Stage | Implementation |
|---|---|
| Acquisition | `scripts/data/acquire.py` (registry `scripts/data/sources.json`) → `data/raw/`, provenance in `data/raw/_provenance/` |
| Dataset definition | `scripts/data/build_v6_manifest.py` under `lungai/configs/dataset_v6_proposal.json` → `data/derived/<build>/manifest_v6.csv` (V5 manifest: history) |
| Splitting and leakage checks | Patient-grouped 70/15/15 in `lungai.data.v6`; groups merged across exact and near-duplicates; checks and split fingerprint in `v6_build_report.json` |
| Preprocessing | `lungai.preprocessing` (one implementation), applied once by the builder and cached; the backend calls the same functions |
| Training | `lungai.training.train_v6` (train and val only) |
| Checkpoint selection | Best validation macro-F1 (`training.checkpoint_selection`) |
| Evaluation | `lungai.evaluation.evaluate_v6` (val; test and external once, with `--final`) → `served_metrics` → `model_metrics.json` |
| Inference API | `backend/ml/inference.py` → `lungai.preprocessing` + `lungai.models.densenet.load_checkpoint` |
| Frontend | `frontend/src/pages/AnalyzePage.js` → `services/prediction.js` (class order and scores read from the response) |

## Endpoint → service → model

| Endpoint | Service | Model / data used |
|---|---|---|
| `POST /api/v1/predict` | `CXRGate.validate` → `InferenceEngine.predict` | Canonical checkpoint (SHA-256 checked), `lungai.preprocessing`, class order from config. 503 if not loaded |
| `GET /api/v1/model` | `InferenceEngine.identity()` | Config identity + load status |
| `GET /api/v1/model-metrics` | `predictions.get_model_metrics` | `artifacts/evaluation_reports/model_metrics.json` (current checkpoint; the V5 record is included under `historical`); `matches_configured_checkpoint` flags a mismatch |
| `GET /api/v1/health` | `InferenceEngine.status()`, `CXRGate.mode` | — |
| `GET /api/v1/predictions[/{id}]` | SQLAlchemy | Stored probabilities, model ID/version, checkpoint SHA-256 |
| `POST /api/v1/reports/generate/{id}` | `reports.py` | Stored prediction |
| `/api/v1/patients*` | `patients.py` | Patient records |

## Dependency direction

`backend` imports `lungai`. `lungai` never imports `backend`. `experiments/` is historical, and nothing active imports it.
