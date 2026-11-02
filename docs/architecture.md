# Architecture

```mermaid
flowchart TB
  subgraph PKG["lungai/ — canonical package"]
    CFG["configs/lungai_densenet121.json<br/>identity · classes · preprocessing · training · evaluation"]
    CONF["config.py"]
    PRE["preprocessing.py"]
    MAN["data/manifest.py · checks.py · splits.py · validate.py"]
    MOD["models/densenet.py<br/>build_model · load_checkpoint (SHA-256)"]
    TRN["training/train.py"]
    EVA["evaluation/metrics.py · evaluate.py · historical.py"]
    CFG --> CONF --> PRE & MAN & MOD & TRN & EVA
  end
  DATA[("data/manifests/unified_manifest_v5.csv<br/>+ images under LUNGAI_DATA_ROOT")] --> MAN
  MAN --> TRN --> RUNS[("runs/&lt;name&gt;/model.h5 + run_metadata.json")]
  MAN --> EVA
  EXP[("experiments/*/…metrics.json<br/>(historical)")] --> EVA
  EVA --> ART[("artifacts/evaluation_reports/")]
  MAN --> ART2[("artifacts/data_validation/")]

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
  ENG --> CK[("models/densenet121_frequency_v5.h5")]
  R1 --> ART
  FE["frontend/ — React<br/>AnalyzePage · HistoryPage · MetricsPage · PatientsPage<br/>services/api.js · prediction.js · metrics.js · errors.js"] -- "HTTP /api/v1 (X-API-Key)" --> MAIN
```

## Data to prediction

| Stage | Implementation |
|---|---|
| Dataset definition | `data/manifests/unified_manifest_v5.csv`, with its checksum checked by `lungai.data.manifest.load_manifest` |
| Data validation | `lungai.data.validate` → `artifacts/data_validation/` |
| Splitting | V5 split preserved as recorded. `lungai.data.splits.grouped_stratified_split` is for the next version |
| Preprocessing | `lungai.preprocessing` (one implementation) |
| Training | `lungai.training.train` (train and val only) |
| Checkpoint selection | Best validation macro-F1 (`training.checkpoint_selection`) |
| Evaluation | `lungai.evaluation.evaluate` (val, or test/external with `--final`) |
| Inference API | `backend/ml/inference.py` → `lungai.preprocessing` + `lungai.models.densenet.load_checkpoint` |
| Frontend | `frontend/src/pages/AnalyzePage.js` → `services/prediction.js` (class order and scores read from the response) |

## Endpoint → service → model

| Endpoint | Service | Model / data used |
|---|---|---|
| `POST /api/v1/predict` | `CXRGate.validate` → `InferenceEngine.predict` | Canonical checkpoint (SHA-256 checked), `lungai.preprocessing`, class order from config. 503 if not loaded |
| `GET /api/v1/model` | `InferenceEngine.identity()` | Config identity + load status |
| `GET /api/v1/model-metrics` | `predictions.get_model_metrics` | `artifacts/evaluation_reports/historical_record_verification.json` (canonical entry) |
| `GET /api/v1/health` | `InferenceEngine.status()`, `CXRGate.mode` | — |
| `GET /api/v1/predictions[/{id}]` | SQLAlchemy | Stored probabilities, model ID/version, checkpoint SHA-256 |
| `POST /api/v1/reports/generate/{id}` | `reports.py` | Stored prediction |
| `/api/v1/patients*` | `patients.py` | Patient records |

## Dependency direction

`backend` imports `lungai`. `lungai` never imports `backend`. `experiments/` is historical, and nothing active imports it.
