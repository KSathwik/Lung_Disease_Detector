# API

Base path: `/api/v1`. Interactive documentation is at `/docs` when the backend is running.

**Authentication.** If `LUNGAI_API_KEY` is set, every `/api/` route except `/api/v1/health` requires the header `X-API-Key: <key>`. If it is not set, only loopback clients are served. Either way, failures return 401.

| Method | Path | Purpose | Status codes |
|---|---|---|---|
| GET | `/health` | Service status (`service_available`) and model readiness (`model_ready`, `model.*`, `cxr_gate_mode`) | 200 |
| GET | `/model` | Canonical identity, class mapping, expected checkpoint SHA-256, load status | 200, 401 |
| GET | `/model-metrics` | Generated evaluation results of the configured checkpoint (`model_metrics.json`), with `evidence_level`, `matches_configured_checkpoint` and the V5 record under `historical` | 200, 401 |
| POST | `/predict` | Multipart `file` (+ optional `patient_id`, `scan_type`, `notes`) → prediction | 200, 400, 401, 404, 413, 422, 500, 503 |
| GET | `/predictions?skip&limit` | List stored predictions | 200, 401 |
| GET | `/predictions/{prediction_id}` | Stored prediction, including the six probabilities | 200, 401, 404 |
| POST | `/reports/generate/{prediction_id}` | Text report for a stored prediction | 200, 401, 404 |
| GET | `/reports` | List reports | 200, 401 |
| GET/POST/PUT | `/patients…` | Patient records | 200/201, 401, 404, 409, 422 |

## `POST /predict` response (200)

```json
{
  "prediction_id": "…", "scan_id": "…",
  "model_id": "lungai-densenet121-6class", "model_version": "v6.0",
  "selected_model": "LungAI DenseNet-121 — Six-Class Chest X-Ray Classifier",
  "model_sha256": "c51ed228…", "checkpoint_verified": true,
  "class_order": ["COVID-19", "Normal", "Pleural Effusion", "Pneumonia", "Pulmonary Nodule / Mass", "Tuberculosis"],
  "predicted_class": "Pneumonia", "predicted_index": 3,
  "confidence": 0.85, "inconclusive": false,
  "probabilities": {"COVID-19": 0.02, "Normal": 0.05, "Pleural Effusion": 0.03, "Pneumonia": 0.85, "Pulmonary Nodule / Mass": 0.02, "Tuberculosis": 0.03},
  "gradcam": {"overlay_image": "data:image/jpeg;base64,…", "original_image": "…", "target_class": "Pneumonia", "disclaimer": "…"},
  "final": {"condition": "Pneumonia", "confidence": 85.0, "inconclusive": false, "urgency": "urgent",
            "alternative_conditions": ["Normal (5.0%)", "…"], "key_findings": [{"label": "…", "value": "…"}],
            "precautions": ["…"], "disclaimer": "…"},
  "external_generalization_limitation": "…",
  "created_at": "2026-…"
}
```

`densenet` is also returned for older clients. It has the same content as `final.condition`, `final.confidence` and `probabilities` (as percentages).

## Error bodies

| Status | Body |
|---|---|
| 422 (gate) | `{"detail": {"error": "INVALID_CXR", "message": "…", "reason": "…", "score": 0.67}}` |
| 503 | `{"detail": {"error": "MODEL_NOT_LOADED", "message": "…", "reason": "<path-free cause>"}}` |
| 400, 404, 413, 500 | `{"detail": "<message>"}` |
