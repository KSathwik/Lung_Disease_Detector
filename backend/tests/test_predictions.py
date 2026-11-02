"""Tests for the prediction endpoints."""

import io
import pytest
import numpy as np
from PIL import Image


def _make_test_image() -> bytes:
    """Generate a minimal valid synthetic radiograph in memory."""
    arr = np.full((64, 64, 3), 40, dtype=np.uint8)
    arr[20:44, 20:44] = 180
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _make_car_photo_bytes() -> bytes:
    """Generate a colorful natural image (e.g. car photo with sky/body/grass)."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:35, :] = [235, 180, 50]   # blue sky
    arr[35:70, :] = [30, 30, 220]  # red car
    arr[70:, :] = [40, 180, 40]    # green grass
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.mark.asyncio
async def test_predict_bad_file_type(client):
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


CONFIDENT = [0.02, 0.05, 0.03, 0.85, 0.02, 0.03]   # Pneumonia
NEAR_UNIFORM = [0.17, 0.16, 0.17, 0.18, 0.16, 0.16]


@pytest.mark.asyncio
async def test_predict_without_model_returns_503_and_stores_nothing(client):
    from ml.inference import InferenceEngine
    if InferenceEngine().is_ready:
        pytest.skip("a real checkpoint is loaded")
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("scan.jpg", _make_test_image(), "image/jpeg")},
        data={"scan_type": "X-Ray"},
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"] == "MODEL_NOT_LOADED"
    listed = await client.get("/api/v1/predictions")
    assert listed.json()["total"] == 0


@pytest.mark.asyncio
async def test_predict_success_persists_model_output(client, stub_model):
    model = stub_model(CONFIDENT)
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("scan.jpg", _make_test_image(), "image/jpeg")},
        data={"scan_type": "X-Ray"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert model.calls == 1
    assert data["predicted_class"] == "Pneumonia"
    assert data["inconclusive"] is False
    assert data["probabilities"]["Pneumonia"] == pytest.approx(0.85, abs=1e-4)
    assert data["final"]["urgency"] == "urgent"
    assert "accuracy" not in (data.get("densenet") or {})
    from lungai.config import CLASS_NAMES, MODEL_DISPLAY_NAME, MODEL_ID
    assert data["model_id"] == MODEL_ID and data["selected_model"] == MODEL_DISPLAY_NAME
    assert data["class_order"] == CLASS_NAMES
    assert data["predicted_index"] == CLASS_NAMES.index("Pneumonia")
    # The API returns the model's scores unchanged (rounded to 4 d.p.), keyed by canonical class name
    assert [data["probabilities"][c] for c in CLASS_NAMES] == pytest.approx(CONFIDENT, abs=1e-4)

    detail = (await client.get(f"/api/v1/predictions/{data['prediction_id']}")).json()
    assert detail["probabilities"] == data["probabilities"]
    assert detail["model_sha256"] == "ab" * 32
    assert detail["final_condition"] == "Pneumonia"
    assert detail["model_used"] == MODEL_ID
    assert detail["model_version"] == data["model_version"]


@pytest.mark.asyncio
async def test_predict_rejects_non_probability_model_output(client, stub_model):
    """A model output that is not a softmax vector is an inference failure (500), never a result."""
    stub_model([0.9, 0.9, 0.0, 0.0, 0.0, 0.0])
    resp = await client.post("/api/v1/predict", files={"file": ("scan.jpg", _make_test_image(), "image/jpeg")})
    assert resp.status_code == 500
    assert (await client.get("/api/v1/predictions")).json()["total"] == 0


@pytest.mark.asyncio
async def test_predict_near_uniform_output_is_inconclusive(client, stub_model):
    stub_model(NEAR_UNIFORM)
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("scan.jpg", _make_test_image(), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["inconclusive"] is True
    assert data["final"]["urgency"] == "review"
    assert data["final"]["key_findings"] == []
    assert any("radiologist" in p for p in data["final"]["precautions"])


@pytest.mark.asyncio
async def test_upload_filename_cannot_escape_uploads_dir(client, stub_model, uploads_dir):
    stub_model(CONFIDENT)
    evil = "x/../../../TRAVERSAL_PROBE.jpg"
    resp = await client.post("/api/v1/predict", files={"file": (evil, _make_test_image(), "image/jpeg")})
    assert resp.status_code == 200
    saved = list(uploads_dir.iterdir())
    assert len(saved) == 1 and saved[0].suffix == ".jpg"
    assert "TRAVERSAL" not in saved[0].name
    assert not (uploads_dir.parent.parent / "TRAVERSAL_PROBE.jpg").exists()
    assert not (uploads_dir.parent / "TRAVERSAL_PROBE.jpg").exists()


def test_safe_display_name_strips_paths():
    from api.routes.predictions import safe_display_name
    assert safe_display_name("x/../../evil.png") == "evil.png"
    assert safe_display_name("..\\..\\Windows\\evil.png") == "evil.png"
    assert safe_display_name("") == "upload"
    assert safe_display_name(None) == "upload"


@pytest.mark.asyncio
async def test_predict_oversized_upload_413(client):
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("big.png", b"\0" * (10 * 1024 * 1024 + 1), "image/png")},
    )
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_predict_empty_upload_400(client):
    resp = await client.post("/api/v1/predict", files={"file": ("e.png", b"", "image/png")})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_remote_client_needs_api_key(remote_client, monkeypatch):
    monkeypatch.delenv("LUNGAI_API_KEY", raising=False)
    assert (await remote_client.get("/api/v1/patients")).status_code == 401
    assert (await remote_client.get("/api/v1/health")).status_code == 200


def test_local_client_detection():
    from types import SimpleNamespace as NS
    from main import _is_local_client
    assert _is_local_client(NS(client=NS(host="127.0.0.1")))
    assert _is_local_client(NS(client=NS(host="::1")))
    assert _is_local_client(NS(client=NS(host=None)))      # uvicorn: fully-local proxy chain
    assert not _is_local_client(NS(client=NS(host="203.0.113.7")))
    assert not _is_local_client(NS(client=None))


@pytest.mark.asyncio
async def test_api_key_enforced_when_configured(client, remote_client, monkeypatch):
    monkeypatch.setenv("LUNGAI_API_KEY", "s3cret-test-key")
    assert (await client.get("/api/v1/patients")).status_code == 401
    assert (await remote_client.get("/api/v1/patients", headers={"X-API-Key": "wrong"})).status_code == 401
    ok = await remote_client.get("/api/v1/patients", headers={"X-API-Key": "s3cret-test-key"})
    assert ok.status_code == 200


@pytest.mark.asyncio
async def test_health_reports_model_state(client):
    data = (await client.get("/api/v1/health")).json()
    assert data["status"] in ("healthy", "degraded")
    assert set(data["model"]) >= {"model_loaded", "checkpoint_sha256", "load_error"}
    assert data["cxr_gate_mode"] in ("trained_model", "heuristic_fallback")
    if not data["model"]["model_loaded"]:
        assert data["status"] == "degraded" and data["model"]["load_error"]
    # /health is unauthenticated: it must not reveal server filesystem paths
    from ml.paths import REPO_ROOT
    assert str(REPO_ROOT) not in resp_text(data)


def resp_text(obj) -> str:
    import json
    return json.dumps(obj)


@pytest.mark.asyncio
async def test_predict_with_nonexistent_patient(client):
    image_bytes = _make_test_image()
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("scan.jpg", image_bytes, "image/jpeg")},
        data={"patient_id": "DOESNOTEXIST"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_predictions_empty(client):
    resp = await client.get("/api/v1/predictions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_get_prediction_not_found(client):
    resp = await client.get("/api/v1/predictions/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_model_metrics_serves_recorded_canonical_results(client):
    resp = await client.get("/api/v1/model-metrics")
    assert resp.status_code == 200
    data = resp.json()
    from lungai.config import CLASS_NAMES, MODEL_ID
    assert data["model_id"] == MODEL_ID
    assert data["available"] is True
    assert data["class_order"] == CLASS_NAMES
    assert data["evidence_level"].startswith("RECORDED")
    internal = data["internal_test"]
    assert internal["consistency"] == "consistent"
    assert internal["n"] == 1570
    assert internal["metrics"]["accuracy"] == pytest.approx(1302 / 1570)
    # The external result is reported with its specificity, never as sensitivity alone
    b = data["external"]["binary_abnormality"]
    assert (b["sensitivity"]["successes"], b["sensitivity"]["n"]) == (58, 58)
    assert (b["specificity"]["successes"], b["specificity"]["n"]) == (0, 80)


@pytest.mark.asyncio
async def test_model_endpoint_reports_identity_and_class_mapping(client):
    from lungai.config import CLASS_NAMES, MODEL_DISPLAY_NAME, MODEL_ID
    data = (await client.get("/api/v1/model")).json()
    assert data["model_id"] == MODEL_ID
    assert data["display_name"] == MODEL_DISPLAY_NAME
    assert [c["name"] for c in data["classes"]] == CLASS_NAMES
    assert [c["index"] for c in data["classes"]] == list(range(6))
    assert "model_loaded" in data and "expected_checkpoint_sha256" in data
    assert "\\" not in str(data) and ":/" not in str(data)  # no filesystem paths


@pytest.mark.asyncio
async def test_predict_reject_natural_color_photo(client):
    """Verify that uploading a natural color photo (e.g. car) is rejected as an invalid radiograph."""
    car_bytes = _make_car_photo_bytes()
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("car.jpg", car_bytes, "image/jpeg")},
        data={"scan_type": "X-Ray"},
    )
    assert resp.status_code in (400, 422)
    detail_str = str(resp.json()["detail"])
    assert "Invalid Radiograph" in detail_str or "Invalid image" in detail_str or "INVALID_CXR" in detail_str

