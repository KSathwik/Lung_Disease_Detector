"""Tests for the prediction endpoints."""

import io
import pytest
import numpy as np
from PIL import Image


def _make_test_image() -> bytes:
    """Generate a minimal valid JPEG image in memory."""
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
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


@pytest.mark.asyncio
async def test_predict_success_demo_mode(client):
    image_bytes = _make_test_image()
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("scan.jpg", image_bytes, "image/jpeg")},
        data={"scan_type": "X-Ray"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction_id" in data
    assert "final" in data
    assert data["final"]["condition"]
    assert data["final"]["confidence"] > 0


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
async def test_model_metrics(client):
    resp = await client.get("/api/v1/model-metrics")
    assert resp.status_code == 200
