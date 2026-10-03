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

