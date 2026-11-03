"""Tests for the health and root endpoints."""

import pytest


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    from main import STATIC_DIR
    if STATIC_DIR.is_dir():  # single-origin deployment: "/" serves the React build (backend/main.py)
        assert "text/html" in resp.headers["content-type"] and "<div id=\"root\">" in resp.text
        return
    data = resp.json()
    assert data["status"] == "running"
    assert "LungAI" in data["message"]
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    # "degraded" when no checkpoint is loaded (the checkpoint is not distributed with the repository)
    assert data["status"] == ("healthy" if data["model"]["model_loaded"] else "degraded")
    assert data["service_available"] is True
    assert data["model_ready"] == data["model"]["model_loaded"]
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_names_the_canonical_model(client):
    from lungai.config import MODEL_ID, CHECKPOINT_FILENAME
    model = (await client.get("/api/v1/health")).json()["model"]
    assert model["model_id"] == MODEL_ID
    assert model["checkpoint_file"] == CHECKPOINT_FILENAME
    if not model["model_loaded"]:
        assert model["load_error"] and "/" not in model["load_error"].replace("docs/inference.md", "")
