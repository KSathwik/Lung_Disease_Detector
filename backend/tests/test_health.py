"""Tests for the health and root endpoints."""

import pytest


@pytest.mark.asyncio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    # "degraded" when no checkpoint is loaded (the checkpoint is not distributed with the repository)
    assert data["status"] == ("healthy" if data["model"]["model_loaded"] else "degraded")
    assert "timestamp" in data
