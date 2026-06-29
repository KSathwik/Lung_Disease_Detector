"""Tests for the patient management endpoints."""

import pytest


@pytest.mark.asyncio
async def test_create_patient(client):
    resp = await client.post("/api/v1/patients", json={
        "name": "Test Patient",
        "age": 45,
        "gender": "Male",
        "email": "test@example.com",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Patient registered."
    assert "patient_id" in data


@pytest.mark.asyncio
async def test_create_patient_invalid_age(client):
    resp = await client.post("/api/v1/patients", json={
        "name": "Bad Age",
        "age": -5,
        "gender": "Male",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_patient_invalid_email(client):
    resp = await client.post("/api/v1/patients", json={
        "name": "Bad Email",
        "age": 30,
        "gender": "Female",
        "email": "not-an-email",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_patient_invalid_gender(client):
    resp = await client.post("/api/v1/patients", json={
        "name": "Bad Gender",
        "age": 30,
        "gender": "InvalidGender",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_patients_empty(client):
    resp = await client.get("/api/v1/patients")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["patients"] == []


@pytest.mark.asyncio
async def test_get_patient_not_found(client):
    resp = await client.get("/api/v1/patients/NONEXISTENT")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_patients_after_create(client):
    await client.post("/api/v1/patients", json={
        "name": "Alice", "age": 30, "gender": "Female",
    })
    resp = await client.get("/api/v1/patients")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["patients"][0]["name"] == "Alice"
