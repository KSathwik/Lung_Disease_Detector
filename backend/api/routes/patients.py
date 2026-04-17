"""
Patient Management API Routes
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
import uuid

from database.connection import get_db, Patient

router = APIRouter()


class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    contact: Optional[str] = None
    email: Optional[str] = None
    medical_history: Optional[str] = None


@router.post("/patients", summary="Register a new patient")
async def create_patient(data: PatientCreate, db: AsyncSession = Depends(get_db)):
    patient = Patient(
        patient_id=str(uuid.uuid4())[:10].upper(),
        **data.model_dump()
    )
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return {"message": "Patient registered.", "patient_id": patient.patient_id, "id": patient.id}


@router.get("/patients", summary="List all patients")
async def list_patients(skip: int = 0, limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Patient).offset(skip).limit(limit))
    patients = result.scalars().all()
    return {"total": len(patients), "patients": patients}


@router.get("/patients/{patient_id}", summary="Get a patient by ID")
async def get_patient(patient_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return patient
