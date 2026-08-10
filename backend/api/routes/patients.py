"""
Patient Management API Routes
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum
import uuid
import re
from datetime import datetime

from database.connection import get_db, Patient

router = APIRouter()


class Gender(str, Enum):
    male = "Male"
    female = "Female"
    other = "Other"


class PatientCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    gender: Gender
    contact: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=100)
    medical_history: Optional[str] = None

    @field_validator("contact", "email", "medical_history", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        # Treat blank/whitespace-only optional fields as "not provided"
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email format")
        return v


class PatientUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[Gender] = None
    contact: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=100)
    medical_history: Optional[str] = None

    @field_validator("contact", "email", "medical_history", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email format")
        return v


class PatientResponse(BaseModel):
    id: int
    patient_id: str
    name: str
    age: int
    gender: str
    contact: Optional[str] = None
    email: Optional[str] = None
    medical_history: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PatientListResponse(BaseModel):
    total: int
    patients: List[PatientResponse]


@router.post("/patients", summary="Register a new patient")
async def create_patient(data: PatientCreate, db: AsyncSession = Depends(get_db)):
    patient = Patient(
        patient_id=str(uuid.uuid4()).upper(),
        **data.model_dump()
    )
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return {"message": "Patient registered.", "patient_id": patient.patient_id, "id": patient.id}


@router.get("/patients", response_model=PatientListResponse, summary="List all patients")
async def list_patients(skip: int = 0, limit: int = 20, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import func
    total_result = await db.execute(select(func.count(Patient.id)))
    total = total_result.scalar()

    result = await db.execute(select(Patient).offset(skip).limit(limit))
    patients = result.scalars().all()
    return PatientListResponse(total=total, patients=patients)


@router.get("/patients/{patient_id}", response_model=PatientResponse, summary="Get a patient by ID")
async def get_patient(patient_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return patient


@router.put("/patients/{patient_id}", response_model=PatientResponse, summary="Update a patient")
async def update_patient(
    patient_id: str, data: PatientUpdate, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    # Apply only the fields that were provided in the request
    updates = data.model_dump(exclude_unset=True)
    for key, value in updates.items():
        setattr(patient, key, value)

    await db.commit()
    await db.refresh(patient)
    return patient
