"""
Prediction API Routes
POST /api/v1/predict          — Analyze a lung image
GET  /api/v1/predictions      — List all predictions
GET  /api/v1/predictions/{id} — Get single prediction
GET  /api/v1/model-metrics    — Training metrics comparison
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

from database.connection import get_db, Prediction, LungScan, Patient
from ml.inference import InferenceEngine
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()
engine = InferenceEngine()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ─── Schemas ────────────────────────────────────────────────────────────────

class ModelResult(BaseModel):
    condition: str
    confidence: float
    accuracy: Optional[float] = None
    all_probabilities: Optional[Dict[str, float]] = None

class FinalResult(BaseModel):
    condition: str
    confidence: float
    urgency: str
    alternative_conditions: List[str]
    key_findings: List[Dict[str, str]]
    precautions: List[str]
    disclaimer: str

class PredictionResponse(BaseModel):
    prediction_id: str
    scan_id: str
    selected_model: str
    cnn: Optional[ModelResult]
    resnet: Optional[ModelResult]
    final: FinalResult
    created_at: str


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse, summary="Analyze a lung image")
async def predict(
    file: UploadFile = File(..., description="Chest X-ray, CT scan, or MRI image"),
    patient_id: Optional[str] = Form(None),
    scan_type: Optional[str] = Form("X-Ray"),
    notes: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a lung image and receive AI-powered disease detection.
    
    - Runs both CNN and ResNet models
    - Returns confidence scores, findings, and precautions
    - Stores results in database for reporting
    """
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Accepted: JPEG, PNG, BMP, TIFF, WebP"
        )

    # Read and validate size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    # Run inference
    try:
        results = engine.predict(image_bytes)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # Save to DB
    scan_id = str(uuid.uuid4())[:12]
    pred_id = str(uuid.uuid4())[:12]
    image_path = f"uploads/{scan_id}_{file.filename}"

    # Create scan record
    scan = LungScan(
        scan_id=scan_id,
        patient_id=1,           # Default patient; link properly via patient_id in prod
        image_path=image_path,
        image_filename=file.filename or "upload.jpg",
        image_size_bytes=len(image_bytes),
        scan_type=scan_type or "X-Ray",
        notes=notes,
        preprocessed=True
    )
    db.add(scan)
    await db.flush()

    final = results["final"]
    cnn_r = results.get("cnn")
    rn_r  = results.get("resnet")

    prediction = Prediction(
        prediction_id=pred_id,
        scan_id=scan.id,
        model_used="Both",
        selected_model=results.get("selected_model", "ResNet"),
        cnn_primary_condition=cnn_r["condition"]  if cnn_r else None,
        cnn_confidence=cnn_r["confidence"]         if cnn_r else None,
        cnn_accuracy=cnn_r.get("accuracy")         if cnn_r else None,
        cnn_all_probabilities=cnn_r.get("all_probabilities") if cnn_r else None,
        resnet_primary_condition=rn_r["condition"] if rn_r else None,
        resnet_confidence=rn_r["confidence"]        if rn_r else None,
        resnet_accuracy=rn_r.get("accuracy")        if rn_r else None,
        resnet_all_probabilities=rn_r.get("all_probabilities") if rn_r else None,
        final_condition=final["condition"],
        final_confidence=final["confidence"],
        urgency_level=final["urgency"],
        alternative_conditions=final["alternative_conditions"],
        key_findings=final["key_findings"],
        precautions=final["precautions"]
    )
    db.add(prediction)
    await db.commit()

    return PredictionResponse(
        prediction_id=pred_id,
        scan_id=scan_id,
        selected_model=results.get("selected_model", "ResNet"),
        cnn=ModelResult(**cnn_r) if cnn_r else None,
        resnet=ModelResult(**rn_r) if rn_r else None,
        final=FinalResult(**final),
        created_at=datetime.utcnow().isoformat()
    )


@router.get("/predictions", summary="List all predictions")
async def list_predictions(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Prediction).offset(skip).limit(limit).order_by(Prediction.created_at.desc())
    )
    preds = result.scalars().all()
    return {
        "total": len(preds),
        "predictions": [
            {
                "prediction_id": p.prediction_id,
                "final_condition": p.final_condition,
                "final_confidence": p.final_confidence,
                "urgency_level": p.urgency_level,
                "selected_model": p.selected_model,
                "created_at": p.created_at.isoformat()
            }
            for p in preds
        ]
    }


@router.get("/predictions/{prediction_id}", summary="Get a single prediction")
async def get_prediction(prediction_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Prediction).where(Prediction.prediction_id == prediction_id)
    )
    pred = result.scalar_one_or_none()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return pred


@router.get("/model-metrics", summary="Get training metrics for both models")
async def get_model_metrics():
    metrics = engine.get_model_metrics()
    if not metrics:
        return {
            "message": "Models not yet trained. Run training script first.",
            "hint": "python backend/ml/train.py"
        }
    return metrics
