"""
Prediction API Routes
POST /api/v1/predict          — Analyze a lung image
GET  /api/v1/predictions      — List all predictions
GET  /api/v1/predictions/{id} — Get single prediction
GET  /api/v1/model            — Canonical model identity, class mapping and readiness
GET  /api/v1/model-metrics    — Recorded evaluation results of the canonical model (with evidence level)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path, PureWindowsPath
import os
import re
import uuid
import json
import aiofiles
from datetime import datetime, timezone

from database.connection import get_db, Prediction, LungScan, Patient
from ml.inference import InferenceEngine, ModelNotLoadedError
from lungai.config import INFERENCE, MODEL_ID, REPO_ROOT
from utils.logger import setup_logger

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)
router = APIRouter()


def get_engine() -> InferenceEngine:
    """Return the singleton InferenceEngine (already initialized at startup)."""
    return InferenceEngine()

# Content type -> file extension used for the stored copy (never derived from the client filename)
ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}
MAX_FILE_SIZE = int(INFERENCE["max_upload_bytes"])  # 10 MB
RECORDED_RESULTS_FILE = REPO_ROOT / "artifacts" / "evaluation_reports" / "historical_record_verification.json"


def safe_display_name(filename: Optional[str]) -> str:
    """Basename of the client-supplied filename with path separators and control characters removed (for display only)."""
    name = PureWindowsPath(filename or "").name  # handles both "/" and "\" separators
    name = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", name).strip(" .")
    return (name or "upload")[:255]


# ─── Schemas ────────────────────────────────────────────────────────────────

class ModelResult(BaseModel):
    condition: str
    confidence: float
    all_probabilities: Optional[Dict[str, float]] = None

class FinalResult(BaseModel):
    condition: str
    confidence: float
    inconclusive: bool = False
    urgency: str
    alternative_conditions: List[str]
    key_findings: List[Dict[str, str]]
    precautions: List[str]
    disclaimer: str

class PredictionResponse(BaseModel):
    prediction_id: str
    scan_id: str
    model_id: str
    model_version: str
    selected_model: str
    model_sha256: Optional[str] = None
    checkpoint_verified: bool
    class_order: List[str]
    predicted_class: Optional[str] = None
    predicted_index: Optional[int] = None
    confidence: Optional[float] = None
    inconclusive: bool = False
    probabilities: Optional[Dict[str, float]] = None
    gradcam: Optional[Dict[str, Any]] = None
    external_generalization_limitation: Optional[str] = None
    densenet: Optional[ModelResult] = None
    final: FinalResult
    created_at: str

    model_config = {"protected_namespaces": ()}


class PredictionDetail(BaseModel):
    """Full prediction detail returned from GET /predictions/{id}."""
    id: int
    prediction_id: str
    scan_id: int
    model_used: str
    selected_model: Optional[str] = None
    model_version: Optional[str] = None
    model_sha256: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    inconclusive: Optional[bool] = None
    final_condition: str
    final_confidence: float
    urgency_level: str
    alternative_conditions: Optional[List[str]] = None
    key_findings: Optional[List[Dict[str, str]]] = None
    precautions: Optional[List[str]] = None
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class PredictionSummary(BaseModel):
    prediction_id: str
    final_condition: str
    final_confidence: float
    inconclusive: Optional[bool] = None
    urgency_level: str
    selected_model: Optional[str] = None
    created_at: str


class PredictionListResponse(BaseModel):
    total: int
    predictions: List[PredictionSummary]


def _invalid_cxr(reason: str, score: Optional[float] = None) -> HTTPException:
    detail = {
        "error": "INVALID_CXR",
        "message": "The uploaded image does not appear to be a chest radiograph. Please upload a valid chest X-ray.",
        "reason": reason,
    }
    if score is not None:
        detail["score"] = round(score, 4)
    return HTTPException(status_code=422, detail=detail)


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictionResponse, summary="Analyze a chest X-ray")
async def predict(
    file: UploadFile = File(..., description="Frontal chest X-ray (JPEG, PNG, BMP, TIFF or WebP, max 10 MB)"),
    patient_id: Optional[str] = Form(None),
    scan_type: Optional[str] = Form("X-Ray"),
    notes: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a chest X-ray and receive a six-class prediction from the canonical LungAI DenseNet-121 classifier.

    - Returns 503 if no model checkpoint is loaded (no placeholder predictions are ever produced)
    - Returns class probabilities, Grad-CAM, and an inconclusive flag for low-confidence outputs
    - Stores the result (including the six probabilities and checkpoint hash) in the database
    """
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Accepted: JPEG, PNG, BMP, TIFF, WebP"
        )

    # Read at most MAX_FILE_SIZE + 1 bytes so oversized uploads are rejected without buffering them fully
    image_bytes = await file.read(MAX_FILE_SIZE + 1)
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    # ── Semantic CXR vs Non-CXR Validation Gate ──
    # Keeps obvious non-radiographs away from the classifier
    from ml.cxr_gate import get_cxr_gate
    gate = get_cxr_gate()
    is_cxr, gate_score, rejection_reason = gate.validate(image_bytes)
    if not is_cxr:
        logger.warning(f"Rejected non-CXR image: {rejection_reason} (score={gate_score:.3f})")
        raise _invalid_cxr(rejection_reason, gate_score)

    # Resolve patient before inference so an unknown patient_id fails fast (nothing is committed on later errors)
    db_patient_id: int
    if patient_id:
        result = await db.execute(
            select(Patient).where(Patient.patient_id == patient_id)
        )
        patient = result.scalar_one_or_none()
        if not patient:
            raise HTTPException(
                status_code=404,
                detail=f"Patient '{patient_id}' not found. Register the patient first."
            )
        db_patient_id = patient.id
    else:
        result = await db.execute(
            select(Patient).where(Patient.patient_id == "DEFAULT")
        )
        default_patient = result.scalar_one_or_none()
        if not default_patient:
            default_patient = Patient(
                patient_id="DEFAULT",
                name="Unregistered Patient",
                age=0,
                gender="Unknown",
            )
            db.add(default_patient)
            await db.flush()
        db_patient_id = default_patient.id

    # Run inference (only reached if gate passes). Without a checkpoint there is no prediction to return.
    engine = get_engine()
    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "MODEL_NOT_LOADED",
                "message": "The prediction model is not loaded, so no analysis can be performed.",
                "reason": engine.public_load_error,
            },
        )
    try:
        results = engine.predict(image_bytes)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail={"error": "MODEL_NOT_LOADED", "message": str(e)})
    except ValueError as e:
        # Preprocessor validation failure
        err_msg = str(e)
        if "Invalid Radiograph" in err_msg or "color photograph" in err_msg or "too small" in err_msg:
            raise _invalid_cxr(err_msg)
        raise HTTPException(status_code=400, detail=f"Invalid image: {err_msg}")
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Model inference failed.")

    # Save uploaded image to disk under a server-generated name; the client filename is stored as metadata only
    scan_id = str(uuid.uuid4())
    pred_id = str(uuid.uuid4())
    image_filename = safe_display_name(file.filename)
    image_path = UPLOADS_DIR / f"{scan_id}{ALLOWED_TYPES[file.content_type]}"

    async with aiofiles.open(image_path, "wb") as f_out:
        await f_out.write(image_bytes)

    # Create scan record
    scan = LungScan(
        scan_id=scan_id,
        patient_id=db_patient_id,
        image_path=str(image_path),
        image_filename=image_filename,
        image_size_bytes=len(image_bytes),
        scan_type=scan_type or "X-Ray",
        notes=notes,
        preprocessed=True
    )
    db.add(scan)
    await db.flush()

    final = results["final"]
    dn_r  = results.get("densenet")

    prediction = Prediction(
        prediction_id=pred_id,
        scan_id=scan.id,
        model_used=results["model_id"],
        selected_model=results["selected_model"],
        model_version=results["model_version"],
        probabilities=results.get("probabilities"),
        model_sha256=results.get("model_sha256"),
        inconclusive=bool(results.get("inconclusive")),
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
        model_id=results["model_id"],
        model_version=results["model_version"],
        selected_model=results["selected_model"],
        model_sha256=results.get("model_sha256"),
        checkpoint_verified=bool(results["checkpoint_verified"]),
        class_order=results["class_order"],
        predicted_class=results.get("predicted_class"),
        predicted_index=results.get("predicted_index"),
        confidence=results.get("confidence"),
        inconclusive=bool(results.get("inconclusive")),
        probabilities=results.get("probabilities"),
        gradcam=results.get("gradcam"),
        external_generalization_limitation=results.get("external_generalization_limitation"),
        densenet=ModelResult(**dn_r) if dn_r else None,
        final=FinalResult(**final),
        created_at=datetime.now(timezone.utc).isoformat()
    )


@router.get("/predictions", response_model=PredictionListResponse, summary="List all predictions")
async def list_predictions(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    total_result = await db.execute(select(func.count(Prediction.id)))
    total = total_result.scalar()

    result = await db.execute(
        select(Prediction).offset(skip).limit(limit).order_by(Prediction.created_at.desc())
    )
    preds = result.scalars().all()
    return PredictionListResponse(
        total=total,
        predictions=[
            PredictionSummary(
                prediction_id=p.prediction_id,
                final_condition=p.final_condition,
                final_confidence=p.final_confidence,
                inconclusive=p.inconclusive,
                urgency_level=p.urgency_level,
                selected_model=p.selected_model,
                created_at=p.created_at.isoformat()
            )
            for p in preds
        ]
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionDetail, summary="Get a single prediction")
async def get_prediction(prediction_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Prediction).where(Prediction.prediction_id == prediction_id)
    )
    pred = result.scalar_one_or_none()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return pred


@router.get("/model", summary="Canonical model identity, class mapping and readiness")
async def get_model():
    return get_engine().identity()


@router.get("/model-metrics", summary="Recorded evaluation results of the canonical model")
async def get_model_metrics():
    """
    Recorded results for the canonical model, regenerated by `python -m lungai.evaluation.historical`.
    These are aggregate results from an earlier run, checked against their own confusion matrices; they are
    not produced by the checkpoint this server has loaded.
    """
    if not RECORDED_RESULTS_FILE.is_file():
        return {
            "model_id": MODEL_ID,
            "available": False,
            "message": "No recorded evaluation results found. Run: python -m lungai.evaluation.historical",
            "model_loaded": get_engine().is_ready,
        }
    record = json.loads(RECORDED_RESULTS_FILE.read_text(encoding="utf-8"))
    canonical = next(e for e in record["entries"] if e["canonical"])
    internal = canonical["internal_test"]
    return {
        "model_id": MODEL_ID,
        "available": True,
        "model": record["canonical_model"],
        "model_loaded": get_engine().is_ready,
        "evidence_level": record["evidence_level"],
        "generated_at": record["generated_at"],
        "dataset_version": record["dataset_version"],
        "class_order": record["class_order"],
        "internal_test": {
            "split": internal["split"],
            "n": internal["n"],
            "consistency": internal["status"],
            "metrics": internal["recomputed"],
            "recorded_unverifiable": internal["recorded_unverifiable"],
        },
        "external": canonical["external"],
        "validation_evidence": canonical["validation_evidence"],
    }
