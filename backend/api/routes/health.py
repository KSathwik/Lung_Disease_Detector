"""Health check route."""
from fastapi import APIRouter
from datetime import datetime, timezone

from ml.inference import InferenceEngine
from ml.cxr_gate import get_cxr_gate


router = APIRouter()

@router.get("/health", summary="API health check")
async def health():
    model = InferenceEngine().status()
    return {
        # The API answering at all means the service is available; "degraded" means it cannot produce predictions
        "status": "healthy" if model["model_loaded"] else "degraded",
        "service_available": True,
        "model_ready": model["model_loaded"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "model": model,
        "cxr_gate_mode": get_cxr_gate().mode,
    }
