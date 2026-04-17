"""Health check route."""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health", summary="API health check")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}
