"""Health check route."""
from fastapi import APIRouter
from datetime import datetime, timezone


router = APIRouter()

@router.get("/health", summary="API health check")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "version": "1.0.0"}

