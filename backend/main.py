"""
Lung Disease Detection API - Main Application Entry Point
Academic Project | FastAPI Backend
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import os
import logging
from contextlib import asynccontextmanager

from api.routes import predictions, patients, reports, health
from database.connection import init_db, get_db
from ml.inference import InferenceEngine
from utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting Lung Disease Detection API...")
    await init_db()
    logger.info("Database initialized successfully.")
    engine = InferenceEngine()
    engine.initialize()
    logger.info("Inference engine initialized.")
    yield
    logger.info("Shutting down Lung Disease Detection API...")


app = FastAPI(
    title="Lung Disease Detection API",
    description="""
    ## Academic Lung Disease Detection System
    
    This API provides AI-powered lung disease detection from medical images.
    
    **Important:** This tool is designed to assist medical professionals,
    not replace clinical judgment. Always consult a qualified physician.
    
    ### Features:
    - Upload chest X-rays, CT scans
    - Two ML models (CNN + ResNet) with accuracy comparison
    - Confidence scoring and differential diagnosis
    - Patient record management
    - Detailed medical reports
    """,
    version="1.0.0",
    lifespan=lifespan
)

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:3001"
).split(",")

# CORS middleware - allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Register routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
app.include_router(patients.router, prefix="/api/v1", tags=["Patients"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])


@app.get("/")
async def root():
    return {
        "message": "Lung Disease Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
