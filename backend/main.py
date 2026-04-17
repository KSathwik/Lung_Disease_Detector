"""
Lung Disease Detection API - Main Application Entry Point
Academic Project | FastAPI Backend
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from api.routes import predictions, patients, reports, health
from database.connection import init_db, get_db
from utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting Lung Disease Detection API...")
    await init_db()
    logger.info("Database initialized successfully.")
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

# CORS middleware - allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
