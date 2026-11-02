"""
Lung Disease Detection API - Main Application Entry Point
Academic Project | FastAPI Backend
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from pathlib import Path
import hmac
import uvicorn
import os
import logging
from contextlib import asynccontextmanager

from api.routes import predictions, patients, reports, health
from lungai.config import MODEL_DISPLAY_NAME
from database.connection import init_db, get_db
from ml.inference import InferenceEngine
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Built React app (present in the Docker image, see Dockerfile)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting Lung Disease Detection API...")
    await init_db()
    logger.info("Database initialized successfully.")
    engine = InferenceEngine()
    engine.initialize()
    if engine.is_ready:
        logger.info("Inference engine initialized.")
    else:
        logger.error(f"Inference engine has NO model loaded; /predict will return 503. {engine.load_error}")
    if not os.getenv("LUNGAI_API_KEY"):
        logger.warning("LUNGAI_API_KEY is not set: API requests are accepted from loopback clients only.")
    yield
    logger.info("Shutting down Lung Disease Detection API...")


app = FastAPI(
    title=f"LungAI API — {MODEL_DISPLAY_NAME}",
    description="""
    ## Academic Lung Disease Detection System (LungAI)

    This API provides AI-powered 6-class lung disease triage and detection from chest radiographs.
    One model: the LungAI DenseNet-121 Six-Class Chest X-Ray Classifier (see `GET /api/v1/model`), with Grad-CAM.

    **Important:** This research prototype is not a medical device and must not be used for clinical decisions.

    ### Features:
    - Chest radiograph (CXR) ingestion (JPEG, PNG, BMP, TIFF, WebP &le; 10MB)
    - Six-class inference with the canonical checkpoint only (503 when it is missing or fails its SHA-256 check)
    - Interactive Grad-CAM attention heatmap overlays
    - Confidence scoring with an inconclusive state for low-confidence outputs
    - Urgency categorization (Routine, Urgent, Emergency, Review)
    - Patient record and historical scan management
    - Structured report generation

    ### Authentication
    Set `LUNGAI_API_KEY` and send it as the `X-API-Key` header. Without it, only loopback clients are served.
    """,
    version="1.0.0",
    lifespan=lifespan
)

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:3001"
).split(",")

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}


def _is_local_client(request: Request) -> bool:
    if request.client is None:
        return False
    # uvicorn's proxy-header handling sets the host to None when the connecting peer is 127.0.0.1 and every
    # X-Forwarded-For hop is 127.0.0.1 too (e.g. the CRA dev-server proxy on the same machine). A remote client
    # behind a local reverse proxy keeps its real address and is still rejected.
    return request.client.host is None or request.client.host in LOOPBACK_HOSTS


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """
    Protects every /api/ route except /api/v1/health.
    - LUNGAI_API_KEY set: requests must carry a matching X-API-Key header.
    - not set: only loopback clients are allowed (safe default for a local demo).
    """
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS" or not path.startswith("/api/") or path == "/api/v1/health":
            return await call_next(request)

        api_key = os.getenv("LUNGAI_API_KEY")
        if api_key:
            supplied = request.headers.get("x-api-key", "")
            if not hmac.compare_digest(supplied.encode(), api_key.encode()):
                return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key."})
        else:
            if not _is_local_client(request):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Remote access requires an API key. Set LUNGAI_API_KEY on the server."},
                )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        return response


# The last middleware added runs first: CORS wraps everything so 401 responses still carry CORS headers.
app.add_middleware(ApiKeyMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
app.include_router(patients.router, prefix="/api/v1", tags=["Patients"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])


if STATIC_DIR.is_dir():
    class SPAStaticFiles(StaticFiles):
        """Serve the React build; unknown paths fall back to index.html for client-side routing."""
        async def get_response(self, path, scope):
            try:
                return await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                # scope["path"] is the raw URL path; `path` is OS-normalised (backslashes on Windows)
                if exc.status_code == 404 and not scope["path"].startswith("/api/"):
                    return FileResponse(STATIC_DIR / "index.html")
                raise

    app.mount("/", SPAStaticFiles(directory=STATIC_DIR, html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return {
            "message": f"LungAI API — {MODEL_DISPLAY_NAME}",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs"
        }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        log_level="info"
    )
