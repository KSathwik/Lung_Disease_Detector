# ── Stage 1: Build React frontend ─────────────────────────────────
FROM node:18-alpine AS frontend-build

# Optional: baked into the JS bundle so the browser sends X-API-Key (must match LUNGAI_API_KEY at runtime)
ARG REACT_APP_API_KEY=""
ENV REACT_APP_API_KEY=$REACT_APP_API_KEY

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + static frontend ────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
# Canonical package (config, class mapping, preprocessing) imported by the backend as `lungai`
COPY lungai/ ./lungai/
# Recorded evaluation results served by GET /api/v1/model-metrics
COPY artifacts/evaluation_reports/historical_record_verification.json ./artifacts/evaluation_reports/

# Served by main.py (SPAStaticFiles) at "/"
COPY --from=frontend-build /app/frontend/build ./static

# The checkpoint is not in the image (.dockerignore excludes *.h5). Mount it at runtime; the backend verifies its SHA-256:
#   -v /path/to/densenet121_frequency_v5.h5:/app/models/densenet121_frequency_v5.h5:ro
RUN mkdir -p models uploads

EXPOSE 8000

# Inside the container the API must listen on all interfaces; requests from outside the container
# are only accepted when LUNGAI_API_KEY is set (see ApiKeyMiddleware in main.py).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
