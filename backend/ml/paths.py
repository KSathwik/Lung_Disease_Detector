"""
Filesystem locations shared by the inference engine and the CXR gate.

Works for both layouts:
- repository checkout: <repo>/backend/ml/paths.py  -> models in <repo>/models
- Docker image:        /app/ml/paths.py            -> models in /app/models
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


def resolve_models_dir() -> Path:
    """MODELS_DIR if set (absolute, or relative to the repo root / backend dir), else the first existing default."""
    env = os.getenv("MODELS_DIR")
    if env:
        p = Path(env)
        if p.is_absolute():
            return p
        for base in (REPO_ROOT, BACKEND_DIR):
            if (base / p).exists():
                return base / p
        return REPO_ROOT / p
    for cand in (REPO_ROOT / "models", BACKEND_DIR / "models"):
        if cand.exists():
            return cand
    return REPO_ROOT / "models"
