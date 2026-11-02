"""Canonical configuration: model identity, class mapping and pipeline settings.

Everything is read from ``lungai/configs/lungai_densenet121.json``. Import the
constants below instead of re-declaring class names or preprocessing values.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
CONFIG_PATH = PACKAGE_DIR / "configs" / "lungai_densenet121.json"


class ConfigError(ValueError):
    """Raised when the canonical configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class ClassInfo:
    index: int
    id: str
    name: str


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Canonical config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: Dict[str, Any]) -> None:
    classes = cfg.get("classes") or []
    indices = [c["index"] for c in classes]
    if indices != list(range(len(classes))):
        raise ConfigError(f"Class indices must be 0..N-1 in order, got {indices}")
    for key in ("id", "name"):
        values = [c[key] for c in classes]
        if len(set(values)) != len(values):
            raise ConfigError(f"Duplicate class {key} in config: {values}")
    if len(classes) != 6:
        raise ConfigError(f"Expected exactly 6 classes, got {len(classes)}")
    if cfg["model"]["input_shape"] != [224, 224, 3]:
        raise ConfigError("model.input_shape must be [224, 224, 3]")


def _classes() -> List[ClassInfo]:
    return [ClassInfo(**c) for c in load_config()["classes"]]


CONFIG = load_config()
CLASSES: List[ClassInfo] = _classes()
CLASS_NAMES: List[str] = [c.name for c in CLASSES]
CLASS_IDS: List[str] = [c.id for c in CLASSES]
CLASS_TO_INDEX: Dict[str, int] = {c.name: c.index for c in CLASSES}
NUM_CLASSES: int = len(CLASSES)
NORMAL_CLASS: str = "Normal"

MODEL = CONFIG["model"]
MODEL_ID: str = MODEL["model_id"]
MODEL_DISPLAY_NAME: str = MODEL["display_name"]
MODEL_VERSION: str = MODEL["version"]
CHECKPOINT_FILENAME: str = MODEL["checkpoint"]["filename"]
CHECKPOINT_SHA256: str = MODEL["checkpoint"]["sha256"].lower()
PREPROCESSING = CONFIG["preprocessing"]
AUGMENTATION = CONFIG["augmentation"]
TRAINING = CONFIG["training"]
DATASET = CONFIG["dataset"]
EXTERNAL_COHORT = CONFIG["external_cohort"]
EVALUATION = CONFIG["evaluation"]
INFERENCE = CONFIG["inference"]


def model_identity() -> Dict[str, Any]:
    """Public, path-free description of the canonical model (used by the API and reports)."""
    return {
        "model_id": MODEL_ID,
        "display_name": MODEL_DISPLAY_NAME,
        "version": MODEL_VERSION,
        "architecture": MODEL["architecture"],
        "task": MODEL["task"],
        "input_shape": MODEL["input_shape"],
        "classes": [{"index": c.index, "id": c.id, "name": c.name} for c in CLASSES],
        "dataset_version": DATASET["version"],
        "expected_checkpoint_sha256": CHECKPOINT_SHA256,
    }


def manifest_path() -> Path:
    return REPO_ROOT / DATASET["manifest"]


def data_root() -> Path:
    """Folder that manifest image paths are relative to (env LUNGAI_DATA_ROOT, default repo root)."""
    env = os.getenv(DATASET["data_root_env"])
    return Path(env).expanduser().resolve() if env else REPO_ROOT


def sha256_file(path: Path, normalize_newlines: bool = False) -> str:
    """SHA-256 of a file. ``normalize_newlines`` hashes CRLF as LF (platform-independent text hash)."""
    h = hashlib.sha256()
    if normalize_newlines:
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return h.hexdigest()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
