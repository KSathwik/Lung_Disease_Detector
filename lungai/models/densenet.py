"""The canonical DenseNet-121 six-class classifier and its checkpoint loader.

Layer names match the v5.0 and v6.0 checkpoints so Grad-CAM and weight loading work
unchanged (see ``model.layer_names`` in the config).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from lungai.config import CHECKPOINT_SHA256, MODEL, NUM_CLASSES, sha256_file


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, has the wrong hash, or is incompatible with the architecture."""


def build_model(weights: Optional[str] = "imagenet"):
    """Return (model, backbone). ``weights=None`` builds a randomly initialised network (tests only)."""
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import DenseNet121

    names = MODEL["layer_names"]
    shape = tuple(MODEL["input_shape"])
    base = DenseNet121(weights=weights, include_top=False, input_shape=shape)
    base.trainable = False

    inputs = keras.Input(shape=shape, name=names["input"])
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name=names["pooling"])(x)
    x = layers.BatchNormalization(name=names["batch_norm"])(x)
    x = layers.Dense(MODEL["head"]["dense_units"], activation="relu", name=names["features"])(x)
    x = layers.Dropout(MODEL["head"]["dropout"], name=names["dropout"])(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name=names["output"])(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="lungai_densenet121"), base


def verify_checkpoint(path: Path, allow_unverified: bool = False) -> Tuple[str, bool]:
    """Return (sha256, matches_canonical). Raises CheckpointError if missing, or on mismatch unless allowed."""
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path.name}")
    digest = sha256_file(path)
    matches = digest == CHECKPOINT_SHA256
    if not matches and not allow_unverified:
        raise CheckpointError(
            f"Checkpoint {path.name} has SHA-256 {digest[:12]}…, expected {CHECKPOINT_SHA256[:12]}… for the canonical "
            "model. Refusing to load a different model."
        )
    return digest, matches


def load_checkpoint(path: Path, allow_unverified: bool = False):
    """Load and validate a checkpoint. Returns (model, sha256, matches_canonical)."""
    from tensorflow import keras

    digest, matches = verify_checkpoint(path, allow_unverified)
    try:
        model = keras.models.load_model(str(path), compile=False)
    except Exception as e:  # keras raises many types for incompatible files
        raise CheckpointError(f"Checkpoint {path.name} could not be loaded: {type(e).__name__}: {e}") from e
    out_shape = tuple(model.output_shape)
    if out_shape != (None, NUM_CLASSES):
        raise CheckpointError(f"Checkpoint {path.name} outputs shape {out_shape}, expected (None, {NUM_CLASSES}).")
    in_shape = tuple(model.input_shape)
    if in_shape != (None, *MODEL["input_shape"]):
        raise CheckpointError(f"Checkpoint {path.name} expects input {in_shape}, expected (None, {MODEL['input_shape']}).")
    return model, digest, matches
