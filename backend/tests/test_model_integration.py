"""
Checkpoint loading and backend-to-model integration.

Two kinds of checkpoint are used:
- the canonical checkpoint (tests marked needs_checkpoint; skipped when models/ does not contain it);
- a randomly initialised network with the canonical architecture, saved as .h5 in a temp dir. It exercises
  the real load / hash-check / preprocessing / Grad-CAM path. Its outputs say nothing about accuracy.
"""

import numpy as np
import pytest

from lungai.config import CHECKPOINT_FILENAME, CLASS_NAMES
from lungai.preprocessing import preprocess_bytes_with_display
from ml.inference import InferenceEngine
from ml.paths import resolve_models_dir


def _cxr_like_bytes() -> bytes:
    """Synthetic thoracic projection that passes the CXR gate (same as tests/test_cxr_gate.py)."""
    from tests.test_cxr_gate import _make_sample_cxr_bytes
    return _make_sample_cxr_bytes()


@pytest.fixture(scope="module")
def surrogate_dir(tmp_path_factory):
    from lungai.models.densenet import build_model
    d = tmp_path_factory.mktemp("models")
    model, _ = build_model(weights=None)
    model.save(d / CHECKPOINT_FILENAME)
    return d


@pytest.fixture
def engine_with(monkeypatch):
    """Re-initialise the engine singleton against another models dir; restore it afterwards."""
    engine = InferenceEngine()

    def load(models_dir, allow_unverified=False):
        monkeypatch.setenv("MODELS_DIR", str(models_dir))
        if allow_unverified:
            monkeypatch.setenv("LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT", "true")
        else:
            monkeypatch.delenv("LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT", raising=False)
        engine.initialize()
        return engine

    yield load
    monkeypatch.undo()
    engine.initialize()


def test_non_canonical_checkpoint_is_refused(surrogate_dir, engine_with):
    engine = engine_with(surrogate_dir)
    assert not engine.is_ready
    assert "SHA-256" in engine.public_load_error
    assert str(surrogate_dir) not in engine.public_load_error


@pytest.mark.asyncio
async def test_refused_checkpoint_gives_503(surrogate_dir, engine_with, client):
    engine_with(surrogate_dir)
    resp = await client.post("/api/v1/predict", files={"file": ("cxr.png", _cxr_like_bytes(), "image/png")})
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"] == "MODEL_NOT_LOADED"


def test_missing_checkpoint_is_reported(tmp_path, engine_with):
    engine = engine_with(tmp_path)
    assert not engine.is_ready
    assert "not found" in engine.public_load_error
    assert engine.status()["checkpoint_sha256"] is None


def test_corrupt_checkpoint_is_refused(tmp_path, engine_with):
    (tmp_path / CHECKPOINT_FILENAME).write_bytes(b"not an hdf5 file")
    engine = engine_with(tmp_path, allow_unverified=True)
    assert not engine.is_ready
    assert "could not be loaded" in engine.public_load_error


def test_wrong_output_shape_is_refused(tmp_path, engine_with):
    from tensorflow import keras
    m = keras.Sequential([keras.Input((224, 224, 3)), keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(5, activation="softmax")])
    m.save(tmp_path / CHECKPOINT_FILENAME)
    engine = engine_with(tmp_path, allow_unverified=True)
    assert not engine.is_ready
    assert "expected (None, 6)" in engine.public_load_error


def test_unverified_override_loads_and_is_flagged(surrogate_dir, engine_with):
    engine = engine_with(surrogate_dir, allow_unverified=True)
    assert engine.is_ready
    assert engine.status()["checkpoint_verified"] is False
    assert engine.base_grad_model is not None, "Grad-CAM wiring must match the canonical layer names"


@pytest.mark.asyncio
async def test_api_output_equals_raw_model_output(surrogate_dir, engine_with, client):
    """Backend preprocessing + model + serialization reproduce the raw model softmax exactly (to 4 d.p.)."""
    engine = engine_with(surrogate_dir, allow_unverified=True)
    raw = _cxr_like_bytes()
    x, _ = preprocess_bytes_with_display(raw)
    expected = engine.densenet_model.predict(x, verbose=0)[0]

    resp = await client.post("/api/v1/predict", files={"file": ("cxr.png", raw, "image/png")})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["checkpoint_verified"] is False
    assert data["class_order"] == CLASS_NAMES
    assert [data["probabilities"][c] for c in CLASS_NAMES] == pytest.approx(expected.tolist(), abs=1e-4)
    assert data["predicted_class"] == CLASS_NAMES[int(np.argmax(expected))]
    assert data["gradcam"]["overlay_image"].startswith("data:image/jpeg;base64,")


def test_prediction_is_deterministic(surrogate_dir, engine_with):
    engine = engine_with(surrogate_dir, allow_unverified=True)
    raw = _cxr_like_bytes()
    a = engine.predict(raw, generate_gradcam=False)["probabilities"]
    b = engine.predict(raw, generate_gradcam=False)["probabilities"]
    assert a == b


# ─── Canonical checkpoint (skipped unless installed) ──────────────────────────

needs_checkpoint = pytest.mark.skipif(
    not (resolve_models_dir() / CHECKPOINT_FILENAME).is_file(),
    reason=f"canonical checkpoint {CHECKPOINT_FILENAME} is not installed in models/ (see docs/inference.md)",
)


@needs_checkpoint
@pytest.mark.needs_checkpoint
def test_canonical_checkpoint_loads_and_verifies():
    engine = InferenceEngine()
    engine.initialize()
    assert engine.is_ready, engine.public_load_error
    assert engine.checkpoint_verified is True
    assert engine.densenet_model.output_shape == (None, 6)


@needs_checkpoint
@pytest.mark.needs_checkpoint
def test_canonical_checkpoint_predicts_six_class_distribution():
    engine = InferenceEngine()
    engine.initialize()
    res = engine.predict(_cxr_like_bytes(), generate_gradcam=True)
    assert list(res["probabilities"]) == CLASS_NAMES
    assert sum(res["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert res["gradcam"] is not None
