"""Architecture and checkpoint loading (randomly initialised weights; says nothing about accuracy)."""

import numpy as np
import pytest

from lungai.config import CHECKPOINT_FILENAME, MODEL, NUM_CLASSES
from lungai.models.densenet import CheckpointError, build_model, load_checkpoint, verify_checkpoint


@pytest.fixture(scope="module")
def model():
    m, _ = build_model(weights=None)
    return m


def test_output_is_six_class_softmax(model):
    assert model.output_shape == (None, NUM_CLASSES)
    assert model.input_shape == (None, 224, 224, 3)
    out = np.asarray(model(np.random.default_rng(0).normal(size=(2, 224, 224, 3)).astype("float32"), training=False))
    assert out.shape == (2, 6)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)
    assert (out >= 0).all()


def test_layer_names_match_the_recorded_checkpoint(model):
    names = {l.name for l in model.layers}
    for key in ("pooling", "batch_norm", "features", "dropout", "output"):
        assert MODEL["layer_names"][key] in names
    assert "densenet121" in names


def test_checkpoint_round_trip_and_hash_policy(model, tmp_path):
    path = tmp_path / CHECKPOINT_FILENAME
    model.save(path)
    with pytest.raises(CheckpointError, match="SHA-256"):
        verify_checkpoint(path)
    loaded, digest, canonical = load_checkpoint(path, allow_unverified=True)
    assert canonical is False and len(digest) == 64
    x = np.zeros((1, 224, 224, 3), "float32")
    np.testing.assert_allclose(np.asarray(loaded(x, training=False)), np.asarray(model(x, training=False)), atol=1e-6)


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "absent.h5", allow_unverified=True)
