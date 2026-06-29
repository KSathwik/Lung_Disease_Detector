"""Tests for the image preprocessing pipeline."""

import io
import numpy as np
import pytest
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.preprocessing import ImagePreprocessor, DISEASE_CLASSES, PRECAUTIONS_MAP


def _make_rgb_bytes() -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _make_grayscale_bytes() -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class TestImagePreprocessor:
    def test_preprocess_rgb_shape(self):
        pp = ImagePreprocessor()
        result = pp.preprocess(_make_rgb_bytes())
        assert result.shape == (1, 224, 224, 3)

    def test_preprocess_grayscale_shape(self):
        pp = ImagePreprocessor()
        result = pp.preprocess(_make_grayscale_bytes())
        assert result.shape == (1, 224, 224, 3)

    def test_load_from_bytes_invalid(self):
        pp = ImagePreprocessor()
        with pytest.raises(ValueError, match="Could not decode"):
            pp.load_from_bytes(b"not-an-image")

    def test_output_dtype(self):
        pp = ImagePreprocessor()
        result = pp.preprocess(_make_rgb_bytes())
        assert result.dtype == np.float32


class TestConstants:
    def test_disease_classes_count(self):
        assert len(DISEASE_CLASSES) == 7

    def test_precautions_map_keys_match(self):
        assert set(PRECAUTIONS_MAP.keys()) == set(DISEASE_CLASSES)

    def test_precautions_non_empty(self):
        for cls, precs in PRECAUTIONS_MAP.items():
            assert len(precs) > 0, f"{cls} has no precautions"
