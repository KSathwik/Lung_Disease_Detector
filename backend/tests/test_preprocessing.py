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
    arr = np.random.randint(20, 220, (100, 100), dtype=np.uint8)
    img = Image.fromarray(np.stack([arr, arr, arr], axis=-1))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _make_grayscale_bytes() -> bytes:
    img = Image.fromarray(np.random.randint(20, 220, (100, 100), dtype=np.uint8), mode="L")
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

    def test_validate_cxr_rejects_natural_photo(self):
        pp = ImagePreprocessor()
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:35, :] = [235, 180, 50]   # blue
        arr[35:70, :] = [30, 30, 220]  # red
        arr[70:, :] = [40, 180, 40]    # green
        with pytest.raises(ValueError, match="natural color photograph"):
            pp.validate_cxr_plausibility(arr)

    def test_validate_cxr_rejects_blank_image(self):
        pp = ImagePreprocessor()
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Insufficient radiographic contrast"):
            pp.validate_cxr_plausibility(blank)

    def test_output_dtype(self):
        pp = ImagePreprocessor()
        result = pp.preprocess(_make_rgb_bytes())
        assert result.dtype == np.float32


class TestConstants:
    def test_disease_classes_count(self):
        assert len(DISEASE_CLASSES) == 6

    def test_precautions_map_keys_contain_classes(self):
        for cls in DISEASE_CLASSES:
            assert cls in PRECAUTIONS_MAP, f"{cls} missing in PRECAUTIONS_MAP"

    def test_precautions_non_empty(self):
        for cls, precs in PRECAUTIONS_MAP.items():
            assert len(precs) > 0, f"{cls} has no precautions"
