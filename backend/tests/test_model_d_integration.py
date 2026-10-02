"""
Automated Test Suite for Model D Integration (Section 13 Requirements)
Covers:
1. Model loading: verify checkpoint loads successfully with exact architecture.
2. Preprocessing: verify dimensions (224x224, 3), ImageNet normalization, Gaussian LP sigma=1.0.
3. Inference: verify six probabilities returned, sum ~ 1.0, predicted class in 6 classes.
4. API: verify valid CXR, invalid file format, corrupted file, and empty file handling.
5. Grad-CAM: verify explanation and attention overlay generation.
"""

import io
import pytest
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import tensorflow as tf

from ml.inference import InferenceEngine
from ml.preprocessing import ImagePreprocessor, DISEASE_CLASSES, apply_gaussian_lp


def _make_test_cxr_bytes(size=(256, 256)) -> bytes:
    """Generate a realistic synthetic chest X-ray in memory."""
    img = np.full((*size, 3), 40, dtype=np.uint8)
    cv2.circle(img, (size[0] // 2, size[1] // 2), size[0] // 3, (180, 180, 180), -1)
    buf = io.BytesIO()
    pil_img = Image.fromarray(img)
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ─── 1. Model Loading Tests ───────────────────────────────────────────────────

def test_model_d_loading():
    """Verify Model D checkpoint loads cleanly with expected architecture."""
    engine = InferenceEngine()
    engine.initialize()
    assert engine.densenet_model is not None, "Model D failed to load."
    
    # Layer verification
    layer_names = [l.name for l in engine.densenet_model.layers]
    assert "densenet121" in layer_names
    assert "class_predictions" in layer_names
    
    # Output shape verification (6 classes)
    output_shape = engine.densenet_model.output_shape
    assert output_shape[-1] == 6, f"Expected 6 classes, got {output_shape[-1]}"


# ─── 2. Preprocessing Tests ───────────────────────────────────────────────────

def test_preprocessing_dimensions_and_normalization():
    """Verify 224x224 dimensions, 3 channels, ImageNet normalization, and Gaussian LP sigma=1.0."""
    pp = ImagePreprocessor(sigma=1.0)
    raw_bytes = _make_test_cxr_bytes((300, 300))
    
    norm_tensor, display_rgb = pp.preprocess_with_display(raw_bytes)
    
    # Output dimensions = 224 x 224 x 3
    assert norm_tensor.shape == (1, 224, 224, 3)
    assert display_rgb.shape == (224, 224, 3)
    assert norm_tensor.dtype == np.float32
    assert display_rgb.dtype == np.uint8
    
    # Normalization sanity: values should be centered around ImageNet mean/std
    assert np.min(norm_tensor) < 0.0
    assert np.max(norm_tensor) > 0.0


def test_gaussian_lp_filtering():
    """Verify Gaussian low-pass spatial filtering with sigma=1.0."""
    sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
    sharp_img[40:60, 40:60] = 255  # sharp square step function
    
    filtered = apply_gaussian_lp(sharp_img, sigma=1.0)
    assert filtered.shape == sharp_img.shape
    # Step edges should be smoothly attenuated
    assert filtered[50, 50, 0] < 255 or filtered[39, 50, 0] > 0


# ─── 3. Inference Tests ───────────────────────────────────────────────────────

def test_inference_six_class_probabilities():
    """Verify six probabilities returned, sum to ~1.0, and predicted class in class mapping."""
    engine = InferenceEngine()
    engine.initialize()
    raw_bytes = _make_test_cxr_bytes((224, 224))
    
    res = engine.predict(raw_bytes, generate_gradcam=True)
    
    # Six classes returned
    assert "probabilities" in res
    probs = res["probabilities"]
    assert len(probs) == 6
    
    for cls in DISEASE_CLASSES:
        assert cls in probs, f"Missing class {cls} in output probabilities"
        assert 0.0 <= probs[cls] <= 1.0
        
    # Probabilities sum approximately to 1
    total_prob = sum(probs.values())
    assert abs(total_prob - 1.0) < 0.01
    
    # Predicted class belongs to 6-class mapping
    assert res["predicted_class"] in DISEASE_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0


# ─── 4. Grad-CAM Tests ────────────────────────────────────────────────────────

def test_gradcam_explanation_generation():
    """Verify that Grad-CAM explanation is generated for a valid image."""
    engine = InferenceEngine()
    engine.initialize()
    raw_bytes = _make_test_cxr_bytes((224, 224))
    
    res = engine.predict(raw_bytes, generate_gradcam=True)
    assert res.get("gradcam") is not None
    gradcam = res["gradcam"]
    
    assert "overlay_image" in gradcam
    assert gradcam["overlay_image"].startswith("data:image/jpeg;base64,")
    assert "original_image" in gradcam
    assert "disclaimer" in gradcam


# ─── 5. API Tests (Integration) ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_api_valid_cxr(client):
    """Test API endpoint with valid CXR upload."""
    raw_bytes = _make_test_cxr_bytes((224, 224))
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("valid_cxr.jpg", raw_bytes, "image/jpeg")},
        data={"scan_type": "X-Ray"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in DISEASE_CLASSES
    assert "probabilities" in data
    assert len(data["probabilities"]) == 6
    assert "gradcam" in data
    assert "final" in data
    assert "Research-stage" in data["final"]["disclaimer"]


@pytest.mark.asyncio
async def test_api_invalid_file_type(client):
    """Test API rejects unsupported file types."""
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("notes.txt", b"Patient clinical notes", "text/plain")}
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_api_corrupted_file(client):
    """Test API gracefully handles corrupted images without crashing."""
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("corrupted.png", b"\x89PNG\r\n\x1a\ncorrupted_data_not_valid", "image/png")}
    )
    assert resp.status_code == 400
    assert "Invalid image" in resp.json()["detail"] or "decode" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_api_empty_file(client):
    """Test API gracefully handles empty uploads."""
    resp = await client.post(
        "/api/v1/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()
