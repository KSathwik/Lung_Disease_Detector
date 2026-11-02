"""Canonical preprocessing must equal the preprocessing the recorded checkpoint was trained with."""

import cv2
import numpy as np
import pytest

from lungai import preprocessing as pp


def _historical_preprocess(img: np.ndarray) -> np.ndarray:
    """Frozen copy of experiments/train_densenet_frequency_v5.py::preprocess_single_image
    (filter_type="gaussian", sigma=1.0), with cv2.imread replaced by an already-decoded image."""
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(np.ascontiguousarray(lab[:, :, 0], dtype=np.uint8))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    sigma = 1.0
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    return ((img.astype(np.float32) / 255.0 - MEAN) / STD).astype(np.float32)


def _synthetic(h, w, channels=3, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h, w), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    if channels == 1:
        return base
    img = np.stack([base] * 3, axis=-1)
    if channels == 4:
        img = np.concatenate([img, np.full((h, w, 1), 255, np.uint8)], axis=-1)
    return img


@pytest.mark.parametrize("h,w,c", [(224, 224, 3), (512, 400, 3), (1024, 1024, 1), (300, 280, 4), (2048, 1800, 3)])
def test_matches_historical_training_preprocessing(h, w, c):
    img = _synthetic(h, w, c, seed=h + w)
    np.testing.assert_allclose(pp.preprocess_array(img), _historical_preprocess(img), atol=1e-6)


def test_backend_path_equals_training_path(tmp_path):
    """Bytes (backend) and file (training/evaluation) entry points give identical tensors."""
    img = _synthetic(600, 500)
    ok, enc = cv2.imencode(".png", img)
    assert ok
    f = tmp_path / "x.png"
    f.write_bytes(enc.tobytes())
    batch, display = pp.preprocess_bytes_with_display(enc.tobytes())
    assert batch.shape == (1, 224, 224, 3) and batch.dtype == np.float32
    assert display.shape == (224, 224, 3) and display.dtype == np.uint8
    np.testing.assert_array_equal(batch[0], pp.preprocess_file(f))


def test_sixteen_bit_png_is_decoded_like_training(tmp_path):
    img16 = (np.linspace(0, 65535, 300 * 300).reshape(300, 300)).astype(np.uint16)
    ok, enc = cv2.imencode(".png", img16)
    f = tmp_path / "x16.png"
    f.write_bytes(enc.tobytes())
    np.testing.assert_allclose(pp.preprocess_file(f), _historical_preprocess(cv2.imread(str(f))), atol=1e-6)


def test_missing_file_raises_instead_of_blank_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        pp.preprocess_file(tmp_path / "nope.png")


def test_corrupt_bytes_raise():
    with pytest.raises(pp.ImageDecodeError):
        pp.decode_bytes(b"\x89PNG\r\n\x1a\nnot really")
    with pytest.raises(pp.ImageDecodeError):
        pp.decode_bytes(b"")


def test_tiny_image_rejected():
    ok, enc = cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(pp.ImageDecodeError, match="too small"):
        pp.decode_bytes(enc.tobytes())


def test_augmentation_is_seeded_and_shape_preserving():
    x = pp.preprocess_array(_synthetic(256, 256))
    a = pp.augment(x, np.random.default_rng(7))
    b = pp.augment(x, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert a.shape == x.shape


def test_evaluation_loader_never_augments(tmp_path):
    from lungai.data.loader import load_batch
    img = _synthetic(300, 300)
    f = tmp_path / "a.png"
    cv2.imwrite(str(f), img)
    np.testing.assert_array_equal(load_batch([f, f])[0], pp.preprocess_file(f))
