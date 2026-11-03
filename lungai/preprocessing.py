"""Canonical image preprocessing shared by training, evaluation and the backend.

Reproduces, step for step, the preprocessing used to train the v5.0 and v6.0
checkpoint (``experiments/train_densenet_frequency_v5.py::preprocess_single_image``
with the selected Gaussian sigma=1.0 filter). ``tests/ml/test_preprocessing.py``
pins this against a frozen copy of that historical function.

Order: decode -> 3-channel -> BGR->RGB -> Gaussian blur (3x3, 0.8) -> CLAHE on LAB L
-> Gaussian low-pass (sigma=1.0) -> Lanczos-4 resize 224x224 -> ImageNet normalisation.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

from lungai.config import AUGMENTATION, PREPROCESSING

TARGET_SIZE: Tuple[int, int] = tuple(PREPROCESSING["resize"]["size"])  # (width, height) for cv2
MEAN = np.array(PREPROCESSING["normalization"]["mean"], dtype=np.float32)
STD = np.array(PREPROCESSING["normalization"]["std"], dtype=np.float32)
PRE_BLUR_KSIZE = int(PREPROCESSING["pre_blur"]["kernel_size"])
PRE_BLUR_SIGMA = float(PREPROCESSING["pre_blur"]["sigma"])
CLAHE_CLIP = float(PREPROCESSING["clahe"]["clip_limit"])
CLAHE_TILES = tuple(PREPROCESSING["clahe"]["tile_grid_size"])
LOWPASS_SIGMA = float(PREPROCESSING["frequency_filter"]["sigma"])
MIN_DIMENSION = 32

_tls = threading.local()


class ImageDecodeError(ValueError):
    """Raised when bytes or a file cannot be decoded into an image."""


def _clahe() -> "cv2.CLAHE":
    # cv2.CLAHE objects are not thread-safe; keep one per thread
    clahe = getattr(_tls, "clahe", None)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES)
        _tls.clahe = clahe
    return clahe


def gaussian_lowpass(img: np.ndarray, sigma: float = LOWPASS_SIGMA) -> np.ndarray:
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def decode_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode encoded image bytes to a BGR uint8 array; raises ImageDecodeError."""
    if not image_bytes:
        raise ImageDecodeError("Image is empty (0 bytes).")
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("Could not decode image. The file is corrupted or not a supported image format.")
    _check_size(img)
    return img


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image file to BGR uint8; raises FileNotFoundError / ImageDecodeError (never a blank image)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image file not found: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        # cv2.imread cannot open non-ASCII paths on Windows; fall back to decoding the bytes
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError(f"Could not decode image file: {p}")
    _check_size(img)
    return img


def _check_size(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        raise ImageDecodeError(f"Image dimensions ({w}x{h}) are too small. Minimum is {MIN_DIMENSION}x{MIN_DIMENSION} pixels.")


def clean_image(img_bgr: np.ndarray) -> np.ndarray:
    """Steps before resizing. Input BGR (or gray/BGRA) uint8; output RGB uint8 at native resolution."""
    img = img_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (PRE_BLUR_KSIZE, PRE_BLUR_KSIZE), PRE_BLUR_SIGMA)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = _clahe().apply(np.ascontiguousarray(lab[:, :, 0], dtype=np.uint8))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return gaussian_lowpass(img)


def resize_display(img_rgb: np.ndarray) -> np.ndarray:
    """Resized RGB uint8 image (224x224x3), used for Grad-CAM display."""
    return cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)


def normalize(img_rgb_224: np.ndarray) -> np.ndarray:
    return ((img_rgb_224.astype(np.float32) / 255.0 - MEAN) / STD).astype(np.float32)


def preprocess_array(img_bgr: np.ndarray) -> np.ndarray:
    """Full pipeline for one decoded image: returns float32 (224, 224, 3)."""
    return normalize(resize_display(clean_image(img_bgr)))


def preprocess_file(path: Union[str, Path]) -> np.ndarray:
    return preprocess_array(read_image(path))


def preprocess_bytes_with_display(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Backend entry point: (batch tensor (1,224,224,3) float32, display RGB uint8 (224,224,3))."""
    display = resize_display(clean_image(decode_bytes(image_bytes)))
    return normalize(display)[None, ...], display


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Training-only augmentation on a normalised (224,224,3) tensor: random h-flip + rotation (reflect border)."""
    if rng.random() < AUGMENTATION["horizontal_flip_probability"]:
        img = np.fliplr(img)
    angle = rng.uniform(-AUGMENTATION["rotation_degrees"], AUGMENTATION["rotation_degrees"])
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(np.ascontiguousarray(img), m, (w, h), borderMode=cv2.BORDER_REFLECT)
