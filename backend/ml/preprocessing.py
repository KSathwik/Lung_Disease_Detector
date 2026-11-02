"""
Backend image preprocessing.

The preprocessing itself is ``lungai.preprocessing`` (shared with training and evaluation).
This module adds the upload-time plausibility checks and the per-class guidance text.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

from lungai.config import CLASS_NAMES
from lungai import preprocessing as canonical

logger = logging.getLogger(__name__)

DISEASE_CLASSES = list(CLASS_NAMES)
IMAGE_SIZE = canonical.TARGET_SIZE
apply_gaussian_lp = canonical.gaussian_lowpass

# General guidance shown with a (non-inconclusive) result. Not individualised medical advice.
PRECAUTIONS_MAP = {
    "Normal": [
        "Continue regular health check-ups every 6–12 months.",
        "Maintain a smoke-free lifestyle.",
        "Exercise regularly to keep lungs healthy.",
        "Ensure proper ventilation in living/working spaces."
    ],
    "Pneumonia": [
        "Seek medical evaluation; pneumonia is diagnosed clinically and may need antibiotics.",
        "Rest and stay well hydrated.",
        "Avoid exposure to cold air and crowds.",
        "Seek urgent care for breathing difficulty or low oxygen saturation."
    ],
    "Tuberculosis": [
        "Seek medical evaluation for sputum testing; TB is confirmed by laboratory tests, not by an X-ray alone.",
        "Wear a mask around others until evaluated.",
        "Inform close contacts if TB is confirmed so they can be screened.",
        "If treatment is prescribed, complete the full course without interruption."
    ],
    "COVID-19": [
        "Get a confirmatory test (PCR or antigen); X-ray findings are not specific to COVID-19.",
        "Follow local health authority isolation guidance.",
        "Monitor oxygen saturation and seek care if it drops or breathing worsens.",
        "Stay hydrated and rest adequately."
    ],
    "Pulmonary Nodule / Mass": [
        "Consult a pulmonologist; CT imaging is normally needed to characterise a nodule or mass.",
        "Compare with any prior chest imaging.",
        "Avoid tobacco smoke exposure.",
        "Follow the imaging follow-up interval your clinician recommends."
    ],
    "Pleural Effusion": [
        "Consult a pulmonologist to identify the underlying cause.",
        "Report worsening breathlessness promptly.",
        "Limit strenuous activity until reviewed.",
        "Further imaging or fluid sampling may be advised."
    ],
}
assert set(PRECAUTIONS_MAP) == set(CLASS_NAMES), "PRECAUTIONS_MAP must cover exactly the canonical classes"


class ImagePreprocessor:
    """Decodes an upload, rejects implausible inputs, and applies the canonical preprocessing."""

    def __init__(self):
        self.target_size = IMAGE_SIZE

    def validate_cxr_plausibility(self, img: np.ndarray) -> None:
        """
        Cheap checks that reject obvious non-radiographs before inference:
        aspect ratio > 2.8:1, intensity std < 10 (blank/flat), and multi-hue colour photographs.
        Raises ValueError with an "Invalid Radiograph" message.
        """
        h, w = img.shape[:2]
        ratio = max(h, w) / max(min(h, w), 1)
        if ratio > 2.8:
            raise ValueError(
                f"Invalid Radiograph: Image aspect ratio ({w}x{h}, {ratio:.1f}:1) is inconsistent with chest radiography. "
                "Standard PA/AP chest radiographs are approximately 0.8:1 to 1.3:1."
            )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        std_intensity = float(np.std(gray))
        if std_intensity < 10.0:
            raise ValueError(
                f"Invalid Radiograph: Insufficient radiographic contrast (intensity std={std_intensity:.1f} < 10.0). "
                "Image appears blank, solid, or severely degraded."
            )
        if img.ndim == 3 and img.shape[2] == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            high_sat_mask = (hsv[:, :, 1] / 255.0) > 0.25
            high_sat_fraction = float(np.mean(high_sat_mask))
            if high_sat_fraction > 0.18:
                hues = hsv[:, :, 0][high_sat_mask]
                hist, _ = np.histogram(hues, bins=12, range=(0, 180))
                non_empty_bins = int(np.sum(hist > (len(hues) * 0.05)))
                if non_empty_bins >= 3 or float(np.std(hues)) > 22.0:
                    raise ValueError(
                        f"Invalid Radiograph: Uploaded file is a natural color photograph ({high_sat_fraction*100:.1f}% color saturation "
                        f"across {non_empty_bins} distinct color bands). LungAI requires a grayscale/monochrome chest radiograph (X-Ray)."
                    )

    def load_from_bytes(self, image_bytes: bytes, validate_plausibility: bool = True) -> np.ndarray:
        """Decode bytes to BGR uint8. Raises ValueError for empty, undecodable, too-small or implausible images."""
        img = canonical.decode_bytes(image_bytes)  # ImageDecodeError is a ValueError
        if validate_plausibility:
            self.validate_cxr_plausibility(img)
        return img

    def preprocess(self, image_input) -> np.ndarray:
        """bytes or BGR array -> (1, 224, 224, 3) float32 model input."""
        img = self.load_from_bytes(image_input) if isinstance(image_input, bytes) else image_input
        return canonical.preprocess_array(img)[None, ...]

    def preprocess_with_display(self, image_input) -> Tuple[np.ndarray, np.ndarray]:
        """Model input batch (1, 224, 224, 3) and the resized RGB uint8 image used for Grad-CAM display."""
        img = self.load_from_bytes(image_input) if isinstance(image_input, bytes) else image_input
        display_rgb = canonical.resize_display(canonical.clean_image(img))
        return canonical.normalize(display_rgb)[None, ...], display_rgb
