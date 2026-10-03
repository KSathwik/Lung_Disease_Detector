"""
Anatomical & Biophysical CXR Validation Gate (CXR vs Non-CXR Input Rejection).

Protects Model D and the clinical diagnostic pipeline from evaluating out-of-distribution
non-medical and non-chest radiograph inputs, including:
- Natural photographs (cars, dogs, cats, selfies/faces, landscapes, buildings, food)
- Monochrome/grayscale natural images
- Screenshots, charts, synthetic documents
- Axial CT scans and non-thoracic radiographs

Two-Stage Gate Architecture:
Stage 1 (Biophysical Heuristics):
  - Dimension check (minimum 32x32 pixels)
  - Aspect ratio screening (standard PA/AP chest radiographs are ~0.8:1 to 1.3:1; reject > 2.2:1 or < 0.45:1)
  - Intensity dynamic range (luminance std >= 10.0; reject flat, blank, solid images)
  - Polychromatic saturation screening (rejects natural color images exhibiting multi-hue dispersion)

Stage 2 (Deep Anatomical & Morphological Classifier):
  - 128x128 grayscale convolutional neural network trained on isolated V5 train CXRs vs CT scans and natural photos.
  - Scores thoracic skeletal geometry, bilateral aerated lung fields, and mediastinal silhouette.
"""

import os
import logging
from typing import Tuple, Optional, Union
import numpy as np
import cv2

logger = logging.getLogger("lungai.ml.cxr_gate")

DEFAULT_GATE_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "cxr_gate_model.keras")
)
FALLBACK_GATE_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "cxr_gate", "models", "cxr_gate_model.keras")
)

class CXRValidationError(ValueError):
    """Raised when an uploaded image fails the CXR plausibility gate."""
    pass


class CXRGate:
    """
    Validation Gate for Chest Radiographs (CXR vs Non-CXR).
    """

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.83):
        self.model_path = model_path or (
            DEFAULT_GATE_MODEL_PATH if os.path.exists(DEFAULT_GATE_MODEL_PATH) else FALLBACK_GATE_MODEL_PATH
        )
        self.threshold = threshold
        self._model = None
        self._model_load_attempted = False

    def _get_model(self):
        """Lazy loader for the binary gate model."""
        if not self._model_load_attempted:
            self._model_load_attempted = True
            if os.path.exists(self.model_path):
                try:
                    import tensorflow as tf
                    from tensorflow import keras
                    self._model = keras.models.load_model(self.model_path)
                    logger.info(f"Loaded CXR Gate model from {self.model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load CXR Gate model from {self.model_path}: {e}")
                    self._model = None
            else:
                logger.info(f"No CXR Gate model file found at {self.model_path}. Using anatomical profile heuristics.")
        return self._model

    def validate_stage1_heuristics(self, img: np.ndarray) -> Tuple[bool, str]:
        """
        Stage 1: Biophysical screening.
        Fast-fail on size, aspect ratio, blankness, or natural color photos.
        """
        if img is None:
            return False, "Could not decode image. Unreadable or corrupted file."

        h, w = img.shape[:2]
        if h < 32 or w < 32:
            return False, f"Image dimensions ({w}x{h}) are too small for diagnostic analysis. Minimum dimensions are 32x32 pixels."

        # Aspect ratio check
        ratio = max(h, w) / max(min(h, w), 1)
        if ratio > 2.2:
            return False, (
                f"Invalid Radiograph: Image aspect ratio ({w}x{h}, {ratio:.1f}:1) is inconsistent with chest radiography. "
                "Standard PA/AP chest radiographs are approximately 0.8:1 to 1.3:1."
            )

        # Contrast / Dynamic Range check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img
        std_intensity = float(np.std(gray))
        if std_intensity < 10.0:
            return False, (
                f"Invalid Radiograph: Insufficient radiographic contrast (intensity std={std_intensity:.1f} < 10.0). "
                "Image appears blank, solid, or severely degraded."
            )

        # Radiographic exposure: X-rays are dark background with illuminated thoracic cage
        mean_intensity = float(np.mean(gray))
        if mean_intensity > 200.0:
            return False, (
                f"Invalid Radiograph: Image has an excessively bright background (mean={mean_intensity:.1f} > 200.0). "
                "Real radiographs have dark ambient exposure."
            )

        # Polychromatic Color Saturation Check
        if len(img.shape) == 3 and img.shape[2] == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1] / 255.0
            high_sat_mask = sat > 0.20
            high_sat_fraction = float(np.mean(high_sat_mask))

            if high_sat_fraction > 0.15:
                hues = hsv[:, :, 0][high_sat_mask]
                hue_std = float(np.std(hues))
                hist, _ = np.histogram(hues, bins=12, range=(0, 180))
                non_empty_bins = int(np.sum(hist > (len(hues) * 0.05)))

                if non_empty_bins >= 3 or hue_std > 20.0:
                    return False, (
                        f"Invalid Radiograph: Uploaded image appears to be a natural color photograph "
                        f"({high_sat_fraction*100:.1f}% saturation across {non_empty_bins} color hues). "
                        "LungAI requires a planar chest radiograph (X-Ray)."
                    )

        return True, "Passed Stage 1 heuristics"

    def _anatomical_profile_score(self, gray: np.ndarray) -> float:
        """
        Morphological & bilateral profile analysis for fallback validation.
        Chest radiographs exhibit:
        1. Darker bilateral lung regions relative to central spine/cardiac mass.
        2. Bilateral horizontal symmetry across vertical axis.
        3. Dark upper periphery (clavicle/neck border).
        """
        resized = cv2.resize(gray, (128, 128)).astype(np.float32) / 255.0
        
        # Mid-thorax horizontal slice (rows 40 to 80)
        mid_thorax = resized[40:80, :]
        left_lung = np.mean(mid_thorax[:, 20:50])
        center_spine = np.mean(mid_thorax[:, 50:78])
        right_lung = np.mean(mid_thorax[:, 78:108])
        
        # Central mediastinum is typically denser (brighter) than lung fields
        mediastinal_contrast = center_spine - min(left_lung, right_lung)
        
        # Horizontal bilateral symmetry
        left_half = resized[:, :64]
        right_half_flipped = np.fliplr(resized[:, 64:])
        diff = np.mean(np.abs(left_half - right_half_flipped))
        symmetry_score = max(0.0, 1.0 - diff * 2.5)
        
        # Axial CT scans typically have a dark exterior ring with circular body
        corners_mean = (resized[:15, :15].mean() + resized[:15, -15:].mean() +
                        resized[-15:, :15].mean() + resized[-15:, -15:].mean()) / 4.0
        
        score = 0.5 * symmetry_score + 0.3 * (1.0 if mediastinal_contrast > 0.05 else 0.0) + 0.2 * (1.0 - corners_mean)
        return float(np.clip(score, 0.0, 1.0))

    def validate_stage2_anatomical(self, img: np.ndarray) -> Tuple[bool, float, str]:
        """
        Stage 2: Deep anatomical & morphological classification.
        Evaluates whether grayscale spatial patterns match human thoracic radiography.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 and img.shape[2] == 3 else img
        
        model = self._get_model()
        if model is not None:
            try:
                resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
                norm = resized.astype(np.float32) / 255.0
                inp = np.expand_dims(np.expand_dims(norm, axis=-1), axis=0)
                
                pred = model.predict(inp, verbose=0)
                score = float(pred[0][0])
                
                if score >= self.threshold:
                    return True, score, f"Passed anatomical gate (confidence={score:.3f})"
                else:
                    return False, score, (
                        f"Invalid Radiograph: The uploaded image failed chest radiograph verification "
                        f"(anatomical plausibility score: {score:.1%} < {self.threshold:.1%}). "
                        "The image does not exhibit standard thoracic anatomical structures (e.g. lung fields, rib cage, cardiac silhouette)."
                    )
            except Exception as e:
                logger.error(f"Error evaluating gate model: {e}")
                
        # Fallback to analytical morphological profile if model not yet loaded
        score = self._anatomical_profile_score(gray)
        if score >= self.threshold:
            return True, score, f"Passed fallback anatomical heuristics (score={score:.3f})"
        else:
            return False, score, (
                f"Invalid Radiograph: Image failed anatomical thoracic screening (score: {score:.1%} < {self.threshold:.1%}). "
                "Uploaded file does not match bilateral chest radiograph structure."
            )

    def validate(self, img_or_bytes: Union[np.ndarray, bytes]) -> Tuple[bool, float, str]:
        """
        Execute full two-stage validation on input.
        Returns: (is_valid_cxr, score, reason)
        """
        if isinstance(img_or_bytes, bytes):
            if not img_or_bytes or len(img_or_bytes) == 0:
                return False, 0.0, "Uploaded file is empty (0 bytes)."
            nparr = np.frombuffer(img_or_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return False, 0.0, "Could not decode image file. Unsupported or corrupted format."
        elif isinstance(img_or_bytes, np.ndarray):
            img = img_or_bytes
        else:
            return False, 0.0, f"Unsupported input type: {type(img_or_bytes)}"

        # Stage 1: Biophysical screening
        s1_ok, s1_msg = self.validate_stage1_heuristics(img)
        if not s1_ok:
            return False, 0.0, s1_msg

        # Stage 2: Deep anatomical screening
        s2_ok, score, s2_msg = self.validate_stage2_anatomical(img)
        if not s2_ok:
            return False, score, s2_msg

        return True, score, "Image validated as a plausible chest radiograph."


# Singleton instance
_gate_instance: Optional[CXRGate] = None

def get_cxr_gate() -> CXRGate:
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = CXRGate()
    return _gate_instance
