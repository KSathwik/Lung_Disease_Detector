"""
Inference engine for the canonical LungAI DenseNet-121 six-class classifier.

One model, one checkpoint path, one preprocessing implementation (lungai.preprocessing).
There is no fallback: if the checkpoint is missing, fails its SHA-256 check or cannot be
loaded, the engine reports "not ready" and every prediction request fails with 503.
"""

import os
import base64
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

from lungai.config import (
    CHECKPOINT_FILENAME, CLASS_NAMES, INFERENCE, MODEL, MODEL_DISPLAY_NAME, MODEL_ID, MODEL_VERSION, model_identity,
)
from lungai.models.densenet import CheckpointError, load_checkpoint
from ml.preprocessing import ImagePreprocessor, PRECAUTIONS_MAP
from ml.paths import resolve_models_dir

logger = logging.getLogger(__name__)

# Top-class probability below which a result is reported as inconclusive instead of as a condition.
# Not clinically validated: it only stops near-uniform outputs (chance = 1/6) being shown as findings.
CONFIDENCE_FLOOR = float(os.getenv("CONFIDENCE_FLOOR", str(INFERENCE["confidence_floor"])))

EXTERNAL_LIMITATION = (
    "Research model, not validated for clinical use. Recorded results: 82.93% accuracy on the internal V5 test split, "
    "which shares patients with the training split for two sources. On the external Montgomery County cohort "
    "(138 film-digitized scans) it labelled every image abnormal: 0/58 tuberculosis scans were classified as "
    "Tuberculosis and 0/80 normal scans as Normal (specificity 0%). Predictions on images from new hospitals or "
    "scanners are not reliable. Scores are uncalibrated softmax outputs, not probabilities of disease."
)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but the canonical checkpoint is not loaded."""


def _allow_unverified() -> bool:
    return os.getenv("LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT", "").strip().lower() in ("1", "true", "yes")


class InferenceEngine:
    """Process-wide singleton; ``initialize()`` is called once from the FastAPI lifespan."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.preprocessor = ImagePreprocessor()
        self.class_names = list(CLASS_NAMES)
        self.densenet_model = None
        self.base_grad_model = None
        self.top_layers = None
        self.model_path: Optional[Path] = None
        self.model_sha256: Optional[str] = None
        self.checkpoint_verified: bool = False
        self.load_error: Optional[str] = "Model not initialized."         # detailed, for server logs
        self.public_load_error: Optional[str] = "Model not initialized."  # no filesystem paths, safe for clients
        self._initialized = True

    @property
    def is_ready(self) -> bool:
        return self.densenet_model is not None

    def status(self) -> Dict:
        return {
            "model_id": MODEL_ID,
            "model_name": MODEL_DISPLAY_NAME,
            "model_version": MODEL_VERSION,
            "model_loaded": self.is_ready,
            "checkpoint_file": CHECKPOINT_FILENAME,
            "checkpoint_sha256": self.model_sha256,
            "checkpoint_verified": self.checkpoint_verified if self.is_ready else None,
            "load_error": None if self.is_ready else self.public_load_error,
            "gradcam_available": self.base_grad_model is not None,
        }

    def initialize(self):
        """Load the canonical checkpoint and configure Grad-CAM."""
        self._load_model()

    def _load_model(self):
        model_path = resolve_models_dir() / CHECKPOINT_FILENAME
        self.densenet_model = self.base_grad_model = self.top_layers = None
        self.model_sha256 = None
        self.checkpoint_verified = False
        if not model_path.is_file():
            self.public_load_error = (
                f"Checkpoint {CHECKPOINT_FILENAME} not found in the models directory. "
                "See docs/inference.md for how to install it."
            )
            self.load_error = f"{self.public_load_error} Looked for: {model_path}."
            logger.error(self.load_error)
            return
        try:
            logger.info(f"Loading {MODEL_ID} checkpoint from: {model_path}")
            model, digest, verified = load_checkpoint(model_path, allow_unverified=_allow_unverified())
        except CheckpointError as e:
            self.public_load_error = str(e)
            self.load_error = f"{e} (path: {model_path})"
            logger.error(self.load_error)
            return
        self.densenet_model = model
        self.model_path = model_path
        self.model_sha256 = digest
        self.checkpoint_verified = verified
        self.load_error = self.public_load_error = None
        if verified:
            logger.info(f"Loaded canonical checkpoint (sha256 {digest[:12]}).")
        else:
            logger.warning(f"Loaded a NON-canonical checkpoint (sha256 {digest[:12]}) because "
                           "LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT is set; responses are flagged checkpoint_verified=false.")
        self._setup_gradcam()

    def _setup_gradcam(self):
        """Sub-models for Grad-CAM on the backbone's last activation."""
        names = MODEL["layer_names"]
        try:
            base_layer = next((l for l in self.densenet_model.layers if "densenet121" in l.name), None)
            if base_layer is None:
                base_layer = self.densenet_model.layers[1]
            last_conv_layer = base_layer.get_layer(names["gradcam_conv"])
            self.base_grad_model = keras.Model(inputs=base_layer.inputs,
                                               outputs=[last_conv_layer.output, base_layer.output])
            self.top_layers = [self.densenet_model.get_layer(names[k])
                               for k in ("pooling", "batch_norm", "features", "dropout", "output")]
            logger.info("Grad-CAM visualization pipeline initialized.")
        except Exception as e:
            self.base_grad_model = self.top_layers = None
            logger.warning(f"Grad-CAM unavailable for this checkpoint: {e}")

    def generate_gradcam_overlay(self, norm_tensor: np.ndarray, display_rgb: np.ndarray, pred_idx: int) -> Optional[Dict]:
        """Grad-CAM heatmap over the preprocessed image (model attention, not clinical reasoning)."""
        if not self.base_grad_model or not self.top_layers:
            return None
        try:
            gap, bn, dense, drop, out = self.top_layers
            with tf.GradientTape() as tape:
                conv_outputs, _ = self.base_grad_model(norm_tensor)
                tape.watch(conv_outputs)
                x = drop(dense(bn(gap(conv_outputs), training=False)), training=False)
                loss = out(x)[:, pred_idx]
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = tf.squeeze(conv_outputs[0] @ pooled_grads[..., tf.newaxis])
            heatmap = (tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)).numpy()

            heatmap_uint8 = np.uint8(255 * cv2.resize(heatmap, display_rgb.shape[1::-1]))
            heatmap_color = cv2.cvtColor(cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            superimposed = np.uint8(heatmap_color * 0.45 + display_rgb * 0.55)

            def _jpeg(rgb):
                _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

            return {
                "title": "Model Attention Visualization",
                "label": "Grad-CAM visualization",
                "target_class": self.class_names[pred_idx],
                "overlay_image": _jpeg(superimposed),
                "original_image": _jpeg(display_rgb),
                "disclaimer": "Grad-CAM highlights regions that influenced the model's prediction. It represents deep feature activations, not biological or radiological reasoning."
            }
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")
            return None

    def _determine_urgency(self, condition: str, confidence: float) -> str:
        emergency = ["COVID-19", "Pulmonary Nodule / Mass"]
        urgent = ["Tuberculosis", "Pleural Effusion", "Pneumonia"]
        if condition in emergency and confidence > 70:
            return "emergency"
        if condition in urgent and confidence > 60:
            return "urgent"
        return "routine"

    def _get_alternatives(self, all_probs: Dict[str, float], primary: str) -> list:
        """Top 2 other classes by score."""
        ranked = sorted(((k, v) for k, v in all_probs.items() if k != primary), key=lambda x: x[1], reverse=True)
        return [f"{k} ({v:.1f}%)" for k, v in ranked[:2]]

    def _get_key_findings(self, condition: str) -> list:
        """Typical radiographic features of the predicted class (generic text, not read from this image)."""
        findings_map = {
            "Normal":                   [("Opacity", "Absent"), ("Consolidation", "None"), ("Pleural effusion", "Absent"), ("Hyperinflation", "Normal")],
            "Pneumonia":                [("Opacity", "Present"), ("Consolidation", "Lobar/patchy"), ("Pleural effusion", "May be present"), ("Air bronchogram", "Present")],
            "Tuberculosis":             [("Opacity", "Upper lobe"), ("Cavitation", "Possible"), ("Lymphadenopathy", "Present"), ("Miliary pattern", "Possible")],
            "COVID-19":                 [("Ground-glass opacity", "Bilateral"), ("Consolidation", "Peripheral"), ("Pleural effusion", "Rare"), ("Distribution", "Bilateral lower")],
            "Pulmonary Nodule / Mass":  [("Nodule/Mass lesion", "Present"), ("Hilar/Mediastinal contour", "Assess"), ("Margins", "Spiculated/Smooth"), ("Pleural retraction", "Possible")],
            "Pleural Effusion":         [("Opacity", "Basal"), ("Blunting of angle", "Present"), ("Mediastinal shift", "Possible"), ("Consolidation", "Compression")],
        }
        return [{"label": k, "value": v} for k, v in findings_map[condition]]

    def predict(self, image_bytes: bytes, generate_gradcam: bool = True) -> Dict:
        """Decode -> canonical preprocessing -> softmax -> structured result (+ optional Grad-CAM)."""
        if self.densenet_model is None:
            raise ModelNotLoadedError(self.public_load_error or "The model checkpoint is not loaded.")

        norm_tensor, display_rgb = self.preprocessor.preprocess_with_display(image_bytes)

        raw_probs = np.asarray(self.densenet_model.predict(norm_tensor, verbose=0)[0], dtype=np.float64)
        if raw_probs.shape != (len(self.class_names),) or not np.all(np.isfinite(raw_probs)):
            raise RuntimeError(f"Model returned an invalid output of shape {raw_probs.shape}.")
        if raw_probs.min() < 0 or abs(raw_probs.sum() - 1.0) > 1e-3:
            raise RuntimeError("Model output is not a probability vector (softmax expected).")
        probs = raw_probs / raw_probs.sum()  # removes float32 rounding only; never reshapes the distribution

        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        conf_float = float(probs[pred_idx])
        conf_pct = round(conf_float * 100, 2)
        inconclusive = conf_float < CONFIDENCE_FLOOR

        probabilities_unit = {cls: round(float(p), 4) for cls, p in zip(self.class_names, probs)}
        probabilities_pct = {cls: round(float(p) * 100, 2) for cls, p in zip(self.class_names, probs)}

        gradcam_res = None
        if generate_gradcam and self.base_grad_model:
            gradcam_res = self.generate_gradcam_overlay(norm_tensor, display_rgb, pred_idx)

        if inconclusive:
            urgency = "review"
            key_findings = []
            precautions = [
                f"The model's top class score ({conf_pct:.1f}%) is below the {CONFIDENCE_FLOOR * 100:.0f}% reporting threshold, "
                "so no condition is reported.",
                "Refer the image for review by a qualified radiologist.",
            ]
        else:
            urgency = self._determine_urgency(pred_class, conf_pct)
            key_findings = self._get_key_findings(pred_class)
            precautions = PRECAUTIONS_MAP[pred_class]

        return {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "selected_model": MODEL_DISPLAY_NAME,
            "model_sha256": self.model_sha256,
            "checkpoint_verified": self.checkpoint_verified,
            "class_order": list(self.class_names),
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": round(conf_float, 4),
            "confidence_pct": conf_pct,
            "inconclusive": inconclusive,
            "probabilities": probabilities_unit,
            "probabilities_pct": probabilities_pct,
            "gradcam": gradcam_res,
            "densenet": {"condition": pred_class, "confidence": conf_pct, "all_probabilities": probabilities_pct},
            "final": {
                "condition": pred_class,
                "confidence": conf_pct,
                "inconclusive": inconclusive,
                "urgency": urgency,
                "alternative_conditions": self._get_alternatives(probabilities_pct, pred_class),
                "key_findings": key_findings,
                "precautions": precautions,
                "disclaimer": (
                    "Research-stage chest X-ray classification result. This output is not a medical diagnosis "
                    "and should not replace evaluation by a qualified healthcare professional."
                ),
            },
            "external_generalization_limitation": EXTERNAL_LIMITATION,
        }

    def identity(self) -> Dict:
        return {**model_identity(), **self.status()}
