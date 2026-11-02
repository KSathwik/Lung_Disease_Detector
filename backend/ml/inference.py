"""
Model Inference Engine — Model D (DenseNet-121 Frequency Preprocessing V5)
Loads the final trained Model D checkpoint and runs six-class lung disease predictions.
"""

import os
import io
import json
import base64
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

from ml.preprocessing import ImagePreprocessor, DISEASE_CLASSES, PRECAUTIONS_MAP
from ml.paths import BACKEND_DIR, REPO_ROOT, resolve_models_dir

logger = logging.getLogger(__name__)

MODEL_FILENAME = "densenet121_frequency_v5.h5"
CLASS_MAPPING_FILE = BACKEND_DIR / "ml" / "class_mapping.json"

# Top-class probability below which a result is reported as inconclusive instead of as a condition.
# Not clinically validated: it only stops near-uniform outputs (chance = 1/6) being shown as findings.
CONFIDENCE_FLOOR = float(os.getenv("CONFIDENCE_FLOOR", "0.5"))

EXTERNAL_LIMITATION = (
    "Important Limitation: Model D reports 82.93% accuracy on the internal V5 test split, which shares patients "
    "with the training split for some sources. On the external Montgomery County cohort (138 film-digitized scans) "
    "it labelled every image abnormal: 0/58 tuberculosis cases were classified as Tuberculosis and 0/80 normal "
    "scans as Normal (specificity 0%). Predictions on images from new hospitals or scanners are not reliable."
)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but no Model D checkpoint is loaded."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class InferenceEngine:
    """
    Singleton inference engine for LungAI.
    Loads Model D (DenseNet-121 + Gaussian Low-Pass Preprocessing sigma=1.0)
    lazily during FastAPI startup lifespan.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.preprocessor = ImagePreprocessor(sigma=1.0)
        self.densenet_model = None
        self.base_grad_model = None
        self.top_layers = None
        self.selected_model_name = "DenseNet-121 Frequency V5"
        self.class_names = [
            "COVID-19",
            "Normal",
            "Pleural Effusion",
            "Pneumonia",
            "Pulmonary Nodule / Mass",
            "Tuberculosis"
        ]
        self.training_results: Optional[Dict] = None
        self.model_path: Optional[Path] = None
        self.model_sha256: Optional[str] = None
        self.load_error: Optional[str] = "Model not initialized."  # detailed, for server logs
        self.public_load_error: Optional[str] = "Model not initialized."  # no filesystem paths, safe for clients
        self._initialized = True

    @property
    def is_ready(self) -> bool:
        return self.densenet_model is not None

    def status(self) -> Dict:
        return {
            "model_loaded": self.is_ready,
            "model_name": self.selected_model_name if self.is_ready else None,
            "checkpoint_file": self.model_path.name if self.model_path else None,
            "checkpoint_sha256": self.model_sha256,
            "load_error": None if self.is_ready else self.public_load_error,
            "gradcam_available": self.base_grad_model is not None,
        }

    def initialize(self):
        """Load Model D weights from disk and configure Grad-CAM. Called once during app startup."""
        self._load_models()

    def _load_models(self):
        """Load Model D checkpoint from disk without retraining or altering architecture."""
        models_dir = resolve_models_dir()
        results_file = models_dir / "training_results.json"

        # 1. Resolve checkpoint path (Strictly Model D)
        candidates = [models_dir / MODEL_FILENAME, REPO_ROOT / "experiments" / "densenet_frequency_v5" / MODEL_FILENAME]
        model_path = next((p for p in candidates if p.exists()), None)

        if model_path is None:
            self.public_load_error = (
                f"Model D checkpoint ({MODEL_FILENAME}) not found in the models directory. "
                "See backend/ml/download_model.py."
            )
            self.load_error = f"{self.public_load_error} Looked for: {', '.join(str(p) for p in candidates)}."
            logger.error(self.load_error)
        else:
            try:
                logger.info(f"Loading final Model D checkpoint from: {model_path}")
                self.densenet_model = tf.keras.models.load_model(str(model_path))
                self.model_path = model_path
                self.model_sha256 = _sha256(model_path)
                self.load_error = self.public_load_error = None
                logger.info(f"Model D loaded successfully: {self.densenet_model.name} (sha256 {self.model_sha256[:12]})")

                # Configure Grad-CAM components
                self._setup_gradcam()
            except Exception as e:
                self.densenet_model = None
                self.public_load_error = f"Model D checkpoint ({model_path.name}) exists but could not be loaded."
                self.load_error = f"Failed to load Model D checkpoint at {model_path}: {e}"
                logger.error(self.load_error, exc_info=True)

        # 2. Load class mapping artifact
        if CLASS_MAPPING_FILE.exists():
            try:
                with open(CLASS_MAPPING_FILE, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    if "classes" in cdata:
                        self.class_names = cdata["classes"]
                        logger.info(f"Loaded class mapping from artifact: {self.class_names}")
            except Exception as e:
                logger.warning(f"Could not load class mapping file: {e}")

        # 3. Load training results / metrics
        if results_file.exists():
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    self.training_results = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load training results file: {e}")

    def _setup_gradcam(self):
        """Builds sub-models for Grad-CAM attention map generation on Model D."""
        if not self.densenet_model:
            return
        try:
            base_layer = None
            for lyr in self.densenet_model.layers:
                if "densenet121" in lyr.name:
                    base_layer = lyr
                    break
            if base_layer is None:
                base_layer = self.densenet_model.layers[1]

            last_conv_layer = base_layer.get_layer("relu")
            self.base_grad_model = keras.Model(
                inputs=base_layer.inputs,
                outputs=[last_conv_layer.output, base_layer.output]
            )
            self.top_layers = [
                self.densenet_model.get_layer("gap"),
                self.densenet_model.get_layer("bn"),
                self.densenet_model.get_layer("dense_features"),
                self.densenet_model.get_layer("dropout"),
                self.densenet_model.get_layer("class_predictions")
            ]
            logger.info("Grad-CAM visualization pipeline initialized.")
        except Exception as e:
            logger.warning(f"Grad-CAM setup deferred or failed: {e}")

    def generate_gradcam_overlay(self, norm_tensor: np.ndarray, display_rgb: np.ndarray, pred_idx: int) -> Optional[Dict]:
        """
        Generates Grad-CAM attention heatmap overlay.
        Labeled strictly as 'Model attention visualization' (not clinical reasoning).
        """
        if not self.base_grad_model or not self.top_layers:
            return None

        try:
            gap, bn, dense, drop, dense_1 = self.top_layers
            with tf.GradientTape() as tape:
                conv_outputs, _ = self.base_grad_model(norm_tensor)
                tape.watch(conv_outputs)
                x = gap(conv_outputs)
                x = bn(x, training=False)
                x = dense(x)
                x = drop(x, training=False)
                preds = dense_1(x)
                loss = preds[:, pred_idx]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            heatmap_np = heatmap.numpy()

            # Resize heatmap to 224x224 and colorize
            heatmap_resized = cv2.resize(heatmap_np, (224, 224))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            # Superimpose on display image
            superimposed = np.uint8(heatmap_color * 0.45 + display_rgb * 0.55)

            # Encode overlay as base64 JPEG
            _, overlay_buf = cv2.imencode(".jpg", cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            overlay_b64 = f"data:image/jpeg;base64,{base64.b64encode(overlay_buf).decode('utf-8')}"

            # Encode original resized image as base64 JPEG
            _, orig_buf = cv2.imencode(".jpg", cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            orig_b64 = f"data:image/jpeg;base64,{base64.b64encode(orig_buf).decode('utf-8')}"

            return {
                "title": "Model Attention Visualization",
                "label": "Grad-CAM visualization",
                "target_class": self.class_names[pred_idx],
                "overlay_image": overlay_b64,
                "original_image": orig_b64,
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
        """Return top 2 alternative diagnoses (excluding primary)."""
        sorted_probs = sorted(
            [(k, v) for k, v in all_probs.items() if k != primary],
            key=lambda x: x[1], reverse=True
        )
        return [f"{k} ({v:.1f}%)" for k, v in sorted_probs[:2]]

    def _get_key_findings(self, condition: str) -> list:
        findings_map = {
            "Normal":                   [("Opacity", "Absent"), ("Consolidation", "None"), ("Pleural effusion", "Absent"), ("Hyperinflation", "Normal")],
            "Pneumonia":                [("Opacity", "Present"), ("Consolidation", "Lobar/patchy"), ("Pleural effusion", "May be present"), ("Air bronchogram", "Present")],
            "Tuberculosis":             [("Opacity", "Upper lobe"), ("Cavitation", "Possible"), ("Lymphadenopathy", "Present"), ("Miliary pattern", "Possible")],
            "COVID-19":                 [("Ground-glass opacity", "Bilateral"), ("Consolidation", "Peripheral"), ("Pleural effusion", "Rare"), ("Distribution", "Bilateral lower")],
            "Pulmonary Nodule / Mass":  [("Nodule/Mass lesion", "Present"), ("Hilar/Mediastinal contour", "Assess"), ("Margins", "Spiculated/Smooth"), ("Pleural retraction", "Possible")],
            "Pleural Effusion":         [("Opacity", "Basal"), ("Blunting of angle", "Present"), ("Mediastinal shift", "Possible"), ("Consolidation", "Compression")],
        }
        raw = findings_map.get(condition, [("Finding", "Indeterminate")])
        return [{"label": k, "value": v} for k, v in raw]

    def predict(self, image_bytes: bytes, generate_gradcam: bool = True) -> Dict:
        """
        Full Model D Prediction Pipeline:
        1. Validate raw image bytes
        2. Exact Model D Preprocessing (BGR->RGB, Gaussian 0.8 blur, LAB CLAHE, Gaussian sigma=1.0 LP, Lanczos-4 224x224, ImageNet norm)
        3. Model D inference (6-class softmax)
        4. Optional Grad-CAM attention heatmap overlay
        5. Return clean structured prediction response
        """
        if self.densenet_model is None:
            raise ModelNotLoadedError(self.public_load_error or "Model D checkpoint is not loaded.")

        # Exact Preprocessing reproducing Model D
        norm_tensor, display_rgb = self.preprocessor.preprocess_with_display(image_bytes)

        # Model D Softmax Predictions
        raw_probs = np.asarray(self.densenet_model.predict(norm_tensor, verbose=0)[0], dtype=np.float64)
        if raw_probs.shape != (len(self.class_names),) or not np.all(np.isfinite(raw_probs)):
            raise RuntimeError(f"Model returned an invalid output of shape {raw_probs.shape}.")
        raw_probs = np.clip(raw_probs, 0.0, 1.0)
        if raw_probs.sum() <= 0:
            raise RuntimeError("Model returned an all-zero probability vector.")
        probs = raw_probs / np.sum(raw_probs)

        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        conf_float = float(probs[pred_idx])
        conf_pct = round(conf_float * 100, 2)
        inconclusive = conf_float < CONFIDENCE_FLOOR

        # Probabilities dictionary formatted as [0.0 - 1.0] and [0 - 100%]
        probabilities_unit = {cls: round(float(p), 4) for cls, p in zip(self.class_names, probs)}
        probabilities_pct = {cls: round(float(p) * 100, 2) for cls, p in zip(self.class_names, probs)}

        # Grad-CAM visualization
        gradcam_res = None
        if generate_gradcam and self.base_grad_model:
            gradcam_res = self.generate_gradcam_overlay(norm_tensor, display_rgb, pred_idx)

        # Final structured response
        densenet_result = {
            "condition": pred_class,
            "confidence": conf_pct,
            "all_probabilities": probabilities_pct,
        }

        if inconclusive:
            urgency = "review"
            key_findings = []
            precautions = [
                f"The model's top class probability ({conf_pct:.1f}%) is below the {CONFIDENCE_FLOOR * 100:.0f}% reporting threshold, "
                "so no condition is reported.",
                "Refer the image for review by a qualified radiologist.",
            ]
        else:
            urgency = self._determine_urgency(pred_class, conf_pct)
            key_findings = self._get_key_findings(pred_class)
            precautions = PRECAUTIONS_MAP.get(pred_class, [
                "Consult a pulmonologist for detailed evaluation.",
                "Undergo further diagnostic imaging as advised."
            ])

        final_result = {
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
            )
        }

        return {
            "predicted_class": pred_class,
            "confidence": round(conf_float, 4),
            "confidence_pct": conf_pct,
            "inconclusive": inconclusive,
            "probabilities": probabilities_unit,
            "probabilities_pct": probabilities_pct,
            "gradcam": gradcam_res,
            "selected_model": "DenseNet-121 Frequency V5 (Model D)",
            "model_sha256": self.model_sha256,
            "densenet": densenet_result,
            "final": final_result,
            "external_generalization_limitation": EXTERNAL_LIMITATION,
        }

    def get_public_metrics(self) -> Optional[Dict]:
        """Model D evaluation metrics only; the legacy CNN/ResNet entries come from different, older datasets."""
        if not self.training_results or "densenet" not in self.training_results:
            return None
        return {
            "selected_model": self.training_results.get("selected_model", "DenseNet"),
            "class_names": self.training_results.get("class_names", self.class_names),
            "densenet": self.training_results["densenet"],
            "source": "experiments/densenet_frequency_v5/frequency_metrics.json (V5 internal test split, N=1,570)",
            "model_loaded": self.is_ready,
        }

    def get_model_metrics(self) -> Optional[Dict]:
        return self.training_results
