"""
Model Inference Engine
Loads trained models and runs predictions on uploaded images.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import logging
from typing import Dict, Optional, Tuple

from ml.preprocessing import ImagePreprocessor, DISEASE_CLASSES, PRECAUTIONS_MAP

logger = logging.getLogger(__name__)

# Search for models directory in current working dir or project root
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
if not (MODELS_DIR / "resnet_model.h5").exists():
    root_models = Path(__file__).resolve().parent.parent.parent / "models"
    if (root_models / "resnet_model.h5").exists():
        MODELS_DIR = root_models

RESULTS_FILE = MODELS_DIR / "training_results.json"


class InferenceEngine:
    """
    Singleton inference engine.
    Loads both CNN and ResNet, uses the best-performing model by default.

    Models are loaded lazily on the first call to ``initialize()`` (called
    during the FastAPI lifespan) so that importing this module does NOT
    trigger TensorFlow model loading at import time.
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
        self.preprocessor = ImagePreprocessor()
        self.cnn_model    = None
        self.resnet_model = None
        self.selected_model_name = "ResNet"
        self.class_names  = DISEASE_CLASSES
        self.training_results: Optional[Dict] = None
        self._initialized = True

    def initialize(self):
        """Load model weights from disk. Call once during app startup."""
        self._load_models()

    def _load_models(self):
        """Load saved model weights from disk."""
        cnn_path    = MODELS_DIR / "cnn_model.h5"
        resnet_path = MODELS_DIR / "resnet_model.h5"

        if cnn_path.exists():
            try:
                self.cnn_model = tf.keras.models.load_model(str(cnn_path))
                logger.info("CNN model loaded.")
            except Exception as e:
                logger.error(f"Failed to load CNN model: {e}")

        if resnet_path.exists():
            try:
                self.resnet_model = tf.keras.models.load_model(str(resnet_path))
                logger.info("ResNet model loaded.")
            except Exception as e:
                logger.error(f"Failed to load ResNet model: {e}")

        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                self.training_results = json.load(f)
                self.selected_model_name = self.training_results.get("selected_model", "ResNet")

            # Use the exact class list the models were trained on (label-encoder
            # order), so prediction indices map to the correct disease names.
            trained_classes = (
                self.training_results.get("class_names")
                or self.training_results.get("cnn", {}).get("class_names")
                or self.training_results.get("resnet", {}).get("class_names")
            )
            if trained_classes:
                self.class_names = trained_classes
                logger.info(f"Using trained class names: {self.class_names}")

            logger.info(f"Selected model from training: {self.selected_model_name}")

    def _get_model(self, model_name: str):
        if model_name == "CNN":
            return self.cnn_model
        return self.resnet_model

    def _predict_with_model(
        self, model, image_array: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """Run inference with a given model."""
        probs = model.predict(image_array, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx]) * 100
        condition = self.class_names[pred_idx]
        all_probs = {cls: round(float(p) * 100, 2) for cls, p in zip(self.class_names, probs)}
        return condition, confidence, all_probs

    def _get_alternatives(self, all_probs: Dict, primary: str) -> list:
        """Return top 2 alternative diagnoses (excluding primary)."""
        sorted_probs = sorted(
            [(k, v) for k, v in all_probs.items() if k != primary],
            key=lambda x: x[1], reverse=True
        )
        return [f"{k} ({v:.1f}%)" for k, v in sorted_probs[:2]]

    def _determine_urgency(self, condition: str, confidence: float) -> str:
        emergency = ["Lung Cancer", "COVID-19"]
        urgent    = ["Tuberculosis", "Pleural Effusion", "Pneumonia"]
        if condition in emergency and confidence > 70:
            return "emergency"
        if condition in urgent and confidence > 60:
            return "urgent"
        return "routine"

    def _get_key_findings(self, condition: str) -> list:
        findings_map = {
            "Normal":           [("Opacity", "Absent"), ("Consolidation", "None"), ("Pleural effusion", "Absent"), ("Hyperinflation", "Normal")],
            "Pneumonia":        [("Opacity", "Present"), ("Consolidation", "Lobar/patchy"), ("Pleural effusion", "May be present"), ("Air bronchogram", "Present")],
            "Tuberculosis":     [("Opacity", "Upper lobe"), ("Cavitation", "Possible"), ("Lymphadenopathy", "Present"), ("Miliary pattern", "Possible")],
            "COVID-19":         [("Ground-glass opacity", "Bilateral"), ("Consolidation", "Peripheral"), ("Pleural effusion", "Rare"), ("Distribution", "Bilateral lower")],
            "Lung Cancer":      [("Mass lesion", "Present"), ("Hilar enlargement", "Possible"), ("Pleural effusion", "Possible"), ("Atelectasis", "Possible")],
            "COPD":             [("Hyperinflation", "Present"), ("Flattened diaphragm", "Present"), ("Bullae", "Possible"), ("Vascular markings", "Reduced")],
            "Pleural Effusion": [("Opacity", "Basal"), ("Blunting of angle", "Present"), ("Mediastinal shift", "Possible"), ("Consolidation", "Compression")],
        }
        raw = findings_map.get(condition, [("Finding", "Indeterminate")])
        return [{"label": k, "value": v} for k, v in raw]

    def predict(self, image_bytes: bytes) -> Dict:
        """
        Full prediction pipeline:
        1. Preprocess image
        2. Run CNN prediction
        3. Run ResNet prediction
        4. Return both results + best model result
        """
        img_array = self.preprocessor.preprocess(image_bytes)

        results = {
            "cnn": None,
            "resnet": None,
            "selected_model": self.selected_model_name,
            "final": {}
        }

        # ── CNN Prediction ────────────────────────────────────────────────────
        if self.cnn_model:
            cnn_condition, cnn_conf, cnn_probs = self._predict_with_model(self.cnn_model, img_array)
            results["cnn"] = {
                "condition": cnn_condition,
                "confidence": round(cnn_conf, 2),
                "all_probabilities": cnn_probs,
                "accuracy": self.training_results["cnn"]["accuracy"] if self.training_results else None
            }

        # ── ResNet Prediction ─────────────────────────────────────────────────
        if self.resnet_model:
            rn_condition, rn_conf, rn_probs = self._predict_with_model(self.resnet_model, img_array)
            results["resnet"] = {
                "condition": rn_condition,
                "confidence": round(rn_conf, 2),
                "all_probabilities": rn_probs,
                "accuracy": self.training_results["resnet"]["accuracy"] if self.training_results else None
            }

        # ── Select Final Result from Best Model ───────────────────────────────
        best_result = results.get(self.selected_model_name.lower()) or results.get("resnet") or results.get("cnn")

        if best_result is None:
            # Fallback: no models loaded — use demo mode
            best_result = self._demo_prediction()
            results["resnet"] = best_result
            results["cnn"] = best_result

        final_condition = best_result["condition"]
        final_confidence = best_result["confidence"]

        results["final"] = {
            "condition": final_condition,
            "confidence": final_confidence,
            "urgency": self._determine_urgency(final_condition, final_confidence),
            "alternative_conditions": self._get_alternatives(
                best_result.get("all_probabilities", {}), final_condition
            ),
            "key_findings": self._get_key_findings(final_condition),
            "precautions": PRECAUTIONS_MAP.get(final_condition, [
                "Consult a pulmonologist for detailed evaluation.",
                "Undergo further diagnostic tests as advised."
            ]),
            "disclaimer": (
                "This is an AI-generated analysis for academic and decision-support purposes only. "
                "It must not replace clinical examination by a qualified medical professional. "
                "Always verify findings with a licensed radiologist or physician."
            )
        }

        return results

    def _demo_prediction(self) -> Dict:
        """Demo result when no models are loaded (for UI testing)."""
        return {
            "condition": "Pneumonia",
            "confidence": 82.4,
            "all_probabilities": {
                "Normal": 4.2, "Pneumonia": 82.4, "Tuberculosis": 7.1,
                "COVID-19": 3.1, "Lung Cancer": 1.5, "COPD": 1.2, "Pleural Effusion": 0.5
            }
        }

    def get_model_metrics(self) -> Optional[Dict]:
        return self.training_results
