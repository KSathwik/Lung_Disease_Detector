"""
ML Model Architectures
Algorithm 1: Custom CNN (Convolutional Neural Network)
Algorithm 2: ResNet50 (Transfer Learning with fine-tuning)

Both models are trained, evaluated, and the better one is selected automatically.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)

NUM_CLASSES = 7
IMAGE_SIZE  = (224, 224, 3)
MODELS_DIR  = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 1: Custom CNN
# ══════════════════════════════════════════════════════════════════════════════

class LungCNN:
    """
    Custom Convolutional Neural Network for lung disease classification.
    
    Architecture:
      Input(224x224x3)
        → Conv Block 1: Conv2D(32) + BN + ReLU + MaxPool
        → Conv Block 2: Conv2D(64) + BN + ReLU + MaxPool
        → Conv Block 3: Conv2D(128) + BN + ReLU + MaxPool
        → Conv Block 4: Conv2D(256) + BN + ReLU + GlobalAvgPool
        → Dense(512) + Dropout(0.5)
        → Dense(NUM_CLASSES) + Softmax
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.model = self._build()
        self.history = None

    def _build(self) -> Model:
        inputs = keras.Input(shape=IMAGE_SIZE, name="input")

        # ── Block 1 ──────────────────────────────────────────────────────────
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv1a")(inputs)
        x = layers.BatchNormalization(name="bn1a")(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv1b")(x)
        x = layers.BatchNormalization(name="bn1b")(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)
        x = layers.Dropout(0.25)(x)

        # ── Block 2 ──────────────────────────────────────────────────────────
        x = layers.Conv2D(64, (3, 3), padding="same", name="conv2a")(x)
        x = layers.BatchNormalization(name="bn2a")(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, (3, 3), padding="same", name="conv2b")(x)
        x = layers.BatchNormalization(name="bn2b")(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)
        x = layers.Dropout(0.25)(x)

        # ── Block 3 ──────────────────────────────────────────────────────────
        x = layers.Conv2D(128, (3, 3), padding="same", name="conv3a")(x)
        x = layers.BatchNormalization(name="bn3a")(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(128, (3, 3), padding="same", name="conv3b")(x)
        x = layers.BatchNormalization(name="bn3b")(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2), name="pool3")(x)
        x = layers.Dropout(0.25)(x)

        # ── Block 4 ──────────────────────────────────────────────────────────
        x = layers.Conv2D(256, (3, 3), padding="same", name="conv4a")(x)
        x = layers.BatchNormalization(name="bn4a")(x)
        x = layers.Activation("relu")(x)
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dropout(0.25)(x)

        # ── Classifier ────────────────────────────────────────────────────────
        x = layers.Dense(512, activation="relu", name="fc1")(x)
        x = layers.BatchNormalization(name="bn_fc")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="output")(x)

        model = Model(inputs, outputs, name="LungCNN")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray,   y_val: np.ndarray,
        epochs: int = 50, batch_size: int = 32
    ):
        logger.info("Training CNN model...")
        callbacks = _get_callbacks("CNN")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        self.model.save(MODELS_DIR / "cnn_model.h5")
        logger.info("CNN model saved.")
        return self.history

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.model.predict(X)
        preds = np.argmax(probs, axis=1)
        return preds, probs

    def summary(self):
        self.model.summary()


# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2: ResNet50 Transfer Learning
# ══════════════════════════════════════════════════════════════════════════════

class LungResNet:
    """
    ResNet50 with Transfer Learning for lung disease classification.
    
    Strategy:
      Phase 1 — Freeze ResNet50 base, train classifier head (10 epochs)
      Phase 2 — Unfreeze top layers of ResNet50, fine-tune with low LR
    
    Architecture:
      ResNet50(pretrained=ImageNet, include_top=False)
        → GlobalAveragePooling2D
        → Dense(1024, relu) + BatchNorm + Dropout(0.5)
        → Dense(512, relu) + BatchNorm + Dropout(0.3)
        → Dense(NUM_CLASSES, softmax)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.base_model, self.model = self._build()
        self.history = None

    def _build(self) -> Tuple[Model, Model]:
        base = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=IMAGE_SIZE
        )
        base.trainable = False  # Freeze for Phase 1

        inputs = keras.Input(shape=IMAGE_SIZE)
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs, outputs, name="LungResNet50")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return base, model

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray,   y_val: np.ndarray,
        epochs: int = 50, batch_size: int = 32
    ):
        logger.info("Training ResNet50 — Phase 1 (frozen base)...")

        # Phase 1: Train head only
        callbacks = _get_callbacks("ResNet_phase1")
        h1 = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Phase 2: Unfreeze top 30 layers and fine-tune
        logger.info("ResNet50 — Phase 2 (fine-tuning top layers)...")
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-30]:
            layer.trainable = False

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = _get_callbacks("ResNet_phase2")
        h2 = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs - 10,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Merge histories
        combined = {}
        for key in h1.history:
            combined[key] = h1.history[key] + h2.history[key]
        self.history = type("History", (), {"history": combined})()

        self.model.save(MODELS_DIR / "resnet_model.h5")
        logger.info("ResNet model saved.")
        return self.history

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.model.predict(X)
        preds = np.argmax(probs, axis=1)
        return preds, probs


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

def _get_callbacks(model_name: str) -> List:
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation & Metrics
# ══════════════════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """
    Evaluates both models and selects the better one.
    
    Metrics computed:
      - Accuracy
      - Precision (weighted)
      - Recall (weighted)
      - F1-Score (weighted)
      - AUC-ROC (macro OvR)
      - Confusion Matrix
      - Per-class Classification Report
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def evaluate(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        history
    ) -> Dict:
        """Compute all metrics for one model."""

        acc       = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm        = confusion_matrix(y_true, y_pred)
        report    = classification_report(y_true, y_pred, target_names=self.class_names)

        try:
            auc = roc_auc_score(
                tf.keras.utils.to_categorical(y_true, len(self.class_names)),
                y_prob,
                multi_class="ovr",
                average="macro"
            )
        except Exception:
            auc = 0.0

        metrics = {
            "model_name":   model_name,
            "accuracy":     round(float(acc),       4),
            "precision":    round(float(precision),  4),
            "recall":       round(float(recall),     4),
            "f1_score":     round(float(f1),         4),
            "auc_roc":      round(float(auc),        4),
            "confusion_matrix": cm.tolist(),
            "class_names":  self.class_names,
            "training_loss":      history.history.get("loss", []),
            "validation_loss":    history.history.get("val_loss", []),
            "training_accuracy":  history.history.get("accuracy", []),
            "validation_accuracy":history.history.get("val_accuracy", []),
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"  {model_name} Metrics")
        logger.info(f"{'='*50}")
        logger.info(f"  Accuracy  : {acc:.4f}")
        logger.info(f"  Precision : {precision:.4f}")
        logger.info(f"  Recall    : {recall:.4f}")
        logger.info(f"  F1-Score  : {f1:.4f}")
        logger.info(f"  AUC-ROC   : {auc:.4f}")
        logger.info(f"\n{report}")

        return metrics

    def compare_and_select(self, cnn_metrics: Dict, resnet_metrics: Dict) -> str:
        """
        Compare CNN vs ResNet on weighted composite score.
        Composite = 0.4*F1 + 0.3*Accuracy + 0.2*AUC + 0.1*Recall
        """
        def score(m):
            return (0.4 * m["f1_score"] +
                    0.3 * m["accuracy"] +
                    0.2 * m["auc_roc"] +
                    0.1 * m["recall"])

        cnn_score    = score(cnn_metrics)
        resnet_score = score(resnet_metrics)

        logger.info(f"\nModel Comparison:")
        logger.info(f"  CNN composite score    : {cnn_score:.4f}")
        logger.info(f"  ResNet composite score : {resnet_score:.4f}")

        if resnet_score >= cnn_score:
            winner = "ResNet"
            logger.info("  → ResNet50 selected as the primary model.")
        else:
            winner = "CNN"
            logger.info("  → CNN selected as the primary model.")

        return winner

    def plot_training_curves(self, cnn_metrics: Dict, resnet_metrics: Dict):
        """Save training/validation loss & accuracy curves for both models."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Model Training Comparison: CNN vs ResNet50", fontsize=14)

        for i, (metrics, name, color) in enumerate([
            (cnn_metrics, "CNN", "#185FA5"),
            (resnet_metrics, "ResNet50", "#0F6E56")
        ]):
            ax_loss = axes[0][i]
            ax_loss.plot(metrics["training_loss"],   label="Train Loss",  color=color, lw=2)
            ax_loss.plot(metrics["validation_loss"], label="Val Loss",    color=color, lw=2, linestyle="--")
            ax_loss.set_title(f"{name} — Loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)

            ax_acc = axes[1][i]
            ax_acc.plot(metrics["training_accuracy"],   label="Train Acc", color=color, lw=2)
            ax_acc.plot(metrics["validation_accuracy"], label="Val Acc",   color=color, lw=2, linestyle="--")
            ax_acc.set_title(f"{name} — Accuracy")
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("docs/training_curves.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self, metrics: Dict):
        """Save confusion matrix heatmap."""
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        ax.set_title(f"Confusion Matrix — {metrics['model_name']}")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(f"docs/confusion_matrix_{metrics['model_name']}.png", dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Training Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def train_and_select(data: Dict, epochs: int = 50, batch_size: int = 32) -> Dict:
    """
    Full training pipeline:
    1. Train CNN
    2. Train ResNet50
    3. Evaluate both on test set
    4. Select best model
    5. Return metrics + winner
    """
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    class_names       = data["class_names"]

    evaluator = ModelEvaluator(class_names)

    # ── Train CNN ─────────────────────────────────────────────────────────────
    cnn = LungCNN(num_classes=len(class_names))
    cnn_history = cnn.train(X_train, y_train, X_val, y_val, epochs, batch_size)
    cnn_preds, cnn_probs = cnn.predict(X_test)
    cnn_metrics = evaluator.evaluate("CNN", y_test, cnn_preds, cnn_probs, cnn_history)

    # ── Train ResNet ──────────────────────────────────────────────────────────
    resnet = LungResNet(num_classes=len(class_names))
    resnet_history = resnet.train(X_train, y_train, X_val, y_val, epochs, batch_size)
    resnet_preds, resnet_probs = resnet.predict(X_test)
    resnet_metrics = evaluator.evaluate("ResNet", y_test, resnet_preds, resnet_probs, resnet_history)

    # ── Compare & Select ──────────────────────────────────────────────────────
    winner = evaluator.compare_and_select(cnn_metrics, resnet_metrics)

    # ── Plots ─────────────────────────────────────────────────────────────────
    Path("docs").mkdir(exist_ok=True)
    evaluator.plot_training_curves(cnn_metrics, resnet_metrics)
    evaluator.plot_confusion_matrix(cnn_metrics)
    evaluator.plot_confusion_matrix(resnet_metrics)

    # Save metrics to JSON
    results = {
        "cnn": cnn_metrics,
        "resnet": resnet_metrics,
        "selected_model": winner
    }
    with open("models/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Training complete. Best model: {winner}")
    return results
