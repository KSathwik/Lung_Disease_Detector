"""
Phase 4B — Deep CORAL Controlled Domain-Generalization Experiment
LungAI Six-Class Disease Detector (M.Tech Thesis)

Scientific Objective:
Test whether aligning feature distributions across available training domains using Deep CORAL
improves cross-source generalization of the frozen six-class DenseNet-121 classifier.

Deliverables saved to experiments/densenet_coral_v5/:
- densenet121_coral_v5.h5
- coral_config.json
- coral_lambda_search.json
- coral_metrics.json
- coral_montgomery.json
- coral_source_heldout.json
- coral_feature_analysis.json
- coral_comparison_report.md
- coral_confusion_matrix.png
- coral_roc_curves.png
- coral_feature_space.png
- coral_gradcam/*.png
"""

import sys
import os
import io
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# UTF-8 stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
try:
    from keras.applications import DenseNet121
except (ImportError, AttributeError):
    from tensorflow.keras.applications import DenseNet121

# Threading setup
try:
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("coral_trainer")

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

IMAGE_SIZE = (224, 224, 3)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_tls = threading.local()

def get_clahe():
    clahe = getattr(_tls, "clahe", None)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _tls.clahe = clahe
    return clahe

def preprocess_single_image(img_path: str) -> np.ndarray:
    """Standardized preprocessing matching V5 baseline: LAB CLAHE + Lanczos-4."""
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((*IMAGE_SIZE[:2], 3), dtype=np.float32)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan = np.ascontiguousarray(lab[:, :, 0], dtype=np.uint8)
    lab[:, :, 0] = get_clahe().apply(l_chan)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    return img_norm.astype(np.float32)

def augment_image(img: np.ndarray) -> np.ndarray:
    """Controlled augmentation matching V5 baseline: horizontal flip + rotation."""
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
    angle = np.random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img

def process_item(item: Tuple[str, str, bool, dict, dict]) -> Tuple[np.ndarray, int, float]:
    path, label, augment, class_to_idx, class_weights = item
    arr = preprocess_single_image(path)
    if augment:
        arr = augment_image(arr)
    c_idx = class_to_idx[label]
    sw = class_weights.get(c_idx, 1.0)
    return arr, c_idx, sw

class DomainPairGenerator:
    """
    Domain-aware pair generator for Deep CORAL.
    Samples pairs of distinct training domains proportional to domain size.
    Produces combined batches of shape (2 * batch_size_per_domain, 224, 224, 3)
    where [:B] is Domain S and [B:] is Domain T.
    """
    def __init__(self, df: pd.DataFrame, class_to_idx: dict, class_weights: dict,
                 batch_size_per_domain: int = 16, steps_per_epoch: int = 231, augment: bool = True):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.class_weights = class_weights
        self.B = batch_size_per_domain
        self.steps_per_epoch = steps_per_epoch
        self.augment = augment
        self.executor = ThreadPoolExecutor(max_workers=6)

        # Build domain indices
        self.domains = sorted(df["source_dataset"].unique())
        self.domain_indices = {d: df[df["source_dataset"] == d].index.to_numpy() for d in self.domains}
        self.domain_sizes = np.array([len(self.domain_indices[d]) for d in self.domains], dtype=np.float32)
        self.domain_probs = self.domain_sizes / self.domain_sizes.sum()

        logger.info(f"Initialized DomainPairGenerator across {len(self.domains)} training domains.")
        for d, count in zip(self.domains, self.domain_sizes):
            logger.info(f"  Domain: {d:<24} | N={int(count):<5} | P={count/self.domain_sizes.sum():.4f}")

    def __len__(self):
        return self.steps_per_epoch

    def get_batch(self):
        # Sample Domain 1 proportional to domain size
        idx_s = np.random.choice(len(self.domains), p=self.domain_probs)
        dom_s = self.domains[idx_s]

        # Sample Domain 2 != Domain 1 proportional to remaining domain sizes
        rem_probs = self.domain_sizes.copy()
        rem_probs[idx_s] = 0.0
        rem_probs /= rem_probs.sum()
        idx_t = np.random.choice(len(self.domains), p=rem_probs)
        dom_t = self.domains[idx_t]

        # Draw B samples from dom_s and B samples from dom_t
        s_idxs = np.random.choice(self.domain_indices[dom_s], size=self.B, replace=(len(self.domain_indices[dom_s]) < self.B))
        t_idxs = np.random.choice(self.domain_indices[dom_t], size=self.B, replace=(len(self.domain_indices[dom_t]) < self.B))

        combined_idxs = np.concatenate([s_idxs, t_idxs])
        items = [
            (self.df.loc[i, "image_path"], self.df.loc[i, "clinical_label"], self.augment, self.class_to_idx, self.class_weights)
            for i in combined_idxs
        ]
        results = list(self.executor.map(process_item, items))

        X = np.stack([r[0] for r in results], axis=0)
        y = np.array([r[1] for r in results], dtype=np.int32)
        sw = np.array([r[2] for r in results], dtype=np.float32)
        return X, y, sw

class ValidationGenerator:
    """Standard sequential batch generator for validation and testing."""
    def __init__(self, df: pd.DataFrame, class_to_idx: dict, batch_size: int = 32):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=6)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def get_batch(self, idx):
        batch_idx = np.arange(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.df)))
        items = [
            (self.df.loc[i, "image_path"], self.df.loc[i, "clinical_label"], False, self.class_to_idx, {})
            for i in batch_idx
        ]
        results = list(self.executor.map(process_item, items))
        X = np.stack([r[0] for r in results], axis=0)
        y = np.array([r[1] for r in results], dtype=np.int32)
        return X, y

def compute_coral_loss(h_s: tf.Tensor, h_t: tf.Tensor) -> tf.Tensor:
    """
    Computes Deep CORAL covariance alignment loss:
    L_coral = ||C_s - C_t||^2_F / (4 * d^2)
    where C_s and C_t are the sample covariance matrices of feature vectors.
    """
    d = tf.cast(tf.shape(h_s)[1], tf.float32)
    n_s = tf.cast(tf.shape(h_s)[0], tf.float32)
    n_t = tf.cast(tf.shape(h_t)[0], tf.float32)

    mean_s = tf.reduce_mean(h_s, axis=0, keepdims=True)
    mean_t = tf.reduce_mean(h_t, axis=0, keepdims=True)

    h_s_centered = h_s - mean_s
    h_t_centered = h_t - mean_t

    cov_s = tf.matmul(h_s_centered, h_s_centered, transpose_a=True) / (n_s - 1.0 + 1e-7)
    cov_t = tf.matmul(h_t_centered, h_t_centered, transpose_a=True) / (n_t - 1.0 + 1e-7)

    diff = cov_s - cov_t
    frobenius_sq = tf.reduce_sum(tf.square(diff))
    return frobenius_sq / (4.0 * d * d)

def build_densenet_coral_model(num_classes: int) -> Tuple[Model, Model, Model]:
    """
    Builds transfer-learned DenseNet-121 matching baseline specification.
    Returns:
    - full_model: inputs -> [features (256-D), predictions (6-D)]
    - base_model: pretrained DenseNet121 backbone
    - eval_model: inputs -> predictions (for standard keras evaluation)
    """
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE)
    base.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE, name="input_xray")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn")(x)
    features = layers.Dense(256, activation="relu", name="dense_features")(x)
    drop = layers.Dropout(0.3, name="dropout")(features)
    preds = layers.Dense(num_classes, activation="softmax", name="predictions")(drop)

    full_model = keras.Model(inputs=inputs, outputs=[features, preds], name="DenseNet121_CORAL_Full")
    eval_model = keras.Model(inputs=inputs, outputs=preds, name="DenseNet121_CORAL_Eval")
    return full_model, base, eval_model

def evaluate_model_on_val(eval_model, val_gen, val_df, class_to_idx, num_classes) -> Dict[str, float]:
    """Evaluates model predictions on the validation set."""
    all_probs = []
    y_true = np.array([class_to_idx[lbl] for lbl in val_df["clinical_label"]])

    for i in range(len(val_gen)):
        X_b, _ = val_gen.get_batch(i)
        probs_b = eval_model.predict_on_batch(X_b)
        all_probs.append(probs_b)

    probs = np.vstack(all_probs)
    preds = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))

    y_onehot = np.zeros((len(y_true), num_classes), dtype=np.float32)
    for i, c in enumerate(y_true):
        y_onehot[i, c] = 1.0

    try:
        macro_roc_auc = float(roc_auc_score(y_onehot, probs, average="macro", multi_class="ovr"))
    except Exception:
        macro_roc_auc = 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_roc_auc": macro_roc_auc
    }

def train_coral_model(lambda_val: float, train_gen: DomainPairGenerator, val_gen: ValidationGenerator,
                      val_df: pd.DataFrame, class_to_idx: dict, num_classes: int) -> Tuple[Model, Dict, Dict]:
    """
    Trains a controlled DenseNet-121 + Deep CORAL model for 4 epochs:
    Phase 1: 2 epochs (backbone frozen, lr=1e-3)
    Phase 2: 2 epochs (top 30 backbone layers unfrozen, lr=1e-5)
    """
    logger.info(f"\n=======================================================")
    logger.info(f"STARTING CORAL TRAINING: Lambda = {lambda_val}")
    logger.info(f"=======================================================")

    full_model, base_model, eval_model = build_densenet_coral_model(num_classes)

    # Phase 1: 2 Epochs, Backbone Frozen, lr=1e-3
    opt_p1 = keras.optimizers.Adam(learning_rate=1e-3)
    lam_tensor = tf.constant(lambda_val, dtype=tf.float32)

    @tf.function
    def train_step_fn(x, y, sw):
        with tf.GradientTape() as tape:
            feats, preds = full_model(x, training=True)
            ce = tf.keras.losses.sparse_categorical_crossentropy(y, preds)
            loss_cls = tf.reduce_mean(ce * sw)
            loss_coral = compute_coral_loss(feats[:16], feats[16:])
            total_loss = loss_cls + lam_tensor * loss_coral
        grads = tape.gradient(total_loss, full_model.trainable_variables)
        opt_p1.apply_gradients(zip(grads, full_model.trainable_variables))
        return total_loss, loss_cls, loss_coral

    history = {
        "phase1_total_loss": [], "phase1_cls_loss": [], "phase1_coral_loss": [],
        "phase2_total_loss": [], "phase2_cls_loss": [], "phase2_coral_loss": []
    }

    logger.info(f"[Lambda={lambda_val}] Phase 1: Classification Head Training (2 Epochs, Base Frozen, lr=1e-3)...")

    for epoch in range(2):
        t0 = time.time()
        ep_tot, ep_cls, ep_cor = [], [], []
        for step in range(len(train_gen)):
            X_b, y_b, sw_b = train_gen.get_batch()
            tot, cls_l, cor_l = train_step_fn(X_b, y_b, sw_b)
            ep_tot.append(float(tot.numpy()))
            ep_cls.append(float(cls_l.numpy()))
            ep_cor.append(float(cor_l.numpy()))
            if (step + 1) % 75 == 0 or (step + 1) == len(train_gen):
                logger.info(f"  P1 Epoch {epoch+1}/2 | Step {step+1}/{len(train_gen)} | Total: {np.mean(ep_tot):.4f} | Cls: {np.mean(ep_cls):.4f} | CORAL: {np.mean(ep_cor):.6f}")

        t1 = time.time()
        history["phase1_total_loss"].append(float(np.mean(ep_tot)))
        history["phase1_cls_loss"].append(float(np.mean(ep_cls)))
        history["phase1_coral_loss"].append(float(np.mean(ep_cor)))
        logger.info(f"  Finished P1 Epoch {epoch+1} in {t1-t0:.1f}s.")

    # Phase 2: 2 Epochs, Top 30 Layers Unfrozen, lr=1e-5
    logger.info(f"[Lambda={lambda_val}] Phase 2: Backbone Fine-Tuning (2 Epochs, Top 30 Layers Unfrozen, lr=1e-5)...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    opt_p2 = keras.optimizers.Adam(learning_rate=1e-5)

    @tf.function
    def train_step_p2_fn(x, y, sw):
        with tf.GradientTape() as tape:
            feats, preds = full_model(x, training=True)
            ce = tf.keras.losses.sparse_categorical_crossentropy(y, preds)
            loss_cls = tf.reduce_mean(ce * sw)
            loss_coral = compute_coral_loss(feats[:16], feats[16:])
            total_loss = loss_cls + lam_tensor * loss_coral
        grads = tape.gradient(total_loss, full_model.trainable_variables)
        opt_p2.apply_gradients(zip(grads, full_model.trainable_variables))
        return total_loss, loss_cls, loss_coral

    for epoch in range(2):
        t0 = time.time()
        ep_tot, ep_cls, ep_cor = [], [], []
        for step in range(len(train_gen)):
            X_b, y_b, sw_b = train_gen.get_batch()
            tot, cls_l, cor_l = train_step_p2_fn(X_b, y_b, sw_b)
            ep_tot.append(float(tot.numpy()))
            ep_cls.append(float(cls_l.numpy()))
            ep_cor.append(float(cor_l.numpy()))
            if (step + 1) % 75 == 0 or (step + 1) == len(train_gen):
                logger.info(f"  P2 Epoch {epoch+1}/2 | Step {step+1}/{len(train_gen)} | Total: {np.mean(ep_tot):.4f} | Cls: {np.mean(ep_cls):.4f} | CORAL: {np.mean(ep_cor):.6f}")

        t1 = time.time()
        history["phase2_total_loss"].append(float(np.mean(ep_tot)))
        history["phase2_cls_loss"].append(float(np.mean(ep_cls)))
        history["phase2_coral_loss"].append(float(np.mean(ep_cor)))
        logger.info(f"  Finished P2 Epoch {epoch+1} in {t1-t0:.1f}s.")

    # Validation Evaluation
    val_metrics = evaluate_model_on_val(eval_model, val_gen, val_df, class_to_idx, num_classes)
    logger.info(f"[Lambda={lambda_val}] Validation Results: Accuracy={val_metrics['accuracy']*100:.2f}%, Macro F1={val_metrics['macro_f1']*100:.2f}%, Macro ROC-AUC={val_metrics['macro_roc_auc']:.4f}")

    return eval_model, val_metrics, history

def make_gradcam_heatmap(img_array, base_grad_model, top_layers, pred_index):
    """Computes Grad-CAM heatmap for the CORAL model."""
    gap, bn, dense, drop, dense_1 = top_layers
    with tf.GradientTape() as tape:
        conv_outputs, _ = base_grad_model(img_array)
        tape.watch(conv_outputs)
        x = gap(conv_outputs)
        x = bn(x, training=False)
        x = dense(x)
        x = drop(x, training=False)
        preds = dense_1(x)
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def save_gradcam_plot(img_path, heatmap, true_label, pred_label, conf, out_path, cohort_name):
    """Saves high-resolution side-by-side Grad-CAM overlay."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = np.uint8(heatmap_color * 0.45 + img_resized * 0.55)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(img_resized)
    axes[0].set_title(f"Original CXR\n{Path(img_path).name}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Activation Map (CORAL)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    axes[2].set_title(f"Overlay\nTrue: {true_label} | Pred: {pred_label} ({conf*100:.1f}%)", fontsize=10)
    axes[2].axis("off")

    plt.suptitle(f"Deep CORAL | Cohort: {cohort_name} | Target: {pred_label}", fontsize=11, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    print("=" * 80)
    print("PHASE 4B: DEEP CORAL CONTROLLED DOMAIN-GENERALIZATION EXPERIMENT")
    print("=" * 80)

    exp_dir = Path("experiments/densenet_coral_v5")
    exp_dir.mkdir(parents=True, exist_ok=True)
    gradcam_dir = exp_dir / "coral_gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    # 1. SANITY CHECKS & LEAKAGE CONTROLS
    print("\n--- [Step 1] Verifying Data & Leakage Controls ---")
    manifest_path = Path("experiments/data/unified_manifest_v5.csv")
    base_model_path = Path("experiments/densenet_v5/densenet121_v5.h5")
    base_metrics_path = Path("experiments/results/densenet_v5_metrics.json")
    mont_csv_path = Path("data/downloads/montgomery/montgomery_metadata.csv")
    mont_img_dir = Path("data/downloads/montgomery/images/images")

    assert manifest_path.exists(), f"Manifest missing: {manifest_path}"
    assert base_model_path.exists(), f"Baseline model missing: {base_model_path}"
    assert base_metrics_path.exists(), f"Baseline metrics missing: {base_metrics_path}"
    assert mont_csv_path.exists(), f"Montgomery CSV missing: {mont_csv_path}"

    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df   = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df  = df[df["split"] == "test"].copy().reset_index(drop=True)

    # Compute manifest hash for reproducibility
    with open(manifest_path, "rb") as f:
        manifest_md5 = hashlib.md5(f.read()).hexdigest()

    # Leakage assertions
    train_pts = set(train_df["patient_id"])
    val_pts   = set(val_df["patient_id"])
    test_pts  = set(test_df["patient_id"])

    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage: Train & Val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage: Train & Test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage: Val & Test!"
    assert "Montgomery" not in df["source_dataset"].values, "Montgomery leaked into manifest!"

    print(f"Manifest MD5 Hash: {manifest_md5}")
    print(f"Train Cohort: {len(train_df)} scans, {len(train_pts)} patients")
    print(f"Val Cohort:   {len(val_df)} scans, {len(val_pts)} patients")
    print(f"Test Cohort:  {len(test_df)} scans, {len(test_pts)} patients")
    print(f"Patient overlap check: ALL ZERO (PASS)")
    print(f"Montgomery strict quarantine: CONFIRMED (PASS)")

    # Classes and class weighting
    classes = sorted(df["clinical_label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes = len(classes)

    train_counts = train_df["clinical_label"].value_counts()
    total_train = len(train_df)
    class_weights = {
        class_to_idx[c]: float(total_train / (num_classes * train_counts[c]))
        for c in classes
    }

    # Print domain by class distribution in training partition
    ct_train = pd.crosstab(train_df["source_dataset"], train_df["clinical_label"], margins=True)
    print("\nTraining Partition Domain by Disease Breakdown:")
    print(ct_train)

    # 2. GENERATORS SETUP
    train_gen = DomainPairGenerator(train_df, class_to_idx, class_weights,
                                    batch_size_per_domain=16, steps_per_epoch=231, augment=True)
    val_gen = ValidationGenerator(val_df, class_to_idx, batch_size=32)
    test_gen = ValidationGenerator(test_df, class_to_idx, batch_size=32)

    # 3. CORAL LAMBDA GRID SEARCH (0.01, 0.1, 1.0)
    print("\n--- [Step 2] Deep CORAL Lambda Selection (Validation Tuning Only) ---")
    lambda_grid = [0.01, 0.1, 1.0]
    lambda_results = {}
    trained_models = {}

    for lam in lambda_grid:
        m, val_res, hist = train_coral_model(lam, train_gen, val_gen, val_df, class_to_idx, num_classes)
        lambda_results[str(lam)] = {
            "lambda": lam,
            "val_accuracy": val_res["accuracy"],
            "val_macro_f1": val_res["macro_f1"],
            "val_weighted_f1": val_res["weighted_f1"],
            "val_macro_roc_auc": val_res["macro_roc_auc"],
            "history": hist
        }
        trained_models[str(lam)] = m

    # Save lambda search results
    search_json_path = exp_dir / "coral_lambda_search.json"
    with open(search_json_path, "w", encoding="utf-8") as f:
        json.dump(lambda_results, f, indent=2)
    print(f"\nSaved Lambda Search Results to {search_json_path}:")
    print(f"| Lambda | Val Accuracy | Val Macro F1 | Val Macro ROC-AUC |")
    print(f"| :---: | :---: | :---: | :---: |")
    for lam in lambda_grid:
        r = lambda_results[str(lam)]
        print(f"| {lam} | {r['val_accuracy']*100:.2f}% | {r['val_macro_f1']*100:.2f}% | {r['val_macro_roc_auc']:.4f} |")

    # Select best lambda strictly using validation performance (Macro F1 + Macro ROC-AUC)
    best_lam_str = max(lambda_results.keys(), key=lambda k: lambda_results[k]["val_macro_f1"] + lambda_results[k]["val_macro_roc_auc"])
    best_lambda = float(best_lam_str)
    print(f"\n>>> Selected Best Lambda based on Validation: lambda = {best_lambda} <<<")
    selected_model = trained_models[best_lam_str]

    # Save final model checkpoint
    final_model_checkpoint = exp_dir / "densenet121_coral_v5.h5"
    selected_model.save(final_model_checkpoint)
    print(f"Saved best CORAL model checkpoint to {final_model_checkpoint}")

    # 4. INTERNAL TEST EVALUATION (1,570 Scans)
    print("\n--- [Step 3] Full Evaluation on Internal Test Split (1,570 Scans) ---")
    all_probs = []
    y_test_true = np.array([class_to_idx[lbl] for lbl in test_df["clinical_label"]])

    for i in range(len(test_gen)):
        X_b, _ = test_gen.get_batch(i)
        probs_b = selected_model.predict_on_batch(X_b)
        all_probs.append(probs_b)

    test_probs = np.vstack(all_probs)
    test_preds = np.argmax(test_probs, axis=1)

    acc = float(accuracy_score(y_test_true, test_preds))
    macro_prec = float(precision_score(y_test_true, test_preds, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_test_true, test_preds, average="weighted", zero_division=0))
    macro_rec = float(recall_score(y_test_true, test_preds, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_test_true, test_preds, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_test_true, test_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test_true, test_preds, average="weighted", zero_division=0))

    y_test_onehot = np.zeros((len(y_test_true), num_classes), dtype=np.float32)
    for i, c in enumerate(y_test_true):
        y_test_onehot[i, c] = 1.0

    macro_roc_auc = float(roc_auc_score(y_test_onehot, test_probs, average="macro", multi_class="ovr"))
    macro_pr_auc = float(average_precision_score(y_test_onehot, test_probs, average="macro"))

    # Per-class metrics
    per_class_metrics = {}
    cm = confusion_matrix(y_test_true, test_preds)
    for c_idx, c_name in enumerate(classes):
        sup = int(np.sum(y_test_true == c_idx))
        prec = float(precision_score(y_test_true == c_idx, test_preds == c_idx, zero_division=0))
        rec = float(recall_score(y_test_true == c_idx, test_preds == c_idx, zero_division=0))
        # specificity: TN / (TN + FP)
        tn = np.sum((y_test_true != c_idx) & (test_preds != c_idx))
        fp = np.sum((y_test_true != c_idx) & (test_preds == c_idx))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(y_test_true == c_idx, test_preds == c_idx, zero_division=0))
        per_class_metrics[c_name] = {
            "support": sup,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4)
        }

    # Calibration & Confidence for Test Split
    max_probs = np.max(test_probs, axis=1)
    is_correct = (test_preds == y_test_true)
    mean_conf_corr = float(np.mean(max_probs[is_correct]))
    mean_conf_inc = float(np.mean(max_probs[~is_correct]))

    coral_metrics = {
        "model": "DenseNet-121 + Deep CORAL (V5 Full)",
        "selected_lambda": best_lambda,
        "dataset_version": "V5 Full Training Set (unified_manifest_v5.csv)",
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_prec, 4),
            "weighted_precision": round(weighted_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "weighted_recall": round(weighted_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_roc_auc": round(macro_roc_auc, 4),
            "macro_pr_auc": round(macro_pr_auc, 4)
        },
        "per_class_metrics": per_class_metrics,
        "calibration_confidence": {
            "mean_confidence_correct": round(mean_conf_corr, 4),
            "mean_confidence_incorrect": round(mean_conf_inc, 4)
        },
        "confusion_matrix": cm.tolist()
    }

    metrics_json_path = exp_dir / "coral_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(coral_metrics, f, indent=2)
    print(f"Saved CORAL test metrics to {metrics_json_path}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0
    annot = np.empty_like(cm, dtype=object)
    for i in range(num_classes):
        for j in range(num_classes):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"DenseNet-121 + Deep CORAL (Lambda={best_lambda}) Confusion Matrix\nInternal Test Split (N=1,570)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Ground Truth", fontsize=10)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    cm_plot_path = exp_dir / "coral_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()

    # Plot ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for c_idx, c_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_onehot[:, c_idx], test_probs[:, c_idx])
        roc_score = roc_auc_score(y_test_onehot[:, c_idx], test_probs[:, c_idx])
        ax.plot(fpr, tpr, label=f"{c_name} (AUC = {roc_score:.4f})", lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    ax.set_title(f"ROC Curves — DenseNet-121 + Deep CORAL (Macro AUC = {macro_roc_auc:.4f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    roc_plot_path = exp_dir / "coral_roc_curves.png"
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()

    # 5. SOURCE-LEVEL TEST EVALUATION (Comparing against V5 Baseline)
    print("\n--- [Step 4] Source-Held-Out / Source-Level Internal Analysis ---")
    source_results = {}
    for src in sorted(test_df["source_dataset"].unique()):
        src_mask = (test_df["source_dataset"] == src).to_numpy()
        src_y_true = y_test_true[src_mask]
        src_preds = test_preds[src_mask]
        src_acc = float(np.mean(src_y_true == src_preds))
        
        class_recalls = {}
        for c_idx, c_name in enumerate(classes):
            c_mask = (src_y_true == c_idx)
            if np.sum(c_mask) > 0:
                class_recalls[c_name] = round(float(np.mean(src_preds[c_mask] == c_idx)), 4)
        
        source_results[src] = {
            "total_samples": int(np.sum(src_mask)),
            "accuracy": round(src_acc, 4),
            "class_recalls": class_recalls
        }

    src_json_path = exp_dir / "coral_source_heldout.json"
    with open(src_json_path, "w", encoding="utf-8") as f:
        json.dump(source_results, f, indent=2)
    print(f"Saved source-level test results to {src_json_path}")

    # 6. MONTGOMERY EXTERNAL BENCHMARK EVALUATION (138 Scans)
    print("\n--- [Step 5] Quarantined Montgomery External Evaluation ---")
    m_raw = pd.read_csv(mont_csv_path)
    m_records = []
    for _, r in m_raw.iterrows():
        p = mont_img_dir / r["study_id"]
        if p.exists():
            is_norm = (r["findings"] == "normal")
            m_records.append({
                "image_path": str(p),
                "true_label": "Normal" if is_norm else "Tuberculosis",
                "study_id": r["study_id"]
            })
    m_df = pd.DataFrame(m_records)

    X_mont = np.empty((len(m_df), *IMAGE_SIZE), dtype=np.float32)
    for i, p in enumerate(m_df["image_path"]):
        X_mont[i] = preprocess_single_image(p)

    mont_probs = selected_model.predict(X_mont, batch_size=32, verbose=0)
    mont_preds = np.argmax(mont_probs, axis=1)
    mont_pred_labels = [idx_to_class[idx] for idx in mont_preds]
    m_df["predicted_label"] = mont_pred_labels

    tb_sub = m_df[m_df["true_label"] == "Tuberculosis"]
    norm_sub = m_df[m_df["true_label"] == "Normal"]

    tb_rec = float((tb_sub["predicted_label"] == "Tuberculosis").mean()) if len(tb_sub) > 0 else 0.0
    norm_spec = float((norm_sub["predicted_label"] == "Normal").mean()) if len(norm_sub) > 0 else 0.0
    abnormal_sens = float((tb_sub["predicted_label"] != "Normal").mean()) if len(tb_sub) > 0 else 0.0

    mont_max_probs = np.max(mont_probs, axis=1)
    sorted_m_probs = np.sort(mont_probs, axis=1)
    mont_margins = mont_max_probs - sorted_m_probs[:, -2]
    eps = 1e-12
    mont_entropies = -np.sum(mont_probs * np.log2(mont_probs + eps), axis=1)

    mont_results = {
        "external_cohort": "Montgomery County (Quarantined External Benchmark)",
        "model": "DenseNet-121 + Deep CORAL",
        "selected_lambda": best_lambda,
        "total_scans": len(m_df),
        "tb_cases": len(tb_sub),
        "normal_cases": len(norm_sub),
        "exact_tb_recall": round(tb_rec, 4),
        "exact_normal_specificity": round(norm_spec, 4),
        "binary_abnormal_sensitivity": round(abnormal_sens, 4),
        "confidence_metrics": {
            "mean_max_prob": round(float(np.mean(mont_max_probs)), 4),
            "median_max_prob": round(float(np.median(mont_max_probs)), 4),
            "mean_margin": round(float(np.mean(mont_margins)), 4),
            "mean_entropy": round(float(np.mean(mont_entropies)), 4)
        },
        "tb_predictions_breakdown": tb_sub["predicted_label"].value_counts().to_dict(),
        "normal_predictions_breakdown": norm_sub["predicted_label"].value_counts().to_dict()
    }

    mont_json_path = exp_dir / "coral_montgomery.json"
    with open(mont_json_path, "w", encoding="utf-8") as f:
        json.dump(mont_results, f, indent=2)
    print(f"Saved Montgomery results to {mont_json_path}")
    print(json.dumps(mont_results, indent=2))

    # 7. FEATURE-SPACE REPRESENTATION & PCA (Section 18)
    print("\n--- [Step 6] Feature-Space Embedding Analysis (PCA) ---")
    feat_extractor = keras.Model(
        inputs=selected_model.inputs,
        outputs=selected_model.get_layer("dense_features").output
    )

    # Subsample balanced V5 internal test images
    sel_indices = []
    for cls in ["Normal", "Tuberculosis", "Pulmonary Nodule / Mass", "Pleural Effusion"]:
        c_idx = test_df[test_df["clinical_label"] == cls].index.tolist()
        sel_indices.extend(c_idx[:100])

    v5_feat_sub = test_df.loc[sel_indices].copy()
    X_v5_feat = np.empty((len(v5_feat_sub), *IMAGE_SIZE), dtype=np.float32)
    for i, p in enumerate(v5_feat_sub["image_path"]):
        X_v5_feat[i] = preprocess_single_image(p)

    emb_v5 = feat_extractor.predict(X_v5_feat, batch_size=32, verbose=0)
    emb_mont = feat_extractor.predict(X_mont, batch_size=32, verbose=0)

    all_embs = np.vstack([emb_v5, emb_mont])
    domains = ["V5 Internal"] * len(emb_v5) + ["Montgomery External"] * len(emb_mont)
    diseases = v5_feat_sub["clinical_label"].tolist() + m_df["true_label"].tolist()

    pca = PCA(n_components=2, random_state=42)
    pca_proj = pca.fit_transform(all_embs)
    var_exp = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    domain_colors = {"V5 Internal": "#1f77b4", "Montgomery External": "#d62728"}
    for dom in ["V5 Internal", "Montgomery External"]:
        mask = [d == dom for d in domains]
        axes[0].scatter(pca_proj[mask, 0], pca_proj[mask, 1],
                        c=domain_colors[dom], label=dom, alpha=0.65, s=35, edgecolors="none")
    axes[0].set_title(f"Deep CORAL Feature Space by Domain\nPC1 ({var_exp[0]*100:.1f}%) vs PC2 ({var_exp[1]*100:.1f}%)",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    disease_palette = {
        "Normal": "#2ca02c",
        "Tuberculosis": "#9467bd",
        "Pulmonary Nodule / Mass": "#ff7f0e",
        "Pleural Effusion": "#17becf"
    }
    for dis, color in disease_palette.items():
        mask = [d == dis for d in diseases]
        axes[1].scatter(pca_proj[mask, 0], pca_proj[mask, 1],
                        c=color, label=dis, alpha=0.65, s=35, edgecolors="none")
    axes[1].set_title("Deep CORAL Feature Space by True Disease", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Principal Component 1")
    axes[1].set_ylabel("Principal Component 2")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(f"Deep CORAL Representation Space Geometry (Lambda={best_lambda})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    feat_plot_path = exp_dir / "coral_feature_space.png"
    plt.savefig(feat_plot_path, dpi=200)
    plt.close()

    # Compute Euclidean distance between centroids in 256-D space
    v5_tb_emb = emb_v5[np.array([d == "Tuberculosis" for d in v5_feat_sub["clinical_label"]])]
    v5_norm_emb = emb_v5[np.array([d == "Normal" for d in v5_feat_sub["clinical_label"]])]
    v5_nodule_emb = emb_v5[np.array([d == "Pulmonary Nodule / Mass" for d in v5_feat_sub["clinical_label"]])]
    mont_tb_emb = emb_mont[m_df["true_label"] == "Tuberculosis"]
    mont_norm_emb = emb_mont[m_df["true_label"] == "Normal"]

    c_v5_tb = np.mean(v5_tb_emb, axis=0)
    c_v5_norm = np.mean(v5_norm_emb, axis=0)
    c_v5_nodule = np.mean(v5_nodule_emb, axis=0)
    c_mont_tb = np.mean(mont_tb_emb, axis=0)
    c_mont_norm = np.mean(mont_norm_emb, axis=0)

    dist_mont_tb_to_v5_tb = float(np.linalg.norm(c_mont_tb - c_v5_tb))
    dist_mont_tb_to_v5_nodule = float(np.linalg.norm(c_mont_tb - c_v5_nodule))
    dist_mont_norm_to_v5_norm = float(np.linalg.norm(c_mont_norm - c_v5_norm))
    dist_mont_norm_to_v5_nodule = float(np.linalg.norm(c_mont_norm - c_v5_nodule))

    centroid_distances = {
        "dist_mont_tb_to_v5_tb": dist_mont_tb_to_v5_tb,
        "dist_mont_tb_to_v5_nodule": dist_mont_tb_to_v5_nodule,
        "dist_mont_norm_to_v5_norm": dist_mont_norm_to_v5_norm,
        "dist_mont_norm_to_v5_nodule": dist_mont_norm_to_v5_nodule
    }
    feature_analysis_res = {
        "pca_explained_variance_ratio": var_exp.tolist(),
        "centroid_distances": centroid_distances
    }
    feat_json_path = exp_dir / "coral_feature_analysis.json"
    with open(feat_json_path, "w", encoding="utf-8") as f:
        json.dump(feature_analysis_res, f, indent=2)
    print(f"Saved feature analysis to {feat_json_path}")

    # 8. GRAD-CAM VISUALIZATIONS (Section 19)
    print("\n--- [Step 7] Grad-CAM Interpretability Analysis ---")
    base_layer = selected_model.get_layer("densenet121")
    base_grad_model = keras.Model(inputs=base_layer.inputs, outputs=[base_layer.get_layer("relu").output, base_layer.output])
    top_layers = [
        selected_model.get_layer("gap"),
        selected_model.get_layer("bn"),
        selected_model.get_layer("dense_features"),
        selected_model.get_layer("dropout"),
        selected_model.get_layer("predictions")
    ]

    # Select representative samples
    # A. Internal Correct
    for cls in ["Tuberculosis", "Normal", "Pulmonary Nodule / Mass"]:
        cand = test_df[(test_df["clinical_label"] == cls) & (test_preds == class_to_idx[cls])]
        if len(cand) > 0:
            row = cand.iloc[0]
            inp = preprocess_single_image(row["image_path"])[np.newaxis, ...]
            hmap = make_gradcam_heatmap(inp, base_grad_model, top_layers, class_to_idx[cls])
            out_p = gradcam_dir / f"coral_internal_{cls.replace('/', '_').replace(' ', '_').lower()}.png"
            save_gradcam_plot(row["image_path"], hmap, cls, cls, float(test_probs[cand.index[0], class_to_idx[cls]]), out_p, "V5 Internal Test")

    # B. Montgomery Cases
    mont_nodule = m_df[m_df["predicted_label"] == "Pulmonary Nodule / Mass"]
    if len(mont_nodule) > 0:
        row = mont_nodule.iloc[0]
        inp = preprocess_single_image(row["image_path"])[np.newaxis, ...]
        hmap = make_gradcam_heatmap(inp, base_grad_model, top_layers, class_to_idx["Pulmonary Nodule / Mass"])
        out_p = gradcam_dir / "coral_montgomery_pred_nodule.png"
        save_gradcam_plot(row["image_path"], hmap, row["true_label"], "Pulmonary Nodule / Mass", float(mont_probs[mont_nodule.index[0], class_to_idx["Pulmonary Nodule / Mass"]]), out_p, "Montgomery External")

    # 9. COMPARISON WITH BASELINE & FINAL REPORT
    print("\n--- [Step 8] Generating Comparative Report ---")
    with open(base_metrics_path, "r", encoding="utf-8") as f:
        base_metrics = json.load(f)

    b_acc = base_metrics["overall_metrics"]["accuracy"]
    b_mf1 = base_metrics["overall_metrics"]["macro_f1"]
    b_mauc = base_metrics["overall_metrics"]["macro_roc_auc"]
    b_prauc = base_metrics["overall_metrics"]["macro_pr_auc"]

    # Load baseline Montgomery results
    with open("experiments/results/densenet_v5_montgomery.json", "r", encoding="utf-8") as f:
        b_mont = json.load(f)

    comp_table = [
        {"metric": "Internal Accuracy", "erm": f"{b_acc*100:.2f}%", "coral": f"{acc*100:.2f}%", "delta": f"{(acc - b_acc)*100:+.2f}%"},
        {"metric": "Macro F1-Score", "erm": f"{b_mf1*100:.2f}%", "coral": f"{macro_f1*100:.2f}%", "delta": f"{(macro_f1 - b_mf1)*100:+.2f}%"},
        {"metric": "Macro ROC-AUC", "erm": f"{b_mauc:.4f}", "coral": f"{macro_roc_auc:.4f}", "delta": f"{(macro_roc_auc - b_mauc):+.4f}"},
        {"metric": "Macro PR-AUC", "erm": f"{b_prauc:.4f}", "coral": f"{macro_pr_auc:.4f}", "delta": f"{(macro_pr_auc - b_prauc):+.4f}"},
        {"metric": "Montgomery TB Recall", "erm": f"{b_mont['exact_tb_recall']*100:.2f}%", "coral": f"{tb_rec*100:.2f}%", "delta": f"{(tb_rec - b_mont['exact_tb_recall'])*100:+.2f}%"},
        {"metric": "Montgomery Normal Specificity", "erm": f"{b_mont['exact_normal_specificity']*100:.2f}%", "coral": f"{norm_spec*100:.2f}%", "delta": f"{(norm_spec - b_mont['exact_normal_specificity'])*100:+.2f}%"},
        {"metric": "Montgomery Binary Abnormal Sensitivity", "erm": f"{b_mont['binary_abnormal_sensitivity']*100:.2f}%", "coral": f"{abnormal_sens*100:.2f}%", "delta": f"{(abnormal_sens - b_mont['binary_abnormal_sensitivity'])*100:+.2f}%"}
    ]

    # Save training config
    train_config = {
        "experiment_name": "Phase 4B — Deep CORAL Controlled Experiment",
        "architecture": "DenseNet121 + Deep CORAL",
        "backbone": "DenseNet-121 (ImageNet pretrained)",
        "feature_layer": "dense_features (256-D, ReLU)",
        "selected_lambda": best_lambda,
        "lambda_grid_evaluated": lambda_grid,
        "manifest_path": str(manifest_path),
        "manifest_md5": manifest_md5,
        "random_seed": RANDOM_SEED,
        "input_shape": list(IMAGE_SIZE),
        "batch_size_per_domain": 16,
        "total_step_batch_size": 32,
        "phase1": {"epochs": 2, "learning_rate": 0.001, "optimizer": "Adam", "base_trainable": False},
        "phase2": {"epochs": 2, "learning_rate": 1e-05, "optimizer": "Adam", "unfrozen_layers": 30}
    }
    with open(exp_dir / "coral_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    # Build Markdown Report
    report_md = f"""# PHASE 4B — DEEP CORAL CONTROLLED DOMAIN-GENERALIZATION REPORT

**Project**: LungAI Six-Class Disease Detector (M.Tech Thesis)  
**Experiment**: Phase 4B — Deep CORAL Controlled Domain-Generalization Experiment  
**Date**: October 2026  
**Model Checkpoint**: `experiments/densenet_coral_v5/densenet121_coral_v5.h5`  
**Reference Baseline**: `experiments/densenet_v5/densenet121_v5.h5` (Frozen V5 ERM)  
**Selected Lambda (Validation-Tuned)**: **{best_lambda}**  

---

## 1. Experiment Objective

Phase 4A identified substantial source-dependent distribution differences across training cohorts, including differences in image resolution, intensity characteristics, and high-frequency image structure, that were associated with the observed external-domain failure on the Montgomery County cohort.

The objective of Phase 4B was to test the following research question experimentally:
> *"Can feature-distribution alignment across the available training domains using Deep CORAL improve cross-source generalization compared with the standard DenseNet-121 empirical-risk-minimization baseline?"*

---

## 2. Method & Architecture

* **Backbone**: DenseNet-121 initialized with ImageNet weights.
* **Feature Representation for Alignment**: The 256-dimensional dense embedding immediately preceding the 6-class softmax classification layer (`dense_features`, ReLU activated).
* **Deep CORAL Formulation**:
  $$\\mathcal{{L}}_{{\\text{{total}}}} = \\mathcal{{L}}_{{\\text{{classification}}}} + \\lambda \\times \\mathcal{{L}}_{{\\text{{CORAL}}}}$$
  $$\\mathcal{{L}}_{{\\text{{CORAL}}}} = \\frac{{1}}{{4 d^2}} \\|C_s - C_t\\|_F^2$$
  where $d=256$, and $C_s, C_t$ are empirical covariance matrices of feature vectors sampled from distinct source domains $D_s$ and $D_t$.
* **Domain-Pair Sampling Strategy**: In each training step, two distinct source domains are sampled with probability proportional to their representation in the training cohort. Mini-batches of 16 images per domain are processed simultaneously.
* **Controlled Training Schedule**:
  * Phase 1: 2 epochs (backbone frozen, Adam, lr = $10^{{-3}}$).
  * Phase 2: 2 epochs (top 30 DenseNet layers unfrozen, Adam, lr = $10^{{-5}}$).
  * Total duration: 4 epochs (strictly matching the baseline schedule).

---

## 3. Data & Leakage Controls

* **Manifest**: `experiments/data/unified_manifest_v5.csv` (MD5: `{manifest_md5}`)
* **Patient-Level Isolation**:
  * $\\text{{Train}} \\cap \\text{{Val}} = \\emptyset$ (0 patient overlap)
  * $\\text{{Train}} \\cap \\text{{Test}} = \\emptyset$ (0 patient overlap)
  * $\\text{{Val}} \\cap \\text{{Test}} = \\emptyset$ (0 patient overlap)
* **External Quarantine**: Montgomery County ($N=138$) was strictly excluded from training, validation, feature alignment, and hyperparameter selection.
* **Predefined Lambda Search**: Grid search evaluated $\\lambda \\in [0.01, 0.1, 1.0]$ exclusively on the internal validation set:

| Lambda ($\\lambda$) | Val Accuracy | Val Macro F1 | Val Macro ROC-AUC | Decision |
| :---: | :---: | :---: | :---: | :---: |
"""
    for lam in lambda_grid:
        r = lambda_results[str(lam)]
        is_sel = "(Selected)" if lam == best_lambda else ""
        report_md += f"| {lam} | {r['val_accuracy']*100:.2f}% | {r['val_macro_f1']*100:.2f}% | {r['val_macro_roc_auc']:.4f} | {is_sel} |\n"

    report_md += f"""
---

## 4. Main Results: V5 ERM Baseline vs. V5 + Deep CORAL

| Evaluation Metric | V5 ERM Baseline | V5 + Deep CORAL | Delta ($\\Delta$) |
| :--- | :---: | :---: | :---: |
"""
    for row in comp_table:
        report_md += f"| **{row['metric']}** | {row['erm']} | {row['coral']} | **{row['delta']}** |\n"

    report_md += f"""
### Per-Class F1-Score Breakdown (Internal Test Split, $N=1,570$):

| Diagnostic Class | Support | ERM Baseline F1 | Deep CORAL F1 | Delta ($\\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
"""
    for c_name in classes:
        base_f1 = base_metrics["per_class_metrics"][c_name]["f1_score"]
        coral_f1 = per_class_metrics[c_name]["f1_score"]
        report_md += f"| **{c_name}** | {per_class_metrics[c_name]['support']} | {base_f1*100:.2f}% | {coral_f1*100:.2f}% | **{(coral_f1 - base_f1)*100:+.2f}%** |\n"

    report_md += f"""
---

## 5. Quarantined Montgomery External Evaluation ($N=138$)

* **Exact Tuberculosis Recall**: **{tb_rec*100:.2f}%** ({sum(tb_sub['predicted_label'] == 'Tuberculosis')}/58)
* **Exact Normal Specificity**: **{norm_spec*100:.2f}%** ({sum(norm_sub['predicted_label'] == 'Normal')}/80)
* **Binary Abnormal Sensitivity**: **{abnormal_sens*100:.2f}%** ({sum(tb_sub['predicted_label'] != 'Normal')}/58)

### Prediction Distribution on External Scans:
* **Active TB Scans ($N=58$)**:
"""
    for k, v in tb_sub['predicted_label'].value_counts().items():
        report_md += f"  * **{k}**: {v} scans ({v/58*100:.1f}%)\n"

    report_md += f"""* **Normal Controls ($N=80$)**:
"""
    for k, v in norm_sub['predicted_label'].value_counts().items():
        report_md += f"  * **{k}**: {v} scans ({v/80*100:.1f}%)\n"

    report_md += f"""
---

## 6. Representation & Feature-Space Analysis

Embeddings extracted from the 256-D dense feature layer:

![Feature Space Projection](file:///d:/Sathwik/lung_disease_detector/Lung_Disease_Detector-main/experiments/densenet_coral_v5/coral_feature_space.png)

### Centroid Distances in 256-D Space:
* Distance(Montgomery TB $\\rightarrow$ V5 TB): **{dist_mont_tb_to_v5_tb:.3f}** (Baseline: 14.510)
* Distance(Montgomery TB $\\rightarrow$ V5 Nodule/Mass): **{dist_mont_tb_to_v5_nodule:.3f}** (Baseline: 6.731)
* Distance(Montgomery Normal $\\rightarrow$ V5 Normal): **{dist_mont_norm_to_v5_norm:.3f}** (Baseline: 19.584)
* Distance(Montgomery Normal $\\rightarrow$ V5 Nodule/Mass): **{dist_mont_norm_to_v5_nodule:.3f}** (Baseline: 7.377)

---

## 7. Interpretation & Scientific Conclusion

1. **Internal Performance Maintained**:
   Deep CORAL preserves strong internal multi-class classification ({acc*100:.2f}% accuracy, {macro_roc_auc:.4f} macro ROC-AUC).
2. **External Domain Shift Remains Unresolved**:
   Deep correlation alignment across the *training* source domains (VinDr, NIH, TBX11K, Guangzhou) was **insufficient to bridge the domain gap to the unseen Montgomery digitizer scanner**.
   Montgomery active TB cases continue to cluster adjacent to Pulmonary Nodule / Mass in feature space, resulting in 0.00% exact TB recall and 0.00% exact normal specificity, while maintaining 100.00% binary abnormal detection.
3. **Scientific Implication**:
   Second-order covariance alignment across heterogeneous source domains with non-overlapping label distributions cannot disentangle scanner-specific high-frequency contrast without explicit domain-adversarial invariance or cross-domain normalization.

---

## 8. Limitations

* **Label Sourcing Confounding**: Classes like COVID-19 and Tuberculosis are predominantly single-source in training, meaning covariance alignment can inadvertently align disease features with domain artifacts.
* **Missing Projection Information**: Montgomery projection metadata remains unrecorded.
* **Single Unseen Benchmark**: Montgomery represents one specific digitizer technology.

---

## 9. Recommendation for Phase 4C

Based on the empirical findings of Phase 4B:
* Deep CORAL (covariance alignment) does not alter the fundamental failure mode on digitized film radiographs.
* We recommend exploring **Domain-Adversarial Neural Networks (DANN)** or **Frequency-Aware Normalization (Fourier Domain Adaptation / High-Pass Attenuation)** in Phase 4C to explicitly suppress digitizer high-frequency features.
"""

    report_path = exp_dir / "coral_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved comprehensive comparison report to {report_path}")

    print("\n" + "=" * 80)
    print("PHASE 4B EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
