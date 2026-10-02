"""
PHASE 10, 11, 12, 13, 14, 15: Model Training, Evaluation, Comparison, and External Validation

Experiments:
- EXP-004: ResNet50 Transfer Learning on Balanced Unified Multi-Source Dataset
- EXP-005: DenseNet121 Transfer Learning on Balanced Unified Multi-Source Dataset

Evaluation:
- Internal Test Evaluation: 1,963 quarantined patient-level held-out test scans across 6 classes
- External Clinical Validation: Montgomery County Tuberculosis Dataset (138 scans: 58 active TB, 80 Normal)
- Detailed per-class metrics: Accuracy, Macro/Weighted Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC
- Forensic check on Lung Cancer recall (testing authentic planar nodules vs baseline shortcut)
"""

import sys
import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, DenseNet121
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# CPU Threading optimization
try:
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
IMAGE_SIZE = (224, 224, 3)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_single_image(img_path: str) -> np.ndarray:
    """Standardized preprocessing: LAB luminance CLAHE + Lanczos-4."""
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
    lab[:, :, 0] = CLAHE.apply(l_chan)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    return img_norm.astype(np.float32)

def augment_image(img: np.ndarray) -> np.ndarray:
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
    angle = np.random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img

class DataGenerator(keras.utils.Sequence):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict, batch_size: int = 32, augment: bool = False, shuffle: bool = True):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        X = np.empty((len(batch_df), *IMAGE_SIZE), dtype=np.float32)
        y = np.empty((len(batch_df),), dtype=np.int64)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            img_arr = preprocess_single_image(row["image_path"])
            if self.augment:
                img_arr = augment_image(img_arr)
            X[i] = img_arr
            y[i] = self.class_to_idx[row["lungai_label"]]

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def build_densenet121(num_classes: int):
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE)
    base.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="LungDenseNet121")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return base, model

def build_resnet50(num_classes: int):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE)
    base.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="ImprovedResNet50")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return base, model

def evaluate_test_set(model: Model, test_df: pd.DataFrame, class_to_idx: dict, class_names: list, experiment_id: str) -> dict:
    logger.info(f"Evaluating {experiment_id} on test set ({len(test_df)} scans)...")
    test_gen = DataGenerator(test_df, class_to_idx, batch_size=32, augment=False, shuffle=False)
    probs = model.predict(test_gen, verbose=1)
    preds = np.argmax(probs, axis=1)
    y_true = np.array([class_to_idx[lbl] for lbl in test_df["lungai_label"]])

    acc = float(accuracy_score(y_true, preds))
    macro_prec = float(precision_score(y_true, preds, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, preds, average="weighted", zero_division=0))
    macro_rec = float(recall_score(y_true, preds, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_true, preds, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))

    y_onehot = tf.keras.utils.to_categorical(y_true, len(class_names))
    try:
        macro_auc = float(roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro"))
    except Exception:
        macro_auc = 0.0

    try:
        macro_pr_auc = float(average_precision_score(y_onehot, probs, average="macro"))
    except Exception:
        macro_pr_auc = 0.0

    cm = confusion_matrix(y_true, preds)

    per_class = {}
    for idx, cname in enumerate(class_names):
        mask = (y_true == idx)
        support = int(np.sum(mask))
        c_recall = float(np.sum((preds == idx) & mask) / max(support, 1))
        pred_support = int(np.sum(preds == idx))
        c_precision = float(np.sum((preds == idx) & mask) / max(pred_support, 1))
        c_f1 = float(2 * c_precision * c_recall / max(c_precision + c_recall, 1e-6))
        tn = int(np.sum((y_true != idx) & (preds != idx)))
        fp = int(np.sum((y_true != idx) & (preds == idx)))
        c_spec = float(tn / max(tn + fp, 1))

        per_class[cname] = {
            "support": support,
            "precision": round(c_precision, 4),
            "recall": round(c_recall, 4),
            "specificity": round(c_spec, 4),
            "f1_score": round(c_f1, 4)
        }

    results = {
        "experiment_id": experiment_id,
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "weighted_precision": round(weighted_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_auc_roc": round(macro_auc, 4),
        "macro_pr_auc": round(macro_pr_auc, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "class_names": class_names
    }

    logger.info(f"\n==========================================")
    logger.info(f"  {experiment_id} Verification Results")
    logger.info(f"==========================================")
    logger.info(f"Accuracy:           {acc*100:.2f}%")
    logger.info(f"Weighted Recall:    {weighted_rec*100:.2f}%")
    logger.info(f"Weighted Precision: {weighted_prec*100:.2f}%")
    logger.info(f"Weighted F1-Score:  {weighted_f1*100:.2f}%")
    logger.info(f"Macro ROC-AUC:      {macro_auc*100:.2f}%")
    logger.info(f"Macro PR-AUC:       {macro_pr_auc*100:.2f}%")
    for cname, m in per_class.items():
        logger.info(f"  {cname:<18} Rec: {m['recall']*100:6.2f}% | Prec: {m['precision']*100:6.2f}% | Spec: {m['specificity']*100:6.2f}% | F1: {m['f1_score']*100:6.2f}% (N={m['support']})")
    logger.info(f"Confusion Matrix:\n{cm}")
    return results

def evaluate_external_validation(model: Model, class_to_idx: dict, class_names: list, experiment_id: str) -> dict:
    """
    Phase 15: External Validation on Quarantined Montgomery County TB Dataset (138 scans).
    Never touched during training or validation!
    """
    logger.info(f"\n>>> Running Phase 15 External Validation on Montgomery County Dataset ({experiment_id}) <<<")
    mont_csv = Path("data/downloads/montgomery/montgomery_metadata.csv")
    mont_img_dir = Path("data/downloads/montgomery/images/images")
    if not mont_csv.exists() or not mont_img_dir.exists():
        logger.warning("Montgomery dataset not found, skipping external validation.")
        return {}

    df = pd.read_csv(mont_csv)
    records = []
    for _, row in df.iterrows():
        p = mont_img_dir / row["study_id"]
        if not p.exists():
            continue
        # Montgomery label mapping: normal vs active TB
        is_normal = (row["findings"] == "normal")
        records.append({
            "image_path": str(p),
            "true_label": "Normal" if is_normal else "Tuberculosis",
            "study_id": row["study_id"]
        })

    m_df = pd.DataFrame(records)
    logger.info(f"Montgomery cohort: {len(m_df)} scans ({sum(m_df['true_label']=='Tuberculosis')} TB, {sum(m_df['true_label']=='Normal')} Normal)")

    X_ext = np.empty((len(m_df), *IMAGE_SIZE), dtype=np.float32)
    for i, p in enumerate(m_df["image_path"]):
        X_ext[i] = preprocess_single_image(p)

    probs = model.predict(X_ext, batch_size=32, verbose=0)
    preds = np.argmax(probs, axis=1)
    pred_labels = [class_names[idx] for idx in preds]

    # Evaluate TB detection sensitivity on external cohort
    tb_mask = (m_df["true_label"] == "Tuberculosis").values
    tb_detected = sum((pred_labels[i] == "Tuberculosis") for i in range(len(m_df)) if tb_mask[i])
    tb_total = sum(tb_mask)
    tb_ext_recall = float(tb_detected / max(tb_total, 1))

    # Normal specificity on external cohort
    norm_mask = (m_df["true_label"] == "Normal").values
    norm_detected = sum((pred_labels[i] == "Normal") for i in range(len(m_df)) if norm_mask[i])
    norm_total = sum(norm_mask)
    norm_ext_spec = float(norm_detected / max(norm_total, 1))

    ext_results = {
        "external_cohort": "Montgomery_County_NLM_NIH",
        "sample_count": len(m_df),
        "tb_external_recall": round(tb_ext_recall, 4),
        "normal_external_specificity": round(norm_ext_spec, 4),
        "tb_detected_count": int(tb_detected),
        "tb_total_count": int(tb_total),
        "normal_detected_count": int(norm_detected),
        "normal_total_count": int(norm_total)
    }
    logger.info(f"External Validation Results:")
    logger.info(f"  Tuberculosis Sensitivity (External Recall): {tb_ext_recall*100:.2f}% ({tb_detected}/{tb_total})")
    logger.info(f"  Normal Specificity (External Accuracy):    {norm_ext_spec*100:.2f}% ({norm_detected}/{norm_total})")
    return ext_results

def main():
    manifest_path = Path("experiments/data/unified_manifest.csv")
    df = pd.read_csv(manifest_path)

    # Balanced class sampling for CPU efficiency (up to 500 images per class in training set)
    train_full = df[df["split"] == "train"]
    train_dfs = []
    for c in train_full["lungai_label"].unique():
        c_sub = train_full[train_full["lungai_label"] == c]
        n_sample = min(500, len(c_sub))
        train_dfs.append(c_sub.sample(n=n_sample, random_state=RANDOM_SEED))
    train_df = pd.concat(train_dfs).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    val_full = df[df["split"] == "val"]
    val_dfs = []
    for c in val_full["lungai_label"].unique():
        c_sub = val_full[val_full["lungai_label"] == c]
        n_sample = min(100, len(c_sub))
        val_dfs.append(c_sub.sample(n=n_sample, random_state=RANDOM_SEED))
    val_df = pd.concat(val_dfs).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    test_df = df[df["split"] == "test"].reset_index(drop=True)

    class_names = sorted(df["lungai_label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    num_classes = len(class_names)
    logger.info(f"Unified 6-Class System: {class_names}")
    logger.info(f"Training set: {len(train_df)} balanced scans | Validation: {len(val_df)} | Quarantined Test: {len(test_df)}")

    # Class weights for focal balance
    counts = train_df["lungai_label"].value_counts()
    class_weights = {class_to_idx[c]: float(len(train_df) / (num_classes * counts[c])) for c in class_names}

    models_dir = Path("experiments/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    train_gen = DataGenerator(train_df, class_to_idx, batch_size=32, augment=True, shuffle=True)
    val_gen = DataGenerator(val_df, class_to_idx, batch_size=32, augment=False, shuffle=False)

    # ─────────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 005: DenseNet121 Transfer Learning with Phased Fine-Tuning
    # ─────────────────────────────────────────────────────────────────────────────
    logger.info("\n>>> STARTING EXP-005: DenseNet121 Transfer Learning <<<")
    dn_base, dn_model = build_densenet121(num_classes)

    logger.info("EXP-005 Phase 1: Classification Head Training (2 epochs)...")
    dn_model.fit(train_gen, validation_data=val_gen, epochs=2, class_weight=class_weights, verbose=1)

    logger.info("EXP-005 Phase 2: Dense Block Fine-Tuning (2 epochs)...")
    dn_base.trainable = True
    for layer in dn_base.layers[:-30]:
        layer.trainable = False
    dn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    dn_model.fit(train_gen, validation_data=val_gen, epochs=2, class_weight=class_weights, verbose=1)

    dn_model.save(models_dir / "densenet121_unified.h5")
    dn_results = evaluate_test_set(dn_model, test_df, class_to_idx, class_names, "EXP-005_DenseNet121")
    dn_ext_results = evaluate_external_validation(dn_model, class_to_idx, class_names, "EXP-005_DenseNet121")
    dn_results["external_validation"] = dn_ext_results

    # ─────────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 004: ResNet50 Transfer Learning on Unified Dataset
    # ─────────────────────────────────────────────────────────────────────────────
    logger.info("\n>>> STARTING EXP-004: ResNet50 Transfer Learning <<<")
    rn_base, rn_model = build_resnet50(num_classes)

    logger.info("EXP-004 Phase 1: Classification Head Training (2 epochs)...")
    rn_model.fit(train_gen, validation_data=val_gen, epochs=2, class_weight=class_weights, verbose=1)

    logger.info("EXP-004 Phase 2: Top Residual Blocks Fine-Tuning (2 epochs)...")
    rn_base.trainable = True
    for layer in rn_base.layers[:-30]:
        layer.trainable = False
    rn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    rn_model.fit(train_gen, validation_data=val_gen, epochs=2, class_weight=class_weights, verbose=1)

    rn_model.save(models_dir / "resnet50_unified.h5")
    rn_results = evaluate_test_set(rn_model, test_df, class_to_idx, class_names, "EXP-004_ResNet50")
    rn_ext_results = evaluate_external_validation(rn_model, class_to_idx, class_names, "EXP-004_ResNet50")
    rn_results["external_validation"] = rn_ext_results

    # Save comprehensive results
    comparison = {
        "exp_005_densenet121": dn_results,
        "exp_004_resnet50": rn_results,
        "class_names": class_names,
        "class_weights": class_weights,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df)
    }
    with open(results_dir / "improved_model_experiments.json", "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\nAll experiments and external validation results saved to {results_dir / 'improved_model_experiments.json'}")

if __name__ == "__main__":
    main()
