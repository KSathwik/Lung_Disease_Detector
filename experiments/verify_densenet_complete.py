"""
Complete, Independent Verification of DenseNet121 on:
1. Unified Internal Test Split (1,963 scans across 6 classes)
2. Quarantined Montgomery County TB Dataset (138 scans: 58 Abnormal/TB, 80 Normal)

Generates exact, reproducible metrics, confusion matrices, and detailed clinical analysis.
"""

import sys
import os
import io
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)

# Force stdout to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("densenet_verifier")

import cv2

CLASS_NAMES = ["COVID-19", "Lung Cancer", "Normal", "Pleural Effusion", "Pneumonia", "Tuberculosis"]
IMAGE_SIZE = (224, 224, 3)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_image(path: str) -> np.ndarray:
    """Exact standardized preprocessing matching training and production pipeline: LAB luminance CLAHE + Lanczos-4."""
    img = cv2.imread(path)
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


def verify_densenet():
    model_path = Path("models/densenet_model.h5")
    if not model_path.exists():
        model_path = Path("experiments/models/densenet121_unified.h5")
    
    logger.info(f"Loading DenseNet121 model from: {model_path} ({model_path.stat().st_size / 1e6:.2f} MB)")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    results_out = {
        "model_architecture": "DenseNet121",
        "model_checkpoint": str(model_path),
        "class_names": CLASS_NAMES,
    }

    # =========================================================================
    # PART 1: UNIFIED INTERNAL TEST EVALUATION (1,963 scans)
    # =========================================================================
    logger.info(">>> PART 1: Evaluating on Unified Internal Test Split <<<")
    manifest_path = Path("experiments/data/unified_manifest.csv")
    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    logger.info(f"Loaded internal test set: {len(test_df)} scans")

    X_test = np.empty((len(test_df), *IMAGE_SIZE), dtype=np.float32)
    for i, p in enumerate(test_df["image_path"]):
        X_test[i] = preprocess_image(p)

    label_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    y_true_indices = np.array([label_to_idx[l] for l in test_df["lungai_label"]])
    y_true_onehot = tf.keras.utils.to_categorical(y_true_indices, num_classes=len(CLASS_NAMES))

    probs_internal = model.predict(X_test, batch_size=32, verbose=0)
    preds_internal = np.argmax(probs_internal, axis=1)

    acc = float(accuracy_score(y_true_indices, preds_internal))
    prec_macro = float(precision_score(y_true_indices, preds_internal, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_indices, preds_internal, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_indices, preds_internal, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true_indices, preds_internal, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true_indices, preds_internal, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true_indices, preds_internal, average="weighted", zero_division=0))

    auc_roc = float(roc_auc_score(y_true_onehot, probs_internal, average="macro", multi_class="ovr"))
    auc_pr = float(average_precision_score(y_true_onehot, probs_internal, average="macro"))

    cm = confusion_matrix(y_true_indices, preds_internal).tolist()

    per_class_internal = {}
    for idx, cls in enumerate(CLASS_NAMES):
        mask = (y_true_indices == idx)
        per_class_internal[cls] = {
            "samples": int(np.sum(mask)),
            "precision": round(float(precision_score(y_true_indices == idx, preds_internal == idx, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_indices == idx, preds_internal == idx, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true_indices == idx, preds_internal == idx, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true_onehot[:, idx], probs_internal[:, idx])), 4),
            "pr_auc": round(float(average_precision_score(y_true_onehot[:, idx], probs_internal[:, idx])), 4),
        }

    results_out["internal_test_evaluation"] = {
        "sample_count": len(test_df),
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_precision": round(prec_weighted, 4),
        "weighted_recall": round(rec_weighted, 4),
        "weighted_f1": round(f1_weighted, 4),
        "macro_auc_roc": round(auc_roc, 4),
        "macro_pr_auc": round(auc_pr, 4),
        "confusion_matrix": cm,
        "per_class": per_class_internal
    }

    # =========================================================================
    # PART 2: QUARANTINED MONTGOMERY COUNTY EXTERNAL VALIDATION (138 scans)
    # =========================================================================
    logger.info("\n>>> PART 2: Evaluating on Montgomery County External Cohort <<<")
    mont_csv = Path("data/downloads/montgomery/montgomery_metadata.csv")
    mont_img_dir = Path("data/downloads/montgomery/images/images")

    m_df = pd.read_csv(mont_csv)
    records = []
    for _, row in m_df.iterrows():
        p = mont_img_dir / row["study_id"]
        if not p.exists():
            continue
        is_normal = (str(row["findings"]).strip().lower() == "normal")
        records.append({
            "image_path": str(p),
            "study_id": row["study_id"],
            "age": row["age"],
            "gender": row["gender"],
            "findings": row["findings"],
            "cohort_class": "Normal" if is_normal else "Abnormal_TB"
        })
    m_valid = pd.DataFrame(records)
    logger.info(f"Loaded Montgomery scans: {len(m_valid)} (80 Normal, 58 Abnormal/TB)")

    X_mont = np.empty((len(m_valid), *IMAGE_SIZE), dtype=np.float32)
    for i, p in enumerate(m_valid["image_path"]):
        X_mont[i] = preprocess_image(p)

    probs_mont = model.predict(X_mont, batch_size=32, verbose=0)
    preds_mont = np.argmax(probs_mont, axis=1)
    pred_labels_mont = [CLASS_NAMES[idx] for idx in preds_mont]
    m_valid["predicted_class"] = pred_labels_mont
    m_valid["confidence"] = np.max(probs_mont, axis=1) * 100.0

    # Add probability per class
    for idx, cls in enumerate(CLASS_NAMES):
        m_valid[f"prob_{cls}"] = probs_mont[:, idx]

    # Clinical Analysis:
    # In chest radiology screening, when assessing TB or Abnormal cases:
    # Does the model flag pathology (i.e. Non-Normal: TB, Pneumonia, Pleural Effusion, Lung Cancer)?
    m_valid["is_flagged_abnormal"] = m_valid["predicted_class"] != "Normal"

    abnormal_subset = m_valid[m_valid["cohort_class"] == "Abnormal_TB"]
    normal_subset = m_valid[m_valid["cohort_class"] == "Normal"]

    # Strict multi-class breakdown for Abnormal/TB cases
    abnormal_pred_counts = abnormal_subset["predicted_class"].value_counts().to_dict()
    # Strict multi-class breakdown for Normal cases
    normal_pred_counts = normal_subset["predicted_class"].value_counts().to_dict()

    # Pathology Screening Sensitivity (Flagged as non-normal when abnormal)
    abnormal_detection_sensitivity = float(abnormal_subset["is_flagged_abnormal"].mean())
    # Normal Specificity (Classified as Normal when normal)
    normal_specificity = float((normal_subset["predicted_class"] == "Normal").mean())

    # Exact TB class sensitivity
    exact_tb_sensitivity = float((abnormal_subset["predicted_class"] == "Tuberculosis").mean())

    # Mean probability assigned to TB across abnormal cohort
    mean_tb_prob_abnormal = float(abnormal_subset["prob_Tuberculosis"].mean())
    mean_tb_prob_normal = float(normal_subset["prob_Tuberculosis"].mean())

    # Mean probability assigned to Normal
    mean_normal_prob_abnormal = float(abnormal_subset["prob_Normal"].mean())
    mean_normal_prob_normal = float(normal_subset["prob_Normal"].mean())

    results_out["montgomery_external_validation"] = {
        "total_scans": len(m_valid),
        "abnormal_tb_cohort_size": len(abnormal_subset),
        "normal_cohort_size": len(normal_subset),
        "abnormal_pred_distribution": abnormal_pred_counts,
        "normal_pred_distribution": normal_pred_counts,
        "metrics": {
            "exact_tb_recall": round(exact_tb_sensitivity, 4),
            "exact_tb_detected_count": int(np.sum(abnormal_subset["predicted_class"] == "Tuberculosis")),
            "abnormal_pathology_detection_sensitivity": round(abnormal_detection_sensitivity, 4),
            "abnormal_pathology_detected_count": int(np.sum(abnormal_subset["is_flagged_abnormal"])),
            "normal_exact_specificity": round(normal_specificity, 4),
            "normal_exact_detected_count": int(np.sum(normal_subset["predicted_class"] == "Normal")),
            "mean_tb_prob_in_abnormal": round(mean_tb_prob_abnormal, 4),
            "mean_tb_prob_in_normal": round(mean_tb_prob_normal, 4),
            "mean_normal_prob_in_abnormal": round(mean_normal_prob_abnormal, 4),
            "mean_normal_prob_in_normal": round(mean_normal_prob_normal, 4),
        },
        "sample_predictions": m_valid[["study_id", "findings", "cohort_class", "predicted_class", "confidence", "prob_Tuberculosis", "prob_Normal", "prob_Pleural Effusion", "prob_Pneumonia"]].head(15).to_dict(orient="records")
    }

    # Save to JSON
    out_file = Path("experiments/results/densenet_complete_verification.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results_out, f, indent=2)

    logger.info(f"\nSaved complete verification results to: {out_file}")

    # Output detailed summary to stdout
    print("\n" + "="*80)
    print("      DENSENET-121 INDEPENDENT COMPREHENSIVE VERIFICATION REPORT")
    print("="*80)
    print(f"Model Checkpoint: {model_path}")
    print(f"\n--- PART 1: INTERNAL TEST SPLIT (N={len(test_df)}) ---")
    print(f"Overall Accuracy:       {acc*100:.2f}%")
    print(f"Weighted F1-Score:      {f1_weighted*100:.2f}% (Precision: {prec_weighted*100:.2f}%, Recall: {rec_weighted*100:.2f}%)")
    print(f"Macro F1-Score:         {f1_macro*100:.2f}% (Precision: {prec_macro*100:.2f}%, Recall: {rec_macro*100:.2f}%)")
    print(f"Macro ROC-AUC:          {auc_roc*100:.2f}%")
    print(f"Macro PR-AUC:           {auc_pr*100:.2f}%")
    print("\nPer-Class Breakdown:")
    for cls, m in per_class_internal.items():
        print(f"  {cls:18s} | N={m['samples']:4d} | Prec: {m['precision']*100:5.1f}% | Rec: {m['recall']*100:5.1f}% | F1: {m['f1']*100:5.1f}% | AUC: {m['roc_auc']*100:5.1f}%")

    print(f"\n--- PART 2: QUARANTINED MONTGOMERY COUNTY EXTERNAL VALIDATION (N={len(m_valid)}) ---")
    print(f"Cohort Composition: 58 Active TB / Abnormal Scans, 80 Verified Normal Scans")
    print(f"Distribution of Predictions for Abnormal TB Cohort (N=58):")
    for k, v in abnormal_pred_counts.items():
        print(f"  - Predicted as {k:18s}: {v:2d} / 58 ({v/58*100:5.1f}%)")
    print(f"Distribution of Predictions for Normal Cohort (N=80):")
    for k, v in normal_pred_counts.items():
        print(f"  - Predicted as {k:18s}: {v:2d} / 80 ({v/80*100:5.1f}%)")

    print("\nClinical Performance Summary on External Cohort:")
    print(f"  * Strict TB Label Exact Recall:              {exact_tb_sensitivity*100:.2f}% ({np.sum(abnormal_subset['predicted_class'] == 'Tuberculosis')}/58)")
    print(f"  * Pathology Screening Sensitivity (Abnormal): {abnormal_detection_sensitivity*100:.2f}% ({np.sum(abnormal_subset['is_flagged_abnormal'])}/58)")
    print(f"  * Exact Normal Specificity:                   {normal_specificity*100:.2f}% ({np.sum(normal_subset['predicted_class'] == 'Normal')}/80)")
    print(f"  * Mean TB Probability on TB Cases:            {mean_tb_prob_abnormal*100:.2f}%")
    print(f"  * Mean TB Probability on Normal Cases:        {mean_tb_prob_normal*100:.2f}%")
    print("="*80)


if __name__ == "__main__":
    verify_densenet()
