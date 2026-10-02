"""
EXP-001 & EXP-002: Evaluate and Verify Existing Baselines (ResNet50 & Custom CNN)
Strictly adheres to random_state=42, test_size=0.15, val_size=0.15.
"""

import sys
import os
from pathlib import Path
import json
import numpy as np

# Ensure backend is on sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from ml.preprocessing import DatasetPreprocessor

def evaluate_model(model_path: str, model_name: str, X_test: np.ndarray, y_test: np.ndarray, class_names: list):
    print(f"\n==========================================")
    print(f"  Evaluating {model_name} from {model_path}")
    print(f"==========================================")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found!")
        return None

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X_test, batch_size=32, verbose=1)
    preds = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_test, preds))
    prec_macro = float(precision_score(y_test, preds, average="macro", zero_division=0))
    prec_wt = float(precision_score(y_test, preds, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, preds, average="macro", zero_division=0))
    rec_wt = float(recall_score(y_test, preds, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, preds, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_test, preds, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, preds).tolist()

    y_onehot = tf.keras.utils.to_categorical(y_test, len(class_names))
    try:
        auc_macro = float(roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro"))
    except Exception as e:
        print("Warning: AUC calculation failed:", e)
        auc_macro = 0.0

    # Per-class metrics
    per_class = {}
    for idx, cname in enumerate(class_names):
        mask = (y_test == idx)
        support = int(np.sum(mask))
        c_recall = float(np.sum((preds == idx) & mask) / max(support, 1))
        pred_support = int(np.sum(preds == idx))
        c_precision = float(np.sum((preds == idx) & mask) / max(pred_support, 1))
        c_f1 = float(2 * c_precision * c_recall / max(c_precision + c_recall, 1e-6))
        per_class[cname] = {
            "support": support,
            "precision": round(c_precision, 4),
            "recall": round(c_recall, 4),
            "f1_score": round(c_f1, 4)
        }

    results = {
        "model_name": model_name,
        "model_path": model_path,
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "weighted_precision": round(prec_wt, 4),
        "macro_recall": round(rec_macro, 4),
        "weighted_recall": round(rec_wt, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_wt, 4),
        "macro_auc_roc": round(auc_macro, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_names": class_names
    }

    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Weighted Precision: {prec_wt * 100:.2f}%")
    print(f"Weighted Recall:    {rec_wt * 100:.2f}%")
    print(f"Weighted F1:        {f1_wt * 100:.2f}%")
    print(f"Macro ROC-AUC:      {auc_macro * 100:.2f}%")
    print("\nPer-class summary:")
    for cname, m in per_class.items():
        print(f"  {cname:<15} Precision: {m['precision']*100:6.2f}% | Recall: {m['recall']*100:6.2f}% | F1: {m['f1_score']*100:6.2f}% | Support: {m['support']}")
    print("\nConfusion Matrix:")
    print(np.array(cm))
    return results

def main():
    print("Initializing DatasetPreprocessor on data/raw...")
    preprocessor = DatasetPreprocessor(
        data_dir="data/raw",
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    df = preprocessor.scan_dataset()
    df_clean = preprocessor.clean_dataset(df)
    train_df, val_df, test_df = preprocessor.split_dataset(df_clean)
    
    class_names = sorted(df_clean["label"].unique().tolist())
    preprocessor.encode_labels(df_clean)
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Held-out test set size: {len(test_df)}")

    print("Loading test images (without augmentation)...")
    X_test, y_test = preprocessor.load_images(test_df, augment=False)
    print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
    print(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")

    # Evaluate ResNet50
    rn_results = evaluate_model("models/resnet_model.h5", "ResNet50", X_test, y_test, class_names)
    
    # Evaluate CNN Baseline
    cnn_results = evaluate_model("models/cnn_model.h5", "Custom_CNN", X_test, y_test, class_names)

    output_path = Path("experiments/results/baseline_verification.json")
    with open(output_path, "w") as f:
        json.dump({
            "resnet50": rn_results,
            "custom_cnn": cnn_results,
            "test_sample_count": len(test_df),
            "random_seed": 42
        }, f, indent=2)
    print(f"\nBaseline verification saved to {output_path}")

if __name__ == "__main__":
    main()
