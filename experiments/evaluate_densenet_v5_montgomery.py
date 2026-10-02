"""
Evaluate Saved DenseNet-121 V5 Checkpoint on Quarantined Montgomery County Cohort (N=138)
LungAI Disease Detector Project
"""

import sys
import os
import io
import json
import logging
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

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
    lab[:, :, 0] = get_clahe().apply(l_chan)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    return img_norm.astype(np.float32)


def main():
    model_path = Path("experiments/densenet_v5/densenet121_v5.h5")
    mapping_path = Path("experiments/densenet_v5/class_mapping.json")

    with open(mapping_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"Loading trained DenseNet-121 V5 model from {model_path}...")
    model = keras.models.load_model(model_path)

    mont_csv = Path("data/downloads/montgomery/montgomery_metadata.csv")
    mont_img_dir = Path("data/downloads/montgomery/images/images")

    if not mont_csv.exists() or not mont_img_dir.exists():
        raise FileNotFoundError("Montgomery files missing.")

    m_raw = pd.read_csv(mont_csv)
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
    print(f"Loaded {len(m_df)} Montgomery scans.")

    X_mont = np.empty((len(m_df), *IMAGE_SIZE), dtype=np.float32)
    for i, p in enumerate(m_df["image_path"]):
        X_mont[i] = preprocess_single_image(p)

    mont_probs = model.predict(X_mont, batch_size=32, verbose=1)
    mont_preds = np.argmax(mont_probs, axis=1)
    mont_pred_labels = [idx_to_class[idx] for idx in mont_preds]
    m_df["predicted_label"] = mont_pred_labels

    tb_sub = m_df[m_df["true_label"] == "Tuberculosis"]
    norm_sub = m_df[m_df["true_label"] == "Normal"]

    tb_rec = float((tb_sub["predicted_label"] == "Tuberculosis").mean()) if len(tb_sub) > 0 else 0.0
    norm_spec = float((norm_sub["predicted_label"] == "Normal").mean()) if len(norm_sub) > 0 else 0.0
    abnormal_sens = float((tb_sub["predicted_label"] != "Normal").mean()) if len(tb_sub) > 0 else 0.0

    tb_breakdown = tb_sub["predicted_label"].value_counts().to_dict()
    norm_breakdown = norm_sub["predicted_label"].value_counts().to_dict()

    mont_results = {
        "external_cohort": "Montgomery County (Quarantined External Benchmark)",
        "total_scans": len(m_df),
        "tb_cases": len(tb_sub),
        "normal_cases": len(norm_sub),
        "exact_tb_recall": round(tb_rec, 4),
        "exact_normal_specificity": round(norm_spec, 4),
        "binary_abnormal_sensitivity": round(abnormal_sens, 4),
        "tb_predictions_breakdown": tb_breakdown,
        "normal_predictions_breakdown": norm_breakdown
    }

    out_json = Path("experiments/results/densenet_v5_montgomery.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(mont_results, f, indent=2)
    print(f"Saved Montgomery results to {out_json}")
    print(json.dumps(mont_results, indent=2))

    # Update densenet_v5_report.md
    report_path = Path("experiments/results/densenet_v5_report.md")
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Build Montgomery section
    mont_sec = f"""## 3. Quarantined Montgomery External Evaluation ($N=138$)

* **Exact Tuberculosis Recall**: **{tb_rec*100:.2f}%** ({sum(tb_sub['predicted_label'] == 'Tuberculosis')}/{len(tb_sub)})
* **Exact Normal Specificity**: **{norm_spec*100:.2f}%** ({sum(norm_sub['predicted_label'] == 'Normal')}/{len(norm_sub)})
* **Binary Abnormal Sensitivity**: **{abnormal_sens*100:.2f}%** ({sum(tb_sub['predicted_label'] != 'Normal')}/{len(tb_sub)})

### Predictions on Active TB Cases ($N={len(tb_sub)}$):
"""
    for k, v in tb_breakdown.items():
        mont_sec += f"* **{k}**: {v} scans ({v/len(tb_sub)*100:.1f}%)\n"

    mont_sec += f"""
### Predictions on Normal Controls ($N={len(norm_sub)}$):
"""
    for k, v in norm_breakdown.items():
        mont_sec += f"* **{k}**: {v} scans ({v/len(norm_sub)*100:.1f}%)\n"

    # Replace section 3
    sec3_start = content.find("## 3. Quarantined Montgomery External Evaluation")
    sec4_start = content.find("## 4. Key Thesis Insights")
    if sec3_start != -1 and sec4_start != -1:
        new_content = content[:sec3_start] + mont_sec + "\n---\n\n" + content[sec4_start:]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Updated densenet_v5_report.md successfully.")


if __name__ == "__main__":
    main()
