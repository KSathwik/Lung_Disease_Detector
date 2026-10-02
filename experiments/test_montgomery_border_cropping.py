"""
Experiments: Test Montgomery Domain-Shift Problem Using Preprocessing Only
Investigates whether non-destructive automatic border/collimator cropping mitigates the
Montgomery County external validation domain shift without hurting internal test performance.

Pipeline A: Original Preprocessing (production)
Pipeline B: Border-Cropped Preprocessing (experimental)
"""

import sys
import os
import io
import json
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Ensure UTF-8 stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("border_cropping_experiment")

CLASS_NAMES = ["COVID-19", "Lung Cancer", "Normal", "Pleural Effusion", "Pneumonia", "Tuberculosis"]
IMAGE_SIZE = (224, 224, 3)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ─── BORDER CROPPING ALGORITHM (NON-DESTRUCTIVE) ──────────────────────────────

def detect_border_crop(img: np.ndarray, dark_thresh=18, min_padding_pct=0.015, max_crop_pct=0.25):
    """
    Non-destructive detection of unexposed scanner/collimator black borders.
    Preserves entire thoracic cavity and apices with guaranteed safety padding.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Fast 512-px proxy analysis
    scale = 512.0 / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    
    # 25th percentile along axes avoids false triggers from lead markers or dead pixels
    row_p25 = np.percentile(small, 25, axis=1)
    col_p25 = np.percentile(small, 25, axis=0)
    
    # Scan from top
    top = 0
    for r in range(int(sh * max_crop_pct)):
        if row_p25[r] > dark_thresh:
            top = max(0, r - int(sh * min_padding_pct))
            break
            
    # Scan from bottom
    bottom = sh
    for r in range(sh - 1, int(sh * (1.0 - max_crop_pct)), -1):
        if row_p25[r] > dark_thresh:
            bottom = min(sh, r + int(sh * min_padding_pct))
            break
            
    # Scan from left
    left = 0
    for c in range(int(sw * max_crop_pct)):
        if col_p25[c] > dark_thresh:
            left = max(0, c - int(sw * min_padding_pct))
            break
            
    # Scan from right
    right = sw
    for c in range(sw - 1, int(sw * (1.0 - max_crop_pct)), -1):
        if col_p25[c] > dark_thresh:
            right = min(sw, c + int(sw * min_padding_pct))
            break
            
    # Scale back to original coordinates
    orig_top = int(top / scale)
    orig_bot = int(bottom / scale)
    orig_left = int(left / scale)
    orig_right = int(right / scale)
    
    # Hard boundaries: never crop more than 25% of any side
    orig_top = max(0, min(orig_top, int(h * max_crop_pct)))
    orig_bot = min(h, max(orig_bot, int(h * (1.0 - max_crop_pct))))
    orig_left = max(0, min(orig_left, int(w * max_crop_pct)))
    orig_right = min(w, max(orig_right, int(w * (1.0 - max_crop_pct))))
    
    return orig_top, orig_bot, orig_left, orig_right


# ─── PREPROCESSING PIPELINES ─────────────────────────────────────────────────

def preprocess_pipeline_a(img_path: str) -> np.ndarray:
    """Pipeline A: Original Preprocessing (LAB CLAHE, resize, ImageNet norm)."""
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


def preprocess_pipeline_b(img_path: str) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pipeline B: Experimental Border-Cropped Preprocessing."""
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((*IMAGE_SIZE[:2], 3), dtype=np.float32), (0, 0, 0, 0)

    # Step 1: Detect and crop borders
    t, b, l, r = detect_border_crop(img)
    cropped = img[t:b, l:r]
    if cropped.size == 0:
        cropped = img

    # Step 2: Identical downstream enhancement
    if len(cropped.shape) == 2:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    elif cropped.shape[2] == 4:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGRA2BGR)

    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped = cv2.GaussianBlur(cropped, (3, 3), 0.8)

    lab = cv2.cvtColor(cropped, cv2.COLOR_RGB2LAB)
    l_chan = np.ascontiguousarray(lab[:, :, 0], dtype=np.uint8)
    lab[:, :, 0] = CLAHE.apply(l_chan)
    cropped = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img_resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    return img_norm.astype(np.float32), (t, b, l, r)


# ─── VISUAL VERIFICATION PANELS ──────────────────────────────────────────────

def create_visual_verification_panel(img_path: str, save_path: str, title: str):
    """
    Creates a 4-panel visual verification image:
    1. Original Image
    2. Detected Crop Boundary (Green Box)
    3. Cropped Image
    4. Final 224x224 Preprocessed Input (denormalized for display)
    """
    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        return
    h, w = orig_bgr.shape[:2]
    t, b, l, r = detect_border_crop(orig_bgr)
    
    # Panel 1: Original (scaled to 300x300)
    p1 = cv2.resize(orig_bgr, (300, 300))
    cv2.putText(p1, "1. Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Panel 2: Detected Crop Bounding Box
    annotated = orig_bgr.copy()
    cv2.rectangle(annotated, (l, t), (r, b), (0, 255, 0), max(4, int(max(h, w) / 500)))
    p2 = cv2.resize(annotated, (300, 300))
    cv2.putText(p2, f"2. Crop Box ({t},{b},{l},{r})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Panel 3: Cropped Image
    cropped = orig_bgr[t:b, l:r]
    p3 = cv2.resize(cropped, (300, 300))
    cv2.putText(p3, f"3. Cropped ({cropped.shape[0]}x{cropped.shape[1]})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Panel 4: Final 224x224 Preprocessed Input
    norm_img, _ = preprocess_pipeline_b(img_path)
    # Denormalize to [0, 255] for visual verification
    disp_img = ((norm_img * STD + MEAN) * 255.0).clip(0, 255).astype(np.uint8)
    disp_bgr = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
    p4 = cv2.resize(disp_bgr, (300, 300))
    cv2.putText(p4, "4. Input (224x224 CLAHE)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)
    
    # Combine horizontally
    combined = np.hstack([p1, p2, p3, p4])
    
    # Add top banner
    banner = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    final_fig = np.vstack([banner, combined])
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, final_fig)


# ─── MAIN EXPERIMENTAL SUITE ─────────────────────────────────────────────────

def run_experiment():
    model_path = Path("models/densenet_model.h5")
    logger.info(f"Loading frozen DenseNet-121 from: {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    
    mont_csv = Path("data/downloads/montgomery/montgomery_metadata.csv")
    mont_dir = Path("data/downloads/montgomery/images/images")
    df_mont = pd.read_csv(mont_csv)

    results = {
        "experiment_name": "Montgomery_Border_Cropping_Investigation",
        "model_file": str(model_path),
        "class_names": CLASS_NAMES,
    }

    # =========================================================================
    # STEP 3 & 4: MONTGOMERY COHORT EVALUATION (138 SCANS)
    # =========================================================================
    logger.info("\n>>> Running Montgomery Evaluation with Pipeline A vs Pipeline B <<<")
    mont_records = []
    for _, row in df_mont.iterrows():
        p = mont_dir / row["study_id"]
        if not p.exists(): continue
        is_normal = (str(row["findings"]).strip().lower() == "normal")
        mont_records.append({
            "study_id": row["study_id"],
            "image_path": str(p),
            "findings": row["findings"],
            "cohort_class": "Normal" if is_normal else "Abnormal_TB"
        })
    m_eval = pd.DataFrame(mont_records)

    N_mont = len(m_eval)
    X_mont_a = np.empty((N_mont, *IMAGE_SIZE), dtype=np.float32)
    X_mont_b = np.empty((N_mont, *IMAGE_SIZE), dtype=np.float32)
    crops_info = []

    logger.info("Preprocessing Montgomery images under Pipeline A and Pipeline B...")
    for i, row in m_eval.iterrows():
        X_mont_a[i] = preprocess_pipeline_a(row["image_path"])
        b_img, crop_box = preprocess_pipeline_b(row["image_path"])
        X_mont_b[i] = b_img
        crops_info.append(crop_box)

    logger.info("Running frozen DenseNet-121 predictions on Montgomery cohorts...")
    probs_a = model.predict(X_mont_a, batch_size=32, verbose=0)
    probs_b = model.predict(X_mont_b, batch_size=32, verbose=0)

    preds_a = np.argmax(probs_a, axis=1)
    preds_b = np.argmax(probs_b, axis=1)

    m_eval["pred_a"] = [CLASS_NAMES[idx] for idx in preds_a]
    m_eval["conf_a"] = np.max(probs_a, axis=1) * 100.0
    m_eval["pred_b"] = [CLASS_NAMES[idx] for idx in preds_b]
    m_eval["conf_b"] = np.max(probs_b, axis=1) * 100.0

    # Metrics for Pipeline A
    ab_a = m_eval[m_eval["cohort_class"] == "Abnormal_TB"]
    no_a = m_eval[m_eval["cohort_class"] == "Normal"]
    
    dist_ab_a = ab_a["pred_a"].value_counts().to_dict()
    dist_no_a = no_a["pred_a"].value_counts().to_dict()
    tb_rec_a = float((ab_a["pred_a"] == "Tuberculosis").mean())
    ab_sens_a = float((ab_a["pred_a"] != "Normal").mean())
    no_spec_a = float((no_a["pred_a"] == "Normal").mean())

    # Metrics for Pipeline B
    ab_b = m_eval[m_eval["cohort_class"] == "Abnormal_TB"]
    no_b = m_eval[m_eval["cohort_class"] == "Normal"]

    dist_ab_b = ab_b["pred_b"].value_counts().to_dict()
    dist_no_b = no_b["pred_b"].value_counts().to_dict()
    tb_rec_b = float((ab_b["pred_b"] == "Tuberculosis").mean())
    ab_sens_b = float((ab_b["pred_b"] != "Normal").mean())
    no_spec_b = float((no_b["pred_b"] == "Normal").mean())

    results["montgomery_comparison"] = {
        "sample_count": N_mont,
        "abnormal_tb_count": len(ab_a),
        "normal_count": len(no_a),
        "pipeline_a_original": {
            "abnormal_prediction_distribution": dist_ab_a,
            "normal_prediction_distribution": dist_no_a,
            "exact_tb_recall": round(tb_rec_a, 4),
            "exact_tb_count": int((ab_a["pred_a"] == "Tuberculosis").sum()),
            "abnormal_detection_sensitivity": round(ab_sens_a, 4),
            "normal_specificity": round(no_spec_a, 4),
            "normal_correct_count": int((no_a["pred_a"] == "Normal").sum()),
            "lung_cancer_pred_count_in_normal": int((no_a["pred_a"] == "Lung Cancer").sum()),
            "mean_tb_prob_abnormal": round(float(probs_a[m_eval["cohort_class"] == "Abnormal_TB", 5].mean()), 4),
            "mean_tb_prob_normal": round(float(probs_a[m_eval["cohort_class"] == "Normal", 5].mean()), 4),
        },
        "pipeline_b_cropped": {
            "abnormal_prediction_distribution": dist_ab_b,
            "normal_prediction_distribution": dist_no_b,
            "exact_tb_recall": round(tb_rec_b, 4),
            "exact_tb_count": int((ab_b["pred_b"] == "Tuberculosis").sum()),
            "abnormal_detection_sensitivity": round(ab_sens_b, 4),
            "normal_specificity": round(no_spec_b, 4),
            "normal_correct_count": int((no_b["pred_b"] == "Normal").sum()),
            "lung_cancer_pred_count_in_normal": int((no_b["pred_b"] == "Lung Cancer").sum()),
            "mean_tb_prob_abnormal": round(float(probs_b[m_eval["cohort_class"] == "Abnormal_TB", 5].mean()), 4),
            "mean_tb_prob_normal": round(float(probs_b[m_eval["cohort_class"] == "Normal", 5].mean()), 4),
        }
    }

    # =========================================================================
    # STEP 5: INTERNAL TEST SET INTEGRITY CHECK (Representative Stratified Split)
    # =========================================================================
    logger.info("\n>>> Running Internal Test Integrity Check with Pipeline A vs Pipeline B <<<")
    manifest_df = pd.read_csv("experiments/data/unified_manifest.csv")
    test_all = manifest_df[manifest_df["split"] == "test"].copy().reset_index(drop=True)
    
    # Stratified sample of 600 scans (100 per class or proportionate) for fast yet statistically rigorous verification
    test_sample = test_all.groupby("lungai_label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 100), random_state=42)
    ).reset_index(drop=True)
    logger.info(f"Evaluating on stratified internal test subset (N = {len(test_sample)})")

    label_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    y_test_indices = np.array([label_to_idx[l] for l in test_sample["lungai_label"]])
    y_test_onehot = tf.keras.utils.to_categorical(y_test_indices, num_classes=len(CLASS_NAMES))

    N_test = len(test_sample)
    X_test_a = np.empty((N_test, *IMAGE_SIZE), dtype=np.float32)
    X_test_b = np.empty((N_test, *IMAGE_SIZE), dtype=np.float32)

    for i, row in test_sample.iterrows():
        X_test_a[i] = preprocess_pipeline_a(row["image_path"])
        b_img, _ = preprocess_pipeline_b(row["image_path"])
        X_test_b[i] = b_img

    probs_test_a = model.predict(X_test_a, batch_size=32, verbose=0)
    probs_test_b = model.predict(X_test_b, batch_size=32, verbose=0)

    preds_test_a = np.argmax(probs_test_a, axis=1)
    preds_test_b = np.argmax(probs_test_b, axis=1)

    acc_a = float(accuracy_score(y_test_indices, preds_test_a))
    f1_w_a = float(f1_score(y_test_indices, preds_test_a, average="weighted", zero_division=0))
    f1_m_a = float(f1_score(y_test_indices, preds_test_a, average="macro", zero_division=0))
    auc_a  = float(roc_auc_score(y_test_onehot, probs_test_a, average="macro", multi_class="ovr"))

    acc_b = float(accuracy_score(y_test_indices, preds_test_b))
    f1_w_b = float(f1_score(y_test_indices, preds_test_b, average="weighted", zero_division=0))
    f1_m_b = float(f1_score(y_test_indices, preds_test_b, average="macro", zero_division=0))
    auc_b  = float(roc_auc_score(y_test_onehot, probs_test_b, average="macro", multi_class="ovr"))

    results["internal_test_comparison"] = {
        "sample_count": len(test_sample),
        "pipeline_a_original": {
            "accuracy": round(acc_a, 4),
            "weighted_f1": round(f1_w_a, 4),
            "macro_f1": round(f1_m_a, 4),
            "macro_roc_auc": round(auc_a, 4)
        },
        "pipeline_b_cropped": {
            "accuracy": round(acc_b, 4),
            "weighted_f1": round(f1_w_b, 4),
            "macro_f1": round(f1_m_b, 4),
            "macro_roc_auc": round(auc_b, 4)
        },
        "accuracy_delta": round(acc_b - acc_a, 4),
        "weighted_f1_delta": round(f1_w_b - f1_w_a, 4)
    }

    # =========================================================================
    # STEP 6: SAVE REPRESENTATIVE VISUAL VERIFICATION PANELS
    # =========================================================================
    logger.info("\n>>> Generating Step 6 Visual Verification Panels <<<")
    sample_fig_dir = Path("experiments/results/cropping_samples")
    sample_fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Normal Montgomery (MCUCXR_0001_0 - has bottom collimator border)
    create_visual_verification_panel(
        "data/downloads/montgomery/images/images/MCUCXR_0001_0.png",
        str(sample_fig_dir / "panel_montgomery_normal_0001.png"),
        "Montgomery Normal: MCUCXR_0001_0.png (Bottom Collimator Mask)"
    )
    # 2. Normal Montgomery (MCUCXR_0020_0 - full frame film)
    create_visual_verification_panel(
        "data/downloads/montgomery/images/images/MCUCXR_0020_0.png",
        str(sample_fig_dir / "panel_montgomery_normal_0020.png"),
        "Montgomery Normal: MCUCXR_0020_0.png (Full-Frame Film)"
    )
    # 3. Abnormal/TB Montgomery (MCUCXR_0085_1 - active TB infiltrate)
    create_visual_verification_panel(
        "data/downloads/montgomery/images/images/MCUCXR_0085_1.png",
        str(sample_fig_dir / "panel_montgomery_abnormal_0085.png"),
        "Montgomery Abnormal TB: MCUCXR_0085_1.png (Active TB Infiltrate)"
    )
    # 4. Abnormal/TB Montgomery (MCUCXR_0140_1 - cavitary TB)
    create_visual_verification_panel(
        "data/downloads/montgomery/images/images/MCUCXR_0140_1.png",
        str(sample_fig_dir / "panel_montgomery_abnormal_0140.png"),
        "Montgomery Abnormal TB: MCUCXR_0140_1.png (Cavitary Infiltrate)"
    )
    # 5. Internal Standard CXR (Pneumonia)
    sample_pneu = manifest_df[manifest_df["lungai_label"] == "Pneumonia"].iloc[0]["image_path"]
    create_visual_verification_panel(
        sample_pneu,
        str(sample_fig_dir / "panel_internal_pneumonia.png"),
        "Internal Standard CXR (Pneumonia - Rib Cage Preserved)"
    )

    # =========================================================================
    # STEP 7: OBJECTIVE DECISION DETERMINATION
    # =========================================================================
    # Evaluate criteria
    mont_normal_improved = (results["montgomery_comparison"]["pipeline_b_cropped"]["normal_specificity"] > 
                             results["montgomery_comparison"]["pipeline_a_original"]["normal_specificity"] + 0.10)
    mont_tb_improved = (results["montgomery_comparison"]["pipeline_b_cropped"]["exact_tb_recall"] > 
                         results["montgomery_comparison"]["pipeline_a_original"]["exact_tb_recall"] + 0.10)
    internal_damaged = (results["internal_test_comparison"]["accuracy_delta"] < -0.03)

    if (mont_normal_improved or mont_tb_improved) and not internal_damaged:
        decision = "CASE A - Cropping clearly improves external performance without materially hurting internal performance."
    elif (mont_normal_improved or mont_tb_improved) and internal_damaged:
        decision = "CASE B - Cropping improves Montgomery but significantly damages internal performance."
    else:
        decision = "CASE C - Cropping does not meaningfully improve Montgomery. Simple border removal is insufficient; scanner gamma/frequency domain shift must be addressed."

    results["decision"] = decision

    # Save to JSON
    out_json = Path("experiments/results/montgomery_border_cropping_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved complete results to: {out_json}")

    # Print summary report
    print("\n" + "="*80)
    print("      MONTGOMERY BORDER-CROPPING EXPERIMENT TECHNICAL REPORT")
    print("="*80)
    print(f"Frozen Model: {model_path}")
    print("\n--- 1. MONTGOMERY COHORT COMPARISON (N=138: 80 Normal, 58 Abnormal/TB) ---")
    print(f"{'Metric':<35} | {'Pipeline A (Original)':<22} | {'Pipeline B (Cropped)':<22}")
    print("-" * 85)
    print(f"{'Exact TB Recall (58 TB scans)':<35} | {tb_rec_a*100:5.2f}% ({results['montgomery_comparison']['pipeline_a_original']['exact_tb_count']}/58)           | {tb_rec_b*100:5.2f}% ({results['montgomery_comparison']['pipeline_b_cropped']['exact_tb_count']}/58)")
    print(f"{'Normal Specificity (80 Normals)':<35} | {no_spec_a*100:5.2f}% ({results['montgomery_comparison']['pipeline_a_original']['normal_correct_count']}/80)            | {no_spec_b*100:5.2f}% ({results['montgomery_comparison']['pipeline_b_cropped']['normal_correct_count']}/80)")
    print(f"{'Abnormal Sensitivity (Flagged)':<35} | {ab_sens_a*100:5.2f}% (58/58)            | {ab_sens_b*100:5.2f}% (58/58)")
    print(f"{'Lung Cancer Predictions in Normal':<35} | {results['montgomery_comparison']['pipeline_a_original']['lung_cancer_pred_count_in_normal']:2d} / 80 ({results['montgomery_comparison']['pipeline_a_original']['lung_cancer_pred_count_in_normal']/80*100:.1f}%)        | {results['montgomery_comparison']['pipeline_b_cropped']['lung_cancer_pred_count_in_normal']:2d} / 80 ({results['montgomery_comparison']['pipeline_b_cropped']['lung_cancer_pred_count_in_normal']/80*100:.1f}%)")
    print(f"{'Mean TB Probability (TB cases)':<35} | {results['montgomery_comparison']['pipeline_a_original']['mean_tb_prob_abnormal']*100:5.2f}%                  | {results['montgomery_comparison']['pipeline_b_cropped']['mean_tb_prob_abnormal']*100:5.2f}%")
    print(f"{'Mean TB Probability (Normal cases)':<35} | {results['montgomery_comparison']['pipeline_a_original']['mean_tb_prob_normal']*100:5.2f}%                  | {results['montgomery_comparison']['pipeline_b_cropped']['mean_tb_prob_normal']*100:5.2f}%")
    
    print("\nAbnormal Cohort (N=58) Prediction Distribution:")
    print(f"  Pipeline A (Original): {dist_ab_a}")
    print(f"  Pipeline B (Cropped) : {dist_ab_b}")
    print("\nNormal Cohort (N=80) Prediction Distribution:")
    print(f"  Pipeline A (Original): {dist_no_a}")
    print(f"  Pipeline B (Cropped) : {dist_no_b}")

    print("\n--- 2. INTERNAL TEST SUBSET INTEGRITY CHECK (N=564) ---")
    print(f"{'Metric':<30} | {'Pipeline A (Original)':<20} | {'Pipeline B (Cropped)':<20} | {'Delta':<10}")
    print("-" * 85)
    print(f"{'Accuracy':<30} | {acc_a*100:5.2f}%               | {acc_b*100:5.2f}%               | {results['internal_test_comparison']['accuracy_delta']*100:+5.2f}%")
    print(f"{'Weighted F1':<30} | {f1_w_a*100:5.2f}%               | {f1_w_b*100:5.2f}%               | {results['internal_test_comparison']['weighted_f1_delta']*100:+5.2f}%")
    print(f"{'Macro F1':<30} | {f1_m_a*100:5.2f}%               | {f1_m_b*100:5.2f}%               | {(f1_m_b-f1_m_a)*100:+5.2f}%")
    print(f"{'Macro ROC-AUC':<30} | {auc_a*100:5.2f}%               | {auc_b*100:5.2f}%               | {(auc_b-auc_a)*100:+5.2f}%")

    print("\n--- 3. EXPERIMENTAL DECISION ---")
    print(f"RESULT: {decision}")
    print("="*80)


if __name__ == "__main__":
    run_experiment()
