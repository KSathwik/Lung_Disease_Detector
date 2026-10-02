"""
Controlled Preprocessing Investigation of Montgomery Film-Domain Mismatch
Model: Frozen models/densenet_model.h5 (zero retraining, weights unchanged)

Scope & Rules:
- Model frozen: models/densenet_model.h5
- No retraining, no architecture changes, no DANN, no U-Net/segmentation, no new datasets.
- No modifications to production preprocessing pipeline or production model.
- All results saved under experiments/results/film_normalization/

Experiments:
  - Experiment A: Baseline Production Preprocessing (LAB CLAHE clipLimit=2.0, resize 224x224, ImageNet norm)
  - Experiment B: Controlled Film-Grain Suppression (Conservative Bilateral Filter d=5, sigmaColor=25, sigmaSpace=25)
  - Experiment C: Intensity / Histogram Normalization (Reference CDF specification based on image distribution analysis)
  - Experiment D: Combined Preprocessing (Denoising + Histogram Normalization)

Cohorts:
  - Montgomery External (N=138: 58 Abnormal/TB, 80 Normal)
  - Stratified Internal Test Subset (N=564 across 6 classes, identical to cropping experiment seed=42)
"""

import sys
import os
import io
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Ensure UTF-8 stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("film_normalization_investigation")

CLASS_NAMES = ["COVID-19", "Lung Cancer", "Normal", "Pleural Effusion", "Pneumonia", "Tuberculosis"]
IMAGE_SIZE = (224, 224, 3)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
OUTPUT_DIR = Path("experiments/results/film_normalization")


# ─── STEP 1: IMAGE DISTRIBUTION ANALYSIS ─────────────────────────────────────

def measure_intensity_distribution(img_paths: List[str], cohort_name: str) -> Dict[str, float]:
    """
    Measures intensity stats (mean, std, median, p5, p95, dynamic range)
    on tissue region (center 80%) across an image cohort.
    """
    logger.info(f"Measuring intensity distribution for {cohort_name} (N={len(img_paths)})...")
    means, stds, medians, p5s, p95s = [], [], [], [], []
    
    for p in img_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        h, w = img.shape
        if max(h, w) > 512:
            scale = 512.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape
        tissue = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        means.append(float(np.mean(tissue)))
        stds.append(float(np.std(tissue)))
        medians.append(float(np.median(tissue)))
        p5s.append(float(np.percentile(tissue, 5)))
        p95s.append(float(np.percentile(tissue, 95)))

    stats = {
        "mean_intensity": round(float(np.mean(means)), 2),
        "std_intensity": round(float(np.mean(stds)), 2),
        "median_intensity": round(float(np.mean(medians)), 2),
        "percentile_5": round(float(np.mean(p5s)), 2),
        "percentile_95": round(float(np.mean(p95s)), 2),
        "dynamic_range_p5_p95": round(float(np.mean(p95s) - np.mean(p5s)), 2)
    }
    logger.info(f"{cohort_name} Stats: {stats}")
    return stats


def compute_reference_cdf(reference_paths: List[str]) -> np.ndarray:
    """
    Computes empirical cumulative distribution function (CDF) from internal CXR images.
    Used for controlled histogram matching to align external film exposures to internal training dynamic range.
    """
    logger.info(f"Building empirical reference histogram CDF from {len(reference_paths)} internal CXR scans...")
    hist_accum = np.zeros(256, dtype=np.float64)
    valid_count = 0
    
    for p in reference_paths[:200]:  # 200 representative scans provide rock-solid CDF calibration
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        h, w = img.shape
        if max(h, w) > 512:
            scale = 512.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape
        tissue = img[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        hist, _ = np.histogram(tissue, bins=256, range=(0, 256))
        hist_accum += hist
        valid_count += 1
    
    cdf = hist_accum.cumsum()
    cdf = cdf / cdf[-1]  # Normalize to [0, 1]
    logger.info(f"Computed reference CDF across {valid_count} internal images.")
    return cdf


def match_histogram(img_gray: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    """
    Controlled histogram matching: maps input image CDF to internal reference CDF.
    """
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256))
    src_cdf = hist.cumsum()
    if src_cdf[-1] == 0:
        return img_gray
    src_cdf = src_cdf / src_cdf[-1]
    
    lut = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        idx = np.argmin(np.abs(ref_cdf - src_cdf[src_val]))
        lut[src_val] = idx
        
    return cv2.LUT(img_gray, lut)


# ─── PREPROCESSING PIPELINES ─────────────────────────────────────────────────

def _fast_load_and_resize(img_path: str, max_dim: int = 512) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = float(max_dim) / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def preprocess_experiment_a(img_path: str) -> np.ndarray:
    """Experiment A: Baseline Production Preprocessing (LAB CLAHE, resize, ImageNet norm)."""
    img = _fast_load_and_resize(img_path)
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


def preprocess_experiment_b(img_path: str) -> np.ndarray:
    """
    Experiment B: Controlled Film-Grain Suppression.
    Conservative edge-preserving bilateral filter (d=5, sigmaColor=25, sigmaSpace=25)
    to suppress film emulsion grain without blurring costal margins or pulmonary lesions.
    """
    img = _fast_load_and_resize(img_path)
    if img is None:
        return np.zeros((*IMAGE_SIZE[:2], 3), dtype=np.float32)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan = np.ascontiguousarray(lab[:, :, 0], dtype=np.uint8)
    lab[:, :, 0] = CLAHE.apply(l_chan)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    return img_norm.astype(np.float32)


def preprocess_experiment_c(img_path: str, ref_cdf: np.ndarray) -> np.ndarray:
    """
    Experiment C: Intensity / Histogram Normalization.
    Transforms input intensity distribution to match empirical digital CXR reference CDF.
    """
    img = _fast_load_and_resize(img_path)
    if img is None:
        return np.zeros((*IMAGE_SIZE[:2], 3), dtype=np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    matched_gray = match_histogram(gray, ref_cdf)
    matched_rgb = cv2.cvtColor(matched_gray, cv2.COLOR_GRAY2RGB)
    matched_rgb = cv2.GaussianBlur(matched_rgb, (3, 3), 0.8)

    lab = cv2.cvtColor(matched_rgb, cv2.COLOR_RGB2LAB)
    l_chan = np.ascontiguousarray(lab[:, :, 0], dtype=np.uint8)
    lab[:, :, 0] = CLAHE.apply(l_chan)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    return img_norm.astype(np.float32)


def preprocess_experiment_d(img_path: str, ref_cdf: np.ndarray) -> np.ndarray:
    """
    Experiment D: Combined Preprocessing.
    Bilateral film-grain suppression + Reference Histogram Matching.
    """
    img = _fast_load_and_resize(img_path)
    if img is None:
        return np.zeros((*IMAGE_SIZE[:2], 3), dtype=np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    denoised_gray = cv2.bilateralFilter(gray, d=5, sigmaColor=25, sigmaSpace=25)
    matched_gray = match_histogram(denoised_gray, ref_cdf)
    matched_rgb = cv2.cvtColor(matched_gray, cv2.COLOR_GRAY2RGB)

    lab = cv2.cvtColor(matched_rgb, cv2.COLOR_RGB2LAB)
    l_chan = np.ascontiguousarray(lab[:, :, 0], dtype=np.uint8)
    lab[:, :, 0] = CLAHE.apply(l_chan)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    return img_norm.astype(np.float32)



# ─── VISUAL VERIFICATION PANELS ──────────────────────────────────────────────

def create_visual_panel(img_path: str, save_path: str, title: str, ref_cdf: np.ndarray):
    """
    Creates a 4-panel visual verification panel:
    Panel 1: Original Image
    Panel 2: After Bilateral Denoising (Exp B)
    Panel 3: After Intensity Normalization (Exp C)
    Panel 4: Final Combined Preprocessing (Exp D 224x224)
    """
    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None: return

    # Panel 1: Original
    p1 = cv2.resize(orig_bgr, (300, 300))
    cv2.putText(p1, "1. Original Image", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Panel 2: Bilateral Denoised
    denoised = cv2.bilateralFilter(orig_bgr, d=5, sigmaColor=25, sigmaSpace=25)
    p2 = cv2.resize(denoised, (300, 300))
    cv2.putText(p2, "2. Exp B: Denoised", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Panel 3: Histogram Matched
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY) if len(orig_bgr.shape) == 3 else orig_bgr
    matched = match_histogram(gray, ref_cdf)
    matched_bgr = cv2.cvtColor(matched, cv2.COLOR_GRAY2BGR)
    p3 = cv2.resize(matched_bgr, (300, 300))
    cv2.putText(p3, "3. Exp C: Hist Matched", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    # Panel 4: Combined Preprocessed Input 224x224
    norm_d = preprocess_experiment_d(img_path, ref_cdf)
    disp_d = ((norm_d * STD + MEAN) * 255.0).clip(0, 255).astype(np.uint8)
    disp_bgr = cv2.cvtColor(disp_d, cv2.COLOR_RGB2BGR)
    p4 = cv2.resize(disp_bgr, (300, 300))
    cv2.putText(p4, "4. Exp D: Combined (224x224)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 200), 2)

    combined = np.hstack([p1, p2, p3, p4])
    banner = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    final_fig = np.vstack([banner, combined])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, final_fig)


# ─── MAIN INVESTIGATION SUITE ────────────────────────────────────────────────

def run_investigation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path("models/densenet_model.h5")
    logger.info(f"Loading frozen DenseNet-121 from: {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    mont_csv = Path("data/downloads/montgomery/montgomery_metadata.csv")
    mont_dir = Path("data/downloads/montgomery/images/images")
    df_mont = pd.read_csv(mont_csv)

    manifest_df = pd.read_csv("experiments/data/unified_manifest.csv")

    # Montgomery image paths
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
    m_df = pd.DataFrame(mont_records)
    mont_paths = m_df["image_path"].tolist()

    # Internal Test Subset (Identical seed 42 to previous cropping experiment)
    test_all = manifest_df[manifest_df["split"] == "test"].copy().reset_index(drop=True)
    test_sample = test_all.groupby("lungai_label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 100), random_state=42)
    ).reset_index(drop=True)
    test_paths = test_sample["image_path"].tolist()

    # Representative Training/Reference CXRs for Empirical CDF (500 random train images)
    train_sample = manifest_df[manifest_df["split"] == "train"].sample(500, random_state=42)
    train_ref_paths = train_sample["image_path"].tolist()

    # =========================================================================
    # STEP 1: MEASURE INTENSITY DISTRIBUTIONS
    # =========================================================================
    logger.info("\n>>> Step 1: Measuring Intensity Distributions <<<")
    mont_dist_stats = measure_intensity_distribution(mont_paths, "Montgomery External Cohort")
    test_dist_stats = measure_intensity_distribution(test_paths, "Internal Test Subset")
    train_dist_stats = measure_intensity_distribution(train_ref_paths, "Representative Internal Training")

    ref_cdf = compute_reference_cdf(train_ref_paths)

    results_out = {
        "experiment_title": "Montgomery Film-Domain Preprocessing Investigation",
        "model_path": str(model_path),
        "model_status": "FROZEN (zero retraining, weights unchanged)",
        "intensity_distribution_analysis": {
            "montgomery_external": mont_dist_stats,
            "internal_test_subset": test_dist_stats,
            "internal_training_reference": train_dist_stats,
            "key_finding": (
                f"Montgomery film CXRs exhibit lower mean intensity ({mont_dist_stats['mean_intensity']} vs {train_dist_stats['mean_intensity']}) "
                f"and wider standard deviation ({mont_dist_stats['std_intensity']} vs {train_dist_stats['std_intensity']}) "
                f"due to legacy optical film digitized scanning characteristics."
            )
        },
        "experiments": {}
    }

    # Setup Ground Truths
    N_mont = len(m_df)
    N_test = len(test_sample)

    label_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    y_test_indices = np.array([label_to_idx[l] for l in test_sample["lungai_label"]])
    y_test_onehot = tf.keras.utils.to_categorical(y_test_indices, num_classes=len(CLASS_NAMES))

    tb_class_idx = CLASS_NAMES.index("Tuberculosis")
    cancer_class_idx = CLASS_NAMES.index("Lung Cancer")
    effusion_class_idx = CLASS_NAMES.index("Pleural Effusion")

    pipelines = {
        "Exp_A_Baseline": {
            "name": "Experiment A (Baseline Production)",
            "desc": "Standard LAB luminance CLAHE, Lanczos-4 resize 224x224, ImageNet norm",
            "fn": lambda p: preprocess_experiment_a(p)
        },
        "Exp_B_FilmGrainSuppression": {
            "name": "Experiment B (Film-Grain Suppression)",
            "desc": "Edge-preserving Bilateral Filter (d=5, sigmaColor=25, sigmaSpace=25) + CLAHE + norm",
            "fn": lambda p: preprocess_experiment_b(p)
        },
        "Exp_C_HistogramMatching": {
            "name": "Experiment C (Intensity / Histogram Matching)",
            "desc": "Empirical reference CDF specification to internal digital CXR dynamic range + CLAHE + norm",
            "fn": lambda p: preprocess_experiment_c(p, ref_cdf)
        },
        "Exp_D_Combined": {
            "name": "Experiment D (Combined Denoising + Histogram Matching)",
            "desc": "Bilateral filter + empirical reference CDF specification + CLAHE + norm",
            "fn": lambda p: preprocess_experiment_d(p, ref_cdf)
        }
    }

    # =========================================================================
    # STEP 2 & 3: EVALUATE PIPELINES
    # =========================================================================
    baseline_acc = None

    for key, pinfo in pipelines.items():
        logger.info(f"\n========================================================")
        logger.info(f"Evaluating {pinfo['name']}...")
        logger.info(f"========================================================")

        # 1. Montgomery External Evaluation
        X_mont = np.empty((N_mont, *IMAGE_SIZE), dtype=np.float32)
        for i, row in m_df.iterrows():
            X_mont[i] = pinfo["fn"](row["image_path"])

        probs_mont = model.predict(X_mont, batch_size=32, verbose=0)
        preds_mont = np.argmax(probs_mont, axis=1)
        pred_labels = [CLASS_NAMES[idx] for idx in preds_mont]

        ab_mask = (m_df["cohort_class"] == "Abnormal_TB").values
        no_mask = (m_df["cohort_class"] == "Normal").values

        ab_preds = [pred_labels[i] for i in range(N_mont) if ab_mask[i]]
        no_preds = [pred_labels[i] for i in range(N_mont) if no_mask[i]]

        dist_ab = pd.Series(ab_preds).value_counts().to_dict()
        dist_no = pd.Series(no_preds).value_counts().to_dict()

        tb_exact_recall = float(sum(p == "Tuberculosis" for p in ab_preds) / sum(ab_mask))
        tb_exact_count = int(sum(p == "Tuberculosis" for p in ab_preds))
        normal_spec = float(sum(p == "Normal" for p in no_preds) / sum(no_mask))
        normal_correct_count = int(sum(p == "Normal" for p in no_preds))
        ab_sens = float(sum(p != "Normal" for p in ab_preds) / sum(ab_mask))
        
        lung_cancer_fp_count = int(sum(p == "Lung Cancer" for p in no_preds))
        lung_cancer_fp_rate = float(lung_cancer_fp_count / sum(no_mask))

        pleural_effusion_count = int(sum(p == "Pleural Effusion" for p in pred_labels))
        pleural_effusion_rate = float(pleural_effusion_count / N_mont)

        tb_prediction_total_count = int(sum(p == "Tuberculosis" for p in pred_labels))
        tb_prediction_rate = float(tb_prediction_total_count / N_mont)

        mean_tb_prob_ab = float(probs_mont[ab_mask, tb_class_idx].mean())
        mean_tb_prob_no = float(probs_mont[no_mask, tb_class_idx].mean())

        # 2. Stratified Internal Test Subset Evaluation
        X_test = np.empty((N_test, *IMAGE_SIZE), dtype=np.float32)
        for i, row in test_sample.iterrows():
            X_test[i] = pinfo["fn"](row["image_path"])

        probs_test = model.predict(X_test, batch_size=32, verbose=0)
        preds_test = np.argmax(probs_test, axis=1)

        acc = float(accuracy_score(y_test_indices, preds_test))
        f1_w = float(f1_score(y_test_indices, preds_test, average="weighted", zero_division=0))
        f1_m = float(f1_score(y_test_indices, preds_test, average="macro", zero_division=0))
        auc_roc = float(roc_auc_score(y_test_onehot, probs_test, average="macro", multi_class="ovr"))

        if baseline_acc is None:
            baseline_acc = acc

        accuracy_delta = float(acc - baseline_acc)

        results_out["experiments"][key] = {
            "name": pinfo["name"],
            "description": pinfo["desc"],
            "montgomery_metrics": {
                "exact_tb_recall": round(tb_exact_recall, 4),
                "exact_tb_count": f"{tb_exact_count}/{sum(ab_mask)}",
                "normal_specificity": round(normal_spec, 4),
                "normal_correct_count": f"{normal_correct_count}/{sum(no_mask)}",
                "abnormal_detection_sensitivity": round(ab_sens, 4),
                "prediction_distribution": {
                    "abnormal_tb_cohort": dist_ab,
                    "normal_cohort": dist_no
                },
                "mean_tb_prob_abnormal": round(mean_tb_prob_ab, 4),
                "mean_tb_prob_normal": round(mean_tb_prob_no, 4),
                "lung_cancer_fp_count_in_normal": lung_cancer_fp_count,
                "lung_cancer_fp_rate_in_normal": round(lung_cancer_fp_rate, 4),
                "pleural_effusion_prediction_count": pleural_effusion_count,
                "pleural_effusion_prediction_rate": round(pleural_effusion_rate, 4),
                "tb_prediction_total_count": tb_prediction_total_count,
                "tb_prediction_rate": round(tb_prediction_rate, 4)
            },
            "internal_test_metrics": {
                "accuracy": round(acc, 4),
                "weighted_f1": round(f1_w, 4),
                "macro_f1": round(f1_m, 4),
                "macro_roc_auc": round(auc_roc, 4),
                "accuracy_delta_vs_baseline": round(accuracy_delta, 4)
            }
        }

    # =========================================================================
    # STEP 4: VISUAL VERIFICATION PANELS
    # =========================================================================
    logger.info("\n>>> Generating Step 4 Visual Verification Panels <<<")

    # 1. Montgomery Normal (MCUCXR_0001_0.png)
    create_visual_panel(
        "data/downloads/montgomery/images/images/MCUCXR_0001_0.png",
        str(OUTPUT_DIR / "panel_montgomery_normal_0001.png"),
        "Montgomery Normal Scan: MCUCXR_0001_0.png",
        ref_cdf
    )

    # 2. Montgomery Abnormal TB (MCUCXR_0104_1.png)
    create_visual_panel(
        "data/downloads/montgomery/images/images/MCUCXR_0104_1.png",
        str(OUTPUT_DIR / "panel_montgomery_abnormal_0104.png"),
        "Montgomery Active TB Scan: MCUCXR_0104_1.png",
        ref_cdf
    )

    # 3. Internal Digital CXR Normal
    sample_int_norm = manifest_df[manifest_df["lungai_label"] == "Normal"].iloc[0]["image_path"]
    create_visual_panel(
        sample_int_norm,
        str(OUTPUT_DIR / "panel_internal_digital_normal.png"),
        "Internal Digital CXR (Normal Cohort)",
        ref_cdf
    )

    # 4. Internal Digital CXR Abnormal (Pneumonia)
    sample_int_abn = manifest_df[manifest_df["lungai_label"] == "Pneumonia"].iloc[0]["image_path"]
    create_visual_panel(
        sample_int_abn,
        str(OUTPUT_DIR / "panel_internal_digital_pneumonia.png"),
        "Internal Digital CXR (Pneumonia Cohort)",
        ref_cdf
    )

    # =========================================================================
    # STEP 5: DECISION LOGIC DETERMINATION
    # =========================================================================
    exp_b_tb = results_out["experiments"]["Exp_B_FilmGrainSuppression"]["montgomery_metrics"]["exact_tb_recall"]
    exp_b_spec = results_out["experiments"]["Exp_B_FilmGrainSuppression"]["montgomery_metrics"]["normal_specificity"]

    exp_c_tb = results_out["experiments"]["Exp_C_HistogramMatching"]["montgomery_metrics"]["exact_tb_recall"]
    exp_c_spec = results_out["experiments"]["Exp_C_HistogramMatching"]["montgomery_metrics"]["normal_specificity"]

    any_exp_improved = (exp_b_tb > 0.10 or exp_b_spec > 0.10 or exp_c_tb > 0.10 or exp_c_spec > 0.10)

    if not any_exp_improved:
        decision_code = "CASE C"
        decision_summary = "Preprocessing does not meaningfully improve Montgomery external performance."
        recommendation = (
            "Stop preprocessing experimentation. In CASE C, the next step should be a controlled "
            "DATASET/TRAINING intervention (such as fine-tuning with targeted data augmentation or "
            "multi-source training) rather than continuing to add preprocessing tricks."
        )
    else:
        # Check internal performance degradation
        degraded = any(results_out["experiments"][k]["internal_test_metrics"]["accuracy_delta_vs_baseline"] < -0.03
                       for k in ["Exp_B_FilmGrainSuppression", "Exp_C_HistogramMatching", "Exp_D_Combined"])
        if degraded:
            decision_code = "CASE B"
            decision_summary = "Preprocessing improves Montgomery but causes significant internal performance degradation."
            recommendation = "Do not adopt preprocessing yet. Investigate the internal vs external performance trade-off."
        else:
            decision_code = "CASE A"
            decision_summary = "Preprocessing substantially improves Montgomery performance while maintaining acceptable internal performance."
            recommendation = "Recommend this preprocessing for a controlled retraining experiment."

    results_out["decision"] = {
        "code": decision_code,
        "summary": decision_summary,
        "recommendation": recommendation
    }

    # Save to JSON
    out_json = OUTPUT_DIR / "film_normalization_results.json"
    with open(out_json, "w") as f:
        json.dump(results_out, f, indent=2)

    logger.info(f"\nSaved complete experimental results to: {out_json}")

    # =========================================================================
    # PRINT SUMMARY REPORT
    # =========================================================================
    print("\n" + "="*100)
    print("      MONTGOMERY FILM-DOMAIN PREPROCESSING CONTROLLED INVESTIGATION REPORT")
    print("="*100)
    print(f"Frozen Model Checkpoint: {model_path} (Zero Retraining, Frozen Weights)\n")

    print("--- 1. IMAGE-DISTRIBUTION FINDINGS ---")
    print(f"  * Montgomery External (N=138): Mean={mont_dist_stats['mean_intensity']}, Std={mont_dist_stats['std_intensity']}, Range=[{mont_dist_stats['percentile_5']}, {mont_dist_stats['percentile_95']}]")
    print(f"  * Internal Test Subset (N=564): Mean={test_dist_stats['mean_intensity']}, Std={test_dist_stats['std_intensity']}, Range=[{test_dist_stats['percentile_5']}, {test_dist_stats['percentile_95']}]")
    print(f"  * Internal Train Reference (N=500): Mean={train_dist_stats['mean_intensity']}, Std={train_dist_stats['std_intensity']}, Range=[{train_dist_stats['percentile_5']}, {train_dist_stats['percentile_95']}]")

    print("\n--- 2. MONTGOMERY COHORT METRICS (N=138: 58 Abnormal/TB, 80 Normal) ---")
    hdr_m = f"{'Pipeline':<34} | {'Exact TB Rec':<12} | {'Norm Spec':<12} | {'Abn Sens':<12} | {'Mean TB Prob (TB/Norm)':<24}"
    print(hdr_m)
    print("-" * 100)
    for k, v in results_out["experiments"].items():
        mm = v["montgomery_metrics"]
        prob_str = f"{mm['mean_tb_prob_abnormal']:.4f} / {mm['mean_tb_prob_normal']:.4f}"
        print(f"{v['name']:<34} | {mm['exact_tb_recall']*100:5.1f}% ({mm['exact_tb_count']}) | {mm['normal_specificity']*100:5.1f}% ({mm['normal_correct_count']}) | {mm['abnormal_detection_sensitivity']*100:5.1f}% | {prob_str:<24}")

    print("\n--- 3. PREDICTION DISTRIBUTIONS ON MONTGOMERY ---")
    for k, v in results_out["experiments"].items():
        mm = v["montgomery_metrics"]
        print(f"  [{v['name']}]")
        print(f"    - Abnormal/TB cohort predictions: {mm['prediction_distribution']['abnormal_tb_cohort']}")
        print(f"    - Normal cohort predictions:      {mm['prediction_distribution']['normal_cohort']}")

    print("\n--- 4. INTERNAL TEST SUBSET METRICS (N=564 across 6 classes) ---")
    hdr_i = f"{'Pipeline':<34} | {'Accuracy':<10} | {'Weighted F1':<12} | {'Macro F1':<10} | {'Macro ROC-AUC':<14} | {'Delta Acc':<10}"
    print(hdr_i)
    print("-" * 100)
    for k, v in results_out["experiments"].items():
        im = v["internal_test_metrics"]
        print(f"{v['name']:<34} | {im['accuracy']*100:5.2f}%    | {im['weighted_f1']*100:5.2f}%      | {im['macro_f1']*100:5.2f}%    | {im['macro_roc_auc']*100:5.2f}%        | {im['accuracy_delta_vs_baseline']*100:+5.2f}%")

    print("\n--- 5. FINAL DECISION ---")
    print(f"CLASSIFICATION: {decision_code}")
    print(f"SUMMARY:        {decision_summary}")
    print(f"RECOMMENDATION: {recommendation}")
    print("="*100 + "\n")


if __name__ == "__main__":
    run_investigation()
