"""
Step 1: Programmatic Inspection of Montgomery Images
Analyzes dimensions, aspect ratios, border characteristics, collimator regions, text/lead markers,
and lung field coverage across the entire 138-image Montgomery cohort compared to training/internal images.
"""

import sys
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

mont_csv = Path("data/downloads/montgomery/montgomery_metadata.csv")
mont_dir = Path("data/downloads/montgomery/images/images")
out_dir = Path("experiments/results/cropping_samples")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(mont_csv)

records = []
for idx, row in df.iterrows():
    p = mont_dir / row["study_id"]
    if not p.exists():
        continue
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # 20-pixel border slices
    top_mean = float(img[:20, :].mean())
    bot_mean = float(img[-20:, :].mean())
    left_mean = float(img[:, :20].mean())
    right_mean = float(img[:, -20:].mean())
    
    # 4 corner 20x20 patches
    tl = float(img[:20, :20].mean())
    tr = float(img[:20, -20:].mean())
    bl = float(img[-20:, :20].mean())
    br = float(img[-20:, -20:].mean())
    
    # Center 500x500
    cy, cx = h // 2, w // 2
    center_mean = float(img[cy-250:cy+250, cx-250:cx+250].mean())
    
    # Check for near-black borders (e.g. mean < 15)
    has_black_top = top_mean < 15.0
    has_black_bot = bot_mean < 15.0
    has_black_left = left_mean < 15.0
    has_black_right = right_mean < 15.0
    
    # Otsu threshold to find foreground/body region bounding box
    # Downsample for fast morphological analysis
    scale = 512 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    # Threshold at low value (e.g. > 15) to detect non-black film/body area
    _, binary = cv2.threshold(small, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest bounding box
        largest_c = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest_c)
        body_area_pct = (bw * bh) / (small.shape[0] * small.shape[1]) * 100.0
    else:
        bx, by, bw, bh = 0, 0, small.shape[1], small.shape[0]
        body_area_pct = 100.0

    records.append({
        "study_id": row["study_id"],
        "findings": row["findings"],
        "h": h, "w": w, "aspect_ratio": round(w / h, 3),
        "top_mean": round(top_mean, 1), "bot_mean": round(bot_mean, 1),
        "left_mean": round(left_mean, 1), "right_mean": round(right_mean, 1),
        "tl": round(tl, 1), "tr": round(tr, 1), "bl": round(bl, 1), "br": round(br, 1),
        "center_mean": round(center_mean, 1),
        "has_black_top": has_black_top, "has_black_bot": has_black_bot,
        "has_black_left": has_black_left, "has_black_right": has_black_right,
        "body_area_pct": round(body_area_pct, 1),
        "crop_box_norm": [round(by / small.shape[0], 3), round(bx / small.shape[1], 3),
                          round((by + bh) / small.shape[0], 3), round((bx + bw) / small.shape[1], 3)]
    })

res_df = pd.DataFrame(records)

print(f"Total Montgomery Scans Analyzed: {len(res_df)}")
print(f"Dimensions: {res_df['h'].value_counts().to_dict()} (h) x {res_df['w'].value_counts().to_dict()} (w)")
print(f"Aspect ratios: min={res_df['aspect_ratio'].min()}, max={res_df['aspect_ratio'].max()}, mean={res_df['aspect_ratio'].mean():.3f}")
print("\nBorder Analysis across all 138 scans:")
print(f"  Scans with black top border (<15):    {res_df['has_black_top'].sum()} / {len(res_df)} ({res_df['has_black_top'].mean()*100:.1f}%)")
print(f"  Scans with black bottom border (<15): {res_df['has_black_bot'].sum()} / {len(res_df)} ({res_df['has_black_bot'].mean()*100:.1f}%)")
print(f"  Scans with black left border (<15):   {res_df['has_black_left'].sum()} / {len(res_df)} ({res_df['has_black_left'].mean()*100:.1f}%)")
print(f"  Scans with black right border (<15):  {res_df['has_black_right'].sum()} / {len(res_df)} ({res_df['has_black_right'].mean()*100:.1f}%)")
print(f"  Scans with AT LEAST ONE black edge:   {(res_df['has_black_top'] | res_df['has_black_bot'] | res_df['has_black_left'] | res_df['has_black_right']).sum()} / {len(res_df)} ({(res_df['has_black_top'] | res_df['has_black_bot'] | res_df['has_black_left'] | res_df['has_black_right']).mean()*100:.1f}%)")
print(f"  Scans with ALL FOUR black edges:      {(res_df['has_black_top'] & res_df['has_black_bot'] & res_df['has_black_left'] & res_df['has_black_right']).sum()} / {len(res_df)}")

print("\nCorner Patch Means (0-255 scale):")
print(f"  Top-Left:     mean={res_df['tl'].mean():.1f}, median={res_df['tl'].median():.1f}")
print(f"  Top-Right:    mean={res_df['tr'].mean():.1f}, median={res_df['tr'].median():.1f}")
print(f"  Bottom-Left:  mean={res_df['bl'].mean():.1f}, median={res_df['bl'].median():.1f}")
print(f"  Bottom-Right: mean={res_df['br'].mean():.1f}, median={res_df['br'].median():.1f}")

print("\nBody Area Coverage Percentage (bounding box of non-dark content):")
print(f"  Mean body area coverage: {res_df['body_area_pct'].mean():.1f}% (min={res_df['body_area_pct'].min():.1f}%, max={res_df['body_area_pct'].max():.1f}%)")

# Compare with internal test dataset
manifest = pd.read_csv("experiments/data/unified_manifest.csv")
sample_test = manifest[manifest["split"] == "test"].sample(50, random_state=42)
int_records = []
for p in sample_test["image_path"]:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    h, w = img.shape
    scale = 512 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, binary = cv2.threshold(small, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest_c)
        area_pct = (bw * bh) / (small.shape[0] * small.shape[1]) * 100.0
    else:
        area_pct = 100.0
    int_records.append({
        "h": h, "w": w, "aspect_ratio": w / h,
        "tl": img[:20, :20].mean(), "br": img[-20:, -20:].mean(),
        "area_pct": area_pct
    })
int_df = pd.DataFrame(int_records)
print("\n=== INTERNAL TEST SAMPLES (N=50) COMPARISON ===")
print(f"  Internal Aspect Ratio: mean={int_df['aspect_ratio'].mean():.3f} (min={int_df['aspect_ratio'].min():.3f}, max={int_df['aspect_ratio'].max():.3f})")
print(f"  Internal Top-Left corner mean: {int_df['tl'].mean():.1f} (median={int_df['tl'].median():.1f})")
print(f"  Internal Bottom-Right corner mean: {int_df['br'].mean():.1f} (median={int_df['br'].median():.1f})")
print(f"  Internal Body Coverage: mean={int_df['area_pct'].mean():.1f}%")

# Save inspection report
with open("experiments/results/montgomery_inspection_report.json", "w") as f:
    json.dump({
        "montgomery_scans_analyzed": len(res_df),
        "dimensions": {"4020x4892": int((res_df['h']==4020).sum()), "4892x4020": int((res_df['h']==4892).sum())},
        "aspect_ratios": {"mean": round(float(res_df['aspect_ratio'].mean()), 3), "min": float(res_df['aspect_ratio'].min()), "max": float(res_df['aspect_ratio'].max())},
        "border_black_frequency": {
            "top": int(res_df['has_black_top'].sum()),
            "bottom": int(res_df['has_black_bot'].sum()),
            "left": int(res_df['has_black_left'].sum()),
            "right": int(res_df['has_black_right'].sum()),
            "at_least_one": int((res_df['has_black_top'] | res_df['has_black_bot'] | res_df['has_black_left'] | res_df['has_black_right']).sum()),
            "all_four": int((res_df['has_black_top'] & res_df['has_black_bot'] & res_df['has_black_left'] & res_df['has_black_right']).sum())
        },
        "mean_corner_intensity": {"tl": float(res_df['tl'].mean()), "tr": float(res_df['tr'].mean()), "bl": float(res_df['bl'].mean()), "br": float(res_df['br'].mean())},
        "internal_comparison": {
            "aspect_ratio_mean": round(float(int_df['aspect_ratio'].mean()), 3),
            "tl_corner_mean": round(float(int_df['tl'].mean()), 1),
            "br_corner_mean": round(float(int_df['br'].mean()), 1),
            "area_pct_mean": round(float(int_df['area_pct'].mean()), 1)
        }
    }, f, indent=2)

print("\nSaved inspection report to experiments/results/montgomery_inspection_report.json")
