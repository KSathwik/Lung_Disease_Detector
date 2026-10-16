"""
Script to prepare submission assets, generate 300-DPI high-res figures,
build the Word .docx manuscript, and compile LaTeX files.
"""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = r"d:\Sathwik\lung_disease_detector\Lung_Disease_Detector-main"
SUB_DIR = os.path.join(ROOT, "submission_package")
FIGS_DIR = os.path.join(SUB_DIR, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# 1. Copy existing high-res rendered assets
copy_mapping = {
    "Figure_1_System_Workflow.png": os.path.join(ROOT, "docs", "diagrams", "rendered", "system_architecture.png"),
    "Figure_2_Dataset_Pipeline.png": os.path.join(ROOT, "docs", "diagrams", "rendered", "dataset_training_pipeline.png"),
    "Figure_4_Training_Curves.png": os.path.join(ROOT, "experiments", "results", "densenet_v5_training_curves.png"),
    "Figure_6_Confusion_Matrix.png": os.path.join(ROOT, "experiments", "results", "densenet_v5_confusion_matrix.png"),
    "Figure_7_ROC_Curves.png": os.path.join(ROOT, "experiments", "results", "densenet_v5_roc_curves.png"),
    "Figure_8_PR_Curves.png": os.path.join(ROOT, "docs", "figures", "performance", "model_d_pr_curves.png"),
    "Figure_10A_Montgomery_CM.png": os.path.join(ROOT, "experiments", "results", "phase4a_montgomery_confusion_matrix.png"),
    "Figure_10B_Feature_Space.png": os.path.join(ROOT, "experiments", "results", "phase4a_feature_space.png")
}

for dest_name, src_path in copy_mapping.items():
    if os.path.exists(src_path):
        target = os.path.join(FIGS_DIR, dest_name)
        shutil.copyfile(src_path, target)
        print(f"Copied {dest_name}")
    else:
        print(f"Warning: {src_path} not found")

# 2. Generate Figure 3 (Class-by-Source Distribution Matrix Stacked Bar Chart)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

classes = ['COVID-19', 'Normal', 'Pleural Effusion', 'Pneumonia', 'Nodule / Mass', 'Tuberculosis']
# Distribution matrix from Table 1
sources = ['Existing COVID', 'Existing Normal', 'Existing Pneumonia', 'Existing TB', 'JSRT', 'NIH ChestX-ray14', 'TBX11K', 'VinDr-CXR']
data = np.array([
    [1942, 0, 0, 0, 0, 0, 0, 0],         # COVID-19
    [0, 1199, 0, 0, 67, 171, 1199, 0],   # Normal
    [0, 0, 0, 0, 0, 131, 0, 931],        # Effusion
    [0, 0, 1395, 0, 0, 15, 1395, 0],     # Pneumonia
    [0, 0, 0, 0, 78, 140, 0, 536],       # Nodule
    [0, 0, 0, 665, 0, 0, 683, 0]         # TB
])

colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#9467bd']
bottom = np.zeros(len(classes))

for s_idx, source_name in enumerate(sources):
    values = data[:, s_idx]
    ax.barh(classes, values, left=bottom, label=source_name, color=colors[s_idx], edgecolor='white', linewidth=0.5)
    bottom += values

ax.set_xlabel('Number of Frontal Radiographs', fontsize=12, fontweight='bold')
ax.set_title('Figure 3: Class-by-Source Acquisition Matrix (Cramer\'s V = 0.7654)', fontsize=13, fontweight='bold', pad=15)
ax.legend(title='Acquisition Repository', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
fig3_path = os.path.join(FIGS_DIR, "Figure_3_Source_Distribution.png")
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()
print("Generated Figure 3 (300 DPI)")

# 3. Generate Figure 9 (Per-Class Diagnostic Performance Comparison)
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
x = np.arange(len(classes))
width = 0.2

precision = [100.0, 89.22, 47.74, 92.16, 28.30, 91.48]
recall = [82.19, 84.01, 62.50, 69.50, 69.44, 80.10]
specificity = [100.0, 96.60, 92.67, 97.82, 87.00, 98.90]
f1 = [90.23, 86.54, 54.13, 79.25, 40.21, 85.41]

ax.bar(x - 1.5*width, precision, width, label='Precision (%)', color='#2b5c8f')
ax.bar(x - 0.5*width, recall, width, label='Recall / Sens. (%)', color='#41924B')
ax.bar(x + 0.5*width, specificity, width, label='Specificity (%)', color='#e67e22')
ax.bar(x + 1.5*width, f1, width, label='F1-Score (%)', color='#9b59b6')

ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 9: Per-Class Diagnostic Performance Profile on Internal Test Set (N=1,570)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=15, ha='right', fontsize=10, fontweight='bold')
ax.set_ylim(0, 110)
ax.axhline(76.18, color='gray', linestyle='--', linewidth=1, label='Overall Accuracy (76.18%)')
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
fig9_path = os.path.join(FIGS_DIR, "Figure_9_Per_Class_Performance.png")
plt.savefig(fig9_path, dpi=300, bbox_inches='tight')
plt.close()
print("Generated Figure 9 (300 DPI)")

# 4. Generate Figure 11 (V3 vs V5 Performance Delta Comparison)
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
metrics_delta = [
    'Overall Accuracy',
    'Macro Precision',
    'Macro Recall',
    'Macro F1-Score',
    'COVID-19 F1',
    'Normal F1',
    'Effusion F1',
    'Pneumonia F1',
    'Nodule F1',
    'TB F1'
]
deltas = [-2.35, +0.56, -0.65, -0.65, -1.62, -3.46, -4.11, -0.59, +3.98, +1.86]
bar_colors = ['#27ae60' if d >= 0 else '#c0392b' for d in deltas]

y_pos = np.arange(len(metrics_delta))
bars = ax.barh(y_pos, deltas, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(metrics_delta, fontsize=10, fontweight='bold')
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Metric Delta (%) between V3 and V5', fontsize=12, fontweight='bold')
ax.set_title('Figure 11: Performance Delta (V5 vs V3 Benchmark)', fontsize=13, fontweight='bold', pad=15)

for bar in bars:
    w = bar.get_width()
    offset = 0.2 if w >= 0 else -0.6
    ax.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:+.2f}%",
            va='center', ha='left' if w >= 0 else 'right', fontsize=9, fontweight='bold')

plt.tight_layout()
fig11_path = os.path.join(FIGS_DIR, "Figure_11_V3_vs_V5_Delta.png")
plt.savefig(fig11_path, dpi=300, bbox_inches='tight')
plt.close()
print("Generated Figure 11 (300 DPI)")
