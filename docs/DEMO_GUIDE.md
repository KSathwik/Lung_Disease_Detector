# 📹 LungAI — Demo & Presentation Guide

This guide outlines the step-by-step workflow for demonstrating the **Lung Disease Classification & Decision Support System** on GitHub and LinkedIn.

---

## 🎬 Recommended Recording Flow (60–90 Seconds)

### Scene 1: Launch & System Overview (10s)
1. Show terminal window starting backend API:
   ```bash
   python backend/main.py
   ```
2. Show React dashboard opening at `http://localhost:3000`. Highlight the clean dark-themed UI and system status indicators.

### Scene 2: Patient Registration & Upload (20s)
1. Navigate to **Patients** tab and show patient record selection or registration.
2. Navigate to **Analyze** tab.
3. Drag & drop a sample chest radiograph (e.g. Pneumonia or COVID-19 X-ray from `data/raw/`).

### Scene 3: Inference & Triage Dashboard (30s)
1. Click **Analyze Radiograph**.
2. Show the real-time classification output:
   - Primary predicted class & confidence score (e.g. `Pneumonia - 96.4%`)
   - Urgency Level badge (`Urgent` / `Emergency`)
   - Differential diagnosis probability comparison across all 5 classes
   - Key radiographic findings & clinical precautions
   - Visible medical disclaimer banner

### Scene 4: Metrics & Model Selection (20s)
1. Navigate to **Model Metrics** page.
2. Show the empirical comparison chart between **Custom CNN** (78.22% accuracy) and **ResNet50** (95.21% accuracy, 99.39% AUC-ROC).
3. Display confusion matrices and training convergence curves.

---

## 📸 Key Screenshot Checklist for GitHub README
- [x] Dashboard Upload Interface
- [x] Prediction & Urgency Triage Results
- [x] Dual-Model Metric Comparison Charts (`docs/training_curves.png`)
- [x] ResNet50 Test Set Confusion Matrix (`docs/confusion_matrix_ResNet.png`)
