# 📹 LungAI — Demo & Presentation Guide

This guide outlines the step-by-step workflow for demonstrating the **Lung Disease Classification & Clinical Decision Support System** during presentations and thesis defenses.

---

## 🎬 Recommended Demonstration Flow (60–90 Seconds)

### Scene 1: Launch & System Overview (10s)
1. Show terminal window starting backend API and React frontend:
   ```bash
   uvicorn main:app --app-dir backend --port 8000
   npm start
   ```
2. Open clinician dashboard at `http://localhost:3000`. Highlight the responsive medical UI and system status indicators.

### Scene 2: Patient Registration & Upload (20s)
1. Navigate to **Patients** tab and demonstrate patient selection or new patient registration.
2. Navigate to **Analyze** tab.
3. Drag & drop a sample chest radiograph (or select one of the built-in CXR sample presets).

### Scene 3: Inference & Triage Dashboard (30s)
1. Click **Analyze Radiograph**.
2. Review real-time diagnostic output powered by Model D (DenseNet-121 Frequency V5):
   - Primary predicted class & confidence score (e.g. `COVID-19 — 98.4%`)
   - Urgency Level badge (`Routine` / `Urgent` / `Emergency`)
   - Differential diagnosis probability bars across all 6 classes (COVID-19, Normal, Pleural Effusion, Pneumonia, Pulmonary Nodule / Mass, Tuberculosis)
   - Interactive Grad-CAM visual attention heatmap overlay toggle
   - Key radiographic findings & clinical precautions
   - Research disclaimer and Montgomery domain-shift limitation notice

### Scene 4: Metrics & Model Selection (20s)
1. Navigate to **Model Metrics** page.
2. Show the empirical comparison chart between initial baselines (Custom CNN, ResNet50) and the production champion **Model D (DenseNet-121 Frequency V5)**:
   - Test Accuracy: **82.93%**
   - Macro F1: **78.35%**
   - Macro ROC-AUC: **97.55%**
   - Macro PR-AUC: **83.91%**
3. Inspect the per-class performance breakdown table and hardware generalization limitation note.

---

## 📸 Key Verification Checklist
- [x] Analyze Workspace with CXR Sample Presets
- [x] Diagnostic Prediction Card with Urgency Triage
- [x] Grad-CAM Saliency Overlay
- [x] Model Metrics Dashboard with Model D vs Baselines
- [x] Patient Management & EMR Registration
- [x] Searchable Diagnostic History Log
- [x] Structured Clinical Report Generation
