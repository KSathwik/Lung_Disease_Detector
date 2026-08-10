# Baseline Evaluation Summary Report

## 1. Overview & Dataset Split
- **Project**: Lung Disease Detection System
- **Dataset Size**: 10,864 total images across 5 active classes
- **Splits**: 70% Train (7,604 images), 15% Validation (1,630 images), 15% Held-Out Test Set (1,630 images)
- **Random Seed**: 42

## 2. Held-Out Test Set Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Composite Score |
|---|---|---|---|---|---|---|
| **Custom CNN** | 0.7822 | 0.7850 | 0.7822 | 0.7216 | 0.8850 | 0.7919 |
| **ResNet50 (Selected)** | **0.9521** | **0.9518** | **0.9521** | **0.9517** | **0.9939** | **0.9603** |

## 3. ResNet50 Classification Breakdown
- **COVID-19**: Precision 0.96 | Recall 0.98 | F1-Score 0.97
- **Lung Cancer**: Precision 1.00 | Recall 1.00 | F1-Score 1.00
- **Normal**: Precision 0.91 | Recall 0.90 | F1-Score 0.91
- **Pneumonia**: Precision 0.96 | Recall 0.97 | F1-Score 0.96
- **Tuberculosis**: Precision 0.92 | Recall 0.81 | F1-Score 0.86

## 4. Operational Assets
- `reports/baseline_results.json` — Structured JSON payload
- `reports/classification_report.json` — Per-class classification breakdown
- `reports/confusion_matrix.png` — Test set confusion matrix plot
- `reports/training_history.png` — Convergence loss & accuracy curves
