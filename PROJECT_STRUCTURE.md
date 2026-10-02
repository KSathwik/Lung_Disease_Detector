# LUNGAI — PROJECT STRUCTURE & ARCHITECTURE

**System**: LungAI Six-Class Chest X-Ray Diagnostic System  
**Degree / Purpose**: M.Tech Thesis Research & Demonstration Platform  
**Final Production Model**: **Model D (DenseNet-121 Frequency Preprocessing V5)**  
**Internal Accuracy**: **82.93%** | **Macro F1**: **78.35%** | **ROC-AUC**: **0.9755**

---

## 1. Top-Level Repository Overview

```text
LungAI/
├── backend/                  # FastAPI Application, Database & Inference Engine
│   ├── api/routes/           # API Endpoints (predict, patients, reports, health)
│   ├── database/             # SQLite connection & SQLAlchemy ORM models
│   ├── ml/                   # Model loading, Preprocessing & Inference Service
│   │   ├── class_mapping.json# Official 6-class mapping source of truth
│   │   ├── inference.py      # Singleton Model D inference engine & Grad-CAM
│   │   ├── preprocessing.py  # Standardized Gaussian LP sigma=1.0 preprocessing
│   │   └── models.py         # Neural network architectural definitions
│   ├── tests/                # Automated pytest unit and integration test suite
│   ├── main.py               # FastAPI application startup & lifespan
│   └── requirements.txt      # Python backend dependencies
│
├── frontend/                 # React 18 User Interface
│   ├── public/               # Static assets & HTML template
│   ├── src/
│   │   ├── pages/            # AnalyzePage, MetricsPage, PatientsPage, HistoryPage
│   │   ├── services/         # Axios API client bindings
│   │   ├── App.js            # Routing and layout navigation
│   │   └── App.css           # Custom medical-themed styling system
│   └── package.json          # Node dependencies & frontend scripts
│
├── experiments/              # Scientifically Controlled Thesis Experiments (Phases 3A–4E)
│   ├── data/                 # Master unified manifest (unified_manifest_v5.csv)
│   ├── densenet_v5/          # Phase 3B: Frozen ERM Baseline (76.18% Acc)
│   ├── densenet_coral_v5/    # Phase 4B: Deep CORAL Latent Alignment (80.89% Acc)
│   ├── densenet_dann_v5/     # Phase 4C: Domain-Adversarial Neural Network (76.24% Acc)
│   ├── densenet_frequency_v5/# Phase 4D: Frequency-Aware Preprocessing (82.93% Acc) [FINAL MODEL]
│   ├── densenet_hybrid_v5/   # Phase 4E: Hybrid Preprocessing + CORAL (78.22% Acc)
│   ├── results/              # Phase 4A failure analysis & validation results
│   └── README.md             # Master thesis experimental index & comparison table
│
├── models/                   # Application Model Weights Directory
│   ├── densenet121_frequency_v5.h5  # Active Model D weights (28.71 MB)
│   ├── training_results.json        # Performance metrics and evaluation records
│   └── baseline_archive/            # Historical legacy baselines
│
├── data/                     # Multicentric Datasets (Quarantined & Structured)
│   ├── raw/                  # Clinical disease categories (COVID, Normal, Pneumonia, etc.)
│   └── downloads/            # Source cohorts (NIH, VinDr, TBX11K, JSRT, Montgomery)
│
├── docs/                     # Academic Documentation, Thesis Chapters & Reports
│   └── thesis/THESIS.md      # Comprehensive academic thesis documentation
│
└── reports/                  # Generated clinical diagnostic PDF/text outputs
```

---

## 2. Final Production Model Specification (Model D)

* **Checkpoint Location**: `experiments/densenet_frequency_v5/densenet121_frequency_v5.h5`  
  *(Also mirrored at `models/densenet121_frequency_v5.h5`)*
* **Architecture**:
  * DenseNet-121 ImageNet Backbone (121 convolutional layers)
  * Global Average Pooling (GAP)
  * Batch Normalization (BN)
  * Dense (256 units, ReLU activation, bottleneck representation)
  * Dropout (rate = 0.3)
  * Dense (6 units, Softmax activation)
* **Input Dimensions**: $224 \times 224 \times 3$
* **Standardized Preprocessing Pipeline**:
  1. Input validation (format, non-empty, minimum $32 \times 32$ pixels)
  2. BGR $\rightarrow$ RGB conversion
  3. Denoising Gaussian blur (kernel: $3 \times 3$, $\sigma = 0.8$)
  4. CIE LAB color conversion & CLAHE on L-channel (clip limit: 2.0, tile size: $8 \times 8$)
  5. Back to RGB
  6. **Frequency Preprocessing**: Gaussian low-pass spatial filter ($\sigma = 1.0$)
  7. Lanczos-4 resize to $224 \times 224$
  8. ImageNet normalization: Mean $[0.485, 0.456, 0.406]$, Std $[0.229, 0.224, 0.225]$

---

## 3. Official Class Mapping (Source of Truth)

| Class Index | Diagnostic Category | Clinical Presentation |
| :---: | :--- | :--- |
| **0** | `COVID-19` | Bilateral peripheral ground-glass opacities |
| **1** | `Normal` | Healthy lung field lucency without pathological opacities |
| **2** | `Pleural Effusion` | Blunting of costophrenic angles with dependent fluid accumulation |
| **3** | `Pneumonia` | Patchy, lobar, or segmental airspace consolidation |
| **4** | `Pulmonary Nodule / Mass` | Focal, circumscribed, or spiculated pulmonary lesions |
| **5** | `Tuberculosis` | Upper-lobe infiltration, cavitary lesions, or lymphadenopathy |

---

## 4. Master Thesis Experimental Series

| Phase | Paradigm | Model Architecture | Key Finding / Thesis Contribution | Internal Acc | Macro F1 | External TB Recall |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **3A** | V5 Reconstruction | — | Leak-free unified cohort ($N=10,547$, 10,270 patients) | — | — | — |
| **3B** | ERM Baseline | DenseNet-121 | Frozen benchmark; documented external domain collapse | 76.18% | 72.63% | 0.00% (0/58) |
| **4A** | Failure Analysis | Biophysical | Identified film digitizer sensor shift (4× edge variance) | — | — | — |
| **4B** | Deep CORAL | DenseNet-121 | 2nd-order covariance alignment improves multi-source F1 | 80.89% | 76.85% | 1.72% (1/58) |
| **4C** | DANN | DenseNet-121 | Minimax gradient reversal; reveals negative transfer risk | 76.24% | 72.99% | 0.00% (0/58) |
| **4D** | **Frequency LP ($\sigma=1.0$)** | **DenseNet-121** | **All-time peak accuracy; suppresses high-freq sensor noise** | **82.93%** | **78.35%** | **0.00% (0/58)** |
| **4E** | Hybrid Synthesis | DenseNet-121 | Synthesizes input LP + CORAL; peak COVID-19 F1 (98.62%) | 78.22% | 73.95% | 0.00% (0/58) |

---

## 5. Development & Execution Instructions

### Backend Startup:
```bash
# Activate virtual environment
.venv\Scripts\activate

# Launch FastAPI development server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation available at: `http://localhost:8000/docs`

### Frontend Startup:
```bash
cd frontend
npm start
```
Web application interface available at: `http://localhost:3000`

### Automated Test Execution:
```bash
# Run complete test suite (unit, integration, API, Grad-CAM)
.venv\Scripts\python.exe -m pytest backend/tests/ -v
```
