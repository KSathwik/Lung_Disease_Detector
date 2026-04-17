# 🫁 LungAI — Lung Disease Detection System
### Academic AI Project | FastAPI + React + TensorFlow

> **Disclaimer:** This is an academic prototype designed to assist medical professionals — not replace them. All predictions must be verified by a qualified physician.

---

## 🏗️ Project Architecture

```
lung_disease_detector/
├── backend/                        # FastAPI Python backend
│   ├── main.py                     # App entry point
│   ├── requirements.txt
│   ├── .env.example
│   ├── api/
│   │   └── routes/
│   │       ├── predictions.py      # POST /predict, GET /predictions
│   │       ├── patients.py         # Patient CRUD
│   │       ├── reports.py          # Report generation
│   │       └── health.py           # Health check
│   ├── ml/
│   │   ├── preprocessing.py        # Data cleaning & normalization
│   │   ├── models.py               # CNN + ResNet50 architectures
│   │   ├── inference.py            # Inference engine
│   │   └── train.py                # Training orchestrator
│   ├── database/
│   │   └── connection.py           # SQLAlchemy ORM models & DB init
│   └── utils/
│       └── logger.py
├── frontend/                       # React + Node.js frontend
│   ├── package.json
│   ├── public/index.html
│   └── src/
│       ├── App.js                  # Router & sidebar layout
│       ├── App.css                 # Full stylesheet
│       ├── index.js
│       ├── services/api.js         # Axios API layer
│       └── pages/
│           ├── AnalyzePage.js      # Main upload & prediction UI
│           ├── MetricsPage.js      # CNN vs ResNet charts
│           ├── HistoryPage.js      # Prediction history
│           └── PatientsPage.js     # Patient management
├── data/
│   ├── raw/                        # ← Put your dataset here
│   └── processed/
├── models/                         # Auto-created after training
├── docs/                           # Training curves, confusion matrices
└── README.md
```

---

## 🧠 ML Pipeline

### Two Algorithms Trained & Compared

| | Algorithm 1: Custom CNN | Algorithm 2: ResNet50 |
|---|---|---|
| **Type** | Custom architecture | Transfer learning |
| **Depth** | 4 conv blocks | 50-layer residual network |
| **Pretrained** | No | ImageNet weights |
| **Training** | End-to-end | 2-phase (freeze → fine-tune) |
| **Best for** | Smaller datasets | Larger, diverse datasets |

### Data Pipeline Steps
1. **Scan** — Index all images and labels from directory
2. **Clean** — Remove duplicates, corrupted files, tiny images
3. **Enhance** — CLAHE contrast enhancement (critical for X-rays)
4. **Denoise** — Gaussian blur
5. **Split** — 70% train / 15% val / 15% test (stratified)
6. **Augment** — Flip, rotation ±15°, brightness jitter (training only)
7. **Normalize** — ImageNet mean/std normalization

### Metrics Computed
- Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion matrix per class
- Training/validation loss curves
- Composite score for model selection: `0.4×F1 + 0.3×Accuracy + 0.2×AUC + 0.1×Recall`

### Diseases Detected
`Normal` · `Pneumonia` · `Tuberculosis` · `COVID-19` · `Lung Cancer` · `COPD` · `Pleural Effusion`

---

## 🗄️ Database Schema

```
patients          → patient records (name, age, gender, history)
lung_scans        → uploaded image metadata
predictions       → ML results (both CNN + ResNet per scan)
model_metrics     → training metrics per model version
reports           → generated PDF/text reports
```

---

## ⚡ Setup & Run

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

---

### Step 1 — Clone & Setup Backend

```bash
git clone https://github.com/YOUR_USERNAME/lung-disease-detector.git
cd lung-disease-detector/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env file
cp .env.example .env

# Start API server
python main.py
# → API running at http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
```

---

### Step 2 — Setup Frontend

```bash
cd ../frontend

# Install Node packages
npm install

# Start React dev server
npm start
# → App running at http://localhost:3000
```

---

### Step 3 — Prepare Dataset

Download one of these free datasets and organize into the structure below:

| Dataset | Link | Classes |
|---|---|---|
| NIH ChestX-ray14 | https://nihcc.app.box.com/v/ChestXray-NIHCC | 14 diseases |
| CheXpert | https://stanfordmlgroup.github.io/competitions/chexpert/ | 14 classes |
| COVID-19 Radiography | https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database | 4 classes |
| RSNA Pneumonia | https://www.kaggle.com/c/rsna-pneumonia-detection-challenge | Pneumonia |

**Organize dataset as:**
```
data/raw/
    Normal/
        image001.jpg
        image002.jpg
        ...
    Pneumonia/
        image001.jpg
        ...
    Tuberculosis/
        ...
    COVID-19/
        ...
```

---

### Step 4 — Train Models

```bash
cd backend

# Train both CNN and ResNet (select best automatically)
python ml/train.py --data_dir ../data/raw --epochs 50 --batch_size 32

# Output:
#   models/cnn_model.h5
#   models/resnet_model.h5
#   models/training_results.json
#   docs/training_curves.png
#   docs/confusion_matrix_CNN.png
#   docs/confusion_matrix_ResNet.png
```

---

## 🚀 Push to GitHub

```bash
# 1. Initialize git (if not already done)
git init

# 2. Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
venv/
.env
*.h5

# Node
frontend/node_modules/
frontend/build/

# Data & Models (too large for git — use Git LFS or share separately)
data/raw/
models/*.h5

# OS
.DS_Store
EOF

# 3. Stage and commit
git add .
git commit -m "feat: lung disease detection system - CNN + ResNet50"

# 4. Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/lung-disease-detector.git
git branch -M main
git push -u origin main
```

> **Tip:** Use [Git LFS](https://git-lfs.github.com/) to push large model files (`.h5`):
> ```bash
> git lfs install
> git lfs track "*.h5"
> git add .gitattributes
> ```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/predict` | Upload image → get prediction |
| `GET` | `/api/v1/predictions` | List all predictions |
| `GET` | `/api/v1/predictions/{id}` | Get single prediction |
| `GET` | `/api/v1/model-metrics` | CNN vs ResNet training metrics |
| `POST` | `/api/v1/patients` | Register patient |
| `GET` | `/api/v1/patients` | List patients |
| `POST` | `/api/v1/reports/generate/{id}` | Generate report |

Full interactive docs: **http://localhost:8000/docs**

---

## 🔬 Academic Notes

- **Not a medical device.** Accuracy depends entirely on dataset size and quality.
- For better results, use 5,000+ images per class.
- ResNet50 generally outperforms the custom CNN on larger datasets due to ImageNet pretrained weights.
- For production, consider DICOM support via `pydicom`, and deploy on GPU (NVIDIA T4+).

---

*Built for academic purposes · LungAI v1.0*
