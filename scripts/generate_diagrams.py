"""
LungAI Architecture & UML Diagram Generator
Generates current, publication-grade diagrams representing Model D,
FastAPI, React 18, SQLite/PostgreSQL, and 6-class lung disease classification.

Outputs:
  - docs/diagrams/source/*.mmd (Mermaid source)
  - docs/diagrams/rendered/*.png (High-resolution rendered PNGs)
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT_DIR / "docs" / "diagrams" / "source"
RENDERED_DIR = ROOT_DIR / "docs" / "diagrams" / "rendered"

SOURCE_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_DIR.mkdir(parents=True, exist_ok=True)


# ─── 1. MERMAID SOURCE FILES ────────────────────────────────────────────────

MERMAID_SYSTEM_ARCH = """graph TD
    subgraph Client_Presentation_Tier ["Client Presentation Tier (React 18 SPA)"]
        UI_Analyze["Analyze Workspace<br/>• CXR Drag & Drop / Presets<br/>• Interactive Saliency & Heatmap View<br/>• Clinical Triage Alerts"]
        UI_Metrics["Model Performance Dashboard<br/>• CNN Baseline vs ResNet50 vs Model D<br/>• Accuracy, F1, AUC Radar & Bar Charts"]
        UI_History["History & Audit Log<br/>• Search, Severity Filter (Routine/Urgent/Emerg)<br/>• Scan Metadata & Diagnostic Details"]
        UI_Patients["Patient Management Portal<br/>• Demographic Records<br/>• Scan Association & EMR Integration"]
    end

    subgraph API_Gateway_Tier ["API Gateway & Application Server (FastAPI + Uvicorn)"]
        Router_Health["/api/v1/health<br/>• System Status & Version"]
        Router_Predict["/api/v1/predict<br/>• Multipart CXR Ingestion<br/>• MIME & Size Verification (&le;10MB)"]
        Router_History["/api/v1/predictions<br/>• Historical Record Retrieval"]
        Router_Metrics["/api/v1/model-metrics<br/>• Benchmark Cache & Stats"]
        Router_Patients["/api/v1/patients<br/>• Patient Record CRUD"]
        Router_Reports["/api/v1/reports/generate/{id}<br/>• Structured PDF/JSON Report"]
    end

    subgraph ML_Inference_Tier ["ML Decision Support Pipeline (Model D Singleton)"]
        Validation["Image Integrity & Decoupling<br/>Grayscale/RGBA &rarr; 3-Channel RGB"]
        FreqPreproc["Model D Frequency Preprocessing<br/>Gaussian Low-Pass Filter (&sigma; = 1.0)<br/>Resize 224&times;224 &times; 3 &bull; Scale [0, 1]"]
        DenseNet121["DenseNet-121 Deep Backbone<br/>4 Dense Blocks (6, 12, 24, 16 layers)<br/>Target Layer: conv5_block16_concat"]
        Classifier["Dense Classification Head<br/>Global Average Pooling &bull; Dropout (0.3)<br/>6-Class Softmax Probability Distribution"]
        DecisionSupport["Clinical Decision & Triage Engine<br/>• Severity Urgency: Routine / Urgent / Emergency<br/>• Differential Diagnosis Ranking<br/>• Key Findings & Clinical Precautions"]
        GradCAM["Explainability Engine (Grad-CAM)<br/>Feature Map Gradient Backpropagation<br/>Jet Heatmap Alpha-Blending onto Original CXR"]
    end

    subgraph Data_Persistence_Tier ["Persistence & Storage Layer (SQLAlchemy ORM)"]
        DB_Patient["Patient Entity<br/>ID, Demographics, History"]
        DB_Scan["LungScan Entity<br/>File Hash, Mime, Path"]
        DB_Prediction["Prediction Entity<br/>Class, Probabilities, Findings"]
        Storage_Disk["Uploads Volume<br/>Raw Radiographs & Scans"]
    end

    UI_Analyze --> Router_Predict
    UI_Metrics --> Router_Metrics
    UI_History --> Router_History
    UI_Patients --> Router_Patients

    Router_Predict --> Validation
    Validation --> FreqPreproc
    FreqPreproc --> DenseNet121
    DenseNet121 --> Classifier
    Classifier --> DecisionSupport
    DenseNet121 --> GradCAM
    GradCAM --> DecisionSupport

    DecisionSupport --> DB_Prediction
    Router_Predict --> DB_Scan
    Router_Patients --> DB_Patient
    Router_Predict --> Storage_Disk

    DecisionSupport -.->|Structured JSON + Base64 Heatmap| UI_Analyze
"""

MERMAID_ML_PIPELINE = """flowchart TD
    RawCXR["Input Chest Radiograph<br/>(JPEG / PNG / BMP / TIFF / WebP &le; 10MB)"]
    
    subgraph Preprocessing_Stage ["Model D Frequency-Domain Preprocessing Pipeline"]
        Decode["Format & Channel Normalization<br/>Ensure 3-Channel RGB (H &times; W &times; 3)"]
        Blur["Initial Gaussian Blur (3&times;3, &sigma;=0.8)<br/>High-Frequency Spike Damping"]
        CLAHE["CIE LAB Conversion & CLAHE<br/>Clip Limit 2.0 &bull; Tile Grid 8&times;8 on L Channel"]
        LPF["Spatial Gaussian Low-Pass Filter (&sigma;=1.0)<br/>Attenuates Scanner Fingerprints & Textures"]
        Resize["ImageNet Geometry Standardization<br/>Lanczos-4 Interpolation &rarr; 224 &times; 224 &times; 3"]
        Norm["ImageNet Z-Score Normalization<br/>(Pixel/255.0 - &mu;) / &sigma; &bull; float32"]
    end

    subgraph Deep_Inference_Stage ["DenseNet-121 Deep Neural Network (Frozen Model D Checkpoint)"]
        ConvInit["Initial 7&times;7 Conv (stride 2) + 3&times;3 MaxPool<br/>64 Feature Maps"]
        DB1["Dense Block 1 (6 layers) + Transition Layer 1"]
        DB2["Dense Block 2 (12 layers) + Transition Layer 2"]
        DB3["Dense Block 3 (24 layers) + Transition Layer 3"]
        DB4["Dense Block 4 (16 layers)<br/>Target Feature Map: conv5_block16_concat / relu"]
        Head["Global Average Pooling (1024-D) &bull; BatchNorm<br/>Dense Bottleneck (256-D, ReLU) &bull; Dropout (0.3)"]
        Softmax["Dense Output (6 Classes) + Softmax &bull; &sum; p_i = 1.0"]
    end

    subgraph Decision_Explainability_Stage ["Clinical Decision Support & Explainability"]
        Classes["6 Active Disease Classes:<br/>0: COVID-19 &bull; 1: Normal &bull; 2: Pleural Effusion<br/>3: Pneumonia &bull; 4: Pulmonary Nodule / Mass &bull; 5: Tuberculosis"]
        GradCAMCalc["Grad-CAM Gradient Computation<br/>&alpha;_k = GAP(&part;y^c / &part;A^k)<br/>L^c = ReLU(&sum; &alpha;_k A^k) on relu"]
        Overlay["Saliency Map Generation<br/>Jet Color Map + Original CXR Alpha-Blending"]
        Triage["Urgency Triage Determination<br/>Routine (&le;60% or Normal)<br/>Urgent (Effusion/Nodule/TB)<br/>Emergency (COVID-19 / Severe Pneumonia)"]
        Findings["Radiological Findings & Advisory<br/>Differential Candidates &bull; Precautions &bull; Disclaimer"]
    end

    RawCXR --> Decode
    Decode --> Blur
    Blur --> CLAHE
    CLAHE --> LPF
    LPF --> Resize
    Resize --> Norm
    Norm --> ConvInit
    ConvInit --> DB1 --> DB2 --> DB3 --> DB4 --> Head --> Softmax
    Softmax --> Classes
    DB4 --> GradCAMCalc
    Softmax --> GradCAMCalc
    GradCAMCalc --> Overlay
    Classes --> Triage
    Triage --> Findings
"""

MERMAID_SEQUENCE = """sequenceDiagram
    autonumber
    actor Clinician as Clinician / Radiologist
    participant React as React 18 Frontend
    participant API as FastAPI Backend (Port 8000)
    participant Engine as InferenceEngine (Model D)
    participant TF as TensorFlow 2.16 DenseNet-121
    participant DB as SQLite DB (AsyncSession)

    Clinician->>React: Drop/Select Chest X-ray & click "Run Diagnostic Analysis"
    React->>React: Validate MIME type & file size (&le; 10 MB)
    React->>API: POST /api/v1/predict (multipart/form-data: file, patient_id, scan_type)
    API->>DB: INSERT INTO lung_scans (scan_id, file_path, patient_id, created_at)
    API->>Engine: predict_image(image_bytes)
    Engine->>Engine: Preprocess image (Gaussian LP &sigma;=1.0, resize 224x224x3, norm)
    Engine->>TF: Forward pass tensor through DenseNet-121
    TF-->>Engine: 6-Class Logits & Softmax probabilities
    Engine->>Engine: Compute Grad-CAM on conv5_block16_concat w.r.t top class
    Engine->>Engine: Render saliency heatmap & encode Base64 JPEG
    Engine->>Engine: Formulate Urgency Triage, Differential Diagnosis & Key Findings
    Engine-->>API: Diagnostic Result Dict (class, conf, probs, gradcam, triage)
    API->>DB: INSERT INTO predictions (prediction_id, scan_id, condition, confidence, probs)
    API-->>React: 200 OK: PredictionResponse JSON (includes Base64 Grad-CAM)
    React->>React: Render Diagnostic Card, Probability Bars, Urgency Badge
    Clinician->>React: Toggle "Grad-CAM Heatmap" & "Model Explanation"
    React-->>Clinician: Display anatomical attention overlay & differential triage
"""

MERMAID_DEPLOYMENT = """graph TD
    subgraph Client_Environment ["Client Tier"]
        Browser["Modern Web Browser<br/>(Chrome / Edge / Firefox)<br/>Port 3000"]
    end

    subgraph Network_Gateway ["Network / Reverse Proxy Tier"]
        Nginx["Reverse Proxy / TLS Termination<br/>Nginx / Docker Gateway<br/>Port 80 / 443"]
    end

    subgraph Application_Environment ["Application Tier (Docker / Host OS)"]
        Uvicorn["Uvicorn ASGI Server<br/>FastAPI Runtime (Python 3.11)<br/>Port 8000"]
        AsyncWorkers["Async Request Handlers<br/>CORS & Security Headers Middleware"]
    end

    subgraph Deep_Learning_Environment ["ML Inference Tier"]
        TFRuntime["TensorFlow 2.16.1 C++ / OneDNN Runtime<br/>Optimized CPU Vectorization (AVX2 / AVX-512)"]
        ModelD["Frozen Model D Checkpoint<br/>densenet121_frequency_v5.h5 (28.4 MB)"]
    end

    subgraph Persistence_Environment ["Persistence Tier"]
        SQLiteDB[("SQLite Database<br/>lung_disease.db<br/>(Optional: PostgreSQL)")]
        UploadsVolume[("File System Storage<br/>/uploads directory<br/>Scan Images")]
    end

    Browser -->|HTTP / HTTPS| Nginx
    Nginx -->|Proxy Pass localhost:3000| Browser
    Nginx -->|Proxy Pass localhost:8000| Uvicorn
    Uvicorn --> AsyncWorkers
    AsyncWorkers --> TFRuntime
    TFRuntime --> ModelD
    AsyncWorkers --> SQLiteDB
    AsyncWorkers --> UploadsVolume
"""

MERMAID_DATABASE_ER = """erDiagram
    PATIENTS ||--o{ LUNG_SCANS : "undergoes"
    LUNG_SCANS ||--o| PREDICTIONS : "generates"

    PATIENTS {
        int id PK "Auto-increment"
        string patient_id UK "Unique Identifier"
        string name "Patient Full Name"
        int age "Age in Years"
        string gender "M / F / Other"
        string contact "Phone Number"
        string email "Email Address"
        text medical_history "Clinical Notes"
        datetime created_at "Registration Timestamp"
        datetime updated_at "Update Timestamp"
    }

    LUNG_SCANS {
        int id PK "Auto-increment"
        string scan_id UK "Unique Scan UUID"
        string patient_id FK "References PATIENTS(patient_id)"
        string file_path "Relative Server Storage Path"
        string file_name "Original Filename"
        int file_size "Bytes"
        string mime_type "image/jpeg, image/png, etc."
        string scan_type "X-Ray / CT"
        datetime created_at "Upload Timestamp"
    }

    PREDICTIONS {
        int id PK "Auto-increment"
        string prediction_id UK "Unique Prediction UUID"
        string scan_id FK "References LUNG_SCANS(scan_id)"
        string selected_model "DenseNet-121 Frequency V5"
        string condition "Predicted Primary Class"
        float confidence "Probability (0.0 - 1.0)"
        string urgency "routine / urgent / emergency"
        json probabilities "6-Class Softmax Distribution"
        json key_findings "Radiological Signs"
        json precautions "Clinical Recommendations"
        datetime created_at "Inference Timestamp"
    }
"""

MERMAID_DATASET_TRAINING = """flowchart TD
    subgraph Data_Sources ["Multi-Source Benchmark Acquisition (10,547 Clinical Images)"]
        S1["TBX11K Benchmark (3,277)<br/>Tuberculosis &bull; Normal"]
        S2["Existing COVID-19 Cohort (1,942)<br/>COVID-19 Radiographs"]
        S3["VinBigData VinDr-CXR (1,467)<br/>Pleural Effusion &bull; Pulmonary Nodule / Mass"]
        S4["Existing Pneumonia Cohort (1,395)<br/>Bacterial &bull; Viral Pneumonia"]
        S5["Existing Normal (1,199) &bull; Existing TB (665)<br/>Standard CXR Controls & Mycobacterial Scans"]
        S6["NIH ChestX-ray14 (457) &bull; JSRT (145)<br/>Effusion &bull; Nodule/Mass Ground Truth"]
    end

    subgraph Unified_Manifest_V5 ["V5 Dataset Harmonization & Quality Control"]
        Harmonize["6-Class Target Mapping<br/>COVID-19 &bull; Normal &bull; Pleural Effusion<br/>Pneumonia &bull; Pulmonary Nodule/Mass &bull; TB"]
        Dedup["Exact & Perceptual Deduplication<br/>MD5 Hashing &bull; 64-bit dHash Analysis"]
        Split["Patient-Strict Stratified Holdout Split (10,270 Patients)<br/>Train: 70.14% (7,398) &bull; Val: 14.97% (1,579) &bull; Test: 14.89% (1,570)"]
    end

    subgraph Frequency_Preprocessing ["Domain Shift Mitigation (Model D Protocol)"]
        LPFilter["Spatial Gaussian Low-Pass Filter (&sigma; = 1.0)<br/>Suppresses High-Frequency Scanner Fingerprints"]
        Augment["Training Data Augmentation<br/>Horizontal Flip &bull; Rotation (&plusmn;10&deg;) &bull; Zoom (&plusmn;10%)"]
    end

    subgraph Model_D_Optimization ["DenseNet-121 Architecture & Optimization"]
        Init["ImageNet Pretrained Weights Transfer"]
        Training["Two-Phase Fine-Tuning Schedule<br/>Phase 1: Head Only &bull; Phase 2: Unfreeze Dense Blocks 3-4<br/>Adam Optimizer (lr=1e-4) &bull; Categorical Cross-Entropy"]
    end

    subgraph Scientific_Evaluation ["Verification & Domain Shift Assessment"]
        TestMetrics["Held-Out Test Set Performance (1,570 Images)<br/>Accuracy: 82.93% &bull; Macro F1: 78.35%<br/>Macro ROC-AUC: 0.9755 &bull; Macro PR-AUC: 0.8391"]
        ExtAudit["Quarantined Montgomery External Audit (138 Scans)<br/>Sensor-Shift Vulnerability Disclosed<br/>0% TB Recall on Film Scans &bull; 100% Binary Abnormal Sensitivity"]
    end

    S1 & S2 & S3 & S4 & S5 & S6 --> Harmonize
    Harmonize --> Dedup
    Dedup --> Split
    Split --> LPFilter
    LPFilter --> Augment
    Augment --> Init
    Init --> Training
    Training --> TestMetrics
    Training --> ExtAudit
"""


def save_source_files():
    files = {
        "system_architecture.mmd": MERMAID_SYSTEM_ARCH,
        "ml_inference_pipeline.mmd": MERMAID_ML_PIPELINE,
        "application_sequence.mmd": MERMAID_SEQUENCE,
        "deployment_diagram.mmd": MERMAID_DEPLOYMENT,
        "database_er.mmd": MERMAID_DATABASE_ER,
        "dataset_training_pipeline.mmd": MERMAID_DATASET_TRAINING,
    }
    for filename, content in files.items():
        p = SOURCE_DIR / filename
        with open(p, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"Saved Mermaid source: {p}")


# ─── 2. HIGH-RES RENDERED DIAGRAMS (Matplotlib) ──────────────────────────────

def render_system_architecture():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(8, 9.6, "LungAI — Current System Architecture (Model D)",
            fontsize=18, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(8, 9.25, "Decoupled 4-Tier Enterprise Architecture · React 18 · FastAPI · DenseNet-121 Frequency V5 · SQLite/Postgres",
            fontsize=11, ha="center", va="center", color="#475569")

    # Box styles
    def draw_box(x, y, w, h, title, subtitle, items, bg_color, border_color, title_color="#0F172A"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      facecolor=bg_color, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, title, fontsize=11, fontweight="bold", ha="center", va="top", color=title_color)
        if subtitle:
            ax.text(x + w/2, y + h - 0.7, subtitle, fontsize=8.5, fontstyle="italic", ha="center", va="top", color="#64748B")
        line_y = y + h - (1.1 if subtitle else 0.75)
        for it in items:
            ax.text(x + 0.25, line_y, f"• {it}", fontsize=8.5, ha="left", va="top", color="#334155")
            line_y -= 0.32

    # Tier 1: Client Presentation
    draw_box(0.5, 5.2, 4.4, 3.6, "Client Presentation Tier", "React 18 SPA (Port 3000)", [
        "Analyze Workspace: CXR Upload & Presets",
        "Interactive Grad-CAM Saliency Overlay",
        "Clinical Urgency Triage & Precautions",
        "Model Performance Dashboard (Charts)",
        "Audit History & Search / Filter Logs",
        "Patient Management & EMR Records",
        "Light / Dark Theme Support"
    ], "#F8FAFC", "#0EA5E9", "#0369A1")

    # Tier 2: API Gateway
    draw_box(5.8, 5.2, 4.4, 3.6, "API Gateway / Server Tier", "FastAPI + Uvicorn (Port 8000)", [
        "/api/v1/predict (Multipart Form)",
        "/api/v1/health (System Status)",
        "/api/v1/predictions (Audit Logs)",
        "/api/v1/model-metrics (Benchmarks)",
        "/api/v1/patients (CRUD EMR)",
        "Security Headers & CORS Middleware",
        "MIME & Size Verification (<=10 MB)"
    ], "#F8FAFC", "#10B981", "#047857")

    # Tier 3: ML Inference (Model D)
    draw_box(11.1, 5.2, 4.4, 3.6, "ML Inference Engine", "DenseNet-121 Frequency V5", [
        "Singleton Engine Lifecycle",
        "Gaussian LP Filter (sigma = 1.0)",
        "Input Resizing (224 x 224 x 3)",
        "DenseNet-121 Deep Backbone",
        "6-Class Softmax Probability Vector",
        "Grad-CAM on conv5_block16_concat",
        "Decision Triage & Precautions Engine"
    ], "#F8FAFC", "#8B5CF6", "#6D28D9")

    # Tier 4: Persistence
    draw_box(3.2, 0.8, 4.6, 3.6, "Relational Persistence Tier", "SQLAlchemy Async ORM", [
        "patients Table (Demographics, History)",
        "lung_scans Table (File Hash, Mime)",
        "predictions Table (Class, Conf, Probs)",
        "SQLite Engine (lung_disease.db)",
        "PostgreSQL Compatible via Asyncpg",
        "Audit Log Timestamping (UTC)"
    ], "#F8FAFC", "#F59E0B", "#B45309")

    # Tier 5: File Storage
    draw_box(8.8, 0.8, 4.4, 3.6, "Radiological File Storage", "Local / Managed Volume", [
        "Uploads Directory (/uploads)",
        "Preserves Original CXR Resolution",
        "UUID File Mapping to Database",
        "JPEG / PNG / BMP / TIFF / WebP",
        "Sanitized Filename Handling",
        "Integrity Hashing (MD5 / dHash)"
    ], "#F8FAFC", "#64748B", "#334155")

    # Connectors
    arrowprops = dict(arrowstyle="->", lw=2, color="#0284C7")
    ax.annotate("", xy=(5.8, 7.0), xytext=(4.9, 7.0), arrowprops=arrowprops)
    ax.annotate("", xy=(11.1, 7.0), xytext=(10.2, 7.0), arrowprops=dict(arrowstyle="->", lw=2, color="#059669"))
    
    # Downward connectors
    ax.annotate("", xy=(5.5, 4.4), xytext=(7.5, 5.2), arrowprops=dict(arrowstyle="->", lw=1.8, color="#D97706"))
    ax.annotate("", xy=(10.5, 4.4), xytext=(8.5, 5.2), arrowprops=dict(arrowstyle="->", lw=1.8, color="#475569"))

    # Return arrow from ML to UI
    ax.annotate("Result JSON + Grad-CAM", xy=(4.9, 6.2), xytext=(11.1, 6.2),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#7C3AED", linestyle="dashed"),
                fontsize=8.5, fontweight="bold", color="#7C3AED", ha="center", va="bottom")

    plt.tight_layout()
    out_path = RENDERED_DIR / "system_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Rendered: {out_path}")


def render_ml_pipeline():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(7.5, 8.1, "LungAI — Model D ML Inference & Explainability Pipeline",
            fontsize=17, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(7.5, 7.75, "Frequency-Domain Gaussian Low-Pass Preprocessing · DenseNet-121 · 6-Class Softmax · Grad-CAM Saliency",
            fontsize=10.5, ha="center", va="center", color="#475569")

    stages = [
        ("1. Ingestion & Validation", [
            "Input Radiograph (X-Ray)",
            "MIME: JPEG/PNG/BMP/TIFF/WebP",
            "Size <= 10MB verification",
            "Decode to 3-Channel RGB"
        ], "#0284C7", "#E0F2FE"),

        ("2. Model D Preprocessing", [
            "Gaussian Low-Pass (sigma=1.0)",
            "Suppresses scanner shortcuts",
            "Resize to 224 x 224 x 3",
            "Intensity Scale to [0.0, 1.0]"
        ], "#0D9488", "#CCFBF1"),

        ("3. DenseNet-121 Backbone", [
            "4 Dense Blocks (6, 12, 24, 16)",
            "Dense Feature Concatenation",
            "1024-D Bottleneck Representation",
            "Target: conv5_block16_concat"
        ], "#7C3AED", "#EDE9FE"),

        ("4. 6-Class Classification", [
            "Global Average Pooling",
            "Dropout (rate = 0.3)",
            "Dense Softmax (6 logits)",
            "Condition & Confidence Score"
        ], "#D97706", "#FEF3C7"),

        ("5. Decision & Explainability", [
            "Grad-CAM Heatmap Calculation",
            "Urgency Triage (Routine/Urg/Emerg)",
            "Differential Diagnosis Ranking",
            "Clinical Precautions & Disclaimer"
        ], "#DC2626", "#FEE2E2")
    ]

    for idx, (title, items, border_c, bg_c) in enumerate(stages):
        x = 0.5 + idx * 2.9
        y = 2.4
        w = 2.5
        h = 4.6
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                      facecolor=bg_c, edgecolor=border_c, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.35, title, fontsize=10, fontweight="bold", ha="center", va="top", color=border_c)
        
        line_y = y + h - 0.85
        for it in items:
            ax.text(x + 0.18, line_y, f"• {it}", fontsize=8.2, ha="left", va="top", color="#1E293B")
            line_y -= 0.45

        if idx < len(stages) - 1:
            ax.annotate("", xy=(x + w + 0.38, y + h/2), xytext=(x + w + 0.02, y + h/2),
                        arrowprops=dict(arrowstyle="->", lw=2.2, color="#475569"))

    # Bottom summary box: 6 classes
    summary_rect = patches.FancyBboxPatch((0.5, 0.4), 14.0, 1.4, boxstyle="round,pad=0.12",
                                          facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2)
    ax.add_patch(summary_rect)
    ax.text(7.5, 1.5, "Active 6-Class Diagnostic Taxonomy (backend/ml/class_mapping.json)",
            fontsize=10, fontweight="bold", ha="center", va="top", color="#0F172A")
    classes_text = "0: COVID-19 (18.4%)   |   1: Normal (25.0%)   |   2: Pleural Effusion (10.1%)   |   3: Pneumonia (26.6%)   |   4: Nodule / Mass (7.2%)   |   5: TB (12.8%)"
    ax.text(7.5, 1.05, classes_text, fontsize=9.2, ha="center", va="top", color="#334155")
    ax.text(7.5, 0.65, "Performance on Held-Out Test Set (1,570 Scans):  Accuracy: 82.93%  |  Macro F1: 78.35%  |  Macro ROC-AUC: 0.9755  |  Macro PR-AUC: 0.8391",
            fontsize=8.8, fontweight="bold", ha="center", va="top", color="#047857")

    plt.tight_layout()
    out_path = RENDERED_DIR / "ml_inference_pipeline.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Rendered: {out_path}")


def render_application_sequence():
    fig, ax = plt.subplots(figsize=(15, 9.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    ax.text(7.5, 9.1, "LungAI — Application Interaction Sequence Diagram",
            fontsize=17, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(7.5, 8.75, "End-to-End Diagnostic Request-Response Lifecycle & Asynchronous Database Storage",
            fontsize=10.5, ha="center", va="center", color="#475569")

    participants = [
        (1.5, "Clinician / User", "#3B82F6"),
        (4.5, "React 18 Frontend", "#06B6D4"),
        (7.5, "FastAPI Backend", "#10B981"),
        (10.5, "Model D Engine", "#8B5CF6"),
        (13.5, "SQLite / ORM", "#F59E0B")
    ]

    for x, label, color in participants:
        rect = patches.FancyBboxPatch((x - 1.1, 7.8), 2.2, 0.7, boxstyle="round,pad=0.1",
                                      facecolor=color, edgecolor="none")
        ax.add_patch(rect)
        ax.text(x, 8.15, label, fontsize=9.5, fontweight="bold", ha="center", va="center", color="white")
        ax.plot([x, x], [1.0, 7.8], linestyle="dashed", color="#94A3B8", linewidth=1.2)

    messages = [
        (1, 1.5, 4.5, 7.3, "1. Drop CXR & Select Patient", "#1E293B", "->"),
        (2, 4.5, 4.5, 6.7, "2. Validate MIME & Size (<=10MB)", "#64748B", "self"),
        (3, 4.5, 7.5, 6.1, "3. POST /api/v1/predict (multipart)", "#0284C7", "->"),
        (4, 7.5, 13.5, 5.5, "4. INSERT INTO lung_scans (UUID, metadata)", "#D97706", "->"),
        (5, 7.5, 10.5, 4.9, "5. predict_image(image_bytes)", "#059669", "->"),
        (6, 10.5, 10.5, 4.3, "6. Preprocess: Gaussian LP (sigma=1.0) & Norm", "#7C3AED", "self"),
        (7, 10.5, 10.5, 3.7, "7. DenseNet Forward Pass -> 6-Class Softmax", "#7C3AED", "self"),
        (8, 10.5, 10.5, 3.1, "8. Compute Grad-CAM Saliency Heatmap", "#7C3AED", "self"),
        (9, 10.5, 7.5, 2.5, "9. Return Diagnostic Dict & Base64 Heatmap", "#059669", "->"),
        (10, 7.5, 13.5, 1.9, "10. INSERT INTO predictions (Class, Conf, Probs)", "#D97706", "->"),
        (11, 7.5, 4.5, 1.3, "11. 200 OK: PredictionResponse JSON", "#0284C7", "->"),
    ]

    for num, x1, x2, y, text, col, style in messages:
        if style == "self":
            ax.annotate("", xy=(x1 + 0.1, y - 0.25), xytext=(x1 + 0.1, y + 0.15),
                        arrowprops=dict(arrowstyle="->", lw=1.4, color=col, connectionstyle="arc3,rad=-0.4"))
            ax.text(x1 + 0.5, y, text, fontsize=8.2, va="center", color=col)
        else:
            ax.annotate("", xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color=col))
            ax.text((x1 + x2)/2, y + 0.12, text, fontsize=8.2, ha="center", va="bottom", color=col)

    plt.tight_layout()
    out_path = RENDERED_DIR / "application_sequence.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Rendered: {out_path}")


def render_database_er():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7, 7.6, "LungAI — Relational Database Entity-Relationship (ER) Diagram",
            fontsize=17, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(7, 7.25, "SQLAlchemy Async ORM Schema · SQLite Development / PostgreSQL Production",
            fontsize=10.5, ha="center", va="center", color="#475569")

    def draw_entity(x, y, w, h, table_name, pk_fields, fk_fields, normal_fields, header_col):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                      facecolor="#FFFFFF", edgecolor=header_col, linewidth=2)
        ax.add_patch(rect)
        hdr_rect = patches.FancyBboxPatch((x, y + h - 0.7), w, 0.7, boxstyle="round,pad=0.05",
                                         facecolor=header_col, edgecolor="none")
        ax.add_patch(hdr_rect)
        ax.text(x + w/2, y + h - 0.35, table_name, fontsize=11, fontweight="bold", ha="center", va="center", color="white")

        line_y = y + h - 1.05
        for pk in pk_fields:
            ax.text(x + 0.2, line_y, f"[PK] {pk}", fontsize=8.5, fontweight="bold", color="#B45309")
            line_y -= 0.35
        for fk in fk_fields:
            ax.text(x + 0.2, line_y, f"[FK] {fk}", fontsize=8.5, fontstyle="italic", color="#0369A1")
            line_y -= 0.35
        for fld in normal_fields:
            ax.text(x + 0.2, line_y, f"   {fld}", fontsize=8.2, color="#334155")
            line_y -= 0.32

    # Patients table
    draw_entity(0.8, 1.2, 3.8, 5.4, "patients",
                ["id (Integer, PK)", "patient_id (String, UK)"],
                [],
                ["name (String)", "age (Integer)", "gender (String)", "contact (String, opt)",
                 "email (String, opt)", "medical_history (Text, opt)", "created_at (DateTime)", "updated_at (DateTime)"],
                "#0284C7")

    # LungScans table
    draw_entity(5.1, 1.2, 3.8, 5.4, "lung_scans",
                ["id (Integer, PK)", "scan_id (String, UK)"],
                ["patient_id (FK -> patients)"],
                ["file_path (String)", "file_name (String)", "file_size (Integer)",
                 "mime_type (String)", "scan_type (String)", "created_at (DateTime)"],
                "#059669")

    # Predictions table
    draw_entity(9.4, 1.2, 3.8, 5.4, "predictions",
                ["id (Integer, PK)", "prediction_id (String, UK)"],
                ["scan_id (FK -> lung_scans)"],
                ["selected_model (String)", "condition (String)", "confidence (Float)",
                 "urgency (String)", "probabilities (JSON)", "key_findings (JSON)",
                 "precautions (JSON)", "created_at (DateTime)"],
                "#7C3AED")

    # Relationships
    ax.annotate("1 : N\nUndergoes", xy=(5.1, 4.0), xytext=(4.6, 4.0),
                arrowprops=dict(arrowstyle="->", lw=2, color="#0284C7"),
                fontsize=9, fontweight="bold", ha="center", va="center", color="#0284C7")

    ax.annotate("1 : 1\nGenerates", xy=(9.4, 4.0), xytext=(8.9, 4.0),
                arrowprops=dict(arrowstyle="->", lw=2, color="#059669"),
                fontsize=9, fontweight="bold", ha="center", va="center", color="#059669")

    plt.tight_layout()
    out_path = RENDERED_DIR / "database_er.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Rendered: {out_path}")


def render_deployment():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(7.5, 8.1, "LungAI — Physical Deployment Architecture",
            fontsize=17, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(7.5, 7.75, "Containerized Multi-Stage Docker Topology · TLS Reverse Proxy · Uvicorn ASGI · Persistent Volumes",
            fontsize=10.5, ha="center", va="center", color="#475569")

    def draw_node(x, y, w, h, title, items, color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      facecolor="#F8FAFC", edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        hdr = patches.FancyBboxPatch((x, y + h - 0.65), w, 0.65, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="none")
        ax.add_patch(hdr)
        ax.text(x + w/2, y + h - 0.32, title, fontsize=10.5, fontweight="bold", ha="center", va="center", color="white")
        line_y = y + h - 1.0
        for it in items:
            ax.text(x + 0.2, line_y, f"• {it}", fontsize=8.5, color="#1E293B")
            line_y -= 0.36

    draw_node(0.5, 2.5, 3.2, 4.6, "Client Browser Tier", [
        "Clinician Workstation",
        "Chrome / Edge / Firefox",
        "React 18 Single Page App",
        "Dynamic Recharts Canvas",
        "Local Storage (Theme State)",
        "Zero Client Processing"
    ], "#0284C7")

    draw_node(4.2, 2.5, 3.2, 4.6, "Reverse Proxy / Gateway", [
        "Nginx / Docker Gateway",
        "Port 80 / 443 (TLS / SSL)",
        "Static File Serving (/static)",
        "CORS Headers Termination",
        "Proxy Pass /api to Uvicorn",
        "Rate Limiting & Gzip Buffering"
    ], "#059669")

    draw_node(7.9, 2.5, 3.2, 4.6, "FastAPI Backend Container", [
        "Python 3.11 Slim Runtime",
        "Uvicorn ASGI Worker Process",
        "Pydantic V2 Request Validation",
        "SQLAlchemy AsyncIO Pool",
        "Model D Singleton Lifecycle",
        "TensorFlow 2.16.1 OneDNN"
    ], "#7C3AED")

    draw_node(11.6, 2.5, 3.0, 4.6, "Persistence Tier", [
        "SQLite (lung_disease.db)",
        "PostgreSQL Ready (Asyncpg)",
        "Docker Persistent Volume",
        "Uploads Directory (/uploads)",
        "Trained Model Checkpoint",
        "densenet121_frequency_v5.h5"
    ], "#F59E0B")

    # Connectors
    ax.annotate("", xy=(4.2, 4.8), xytext=(3.7, 4.8), arrowprops=dict(arrowstyle="<->", lw=2, color="#0284C7"))
    ax.annotate("", xy=(7.9, 4.8), xytext=(7.4, 4.8), arrowprops=dict(arrowstyle="<->", lw=2, color="#059669"))
    ax.annotate("", xy=(11.6, 4.8), xytext=(11.1, 4.8), arrowprops=dict(arrowstyle="<->", lw=2, color="#7C3AED"))

    plt.tight_layout()
    out_path = RENDERED_DIR / "deployment_diagram.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Rendered: {out_path}")


def render_dataset_training_pipeline():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(7.5, 8.1, "LungAI — Unified V5 Dataset & Training Protocol",
            fontsize=17, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(7.5, 7.75, "Multi-Source Clinical Harmonization · Patient-Level Stratification · Gaussian Low-Pass · Quarantined External Audit",
            fontsize=10.5, ha="center", va="center", color="#475569")

    steps = [
        ("1. Multi-Source Ingestion", [
            "TBX11K Benchmark (3,277)",
            "Existing COVID-19 (1,942)",
            "VinBigData VinDr-CXR (1,467)",
            "Existing Pneumonia (1,395)",
            "Existing Normal/TB (1,864)",
            "NIH & JSRT Scans (602)",
            "Total: 10,547 Clinical Images"
        ], "#0284C7"),

        ("2. V5 Quality & Stratification", [
            "6-Class Harmonization Mapping",
            "MD5 & dHash Deduplication",
            "Patient-Strict Split (10,270 Pts)",
            "Train: 70.14% (7,398 Images)",
            "Val: 14.97% (1,579 Images)",
            "Test: 14.89% (1,570 Images)"
        ], "#059669"),

        ("3. Model D Training", [
            "Gaussian LP Filter (sigma=1.0)",
            "Suppresses Scanner Shortcuts",
            "ImageNet Pretrained Transfer",
            "2-Phase DenseNet-121 Fine-Tune",
            "Adam (1e-4) & Categorical CE",
            "Validation Early Stopping"
        ], "#7C3AED"),

        ("4. Scientific Evaluation", [
            "Held-Out Test Set (1,570 Scans)",
            "Accuracy: 82.93%",
            "Macro F1-Score: 78.35%",
            "Macro ROC-AUC: 0.9755",
            "Macro PR-AUC: 0.8391",
            "Optimal Sensitivity/Specificity"
        ], "#D97706"),

        ("5. External Audit Limitation", [
            "Quarantined Montgomery (138)",
            "Film-Digitized Hardware Shift",
            "0% TB Recall Disclosed",
            "100% Binary Abnormal Sens.",
            "Documents Clinical Boundary",
            "Prevents False Generalization"
        ], "#DC2626")
    ]

    for idx, (title, items, col) in enumerate(steps):
        x = 0.5 + idx * 2.9
        y = 1.2
        w = 2.5
        h = 6.0
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                      facecolor="#F8FAFC", edgecolor=col, linewidth=1.8)
        ax.add_patch(rect)
        hdr = patches.FancyBboxPatch((x, y + h - 0.7), w, 0.7, boxstyle="round,pad=0.05",
                                    facecolor=col, edgecolor="none")
        ax.add_patch(hdr)
        ax.text(x + w/2, y + h - 0.35, title, fontsize=9.5, fontweight="bold", ha="center", va="center", color="white")
        
        line_y = y + h - 1.1
        for it in items:
            ax.text(x + 0.18, line_y, f"• {it}", fontsize=8.2, ha="left", va="top", color="#1E293B")
            line_y -= 0.55

        if idx < len(steps) - 1:
            ax.annotate("", xy=(x + w + 0.38, y + h/2), xytext=(x + w + 0.02, y + h/2),
                        arrowprops=dict(arrowstyle="->", lw=2.2, color="#475569"))

    plt.tight_layout()
    out_path = RENDERED_DIR / "dataset_training_pipeline.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Rendered: {out_path}")


if __name__ == "__main__":
    print("Generating Mermaid source diagrams...")
    save_source_files()
    print("\nRendering high-resolution publication PNG diagrams...")
    render_system_architecture()
    render_ml_pipeline()
    render_application_sequence()
    render_database_er()
    render_deployment()
    render_dataset_training_pipeline()
    print("\nAll diagrams successfully generated and rendered!")
