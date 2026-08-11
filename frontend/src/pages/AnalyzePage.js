import React, { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { analyzeLungImage, generateReport, listPatients } from "../services/api";

const URGENCY_CONFIG = {
  routine:   { color: "#0D9488", bg: "#CCFBF1", label: "Routine" },
  urgent:    { color: "#D97706", bg: "#FEF3C7", label: "Urgent" },
  emergency: { color: "#DC2626", bg: "#FEE2E2", label: "Emergency" },
};

// Synthetic radiology preset scan generator
const generatePresetScanFile = (type) => {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");

  // Background dark radiology gray
  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, 512, 512);

  // Soft background gradient for chest cage
  const bgGrad = ctx.createRadialGradient(256, 256, 20, 256, 256, 250);
  bgGrad.addColorStop(0, "#2a3442");
  bgGrad.addColorStop(0.7, "#141a24");
  bgGrad.addColorStop(1, "#07090e");
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, 512, 512);

  // Lungs lucency (left and right lung fields)
  ctx.fillStyle = "#05070a";
  ctx.beginPath(); ctx.ellipse(170, 250, 75, 140, -0.05, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.ellipse(342, 250, 75, 140, 0.05, 0, Math.PI * 2); ctx.fill();

  // Spine & Cardiac silhouette
  ctx.fillStyle = "#3a4659";
  ctx.fillRect(244, 80, 24, 340);
  ctx.beginPath(); ctx.ellipse(225, 290, 55, 65, -0.3, 0, Math.PI * 2); ctx.fill();

  // Rib cage highlights
  ctx.strokeStyle = "#4b596e";
  ctx.lineWidth = 4;
  for (let y = 140; y <= 360; y += 35) {
    ctx.beginPath(); ctx.arc(170, y, 70, Math.PI * 0.9, Math.PI * 1.8); ctx.stroke();
    ctx.beginPath(); ctx.arc(342, y, 70, Math.PI * 1.2, Math.PI * 0.1, true); ctx.stroke();
  }

  // Disease-specific opacities
  if (type === "pneumonia") {
    const opacityGrad = ctx.createRadialGradient(340, 310, 5, 340, 310, 55);
    opacityGrad.addColorStop(0, "rgba(230, 240, 255, 0.85)");
    opacityGrad.addColorStop(0.6, "rgba(180, 195, 215, 0.5)");
    opacityGrad.addColorStop(1, "transparent");
    ctx.fillStyle = opacityGrad;
    ctx.beginPath(); ctx.arc(340, 310, 55, 0, Math.PI * 2); ctx.fill();
  } else if (type === "covid") {
    const ggo1 = ctx.createRadialGradient(160, 240, 5, 160, 240, 45);
    ggo1.addColorStop(0, "rgba(220, 235, 250, 0.75)");
    ggo1.addColorStop(1, "transparent");
    ctx.fillStyle = ggo1;
    ctx.beginPath(); ctx.arc(160, 240, 45, 0, Math.PI * 2); ctx.fill();

    const ggo2 = ctx.createRadialGradient(350, 260, 5, 350, 260, 50);
    ggo2.addColorStop(0, "rgba(220, 235, 250, 0.8)");
    ggo2.addColorStop(1, "transparent");
    ctx.fillStyle = ggo2;
    ctx.beginPath(); ctx.arc(350, 260, 50, 0, Math.PI * 2); ctx.fill();
  } else if (type === "tb") {
    const tbGrad = ctx.createRadialGradient(350, 160, 5, 350, 160, 35);
    tbGrad.addColorStop(0, "rgba(245, 250, 255, 0.9)");
    tbGrad.addColorStop(0.5, "rgba(170, 190, 215, 0.6)");
    tbGrad.addColorStop(1, "transparent");
    ctx.fillStyle = tbGrad;
    ctx.beginPath(); ctx.arc(350, 160, 35, 0, Math.PI * 2); ctx.fill();
  }

  const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
  const arr = dataUrl.split(",");
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) u8arr[n] = bstr.charCodeAt(n);

  return new File([u8arr], `sample_${type}_chest_xray.jpg`, { type: mime });
};

export default function AnalyzePage() {
  const [file,        setFile]        = useState(null);
  const [preview,     setPreview]     = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);
  const [error,       setError]       = useState(null);
  const [reportData,  setReportData]  = useState(null);
  const [showReport,  setShowReport]  = useState(false);
  const [scanType,    setScanType]    = useState("X-Ray");
  const [patients,    setPatients]    = useState([]);
  const [patientId,   setPatientId]   = useState("");
  const [inferTime,   setInferTime]   = useState(null);

  // Viewer controls state
  const [zoomLevel,   setZoomLevel]   = useState(1);
  const [highContrast,setHighContrast]= useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [copied,      setCopied]      = useState(false);

  useEffect(() => {
    listPatients()
      .then(d => setPatients((d.patients || []).filter(p => p.patient_id !== "DEFAULT")))
      .catch(() => {});
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const selectFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(f);
    });
    setResult(null);
    setError(null);
    setReportData(null);
    setShowReport(false);
    setZoomLevel(1);
    setHighContrast(false);
    setShowHeatmap(false);
    setInferTime(null);
  };

  const onDrop = useCallback((accepted) => {
    if (accepted[0]) selectFile(accepted[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"] },
    maxSize: 10 * 1024 * 1024,
    multiple: false,
  });

  const handlePresetSelect = (presetType) => {
    const presetFile = generatePresetScanFile(presetType);
    selectFile(presetFile);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const startTime = performance.now();
    try {
      const data = await analyzeLungImage(file, { scanType, patientId: patientId || undefined });
      const elapsedSec = ((performance.now() - startTime) / 1000).toFixed(2);
      setInferTime(`${elapsedSec}s`);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!result?.prediction_id) return;
    try {
      const r = await generateReport(result.prediction_id);
      setReportData(r);
      setShowReport(true);
    } catch (err) {
      setError("Failed to generate clinical report.");
    }
  };

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null);
    setResult(null); setError(null); setReportData(null); setShowReport(false); setInferTime(null);
  };

  const copyReportText = () => {
    if (reportData?.content) {
      navigator.clipboard.writeText(reportData.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const urgency = result?.final?.urgency || "routine";
  const urg = URGENCY_CONFIG[urgency] || URGENCY_CONFIG.routine;

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Analyze a Lung Image</h1>
        <p className="page-subtitle">Upload a chest X-ray to analyze the image and view the model's classification results.</p>
      </div>

      <div className="analyze-layout">
        {/* ── Selectors Row (Section 3) ─────────────────────────── */}
        <div className="selectors-card">
          <div>
            <label className="field-label">Patient</label>
            <select className="select-input" value={patientId} onChange={e => setPatientId(e.target.value)}>
              <option value="">Unregistered (default)</option>
              {patients.map(p => (
                <option key={p.patient_id} value={p.patient_id}>
                  {p.name} · {p.age} yrs · {p.gender}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="field-label">Scan type</label>
            <select className="select-input" value={scanType} onChange={e => setScanType(e.target.value)}>
              <option>X-Ray</option>
              <option>CT Scan</option>
              <option>MRI</option>
              <option>PET Scan</option>
            </select>
          </div>
        </div>

        <div className="analyze-main-grid">
          {/* ── Left: Primary Upload Focal Point (Section 4) ──────── */}
          <div className="upload-panel">
            {!preview ? (
              <>
                <div {...getRootProps()} className={`drop-zone ${isDragActive ? "drag-active" : ""}`}>
                  <input {...getInputProps()} />
                  <div className="drop-icon-box">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
                    </svg>
                  </div>
                  <p className="drop-text">{isDragActive ? "Drop image file here" : "Drop image here or Browse files"}</p>
                  <p className="drop-sub">JPG · PNG · BMP · TIFF · Maximum 10 MB</p>
                </div>

                {/* 1-Click Sample Presets */}
                <div className="preset-section">
                  <p className="preset-title">Or test with 1-click sample presets</p>
                  <div className="preset-grid">
                    <button className="preset-btn" onClick={() => handlePresetSelect("normal")}>
                      <span className="preset-name">🫁 Normal</span>
                      <span className="preset-desc">Healthy lungs</span>
                    </button>
                    <button className="preset-btn" onClick={() => handlePresetSelect("pneumonia")}>
                      <span className="preset-name">🧫 Pneumonia</span>
                      <span className="preset-desc">Right lobe opacity</span>
                    </button>
                    <button className="preset-btn" onClick={() => handlePresetSelect("covid")}>
                      <span className="preset-name">🦠 COVID-19</span>
                      <span className="preset-desc">Bilateral opacity</span>
                    </button>
                    <button className="preset-btn" onClick={() => handlePresetSelect("tb")}>
                      <span className="preset-name">🧪 Tuberculosis</span>
                      <span className="preset-desc">Apical cavity scan</span>
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div className="preview-panel card">
                {/* Scan Image Viewer */}
                <div className="scan-viewer-wrapper">
                  <img
                    src={preview}
                    alt="Uploaded chest X-ray"
                    className="scan-viewer-img"
                    style={{
                      transform: `scale(${zoomLevel})`,
                      filter: highContrast ? "contrast(180%) brightness(110%)" : "contrast(100%) brightness(100%)",
                    }}
                  />
                  {showHeatmap && <div className="heat-map-overlay" title="AI Saliency Region Highlight" />}
                </div>

                <div className="viewer-toolbar">
                  <div className="viewer-btn-group">
                    <button className="icon-btn" onClick={() => setZoomLevel(z => Math.min(z + 0.3, 2.5))} title="Zoom In">+</button>
                    <button className="icon-btn" onClick={() => setZoomLevel(z => Math.max(z - 0.3, 1))} title="Zoom Out">-</button>
                    <button className="icon-btn" onClick={() => setZoomLevel(1)} title="Reset Zoom">Reset</button>
                  </div>

                  <div className="viewer-btn-group">
                    <button
                      className={`icon-btn ${highContrast ? "active" : ""}`}
                      onClick={() => setHighContrast(h => !h)}
                      title="Toggle High Contrast Windowing"
                    >
                      Contrast
                    </button>
                    <button
                      className={`icon-btn ${showHeatmap ? "active" : ""}`}
                      onClick={() => setShowHeatmap(s => !s)}
                      title="Toggle AI Saliency Heatmap Overlay"
                    >
                      🔥 Heatmap
                    </button>
                  </div>
                </div>

                <p className="file-name">{file?.name}</p>
                <div className="btn-group">
                  <button className="btn-primary" onClick={handleAnalyze} disabled={loading}>
                    {loading ? <><span className="spinner" /> Analyzing…</> : "Analyze Image"}
                  </button>
                  <button className="btn-ghost" onClick={reset}>Change Scan</button>
                </div>
              </div>
            )}

            {error && <div className="alert-error">{error}</div>}
          </div>

          {/* ── Right: Result Experience (Section 5) ──────────────── */}
          <div className="results-panel">
            {!result && !loading && (
              <div className="empty-state card">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--brand-color)" strokeWidth="1.5">
                  <rect x="3" y="3" width="18" height="18" rx="2"/>
                  <path d="M9 12h6M12 9v6"/>
                </svg>
                <p style={{fontWeight:"600"}}>Select a scan or upload a chest X-ray to view classification results.</p>
              </div>
            )}

            {loading && (
              <div className="loading-state card">
                <div className="pulse-ring" />
                <p style={{fontWeight:"600"}}>Running deep learning inference…</p>
                <p className="loading-sub">Evaluating CNN and ResNet50 classification models</p>
              </div>
            )}

            {result && (
              <>
                {/* Prediction Result Hero */}
                <div className="card result-hero">
                  <div className="result-hero-top">
                    <div>
                      <p className="result-label">Predicted Classification</p>
                      <h2 className="result-condition">{result.final.condition}</h2>
                    </div>
                    <span className="urgency-badge" style={{ color: urg.color, background: urg.bg }}>
                      {urg.label}
                    </span>
                  </div>

                  <div className="confidence-row">
                    <div className="confidence-bar-track">
                      <div className="confidence-bar-fill" style={{ width: `${result.final.confidence}%` }} />
                    </div>
                    <span className="confidence-pct">{result.final.confidence.toFixed(1)}%</span>
                  </div>

                  {/* Metadata Row */}
                  <div className="meta-info-row">
                    <div className="meta-item">
                      <span className="meta-label">Selected Model</span>
                      <span className="meta-val">{result.selected_model === "ResNet" ? "ResNet50" : result.selected_model}</span>
                    </div>
                    {inferTime && (
                      <div className="meta-item">
                        <span className="meta-label">Inference Time</span>
                        <span className="meta-val">{inferTime}</span>
                      </div>
                    )}
                    <div className="meta-item">
                      <span className="meta-label">Scan Mode</span>
                      <span className="meta-val">{scanType}</span>
                    </div>
                  </div>

                  {result.final.alternative_conditions?.length > 0 && (
                    <div className="alt-section">
                      <p className="section-label">Consider also</p>
                      <div className="chip-row">
                        {result.final.alternative_conditions.map((c, i) => (
                          <span key={i} className="chip">{c}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Model Comparison */}
                <div className="card">
                  <p className="section-label">Model Ensemble Comparison</p>
                  <div className="model-compare-grid">
                    {[
                      { name: "CNN", data: result.cnn,    selected: result.selected_model === "CNN" },
                      { name: "ResNet50", data: result.resnet, selected: result.selected_model === "ResNet" },
                    ].map(({ name, data, selected }) => (
                      <div key={name} className={`model-card ${selected ? "model-selected" : ""}`}>
                        <div className="model-card-header">
                          <span className="model-name">{name}</span>
                          {selected && <span className="selected-badge">Selected ✓</span>}
                        </div>
                        {data ? (
                          <>
                            <p className="model-condition">{data.condition}</p>
                            <p className="model-conf">{data.confidence.toFixed(1)}% confidence</p>
                            {data.accuracy && <p className="model-acc">Train Accuracy: {(data.accuracy * 100).toFixed(1)}%</p>}
                          </>
                        ) : (
                          <p className="model-na">Model not available</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Key Radiological Findings */}
                {result.final.key_findings?.length > 0 && (
                  <div className="card">
                    <p className="section-label">Key Radiological Findings</p>
                    <div className="findings-grid">
                      {result.final.key_findings.map((f, i) => (
                        <div key={i} className="finding-chip">
                          <span className="finding-label">{f.label}</span>
                          <span className="finding-value">{f.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Precautions */}
                {result.final.precautions?.length > 0 && (
                  <div className="card">
                    <p className="section-label">Recommended Precautions</p>
                    <ul className="precaution-list">
                      {result.final.precautions.map((p, i) => (
                        <li key={i} className="precaution-item">
                          <span className="precaution-dot" />
                          {p}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="action-row">
                  <button className="btn-secondary" onClick={handleGenerateReport}>📄 Generate Report</button>
                  <button className="btn-ghost" onClick={reset}>New Analysis</button>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Clean Subtle Disclaimer Footer (Section 7) */}
        <p className="subtle-disclaimer">
          Academic Research Prototype · Not a medical device · Do not use for clinical decisions
        </p>
      </div>

      {/* Clinical PDF Report Modal */}
      {showReport && reportData && (
        <div className="modal-backdrop" onClick={() => setShowReport(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">📄 Clinical Classification Report</h3>
              <button className="modal-close" onClick={() => setShowReport(false)}>✕</button>
            </div>
            <div className="modal-body">
              <pre className="report-document">{reportData.content}</pre>
            </div>
            <div className="modal-footer">
              <button className="btn-secondary" onClick={() => window.print()}>🖨️ Print / Save PDF</button>
              <button className="btn-secondary" onClick={copyReportText}>{copied ? "Copied! ✓" : "📋 Copy Text"}</button>
              <button className="btn-primary" onClick={() => setShowReport(false)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
