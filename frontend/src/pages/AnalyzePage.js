import React, { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { analyzeLungImage, generateReport, listPatients } from "../services/api";

const URGENCY_CONFIG = {
  routine:   { color: "#0D9488", bg: "#CCFBF1", label: "Routine" },
  urgent:    { color: "#D97706", bg: "#FEF3C7", label: "Urgent" },
  emergency: { color: "#DC2626", bg: "#FEE2E2", label: "Emergency" },
  review:    { color: "#475569", bg: "#E2E8F0", label: "Needs review" },
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
  const [showExplanation, setShowExplanation] = useState(false);
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
    setShowExplanation(false);
    setInferTime(null);
  };

  const onDrop = useCallback((accepted) => {
    if (accepted[0]) selectFile(accepted[0]);
  }, []);

  const onDropRejected = useCallback((rejections) => {
    const reason = rejections?.[0]?.errors?.[0];
    setError({
      title: reason?.code === "file-too-large" ? "File Too Large" : "File Not Accepted",
      message: reason?.code === "file-too-large"
        ? "The file is larger than the 10 MB limit."
        : "Please choose a single JPG, PNG, BMP, TIFF or WebP chest X-ray image.",
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"] },
    maxSize: 10 * 1024 * 1024,
    multiple: false,
  });

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
      setError({ title: err.title || "Analysis Failed", message: err.message });
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
      setError({ title: "Report Not Generated", message: err.message });
    }
  };

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null);
    setResult(null); setError(null); setReportData(null); setShowReport(false); setShowExplanation(false); setInferTime(null);
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
  const inconclusive = Boolean(result?.inconclusive ?? result?.final?.inconclusive);
  const gradcam = result?.gradcam || null;

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
            {/* Model D was trained on frontal chest X-rays only */}
            <select className="select-input" value={scanType} onChange={e => setScanType(e.target.value)}>
              <option>X-Ray</option>
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
                  <p className="drop-sub">JPG · PNG · BMP · TIFF · WebP · Frontal chest X-ray · Maximum 10 MB</p>
                </div>
              </>
            ) : (
              <div className="preview-panel card">
                {/* Scan Image Viewer */}
                <div className="scan-viewer-wrapper">
                  <img
                    src={showHeatmap && gradcam ? gradcam.overlay_image : preview}
                    alt={showHeatmap && gradcam ? "Grad-CAM overlay" : "Uploaded chest X-ray"}
                    className="scan-viewer-img"
                    style={{
                      transform: `scale(${zoomLevel})`,
                      filter: highContrast ? "contrast(180%) brightness(110%)" : "contrast(100%) brightness(100%)",
                    }}
                  />
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
                      disabled={!gradcam}
                      title={gradcam ? "Toggle Grad-CAM overlay" : "Grad-CAM overlay is available after a successful analysis"}
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

            {error && (
              <div className="card" style={{
                background: "rgba(239, 68, 68, 0.08)",
                border: "1px solid rgba(239, 68, 68, 0.35)",
                borderLeft: "4px solid #EF4444",
                borderRadius: "8px",
                padding: "1rem 1.25rem",
                marginTop: "1rem"
              }}>
                <div style={{ display: "flex", alignItems: "flex-start", gap: "10px" }}>
                  <span style={{ fontSize: "20px", lineHeight: "1" }}>⚠️</span>
                  <div>
                    <p style={{ fontWeight: 700, color: "#DC2626", margin: "0 0 4px 0", fontSize: "14px" }}>
                      {error.title}
                    </p>
                    <p style={{ margin: 0, color: "var(--text-secondary)", fontSize: "13px", lineHeight: "1.4" }}>
                      {error.message}
                    </p>
                  </div>
                </div>
              </div>
            )}
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
                <p className="loading-sub">Evaluating Model D (DenseNet-121 Frequency V5) and computing Grad-CAM</p>
              </div>
            )}

            {result && (
              <>
                {/* Research-Stage Important Medical Disclaimer Banner (Section 9) */}
                <div className="card medical-disclaimer-card" style={{
                  background: "#fffbeb",
                  borderColor: "#fde68a",
                  borderLeft: "4px solid #d97706",
                  marginBottom: "1rem"
                }}>
                  <div style={{ display: "flex", alignItems: "flex-start", gap: "10px" }}>
                    <span style={{ fontSize: "18px" }}>⚠️</span>
                    <div>
                      <p style={{ margin: "0 0 2px 0", fontWeight: "700", color: "#92400e", fontSize: "13px" }}>
                        Research-Stage System Notice
                      </p>
                      <p style={{ margin: 0, fontSize: "12.5px", color: "#78350f", lineHeight: "1.4" }}>
                        Research-stage chest X-ray classification result. This output is not a medical diagnosis and should not replace evaluation by a qualified healthcare professional.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Prediction Result Hero */}
                <div className="card result-hero">
                  <div className="result-hero-top">
                    <div>
                      <p className="result-label">{inconclusive ? "Result" : "Predicted Condition"}</p>
                      <h2 className="result-condition">
                        {inconclusive ? "Inconclusive" : (result.predicted_class || result.final.condition)}
                      </h2>
                      {inconclusive && (
                        <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", margin: "4px 0 0 0" }}>
                          Highest-scoring class: {result.predicted_class || result.final.condition}, below the reporting threshold.
                        </p>
                      )}
                    </div>
                    <span className="urgency-badge" style={{ color: urg.color, background: urg.bg }}>
                      {urg.label}
                    </span>
                  </div>

                  <div className="confidence-row">
                    <div className="confidence-bar-track">
                      <div
                        className="confidence-bar-fill"
                        style={{
                          width: `${(result.confidence !== undefined && result.confidence <= 1 ? result.confidence * 100 : result.final.confidence)}%`
                        }}
                      />
                    </div>
                    <span className="confidence-pct">
                      {((result.confidence !== undefined && result.confidence <= 1 ? result.confidence * 100 : result.final.confidence)).toFixed(1)}%
                    </span>
                  </div>

                  {/* Metadata Row */}
                  <p style={{ fontSize: "11.5px", color: "var(--text-muted)", margin: "4px 0 0 0" }}>
                    Top-class softmax score, not a calibrated probability of disease.
                  </p>

                  <div className="meta-info-row">
                    <div className="meta-item">
                      <span className="meta-label">Selected Model</span>
                      <span className="meta-val" title={result.model_sha256 ? `Checkpoint SHA-256: ${result.model_sha256}` : undefined}>
                        {result.selected_model || "DenseNet-121 Frequency V5"}
                      </span>
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

                {/* Six-Class Probability Distribution (Section 7) */}
                <div className="card">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.85rem" }}>
                    <div>
                      <p className="section-label" style={{ margin: 0 }}>Probability Distribution</p>
                      <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>Six-class diagnostic probability outputs</span>
                    </div>
                    <span style={{
                      fontSize: "11px",
                      fontWeight: "600",
                      padding: "2px 8px",
                      borderRadius: "6px",
                      background: "var(--brand-light-bg)",
                      color: "var(--brand-color)"
                    }}>
                      Model D (DenseNet-121)
                    </span>
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: "9px" }}>
                    {[
                      "Normal",
                      "Pneumonia",
                      "COVID-19",
                      "Tuberculosis",
                      "Pleural Effusion",
                      "Pulmonary Nodule / Mass"
                    ].map(cls => {
                      const probVal = result.probabilities?.[cls] !== undefined
                        ? result.probabilities[cls] * 100
                        : (result.probabilities_pct?.[cls] ?? (result.densenet?.all_probabilities?.[cls] ?? 0));
                      const isPredicted = (result.predicted_class || result.final.condition) === cls;

                      return (
                        <div
                          key={cls}
                          style={{
                            padding: "8px 12px",
                            borderRadius: "8px",
                            background: isPredicted ? "rgba(13, 148, 136, 0.08)" : "var(--bg-primary)",
                            border: isPredicted ? "1.5px solid var(--brand-color)" : "1px solid var(--border-color)",
                            transition: "all 0.2s ease"
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "5px", alignItems: "center" }}>
                            <span style={{
                              fontWeight: isPredicted ? "700" : "500",
                              color: isPredicted ? "var(--brand-color)" : "var(--text-primary)",
                              fontSize: "13.5px",
                              display: "flex",
                              alignItems: "center",
                              gap: "6px"
                            }}>
                              {cls}
                              {isPredicted && (
                                <span style={{
                                  fontSize: "10.5px",
                                  fontWeight: "700",
                                  background: "var(--brand-color)",
                                  color: "#fff",
                                  padding: "1px 6px",
                                  borderRadius: "10px",
                                  textTransform: "uppercase"
                                }}>
                                  {inconclusive ? "Top score" : "Predicted"}
                                </span>
                              )}
                            </span>
                            <span style={{
                              fontWeight: "700",
                              fontSize: "13.5px",
                              color: isPredicted ? "var(--brand-color)" : "var(--text-secondary)"
                            }}>
                              {probVal.toFixed(1)}%
                            </span>
                          </div>
                          <div style={{ width: "100%", height: "7px", background: "rgba(0,0,0,0.06)", borderRadius: "4px", overflow: "hidden" }}>
                            <div style={{
                              width: `${Math.max(probVal, 1.2)}%`,
                              height: "100%",
                              borderRadius: "4px",
                              background: isPredicted ? "var(--brand-color)" : "#64748b",
                              transition: "width 0.4s ease"
                            }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Grad-CAM Model Attention Visualization (Section 8) */}
                <div className="card">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div>
                      <p className="section-label" style={{ margin: 0 }}>Model Attention Visualization</p>
                      <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                        Grad-CAM gradient-weighted class activation mapping
                      </span>
                    </div>
                    <button
                      className="btn-secondary"
                      style={{ padding: "6px 14px", fontSize: "13px" }}
                      onClick={() => setShowExplanation(s => !s)}
                    >
                      {showExplanation ? "Hide Explanation" : "🔍 View Explanation"}
                    </button>
                  </div>

                  {showExplanation && !gradcam && (
                    <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", marginTop: "1rem" }}>
                      No Grad-CAM explanation was produced for this result.
                    </p>
                  )}

                  {showExplanation && gradcam && (
                    <div style={{ marginTop: "1rem" }}>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                        <div style={{ textAlign: "center" }}>
                          <p style={{ fontSize: "12px", fontWeight: "600", color: "var(--text-secondary)", marginBottom: "6px" }}>
                            Filtered Chest X-Ray
                          </p>
                          <img
                            src={gradcam.original_image}
                            alt="Preprocessed X-ray"
                            style={{
                              width: "100%",
                              maxHeight: "220px",
                              objectFit: "contain",
                              borderRadius: "8px",
                              border: "1px solid var(--border-color)",
                              background: "#000"
                            }}
                          />
                        </div>
                        <div style={{ textAlign: "center" }}>
                          <p style={{ fontSize: "12px", fontWeight: "600", color: "var(--brand-color)", marginBottom: "6px" }}>
                            Grad-CAM Visualization
                          </p>
                          <img
                            src={gradcam.overlay_image}
                            alt="Grad-CAM Overlay"
                            style={{
                              width: "100%",
                              maxHeight: "220px",
                              objectFit: "contain",
                              borderRadius: "8px",
                              border: "1px solid var(--border-color)",
                              background: "#000"
                            }}
                          />
                        </div>
                      </div>
                      <p style={{
                        fontSize: "11.5px",
                        color: "var(--text-muted)",
                        marginTop: "10px",
                        background: "var(--bg-primary)",
                        padding: "8px 12px",
                        borderRadius: "6px",
                        lineHeight: 1.4
                      }}>
                        <strong>Model attention visualization:</strong> Grad-CAM highlights image regions that contributed most strongly to the model's output activations. It does not represent verified biological or radiological reasoning.
                      </p>
                    </div>
                  )}
                </div>

                {/* External Generalization Limitation Boundary (Section 14) */}
                <div className="card" style={{ borderLeft: "4px solid #f59e0b", background: "var(--bg-card)" }}>
                  <p className="section-label" style={{ color: "#d97706", marginBottom: "4px" }}>
                    ⚠️ External Generalization Boundary
                  </p>
                  <p style={{ fontSize: "12px", color: "var(--text-secondary)", lineHeight: 1.45, margin: 0 }}>
                    {result.external_generalization_limitation}
                  </p>
                </div>

                {/* Typical features of the predicted class (static text, not detected in this image) */}
                {result.final.key_findings?.length > 0 && (
                  <div className="card">
                    <p className="section-label">Typical Features of {result.final.condition}</p>
                    <p style={{ fontSize: "11.5px", color: "var(--text-muted)", margin: "0 0 8px 0" }}>
                      General radiographic features of this class for reference. They were not detected or measured in this image.
                    </p>
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
                    <p className="section-label">{inconclusive ? "Next Steps" : "General Precautions for This Class"}</p>
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
