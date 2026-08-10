import React, { useState, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { analyzeLungImage, generateReport, listPatients } from "../services/api";

const URGENCY_CONFIG = {
  routine:   { color: "#0F6E56", bg: "#E1F5EE", label: "Routine" },
  urgent:    { color: "#854F0B", bg: "#FAEEDA", label: "Urgent" },
  emergency: { color: "#A32D2D", bg: "#FCEBEB", label: "Emergency" },
};

export default function AnalyzePage() {
  const [file,        setFile]        = useState(null);
  const [preview,     setPreview]     = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [result,      setResult]      = useState(null);
  const [error,       setError]       = useState(null);
  const [reportMsg,   setReportMsg]   = useState(null);
  const [scanType,    setScanType]    = useState("X-Ray");
  const [patients,    setPatients]    = useState([]);
  const [patientId,   setPatientId]   = useState("");

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

  const onDrop = useCallback((accepted) => {
    const f = accepted[0];
    if (!f) return;
    setFile(f);
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(f);
    });
    setResult(null);
    setError(null);
    setReportMsg(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"] },
    maxSize: 10 * 1024 * 1024,
    multiple: false,
  });

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await analyzeLungImage(file, { scanType, patientId: patientId || undefined });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReport = async () => {
    if (!result?.prediction_id) return;
    try {
      const r = await generateReport(result.prediction_id);
      setReportMsg(`Report generated: ${r.report_id}`);
    } catch (err) {
      setReportMsg("Failed to generate report.");
    }
  };

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(null);
    setResult(null); setError(null); setReportMsg(null);
  };

  const urgency = result?.final?.urgency || "routine";
  const urg = URGENCY_CONFIG[urgency] || URGENCY_CONFIG.routine;

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Lung Image Analysis</h1>
        <p className="page-subtitle">Upload a chest X-ray, CT scan or MRI — the AI will analyze it and suggest possible diagnoses with precautions.</p>
      </div>

      <div className="analyze-grid">
        {/* ── Left: Upload panel ─────────────────────────────────── */}
        <div className="upload-panel">
          <div className="card">
            <label className="field-label">Patient</label>
            <select className="select-input" value={patientId} onChange={e => setPatientId(e.target.value)}>
              <option value="">Unregistered (default)</option>
              {patients.map(p => (
                <option key={p.patient_id} value={p.patient_id}>
                  {p.name} · {p.age} yrs · {p.gender}
                </option>
              ))}
            </select>

            <label className="field-label" style={{marginTop:"1rem"}}>Scan type</label>
            <select className="select-input" value={scanType} onChange={e => setScanType(e.target.value)}>
              <option>X-Ray</option>
              <option>CT Scan</option>
              <option>MRI</option>
              <option>PET Scan</option>
            </select>
          </div>

          {!preview ? (
            <div {...getRootProps()} className={`drop-zone ${isDragActive ? "drag-active" : ""}`}>
              <input {...getInputProps()} />
              <div className="drop-icon">
                <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                  <path d="M20 8v16M13 15l7-7 7 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M8 28c0 2.2 1.8 4 4 4h16c2.2 0 4-1.8 4-4" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                </svg>
              </div>
              <p className="drop-text">{isDragActive ? "Drop image here" : "Click or drag to upload"}</p>
              <p className="drop-sub">JPEG · PNG · BMP · TIFF · Max 10 MB</p>
            </div>
          ) : (
            <div className="preview-panel">
              <img src={preview} alt="Lung scan preview" className="preview-img" />
              <p className="file-name">{file?.name}</p>
              <div className="btn-group">
                <button className="btn-primary" onClick={handleAnalyze} disabled={loading}>
                  {loading ? <><span className="spinner" /> Analyzing…</> : "Analyze image"}
                </button>
                <button className="btn-ghost" onClick={reset}>Remove</button>
              </div>
            </div>
          )}

          {error && <div className="alert-error">{error}</div>}

          <div className="disclaimer-box">
            <strong>Academic prototype</strong> — For educational and decision-support use only.
            Always consult a qualified physician before any clinical decisions.
          </div>
        </div>

        {/* ── Right: Results panel ───────────────────────────────── */}
        <div className="results-panel">
          {!result && !loading && (
            <div className="empty-state">
              <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                <rect width="64" height="64" rx="16" fill="#E1F5EE"/>
                <path d="M32 18c-7.7 0-14 6.3-14 14s6.3 14 14 14 14-6.3 14-14S39.7 18 32 18z" stroke="#1D9E75" strokeWidth="2"/>
                <path d="M28 32h8M32 28v8" stroke="#1D9E75" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              <p>Upload and analyze a lung image to see results here.</p>
            </div>
          )}

          {loading && (
            <div className="loading-state">
              <div className="pulse-ring" />
              <p>Running CNN and ResNet models…</p>
              <p className="loading-sub">This may take 10–30 seconds</p>
            </div>
          )}

          {result && (
            <>
              {/* ── Final Diagnosis Card ─────────────────── */}
              <div className="card result-hero">
                <div className="result-hero-top">
                  <div>
                    <p className="result-label">Primary diagnosis</p>
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

              {/* ── Model Comparison ─────────────────────── */}
              <div className="card">
                <p className="section-label">Model comparison</p>
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
                          {data.accuracy && <p className="model-acc">Train accuracy: {(data.accuracy * 100).toFixed(1)}%</p>}
                        </>
                      ) : (
                        <p className="model-na">Model not available</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* ── Key Findings ──────────────────────────── */}
              {result.final.key_findings?.length > 0 && (
                <div className="card">
                  <p className="section-label">Radiological findings</p>
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

              {/* ── Precautions ───────────────────────────── */}
              {result.final.precautions?.length > 0 && (
                <div className="card">
                  <p className="section-label">Recommended precautions</p>
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

              {/* ── Actions ───────────────────────────────── */}
              <div className="action-row">
                <button className="btn-secondary" onClick={handleReport}>Generate Report</button>
                <button className="btn-ghost" onClick={reset}>New Analysis</button>
              </div>
              {reportMsg && <p className="report-msg">{reportMsg}</p>}

              <p className="disclaimer-text">{result.final.disclaimer}</p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
