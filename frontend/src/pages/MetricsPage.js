import React, { useEffect, useState } from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { getModelMetrics } from "../services/api";

const METRICS_KEYS = ["accuracy", "precision", "recall", "f1_score", "auc_roc"];
const METRIC_LABELS = {
  accuracy: "Accuracy", precision: "Macro Precision",
  recall: "Macro Recall", f1_score: "Macro F1-Score", auc_roc: "Macro ROC-AUC"
};

const pct = (v) => (v === undefined || v === null ? "—" : `${(v * 100).toFixed(1)}%`);

export default function MetricsPage() {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    getModelMetrics()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page"><p className="loading-text">Loading model metrics…</p></div>;
  if (error)   return <div className="page"><div className="alert-error">{error}</div></div>;
  if (!data?.densenet) return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Model Performance</h1>
        <p className="page-subtitle">No model evaluation metrics found.</p>
      </div>
      <div className="card">
        <p style={{color:"var(--color-text-secondary)"}}>{data?.hint || "Model metrics file missing or uninitialized:"}</p>
        <code className="code-block">models/training_results.json</code>
      </div>
    </div>
  );

  const densenet = data.densenet;
  const perClass = densenet.per_class || {};
  const perClassKeys = Object.keys(perClass);
  const ext = densenet.external_evaluation;

  const perClassChart = perClassKeys.map(cls => ({
    name: cls.replace("Pulmonary Nodule / Mass", "Nodule / Mass"),
    Precision: +(perClass[cls].precision * 100).toFixed(1),
    Recall: +(perClass[cls].recall * 100).toFixed(1),
    F1: +(perClass[cls].f1_score * 100).toFixed(1),
  }));

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Model Performance & Evaluation</h1>
        <p className="page-subtitle">
          Recorded evaluation of <strong style={{color:"#7C3AED"}}>{densenet.model_name || "DenseNet-121 Frequency V5 (Model D)"}</strong>.
          {data.source ? ` Source: ${data.source}.` : ""}
        </p>
      </div>

      {data.model_loaded === false && (
        <div className="alert-error" style={{ marginBottom: "1rem" }}>
          The Model D checkpoint is not loaded on this server, so these are recorded results only and the Analyze page cannot produce predictions.
        </div>
      )}

      {/* ── Summary Cards ──────────────────────────────────────── */}
      <div className="metrics-grid">
        {METRICS_KEYS.map(k => (
          <div key={k} className="metric-card">
            <p className="metric-label">{METRIC_LABELS[k]}</p>
            <div className="metric-values">
              <div className="metric-val metric-winner" style={{ borderLeft: "3px solid #7C3AED" }}>
                <span className="metric-model-name" style={{ fontWeight: 600, color: "#7C3AED" }}>Model D</span>
                <span className="metric-number">{pct(densenet[k])}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* ── Per-Class Chart ────────────────────────────────────── */}
      {perClassChart.length > 0 && (
        <div className="card chart-card">
          <p className="section-label">Per-Class Precision, Recall and F1 (internal test split)</p>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={perClassChart} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} />
              <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 12 }} />
              <Tooltip formatter={v => `${v}%`} />
              <Legend />
              <Bar dataKey="Precision" fill="#185FA5" radius={[4,4,0,0]} />
              <Bar dataKey="Recall" fill="#1D9E75" radius={[4,4,0,0]} />
              <Bar dataKey="F1" fill="#7C3AED" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Per-Class Breakdown Table (Model D) ──────────────────── */}
      {perClassKeys.length > 0 && (
        <div className="card chart-card">
          <p className="section-label">Model D Internal Test Split Per-Class Metrics (N=1,570)</p>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", marginTop: "8px" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-color)", textAlign: "left", color: "var(--text-secondary)" }}>
                  <th style={{ padding: "8px 12px" }}>Condition</th>
                  <th style={{ padding: "8px 12px" }}>Support (Test)</th>
                  <th style={{ padding: "8px 12px" }}>Precision</th>
                  <th style={{ padding: "8px 12px" }}>Recall / Sensitivity</th>
                  <th style={{ padding: "8px 12px" }}>Specificity</th>
                  <th style={{ padding: "8px 12px" }}>F1-Score</th>
                </tr>
              </thead>
              <tbody>
                {perClassKeys.map(cls => {
                  const m = perClass[cls];
                  return (
                    <tr key={cls} style={{ borderBottom: "1px solid var(--border-color)" }}>
                      <td style={{ padding: "8px 12px", fontWeight: 600, color: "var(--text-primary)" }}>{cls}</td>
                      <td style={{ padding: "8px 12px", color: "var(--text-secondary)" }}>{m.support}</td>
                      <td style={{ padding: "8px 12px", color: "var(--text-secondary)" }}>{pct(m.precision)}</td>
                      <td style={{ padding: "8px 12px", color: "var(--text-secondary)" }}>{pct(m.recall)}</td>
                      <td style={{ padding: "8px 12px", color: "var(--text-secondary)" }}>{pct(m.specificity)}</td>
                      <td style={{ padding: "8px 12px", fontWeight: 600, color: "#7C3AED" }}>{pct(m.f1_score)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── External Evaluation ────────────────────────────────── */}
      {ext && (
        <div className="card chart-card">
          <p className="section-label">External Evaluation: {ext.cohort}</p>
          <div className="metrics-grid" style={{ marginTop: "8px" }}>
            <div className="metric-card"><p className="metric-label">TB classified as Tuberculosis</p><span className="metric-number">{pct(ext.tb_recall)}</span></div>
            <div className="metric-card"><p className="metric-label">Normal classified as Normal (specificity)</p><span className="metric-number">{pct(ext.normal_specificity)}</span></div>
            <div className="metric-card"><p className="metric-label">TB flagged as any abnormal class</p><span className="metric-number">{pct(ext.binary_abnormal_sensitivity)}</span></div>
          </div>
          <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", lineHeight: 1.5, margin: "8px 0 0 0" }}>
            The model labelled every external image abnormal, including all 80 normal scans, so the 100% "abnormal" rate
            does not indicate screening ability.
          </p>
        </div>
      )}

      {/* ── Limitations ────────────────────────────────────────── */}
      <div className="card" style={{ borderLeft: "4px solid #D97706", backgroundColor: "rgba(217, 119, 6, 0.04)" }}>
        <p style={{ fontWeight: 700, color: "#D97706", fontSize: "14px", margin: "0 0 4px 0" }}>
          How to read these numbers
        </p>
        <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", lineHeight: "1.5", margin: 0 }}>
          These are internal test-split results from a single training run. Some sources share patients between the
          training and test splits, and several classes come almost entirely from one source dataset, so internal
          accuracy likely overstates performance on new hospitals or scanners. ROC-AUC cannot be independently
          recomputed because per-image predictions were not saved.
        </p>
      </div>
    </div>
  );
}
