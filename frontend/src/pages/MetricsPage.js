import React, { useEffect, useState } from "react";
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
         BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
         LineChart, Line } from "recharts";
import { getModelMetrics } from "../services/api";

const METRICS_KEYS = ["accuracy", "precision", "recall", "f1_score", "auc_roc"];
const METRIC_LABELS = {
  accuracy: "Accuracy", precision: "Precision",
  recall: "Recall", f1_score: "F1-Score", auc_roc: "AUC-ROC"
};

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
  if (!data?.densenet && !data?.cnn && !data?.resnet) return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Model Performance</h1>
        <p className="page-subtitle">No model evaluation metrics found.</p>
      </div>
      <div className="card">
        <p style={{color:"var(--color-text-secondary)"}}>Model metrics file missing or uninitialized:</p>
        <code className="code-block">models/training_results.json</code>
      </div>
    </div>
  );

  const { cnn, resnet, densenet, selected_model } = data;

  // Build radar data
  const radarData = METRICS_KEYS.map(k => {
    const item = { metric: METRIC_LABELS[k] };
    if (densenet?.[k] !== undefined) item["Model D"] = +(densenet[k] * 100).toFixed(1);
    if (cnn?.[k] !== undefined) item["CNN"] = +(cnn[k] * 100).toFixed(1);
    if (resnet?.[k] !== undefined) item["ResNet"] = +(resnet[k] * 100).toFixed(1);
    return item;
  });

  // Build bar data
  const barData = METRICS_KEYS.map(k => {
    const item = { name: METRIC_LABELS[k] };
    if (densenet?.[k] !== undefined) item["Model D (Prod)"] = +(densenet[k] * 100).toFixed(1);
    if (cnn?.[k] !== undefined) item["CNN Baseline"] = +(cnn[k] * 100).toFixed(1);
    if (resnet?.[k] !== undefined) item["ResNet50"] = +(resnet[k] * 100).toFixed(1);
    return item;
  });

  // Build loss curves if available
  const hasLossCurves = (cnn?.training_loss?.length > 0) || (resnet?.training_loss?.length > 0);
  const maxLen = Math.max(
    (cnn?.training_loss || []).length,
    (resnet?.training_loss || []).length
  );
  const lossData = Array.from({ length: maxLen }, (_, i) => ({
    epoch:       i + 1,
    "CNN Train": cnn?.training_loss?.[i]?.toFixed(4),
    "CNN Val":   cnn?.validation_loss?.[i]?.toFixed(4),
    "ResNet Train": resnet?.training_loss?.[i]?.toFixed(4),
    "ResNet Val":   resnet?.validation_loss?.[i]?.toFixed(4),
  }));

  const MetricCard = ({ label, metricKey }) => {
    const dnVal = densenet?.[metricKey];
    const cnnVal = cnn?.[metricKey];
    const rnVal = resnet?.[metricKey];
    return (
      <div className="metric-card">
        <p className="metric-label">{label}</p>
        <div className="metric-values">
          {dnVal !== undefined && (
            <div className="metric-val metric-winner" style={{ borderLeft: "3px solid #7C3AED" }}>
              <span className="metric-model-name" style={{ fontWeight: 600, color: "#7C3AED" }}>
                Model D (Prod)
              </span>
              <span className="metric-number">{(dnVal * 100).toFixed(1)}%</span>
            </div>
          )}
          {cnnVal !== undefined && (
            <div className="metric-val">
              <span className="metric-model-name">CNN Baseline</span>
              <span className="metric-number">{(cnnVal * 100).toFixed(1)}%</span>
            </div>
          )}
          {rnVal !== undefined && (
            <div className="metric-val">
              <span className="metric-model-name">ResNet50 Baseline</span>
              <span className="metric-number">{(rnVal * 100).toFixed(1)}%</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  const perClass = densenet?.per_class || {};
  const perClassKeys = Object.keys(perClass);

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Model Performance & Evaluation</h1>
        <p className="page-subtitle">
          Comparative benchmark across Phase 1 Baselines and Phase 4D Production Champion. Active model:&nbsp;
          <strong style={{color:"#7C3AED"}}>
            {selected_model === "DenseNet" ? "DenseNet-121 Frequency V5 (Model D)" : selected_model}
          </strong>
        </p>
      </div>

      {/* ── Summary Cards ──────────────────────────────────────── */}
      <div className="metrics-grid">
        {METRICS_KEYS.map(k => (
          <MetricCard key={k} label={METRIC_LABELS[k]} metricKey={k} />
        ))}
      </div>

      {/* ── Bar Chart ──────────────────────────────────────────── */}
      <div className="card chart-card">
        <p className="section-label">Model Architecture Comparison</p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={barData} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 12 }} />
            <Tooltip formatter={v => `${v}%`} />
            <Legend />
            {densenet && <Bar dataKey="Model D (Prod)" fill="#7C3AED" radius={[4,4,0,0]} />}
            {cnn && <Bar dataKey="CNN Baseline" fill="#185FA5" radius={[4,4,0,0]} />}
            {resnet && <Bar dataKey="ResNet50" fill="#1D9E75" radius={[4,4,0,0]} />}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Radar Chart ────────────────────────────────────────── */}
      <div className="card chart-card">
        <p className="section-label">Performance Radar Profile</p>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
            {densenet && <Radar name="Model D (Prod)" dataKey="Model D" stroke="#7C3AED" fill="#7C3AED" fillOpacity={0.25} />}
            {cnn && <Radar name="CNN" dataKey="CNN" stroke="#185FA5" fill="#185FA5" fillOpacity={0.2} />}
            {resnet && <Radar name="ResNet" dataKey="ResNet" stroke="#1D9E75" fill="#1D9E75" fillOpacity={0.2} />}
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Per-Class Breakdown Table (Model D) ──────────────────── */}
      {perClassKeys.length > 0 && (
        <div className="card chart-card">
          <p className="section-label">Model D Held-Out Test Set Per-Class Metrics (N=1,570)</p>
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
                      <td style={{ padding: "8px 12px", color: "var(--text-secondary)" }}>{(m.precision * 100).toFixed(1)}%</td>
                      <td style={{ padding: "8px 12px", color: "var(--text-secondary)" }}>{(m.recall * 100).toFixed(1)}%</td>
                      <td style={{ padding: "8px 12px", color: "var(--text-secondary)" }}>{(m.specificity * 100).toFixed(1)}%</td>
                      <td style={{ padding: "8px 12px", fontWeight: 600, color: "#7C3AED" }}>{(m.f1_score * 100).toFixed(1)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Loss Curves ────────────────────────────────────────── */}
      {hasLossCurves && (
        <div className="card chart-card">
          <p className="section-label">Baseline Training & Validation Loss</p>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={lossData} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="epoch" label={{ value: "Epoch", position: "insideBottom", offset: -4 }} tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              {cnn && <Line type="monotone" dataKey="CNN Train" stroke="#185FA5" dot={false} strokeWidth={2} />}
              {cnn && <Line type="monotone" dataKey="CNN Val" stroke="#185FA5" dot={false} strokeWidth={2} strokeDasharray="5 5" />}
              {resnet && <Line type="monotone" dataKey="ResNet Train" stroke="#1D9E75" dot={false} strokeWidth={2} />}
              {resnet && <Line type="monotone" dataKey="ResNet Val" stroke="#1D9E75" dot={false} strokeWidth={2} strokeDasharray="5 5" />}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Clinical Generalization Note ───────────────────────── */}
      <div className="card" style={{ borderLeft: "4px solid #D97706", backgroundColor: "rgba(217, 119, 6, 0.04)" }}>
        <p style={{ fontWeight: 700, color: "#D97706", fontSize: "14px", margin: "0 0 4px 0" }}>
          External Generalization & Hardware Boundary Note
        </p>
        <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", lineHeight: "1.5", margin: 0 }}>
          While Model D achieves 82.93% accuracy and 0.9755 ROC-AUC on internal multi-source test data (1,570 scans), external evaluation on quarantined Montgomery County digitized film radiographs demonstrated sensor-domain shift (4&times; edge variance from digitizer film grain). Low-pass filtering (&sigma;=1.0) was engineered as a frequency-domain defense.
        </p>
      </div>
    </div>
  );
}
