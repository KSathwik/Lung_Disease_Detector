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
  if (!data?.cnn) return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Model Performance</h1>
        <p className="page-subtitle">No model evaluation metrics found.</p>
      </div>
      <div className="card">
        <p style={{color:"var(--color-text-secondary)"}}>Train models first:</p>
        <code className="code-block">python backend/ml/train.py --data_dir data/raw --epochs 50</code>
      </div>
    </div>
  );

  const { cnn, resnet, selected_model } = data;

  // Build radar data
  const radarData = METRICS_KEYS.map(k => ({
    metric: METRIC_LABELS[k],
    CNN:    +(cnn[k]    * 100).toFixed(1),
    ResNet: +(resnet[k] * 100).toFixed(1),
  }));

  // Build bar data
  const barData = METRICS_KEYS.map(k => ({
    name:   METRIC_LABELS[k],
    CNN:    +(cnn[k]    * 100).toFixed(1),
    ResNet: +(resnet[k] * 100).toFixed(1),
  }));

  // Build loss curves
  const maxLen = Math.max(
    (cnn.training_loss || []).length,
    (resnet.training_loss || []).length
  );
  const lossData = Array.from({ length: maxLen }, (_, i) => ({
    epoch:       i + 1,
    "CNN Train": cnn.training_loss?.[i]?.toFixed(4),
    "CNN Val":   cnn.validation_loss?.[i]?.toFixed(4),
    "ResNet Train": resnet.training_loss?.[i]?.toFixed(4),
    "ResNet Val":   resnet.validation_loss?.[i]?.toFixed(4),
  }));

  const MetricCard = ({ label, cnnVal, resnetVal }) => {
    const winner = cnnVal >= resnetVal ? "CNN" : "ResNet";
    return (
      <div className="metric-card">
        <p className="metric-label">{label}</p>
        <div className="metric-values">
          <div className={`metric-val ${selected_model === "CNN" || winner === "CNN" ? "metric-winner" : ""}`}>
            <span className="metric-model-name">CNN</span>
            <span className="metric-number">{(cnnVal * 100).toFixed(1)}%</span>
          </div>
          <div className={`metric-val ${selected_model === "ResNet" || winner === "ResNet" ? "metric-winner" : ""}`}>
            <span className="metric-model-name">ResNet50</span>
            <span className="metric-number">{(resnetVal * 100).toFixed(1)}%</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Model Performance</h1>
        <p className="page-subtitle">
          Training and evaluation metrics for CNN and ResNet50 models. Selected ensemble model:&nbsp;
          <strong style={{color:"var(--brand-color)"}}>{selected_model === "ResNet" ? "ResNet50" : selected_model}</strong>
        </p>
      </div>

      {/* ── Summary Cards ──────────────────────────────────────── */}
      <div className="metrics-grid">
        {METRICS_KEYS.map(k => (
          <MetricCard key={k} label={METRIC_LABELS[k]} cnnVal={cnn[k]} resnetVal={resnet[k]} />
        ))}
      </div>

      {/* ── Bar Chart ──────────────────────────────────────────── */}
      <div className="card chart-card">
        <p className="section-label">Metrics comparison</p>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={barData} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 12 }} />
            <Tooltip formatter={v => `${v}%`} />
            <Legend />
            <Bar dataKey="CNN"    fill="#185FA5" radius={[4,4,0,0]} />
            <Bar dataKey="ResNet" fill="#1D9E75" radius={[4,4,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Radar Chart ────────────────────────────────────────── */}
      <div className="card chart-card">
        <p className="section-label">Radar comparison</p>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
            <Radar name="CNN"    dataKey="CNN"    stroke="#185FA5" fill="#185FA5" fillOpacity={0.25} />
            <Radar name="ResNet" dataKey="ResNet" stroke="#1D9E75" fill="#1D9E75" fillOpacity={0.25} />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Loss Curves ────────────────────────────────────────── */}
      {lossData.length > 0 && (
        <div className="card chart-card">
          <p className="section-label">Training & validation loss</p>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={lossData} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="epoch" label={{ value: "Epoch", position: "insideBottom", offset: -4 }} tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="CNN Train"    stroke="#185FA5" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="CNN Val"      stroke="#185FA5" dot={false} strokeWidth={2} strokeDasharray="5 5" />
              <Line type="monotone" dataKey="ResNet Train" stroke="#1D9E75" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="ResNet Val"   stroke="#1D9E75" dot={false} strokeWidth={2} strokeDasharray="5 5" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
