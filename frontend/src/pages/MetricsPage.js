import React, { useEffect, useState } from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { getModelMetrics } from "../services/api";
import { summarizeModelMetrics } from "../services/metrics";

const pct = (v) => (v === undefined || v === null ? "—" : `${(v * 100).toFixed(1)}%`);
const ci = (c) => (c && c.lower !== null && c.lower !== undefined ? `${pct(c.lower)}–${pct(c.upper)}` : "");

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

  let summary;
  try {
    summary = summarizeModelMetrics(data);
  } catch (e) {
    return <div className="page"><div className="alert-error">{e.message}</div></div>;
  }
  if (!summary) return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Model Performance</h1>
        <p className="page-subtitle">No evaluation results are available on this server.</p>
      </div>
      <div className="card"><p style={{color:"var(--color-text-secondary)"}}>{data?.message}</p></div>
    </div>
  );

  const { model, internal, perClass, external, validation, historical } = summary;
  const accent = summary.reproduced && summary.matchesConfiguredCheckpoint ? "#1D9E75" : "#D97706";
  const perClassChart = perClass.map(r => ({
    name: r.name.replace("Pulmonary Nodule / Mass", "Nodule / Mass"),
    Precision: +(r.precision * 100).toFixed(1),
    Recall: +(r.recall * 100).toFixed(1),
    F1: +(r.f1 * 100).toFixed(1),
  }));

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Model Performance & Evaluation</h1>
        <p className="page-subtitle">
          Evaluation of <strong style={{color:"#7C3AED"}}>{model.display_name}</strong> ({model.version}),
          dataset {summary.datasetVersion}.
        </p>
      </div>

      <div className="card" style={{ borderLeft: `4px solid ${accent}` }}>
        <p style={{ fontWeight: 700, color: accent, fontSize: "14px", margin: "0 0 4px 0" }}>
          Evidence level: {summary.reproduced ? "computed from per-image predictions of this checkpoint" : "recorded, not reproduced"}
        </p>
        <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", lineHeight: "1.5", margin: 0 }}>{summary.evidenceLevel}</p>
        {!summary.matchesConfiguredCheckpoint && (
          <p style={{ fontSize: "12.5px", color: "#D97706", lineHeight: "1.5", margin: "6px 0 0 0" }}>
            These results belong to a different checkpoint from the one this server is configured to load.
          </p>
        )}
        {validation && (
          <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", lineHeight: "1.5", margin: "6px 0 0 0" }}>
            Checkpoint chosen on the validation split (epoch {validation.selected_epoch} of {validation.epochs_trained};
            validation macro F1 {pct(validation.macro_f1)}). The test split and the external cohort were each evaluated once, afterwards.
          </p>
        )}
        {data.model_loaded === false && (
          <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", lineHeight: "1.5", margin: "6px 0 0 0" }}>
            The checkpoint is not loaded on this server, so the Analyze page cannot produce predictions.
          </p>
        )}
      </div>

      <div className="metrics-grid">
        {internal.map(m => (
          <div key={m.label} className="metric-card">
            <p className="metric-label">{m.label}</p>
            <span className="metric-number">{m.format === "ratio" ? (m.value === undefined || m.value === null ? "—" : m.value.toFixed(3)) : pct(m.value)}</span>
            {m.ci && <p style={{ fontSize: "11.5px", color: "var(--text-muted)", margin: "2px 0 0 0" }}>95% CI {ci(m.ci)}</p>}
            {m.note && <p style={{ fontSize: "11.5px", color: "var(--text-muted)", margin: "2px 0 0 0" }}>{m.note}</p>}
          </div>
        ))}
      </div>

      <div className="card chart-card">
        <p className="section-label">Per-class precision, recall and F1 — internal test split (N={summary.n.toLocaleString()})</p>
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
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px", marginTop: "8px" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border-color)", textAlign: "left", color: "var(--text-secondary)" }}>
                <th style={{ padding: "8px 12px" }}>Class</th>
                <th style={{ padding: "8px 12px" }}>Support</th>
                <th style={{ padding: "8px 12px" }}>Precision</th>
                <th style={{ padding: "8px 12px" }}>Recall (95% CI)</th>
                <th style={{ padding: "8px 12px" }}>Specificity</th>
                <th style={{ padding: "8px 12px" }}>F1</th>
                <th style={{ padding: "8px 12px" }}>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {perClass.map(r => (
                <tr key={r.name} style={{ borderBottom: "1px solid var(--border-color)" }}>
                  <td style={{ padding: "8px 12px", fontWeight: 600 }}>{r.name}</td>
                  <td style={{ padding: "8px 12px" }}>{r.support}</td>
                  <td style={{ padding: "8px 12px" }}>{pct(r.precision)}</td>
                  <td style={{ padding: "8px 12px" }}>{pct(r.recall)} ({ci(r.recallCi)})</td>
                  <td style={{ padding: "8px 12px" }}>{pct(r.specificity)}</td>
                  <td style={{ padding: "8px 12px", fontWeight: 600 }}>{pct(r.f1)}</td>
                  <td style={{ padding: "8px 12px" }}>{r.rocAuc === undefined || r.rocAuc === null ? "—" : r.rocAuc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {external && (
        <div className="card chart-card">
          <p className="section-label">External cohort: {external.cohort}</p>
          <div className="metrics-grid" style={{ marginTop: "8px" }}>
            {external.rows.map(r => (
              <div key={r.label} className="metric-card">
                <p className="metric-label">{r.label}</p>
                <span className="metric-number">{r.successes}/{r.n}</span>
                <p style={{ fontSize: "11.5px", color: "var(--text-muted)", margin: "2px 0 0 0" }}>{pct(r.estimate)} (95% CI {ci(r)})</p>
              </div>
            ))}
          </div>
          <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", lineHeight: 1.5, margin: "8px 0 0 0" }}>
            {external.interpretation}
            {external.rocAuc !== null && ` ROC-AUC of P(abnormal): ${external.rocAuc.toFixed(3)}.`}
          </p>
        </div>
      )}

      {historical && (
        <div className="card" style={{ borderLeft: "4px solid var(--border-color)" }}>
          <p className="section-label">History: previous model {historical.version} (dataset {historical.datasetVersion})</p>
          <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", lineHeight: 1.5, margin: "4px 0 0 0" }}>
            Internal test accuracy {pct(historical.accuracy)}, macro F1 {pct(historical.macroF1)} (N={historical.n});
            external sensitivity {historical.externalSensitivity?.successes}/{historical.externalSensitivity?.n},
            specificity {historical.externalSpecificity?.successes}/{historical.externalSpecificity?.n}.
          </p>
          <p style={{ fontSize: "12.5px", color: "var(--text-muted)", lineHeight: 1.5, margin: "4px 0 0 0" }}>{historical.note}</p>
        </div>
      )}
    </div>
  );
}
