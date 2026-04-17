import React, { useEffect, useState } from "react";
import { listPredictions } from "../services/api";

const URGENCY_COLOR = { routine: "#1D9E75", urgent: "#854F0B", emergency: "#A32D2D" };

export default function HistoryPage() {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    listPredictions()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page"><p className="loading-text">Loading history…</p></div>;
  if (error)   return <div className="page"><div className="alert-error">{error}</div></div>;

  const preds = data?.predictions || [];

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Analysis History</h1>
        <p className="page-subtitle">{preds.length} analyses recorded.</p>
      </div>

      {preds.length === 0 ? (
        <div className="empty-state card">
          <p>No analyses yet. Upload a lung image to get started.</p>
        </div>
      ) : (
        <div className="history-list">
          {preds.map(p => (
            <div key={p.prediction_id} className="history-item card">
              <div className="history-left">
                <p className="history-condition">{p.final_condition}</p>
                <p className="history-meta">
                  {p.selected_model} · {p.final_confidence.toFixed(1)}% confidence
                </p>
              </div>
              <div className="history-right">
                <span className="urgency-dot" style={{ background: URGENCY_COLOR[p.urgency_level] || "#1D9E75" }} />
                <span className="history-date">{new Date(p.created_at).toLocaleDateString()}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
