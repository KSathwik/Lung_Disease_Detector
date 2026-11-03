import React, { useEffect, useState } from "react";
import { listPredictions, getPrediction, generateReport } from "../services/api";

const URGENCY_COLOR = { routine: "#1D9E75", urgent: "#854F0B", emergency: "#A32D2D", review: "#475569" };

export default function HistoryPage() {
  const [data,        setData]        = useState(null);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState(null);
  const [search,      setSearch]      = useState("");
  const [filterUrgent,setFilterUrgent]= useState("all");
  const [selectedPred,setSelectedPred]= useState(null);
  const [reportData,  setReportData]  = useState(null);
  const [generating,  setGenerating]  = useState(false);

  useEffect(() => {
    listPredictions()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const preds = data?.predictions || [];

  const filteredPreds = preds.filter((p) => {
    const matchesSearch =
      p.final_condition.toLowerCase().includes(search.toLowerCase()) ||
      p.selected_model.toLowerCase().includes(search.toLowerCase()) ||
      p.prediction_id.toLowerCase().includes(search.toLowerCase());
    const matchesUrgency = filterUrgent === "all" || p.urgency_level === filterUrgent;
    return matchesSearch && matchesUrgency;
  });

  const openDetail = (pred) => {
    // Show the summary immediately, then replace it with the full stored record
    setSelectedPred(pred);
    setReportData(null);
    getPrediction(pred.prediction_id)
      .then(full => setSelectedPred(cur => (cur?.prediction_id === pred.prediction_id ? full : cur)))
      .catch(e => setError(e.message));
  };

  const handleGenerateReportModal = async () => {
    if (!selectedPred?.prediction_id) return;
    setGenerating(true);
    try {
      const r = await generateReport(selectedPred.prediction_id);
      setReportData(r);
    } catch (e) {
      setError("Failed to generate report.");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Analysis History</h1>
        <p className="page-subtitle">{preds.length} diagnostic analyses recorded.</p>
      </div>

      {/* ── Toolbar: Search & Urgency Filter Pills (Item 3) ──────── */}
      <div className="toolbar-row">
        <div className="search-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="search-input"
            placeholder="Search by condition, model, or ID…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div className="filter-pills">
          {["all", "routine", "urgent", "emergency"].map((level) => (
            <button
              key={level}
              className={`filter-pill ${filterUrgent === level ? "active" : ""}`}
              onClick={() => setFilterUrgent(level)}
            >
              {level.charAt(0).toUpperCase() + level.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="alert-error">{error}</div>}

      {/* ── Skeleton Loaders (Item 2) ────────────────────────────── */}
      {loading ? (
        <div className="history-list">
          {[1, 2, 3, 4].map((n) => (
            <div key={n} className="card skeleton skeleton-card" />
          ))}
        </div>
      ) : filteredPreds.length === 0 ? (
        <div className="empty-state card">
          <p>No matching diagnostic records found.</p>
        </div>
      ) : (
        <div className="history-list">
          {filteredPreds.map((p) => (
            <div key={p.prediction_id} className="history-item card" onClick={() => openDetail(p)}>
              <div className="history-left">
                <p className="history-condition">{p.inconclusive ? `Inconclusive (top: ${p.final_condition})` : p.final_condition}</p>
                <p className="history-meta">
                  {p.selected_model}{p.model_version ? ` ${p.model_version}` : ""} · {p.final_confidence.toFixed(1)}% top-class score · ID: {p.prediction_id.slice(0, 8)}
                </p>
              </div>
              <div className="history-right">
                <span
                  className="urgency-dot"
                  style={{ background: URGENCY_COLOR[p.urgency_level] || "#1D9E75" }}
                  title={`Urgency: ${p.urgency_level}`}
                />
                <span className="history-date">
                  {new Date(p.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Deep History Detail Modal (Item 3) ───────────────────── */}
      {selectedPred && (
        <div className="modal-backdrop" onClick={() => setSelectedPred(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3 className="modal-title">
                  {selectedPred.inconclusive ? `Inconclusive (top: ${selectedPred.final_condition})` : selectedPred.final_condition}
                </h3>
                <p style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "2px" }}>
                  Record ID: {selectedPred.prediction_id}
                </p>
              </div>
              <button className="modal-close" onClick={() => setSelectedPred(null)}>✕</button>
            </div>

            <div className="modal-body">
              <div className="card" style={{ marginBottom: "1rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                  <span className="section-label">Diagnostic Summary</span>
                  <span
                    className="urgency-badge"
                    style={{
                      color: URGENCY_COLOR[selectedPred.urgency_level],
                      background: "var(--brand-light)",
                    }}
                  >
                    {selectedPred.urgency_level}
                  </span>
                </div>
                <p style={{ fontSize: "16px", fontWeight: "700", marginBottom: "4px" }}>
                  {selectedPred.final_confidence.toFixed(1)}% top-class score
                </p>
                <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
                  Model: <strong>{selectedPred.selected_model}</strong>
                  {selectedPred.model_version && <> · version <strong>{selectedPred.model_version}</strong></>}
                </p>
                {selectedPred.model_sha256 && (
                  <p style={{ fontSize: "11.5px", color: "var(--text-muted)", marginTop: "4px", wordBreak: "break-all" }}>
                    Checkpoint SHA-256: {selectedPred.model_sha256}
                  </p>
                )}
              </div>

              {/* Stored six-class model output */}
              <div className="card" style={{ marginBottom: "1rem" }}>
                <span className="section-label">Class Probabilities</span>
                {selectedPred.probabilities ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginTop: "8px" }}>
                    {Object.entries(selectedPred.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([cls, p]) => (
                        <div key={cls} style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
                          <span>{cls}</span>
                          <strong>{(p * 100).toFixed(1)}%</strong>
                        </div>
                      ))}
                  </div>
                ) : (
                  <p style={{ fontSize: "12.5px", color: "var(--text-secondary)", marginTop: "8px" }}>
                    {!("probabilities" in selectedPred)
                      ? "Loading…"
                      : "Not recorded for this prediction (created before probabilities were stored)."}
                  </p>
                )}
              </div>

              {/* Precautions */}
              {selectedPred.precautions?.length > 0 && (
                <div className="card" style={{ marginBottom: "1rem" }}>
                  <span className="section-label">Precautions</span>
                  <ul className="precaution-list" style={{ marginTop: "8px" }}>
                    {selectedPred.precautions.map((pr, i) => (
                      <li key={i} className="precaution-item">
                        <span className="precaution-dot" />
                        {pr}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Report Document if Generated */}
              {reportData && (
                <div className="card">
                  <span className="section-label">Generated Report</span>
                  <pre className="report-document" style={{ marginTop: "8px" }}>{reportData.content}</pre>
                </div>
              )}
            </div>

            <div className="modal-footer">
              {!reportData ? (
                <button className="btn-secondary" onClick={handleGenerateReportModal} disabled={generating}>
                  {generating ? "Generating…" : "📄 Generate Clinical Report"}
                </button>
              ) : (
                <button className="btn-secondary" onClick={() => window.print()}>🖨️ Print / Save PDF</button>
              )}
              <button className="btn-primary" onClick={() => setSelectedPred(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
