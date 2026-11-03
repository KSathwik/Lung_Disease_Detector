/**
 * Shape the /model-metrics payload for display. Values are passed through unchanged.
 * The current model's metrics are computed from per-image predictions; the V5 record,
 * when present, is returned separately as history and never mixed with them.
 */

export function summarizeModelMetrics(data) {
  if (!data || data.available === false) return null;
  const internal = data.internal_test;
  const m = internal?.metrics;
  if (!m || !Array.isArray(data.class_order)) {
    throw new Error("The metrics response is missing the internal test results or the class order.");
  }
  const perClass = data.class_order.map((name) => {
    const v = m.per_class?.[name];
    if (!v) throw new Error(`The metrics response has no per-class result for "${name}".`);
    return {
      name, support: v.support, precision: v.precision, recall: v.recall,
      recallCi: v.recall_ci95, specificity: v.specificity, f1: v.f1, rocAuc: v.roc_auc,
    };
  });
  const ext = data.external?.binary_abnormality;
  const hist = data.historical;
  return {
    model: data.model,
    datasetVersion: data.dataset_version,
    evidenceLevel: data.evidence_level,
    reproduced: String(data.evidence_level || "").startsWith("REPRODUCED"),
    matchesConfiguredCheckpoint: data.matches_configured_checkpoint !== false,
    n: internal.n,
    internal: [
      { label: "Accuracy", value: m.accuracy, ci: m.accuracy_ci95 },
      { label: "Macro F1", value: m.macro_f1 },
      { label: "Weighted F1", value: m.weighted_f1 },
      { label: "Macro recall", value: m.macro_recall },
      { label: "Macro ROC-AUC", value: m.macro_roc_auc ?? null, format: "ratio", note: "One-vs-rest, from per-image probabilities." },
    ],
    perClass,
    validation: data.validation_evidence ?? null,
    external: ext
      ? {
          cohort: data.external.cohort,
          interpretation: data.external.interpretation,
          rocAuc: ext.roc_auc_p_abnormal ?? null,
          rows: [
            { label: "TB scans flagged abnormal (sensitivity)", ...ext.sensitivity },
            { label: "Normal scans called Normal (specificity)", ...ext.specificity },
            { label: "TB scans called Tuberculosis", ...data.external.exact_tuberculosis_recall },
          ],
        }
      : null,
    historical: hist
      ? {
          note: hist.note,
          version: hist.model?.version,
          datasetVersion: hist.dataset_version,
          n: hist.internal_test?.n,
          accuracy: hist.internal_test?.metrics?.accuracy,
          macroF1: hist.internal_test?.metrics?.macro_f1,
          externalSensitivity: hist.external_binary_abnormality?.sensitivity,
          externalSpecificity: hist.external_binary_abnormality?.specificity,
        }
      : null,
  };
}
