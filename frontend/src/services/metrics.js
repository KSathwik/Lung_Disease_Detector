/**
 * Shape the /model-metrics payload for display. Values are passed through unchanged;
 * ROC-AUC is labelled as recorded-but-unverifiable because no per-image predictions exist.
 */

export function summarizeRecordedMetrics(data) {
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
      recallCi: v.recall_ci95, specificity: v.specificity, f1: v.f1,
    };
  });
  const ext = data.external?.binary_abnormality;
  return {
    model: data.model,
    datasetVersion: data.dataset_version,
    evidenceLevel: data.evidence_level,
    n: internal.n,
    internal: [
      { label: "Accuracy", value: m.accuracy, ci: m.accuracy_ci95 },
      { label: "Macro F1", value: m.macro_f1 },
      { label: "Weighted F1", value: m.weighted_f1 },
      { label: "Macro recall", value: m.macro_recall },
      {
        label: "Macro ROC-AUC",
        value: internal.recorded_unverifiable?.macro_roc_auc ?? null,
        format: "ratio",
        note: "Recorded value; cannot be recomputed (no per-image predictions saved).",
      },
    ],
    perClass,
    external: ext
      ? {
          cohort: data.external.cohort,
          interpretation: data.external.interpretation,
          rows: [
            { label: "TB scans flagged abnormal (sensitivity)", ...ext.sensitivity },
            { label: "Normal scans called Normal (specificity)", ...ext.specificity },
            { label: "TB scans called Tuberculosis", ...data.external.exact_tuberculosis_recall },
          ],
        }
      : null,
  };
}
