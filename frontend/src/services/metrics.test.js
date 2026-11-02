import { summarizeRecordedMetrics } from "./metrics";

const ORDER = ["COVID-19", "Normal", "Pleural Effusion", "Pneumonia", "Pulmonary Nodule / Mass", "Tuberculosis"];
const perClass = Object.fromEntries(ORDER.map((c, i) => [c, {
  support: 100 + i, precision: 0.9, recall: 0.8, recall_ci95: { lower: 0.7, upper: 0.87 }, specificity: 0.95, f1: 0.85,
}]));
const payload = {
  available: true,
  model: { display_name: "LungAI DenseNet-121 — Six-Class Chest X-Ray Classifier", version: "v5.0" },
  dataset_version: "V5",
  evidence_level: "RECORDED: ...",
  class_order: ORDER,
  internal_test: {
    n: 1570,
    metrics: { accuracy: 0.8293, accuracy_ci95: { lower: 0.81, upper: 0.847 }, macro_f1: 0.7835, weighted_f1: 0.8418, macro_recall: 0.8073, per_class: perClass },
    recorded_unverifiable: { macro_roc_auc: 0.9755 },
  },
  external: {
    cohort: "Montgomery County (external, binary TB vs normal)",
    interpretation: "Degenerate: ...",
    binary_abnormality: {
      sensitivity: { successes: 58, n: 58, estimate: 1, lower: 0.938, upper: 1 },
      specificity: { successes: 0, n: 80, estimate: 0, lower: 0, upper: 0.046 },
    },
    exact_tuberculosis_recall: { successes: 0, n: 58, estimate: 0, lower: 0, upper: 0.062 },
  },
};

test("values pass through unchanged, per-class rows in the API's class order", () => {
  const s = summarizeRecordedMetrics(payload);
  expect(s.perClass.map((r) => r.name)).toEqual(ORDER);
  expect(s.internal.find((m) => m.label === "Accuracy").value).toBe(0.8293);
  expect(s.n).toBe(1570);
});

test("ROC-AUC is labelled as recorded but unverifiable", () => {
  const auc = summarizeRecordedMetrics(payload).internal.find((m) => m.label === "Macro ROC-AUC");
  expect(auc.value).toBe(0.9755);
  expect(auc.format).toBe("ratio"); // an AUC is shown as 0.9755, not as a percentage
  expect(auc.note).toMatch(/cannot be recomputed/);
});

test("external sensitivity is always shown with specificity", () => {
  const rows = summarizeRecordedMetrics(payload).external.rows;
  expect(rows.map((r) => `${r.successes}/${r.n}`)).toEqual(["58/58", "0/80", "0/58"]);
});

test("unavailable metrics return null; malformed payload throws", () => {
  expect(summarizeRecordedMetrics({ available: false })).toBeNull();
  expect(() => summarizeRecordedMetrics({ ...payload, class_order: undefined })).toThrow();
  const broken = JSON.parse(JSON.stringify(payload));
  delete broken.internal_test.metrics.per_class.Normal;
  expect(() => summarizeRecordedMetrics(broken)).toThrow(/Normal/);
});
