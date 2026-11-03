import { summarizeModelMetrics } from "./metrics";

const ORDER = ["COVID-19", "Normal", "Pleural Effusion", "Pneumonia", "Pulmonary Nodule / Mass", "Tuberculosis"];
const perClass = Object.fromEntries(ORDER.map((c, i) => [c, {
  support: 100 + i, precision: 0.9, recall: 0.8, recall_ci95: { lower: 0.7, upper: 0.87 }, specificity: 0.95, f1: 0.85, roc_auc: 0.9,
}]));
const payload = {
  available: true,
  model: { display_name: "LungAI DenseNet-121 — Six-Class Chest X-Ray Classifier", version: "v6.0" },
  matches_configured_checkpoint: true,
  dataset_version: "V6",
  evidence_level: "REPRODUCED: ...",
  class_order: ORDER,
  internal_test: {
    n: 1724,
    metrics: { accuracy: 0.6566, accuracy_ci95: { lower: 0.634, upper: 0.679 }, macro_f1: 0.6772, weighted_f1: 0.6588, macro_recall: 0.7285, macro_roc_auc: 0.921, per_class: perClass },
  },
  external: {
    cohort: "Montgomery County CXR (NLM)",
    interpretation: "...",
    binary_abnormality: {
      sensitivity: { successes: 45, n: 58, estimate: 0.776, lower: 0.653, upper: 0.864 },
      specificity: { successes: 43, n: 80, estimate: 0.5375, lower: 0.429, upper: 0.643 },
      roc_auc_p_abnormal: 0.73,
    },
    exact_tuberculosis_recall: { successes: 1, n: 58, estimate: 0.017, lower: 0.003, upper: 0.091 },
  },
  historical: {
    note: "Earlier model ...", model: { version: "v5.0" }, dataset_version: "V5",
    internal_test: { n: 1570, metrics: { accuracy: 0.8293, macro_f1: 0.7835 } },
    external_binary_abnormality: { sensitivity: { successes: 58, n: 58 }, specificity: { successes: 0, n: 80 } },
  },
};

test("values pass through unchanged, per-class rows in the API's class order", () => {
  const s = summarizeModelMetrics(payload);
  expect(s.perClass.map((r) => r.name)).toEqual(ORDER);
  expect(s.internal.find((m) => m.label === "Accuracy").value).toBe(0.6566);
  expect(s.n).toBe(1724);
  expect(s.reproduced).toBe(true);
  expect(s.matchesConfiguredCheckpoint).toBe(true);
});

test("ROC-AUC comes from the payload and is shown as a ratio", () => {
  const auc = summarizeModelMetrics(payload).internal.find((m) => m.label === "Macro ROC-AUC");
  expect(auc.value).toBe(0.921);
  expect(auc.format).toBe("ratio"); // an AUC is shown as 0.921, not as a percentage
});

test("external sensitivity is always shown with specificity", () => {
  const rows = summarizeModelMetrics(payload).external.rows;
  expect(rows.map((r) => `${r.successes}/${r.n}`)).toEqual(["45/58", "43/80", "1/58"]);
});

test("the V5 record is kept apart as history", () => {
  const h = summarizeModelMetrics(payload).historical;
  expect(h.version).toBe("v5.0");
  expect(`${h.externalSpecificity.successes}/${h.externalSpecificity.n}`).toBe("0/80");
  expect(summarizeModelMetrics({ ...payload, historical: undefined }).historical).toBeNull();
});

test("unavailable metrics return null; malformed payload throws", () => {
  expect(summarizeModelMetrics({ available: false })).toBeNull();
  expect(() => summarizeModelMetrics({ ...payload, class_order: undefined })).toThrow();
  const broken = JSON.parse(JSON.stringify(payload));
  delete broken.internal_test.metrics.per_class.Normal;
  expect(() => summarizeModelMetrics(broken)).toThrow(/Normal/);
});
