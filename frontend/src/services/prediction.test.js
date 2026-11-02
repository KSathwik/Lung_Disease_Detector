import { classScores, formatScore, topScore, ResponseShapeError } from "./prediction";

const ORDER = ["COVID-19", "Normal", "Pleural Effusion", "Pneumonia", "Pulmonary Nodule / Mass", "Tuberculosis"];
const response = (overrides = {}) => ({
  class_order: ORDER,
  predicted_class: "Pneumonia",
  confidence: 0.85,
  probabilities: {
    "COVID-19": 0.02, "Normal": 0.05, "Pleural Effusion": 0.03,
    "Pneumonia": 0.85, "Pulmonary Nodule / Mass": 0.02, "Tuberculosis": 0.03,
  },
  ...overrides,
});

test("rows follow the class order sent by the API, with the API's scores unchanged", () => {
  const rows = classScores(response());
  expect(rows.map((r) => r.name)).toEqual(ORDER);
  expect(rows.map((r) => r.score)).toEqual([0.02, 0.05, 0.03, 0.85, 0.02, 0.03]);
  expect(rows.filter((r) => r.isTop).map((r) => r.name)).toEqual(["Pneumonia"]);
});

test("a different class order from the API is respected (no hardcoded order)", () => {
  const reversed = [...ORDER].reverse();
  expect(classScores(response({ class_order: reversed })).map((r) => r.name)).toEqual(reversed);
});

test("a missing class score is an error, never shown as 0%", () => {
  const probs = { ...response().probabilities };
  delete probs["Tuberculosis"];
  expect(() => classScores(response({ probabilities: probs }))).toThrow(ResponseShapeError);
});

test("out-of-range or non-numeric scores are rejected", () => {
  expect(() => classScores(response({ probabilities: { ...response().probabilities, Normal: 1.5 } }))).toThrow(ResponseShapeError);
  expect(() => classScores(response({ probabilities: { ...response().probabilities, Normal: "0.1" } }))).toThrow(ResponseShapeError);
});

test("missing class order or predicted class outside it is rejected", () => {
  expect(() => classScores(response({ class_order: undefined }))).toThrow(/class order/);
  expect(() => classScores(response({ predicted_class: "Lung Cancer" }))).toThrow(/predicted class/);
});

test("top score comes from the response and must be a unit value", () => {
  expect(topScore(response())).toBe(0.85);
  expect(() => topScore(response({ confidence: 85 }))).toThrow(ResponseShapeError);
  expect(() => topScore(response({ confidence: undefined }))).toThrow(ResponseShapeError);
});

test("formatting rounds for display only", () => {
  expect(formatScore(0.85123)).toBe("85.1%");
  expect(formatScore(0)).toBe("0.0%");
  expect(formatScore(1)).toBe("100.0%");
});
