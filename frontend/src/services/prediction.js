/**
 * Reading a /predict response for display.
 *
 * The class order and every score come from the API response (class_order, probabilities);
 * nothing here re-declares the six classes. A missing or invalid score is an error, never a 0.
 */

export class ResponseShapeError extends Error {
  constructor(message) {
    super(message);
    this.name = "ResponseShapeError";
    this.kind = "bad_response";
    this.title = "Unexpected Server Response";
  }
}

const isUnitScore = (v) => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1;

/** Rows in the model's class order: { index, name, score (0..1), isTop }. Throws ResponseShapeError. */
export function classScores(result) {
  const order = result?.class_order;
  const probs = result?.probabilities;
  if (!Array.isArray(order) || order.length === 0) {
    throw new ResponseShapeError("The server response does not include the model's class order.");
  }
  if (!probs || typeof probs !== "object") {
    throw new ResponseShapeError("The server response does not include class scores.");
  }
  const rows = order.map((name, index) => {
    const score = probs[name];
    if (!isUnitScore(score)) {
      throw new ResponseShapeError(`The server response has no valid score for "${name}".`);
    }
    return { index, name, score, isTop: name === result.predicted_class };
  });
  if (!rows.some((r) => r.isTop)) {
    throw new ResponseShapeError("The predicted class is not one of the model's classes.");
  }
  return rows;
}

/** Top-class softmax score (0..1) from the response. Throws ResponseShapeError. */
export function topScore(result) {
  if (!isUnitScore(result?.confidence)) {
    throw new ResponseShapeError("The server response does not include a valid top-class score.");
  }
  return result.confidence;
}

/** 0.8512 -> "85.1%". Display only; the stored value is not changed. */
export const formatScore = (v) => `${(v * 100).toFixed(1)}%`;
