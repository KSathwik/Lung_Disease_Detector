"""Metric definitions for the six-class task and the external binary assessment.

Conventions (also stated in docs/evaluation.md):
- Labels are class indices 0..5 in the canonical order; predictions are argmax of the probabilities.
- Per-class precision/recall/F1/specificity are one-vs-rest. Macro = unweighted mean over the 6 classes;
  weighted = mean weighted by true-class support.
- Macro ROC-AUC / PR-AUC: one-vs-rest per class, unweighted mean (sklearn multi_class="ovr", average="macro").
- Proportion CIs: Wilson score 95%. Accuracy / macro-F1 / macro-AUC CIs: percentile bootstrap over images.
- ECE: equal-width confidence bins on the top-class softmax.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix as sk_confusion_matrix, roc_auc_score

Z95 = 1.959963984540054


class EvaluationInputError(ValueError):
    """Raised when labels/probabilities are inconsistent and metrics would be meaningless."""


def validate_inputs(y_true: Sequence[int], probs: Optional[np.ndarray], num_classes: int,
                    y_pred: Optional[Sequence[int]] = None, atol: float = 1e-3) -> None:
    y = np.asarray(y_true)
    if y.ndim != 1 or len(y) == 0:
        raise EvaluationInputError(f"y_true must be a non-empty 1-D array, got shape {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        raise EvaluationInputError(f"y_true must contain integer class indices, got dtype {y.dtype}")
    if y.min() < 0 or y.max() >= num_classes:
        raise EvaluationInputError(f"y_true has class indices outside 0..{num_classes - 1}: {sorted(set(y.tolist()) - set(range(num_classes)))}")
    if probs is not None:
        p = np.asarray(probs, dtype=np.float64)
        if p.shape != (len(y), num_classes):
            raise EvaluationInputError(f"probabilities shape {p.shape} does not match ({len(y)}, {num_classes})")
        if not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
            raise EvaluationInputError("probabilities must be finite values in [0, 1]")
        if np.max(np.abs(p.sum(axis=1) - 1.0)) > atol:
            raise EvaluationInputError("probability rows must sum to 1 (softmax outputs expected)")
    if y_pred is not None:
        yp = np.asarray(y_pred)
        if yp.shape != y.shape:
            raise EvaluationInputError(f"y_pred shape {yp.shape} does not match y_true shape {y.shape}")
        if yp.min() < 0 or yp.max() >= num_classes:
            raise EvaluationInputError("y_pred has class indices outside the canonical range")


def wilson_ci(successes: int, n: int, z: float = Z95) -> Dict[str, Optional[float]]:
    if n <= 0:
        return {"estimate": None, "lower": None, "upper": None, "successes": successes, "n": n}
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"estimate": p, "lower": max(0.0, centre - half), "upper": min(1.0, centre + half), "successes": successes, "n": n}


def confusion(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    return sk_confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def metrics_from_confusion(cm: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """Threshold metrics derived only from a confusion matrix (rows = true, columns = predicted)."""
    cm = np.asarray(cm, dtype=np.int64)
    k = len(class_names)
    if cm.shape != (k, k):
        raise EvaluationInputError(f"confusion matrix shape {cm.shape} does not match {k} classes")
    n = int(cm.sum())
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=(precision + recall) > 0)
    tn = n - support - predicted + tp
    spec = np.divide(tn, n - support, out=np.zeros(k), where=(n - support) > 0)
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "support": int(support[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "recall_ci95": wilson_ci(int(tp[i]), int(support[i])),
            "specificity": float(spec[i]),
            "f1": float(f1[i]),
        }
    correct = int(tp.sum())
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "accuracy_ci95": wilson_ci(correct, n),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()) if support.sum() else None,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def ranking_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    k = len(class_names)
    onehot = np.eye(k)[y_true]
    per_auc, per_ap = {}, {}
    for i, name in enumerate(class_names):
        if onehot[:, i].min() == onehot[:, i].max():
            per_auc[name] = per_ap[name] = None  # undefined when a class is absent or is the only class
            continue
        per_auc[name] = float(roc_auc_score(onehot[:, i], probs[:, i]))
        per_ap[name] = float(average_precision_score(onehot[:, i], probs[:, i]))
    defined = [v for v in per_auc.values() if v is not None]
    defined_ap = [v for v in per_ap.values() if v is not None]
    return {
        "macro_roc_auc_ovr": float(np.mean(defined)) if len(defined) == k else None,
        "macro_pr_auc": float(np.mean(defined_ap)) if len(defined_ap) == k else None,
        "per_class_roc_auc": per_auc,
        "per_class_pr_auc": per_ap,
    }


def calibration(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        m = int(mask.sum())
        if m == 0:
            continue
        acc, avg_conf = float(correct[mask].mean()), float(conf[mask].mean())
        ece += m / len(conf) * abs(acc - avg_conf)
        bins.append({"lower": float(lo), "upper": float(hi), "count": m, "accuracy": acc, "mean_confidence": avg_conf})
    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(conf[~correct].mean()) if (~correct).any() else None,
        "bins": bins,
        "note": "Softmax scores are not calibrated probabilities; this measures how far they are from calibrated.",
    }


def bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, class_names: List[str],
                 n_resamples: int, seed: int) -> Dict[str, Dict[str, float]]:
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    n, k = len(y_true), len(class_names)
    preds = probs.argmax(axis=1)
    acc, mf1, auc = [], [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yt, yp, pr = y_true[idx], preds[idx], probs[idx]
        acc.append(float((yt == yp).mean()))
        mf1.append(float(f1_score(yt, yp, labels=list(range(k)), average="macro", zero_division=0)))
        if len(np.unique(yt)) == k:
            auc.append(float(roc_auc_score(np.eye(k)[yt], pr, average="macro", multi_class="ovr")))
    out = {}
    for name, vals in (("accuracy", acc), ("macro_f1", mf1), ("macro_roc_auc_ovr", auc)):
        if vals:
            out[name] = {"lower": float(np.percentile(vals, 2.5)), "upper": float(np.percentile(vals, 97.5)), "resamples": len(vals)}
    return out


def multiclass_report(y_true: Sequence[int], probs: np.ndarray, class_names: List[str],
                      n_bootstrap: int = 0, seed: int = 42, calibration_bins: int = 15) -> Dict[str, Any]:
    """All six-class metrics from labels and softmax probabilities."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    validate_inputs(y, p, len(class_names))
    preds = p.argmax(axis=1)
    report = metrics_from_confusion(confusion(y, preds, len(class_names)), class_names)
    report.update(ranking_metrics(y, p, class_names))
    report["calibration"] = calibration(y, p, calibration_bins)
    if n_bootstrap:
        report["bootstrap_ci95"] = bootstrap_ci(y, p, class_names, n_bootstrap, seed)
    return report


def binary_abnormality_report(true_abnormal: Sequence[bool], predicted_abnormal: Sequence[bool],
                              abnormal_score: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """External binary assessment: abnormal = any predicted class other than Normal."""
    t = np.asarray(true_abnormal, dtype=bool)
    p = np.asarray(predicted_abnormal, dtype=bool)
    if t.shape != p.shape or t.ndim != 1:
        raise EvaluationInputError("true/predicted abnormality arrays must be 1-D and the same length")
    tp, fn = int((t & p).sum()), int((t & ~p).sum())
    tn, fp = int((~t & ~p).sum()), int((~t & p).sum())
    out: Dict[str, Any] = {
        "definition": "positive = predicted class is not 'Normal'; reference positive = TB-confirmed scan",
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": wilson_ci(tp, tp + fn),
        "specificity": wilson_ci(tn, tn + fp),
        "accuracy": wilson_ci(tp + tn, len(t)),
        "prevalence": float(t.mean()) if len(t) else None,
    }
    if abnormal_score is not None and t.min() != t.max():
        out["roc_auc_p_abnormal"] = float(roc_auc_score(t, np.asarray(abnormal_score, dtype=float)))
    return out
