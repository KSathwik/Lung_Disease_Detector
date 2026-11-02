"""Metric definitions checked against hand-computed values and sklearn."""

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from lungai.config import CLASS_NAMES
from lungai.evaluation.metrics import (
    EvaluationInputError, binary_abnormality_report, calibration, metrics_from_confusion, multiclass_report,
    validate_inputs, wilson_ci,
)


def _onehotish(y, k=6, p=0.7):
    probs = np.full((len(y), k), (1 - p) / (k - 1))
    probs[np.arange(len(y)), y] = p
    return probs


def test_perfect_predictions():
    y = np.repeat(np.arange(6), 5)
    r = multiclass_report(y, _onehotish(y), CLASS_NAMES)
    assert r["accuracy"] == 1.0 and r["macro_f1"] == 1.0 and r["macro_roc_auc_ovr"] == 1.0
    assert all(v["specificity"] == 1.0 for v in r["per_class"].values())


def test_known_confusion_matrix():
    cm = np.zeros((6, 6), int)
    cm[0, 0], cm[0, 1] = 8, 2   # class 0: 10 true, 8 right
    cm[1, 1] = 10
    for i in range(2, 6):
        cm[i, i] = 5
    r = metrics_from_confusion(cm, CLASS_NAMES)
    assert r["n"] == 40 and r["accuracy"] == pytest.approx(38 / 40)
    assert r["per_class"]["COVID-19"]["recall"] == pytest.approx(0.8)
    assert r["per_class"]["Normal"]["precision"] == pytest.approx(10 / 12)
    assert r["per_class"]["Normal"]["specificity"] == pytest.approx(28 / 30)


def test_matches_sklearn_on_random_data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 6, 400)
    logits = rng.normal(size=(400, 6)) + 2 * np.eye(6)[y]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    r = multiclass_report(y, probs, CLASS_NAMES)
    pred = probs.argmax(axis=1)
    assert r["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert r["weighted_f1"] == pytest.approx(f1_score(y, pred, average="weighted"))
    assert r["macro_roc_auc_ovr"] == pytest.approx(roc_auc_score(y, probs, multi_class="ovr", average="macro"))


def test_wilson_interval_reference_values():
    # Montgomery-style degenerate result: 58/58 and 0/80
    s = wilson_ci(58, 58)
    assert s["estimate"] == 1.0 and s["lower"] == pytest.approx(0.9379, abs=1e-4) and s["upper"] == 1.0
    sp = wilson_ci(0, 80)
    assert sp["lower"] == 0.0 and sp["upper"] == pytest.approx(0.0458, abs=1e-4)
    assert wilson_ci(0, 0)["estimate"] is None


def test_binary_abnormality_degenerate_classifier():
    truth = [True] * 58 + [False] * 80
    r = binary_abnormality_report(truth, [True] * 138, abnormal_score=[0.9] * 138)
    assert (r["tp"], r["fn"], r["tn"], r["fp"]) == (58, 0, 0, 80)
    assert r["specificity"]["estimate"] == 0.0
    assert r["roc_auc_p_abnormal"] == 0.5


def test_calibration_of_perfectly_calibrated_bins():
    y = np.array([0, 0, 0, 1] * 25)
    probs = np.zeros((100, 6))
    probs[:, 0], probs[:, 1] = 0.75, 0.25
    c = calibration(y, probs, n_bins=10)
    assert c["ece"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("y,probs,match", [
    (np.array([0, 1]), np.ones((3, 6)) / 6, "shape"),
    (np.array([0, 7]), np.ones((2, 6)) / 6, "outside"),
    (np.array([0.0, 1.0]), np.ones((2, 6)) / 6, "integer"),
    (np.array([0, 1]), np.full((2, 6), 0.5), "sum to 1"),
    (np.array([0, 1]), np.array([[np.nan] * 6, [1 / 6] * 6]), "finite"),
])
def test_invalid_inputs_fail(y, probs, match):
    with pytest.raises(EvaluationInputError, match=match):
        validate_inputs(y, probs, 6)


def test_absent_class_makes_macro_auc_undefined_not_wrong():
    y = np.array([0, 1, 2, 3, 4] * 4)  # no class 5
    r = multiclass_report(y, _onehotish(y), CLASS_NAMES)
    assert r["macro_roc_auc_ovr"] is None
    assert r["per_class_roc_auc"]["Tuberculosis"] is None
