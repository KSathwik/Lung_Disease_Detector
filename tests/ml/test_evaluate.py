"""The evaluation entry point: predictions-file validation and recomputation (no checkpoint needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from lungai.config import CLASS_NAMES
from lungai.data.manifest import load_manifest, split_frame
from lungai.evaluation import evaluate as ev
from lungai.evaluation.metrics import EvaluationInputError


def _val_predictions(correct_fraction=0.8, seed=0):
    val = split_frame(load_manifest(), "val")
    rng = np.random.default_rng(seed)
    y = val["clinical_label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).to_numpy()
    pred = np.where(rng.random(len(y)) < correct_fraction, y, rng.integers(0, 6, len(y)))
    probs = np.full((len(y), 6), 0.04)
    probs[np.arange(len(y)), pred] = 0.8
    samples = pd.DataFrame({"image_id": val["image_id"], "true_label": val["clinical_label"], "source_dataset": val["source_dataset"]})
    return ev.predictions_frame(samples, probs, "val", "f" * 64)


def test_metrics_recomputed_from_a_valid_predictions_file():
    pred = _val_predictions()
    r = ev.compute_metrics(pred)
    assert r["split"] == "val" and r["n_evaluated"] == 1579 and r["n_excluded"] == 0
    m = r["metrics"]
    assert m["accuracy"] == pytest.approx((pred["pred_label"] == pred["true_label"]).mean())
    assert set(r["per_source"]) == set(pred["source_dataset"])
    assert "accuracy" in m["bootstrap_ci95"]


def test_missing_sample_is_detected():
    with pytest.raises(EvaluationInputError, match="identities"):
        ev.compute_metrics(_val_predictions().iloc[1:])


def test_wrong_label_is_detected():
    pred = _val_predictions()
    pred.loc[0, "true_label"] = "Normal" if pred.loc[0, "true_label"] != "Normal" else "COVID-19"
    with pytest.raises(EvaluationInputError, match="disagree"):
        ev.compute_metrics(pred)


def test_tampered_prediction_index_is_detected():
    pred = _val_predictions()
    pred.loc[0, "pred_index"] = (pred.loc[0, "pred_index"] + 1) % 6
    with pytest.raises(EvaluationInputError, match="argmax"):
        ev.compute_metrics(pred)


def test_mixed_checkpoints_are_detected():
    pred = _val_predictions()
    pred.loc[0, "checkpoint_sha256"] = "0" * 64
    with pytest.raises(EvaluationInputError, match="more than one checkpoint"):
        ev.compute_metrics(pred)


def test_external_predictions_report_binary_task():
    n_tb, n_norm = 5, 7
    samples = pd.DataFrame({"image_id": [f"m{i}.png" for i in range(n_tb + n_norm)],
                            "true_label": ["Tuberculosis"] * n_tb + ["Normal"] * n_norm, "source_dataset": "Montgomery"})
    probs = np.full((12, 6), 0.02)
    probs[:, 4] = 0.90  # everything predicted Nodule/Mass
    r = ev.compute_metrics(ev.predictions_frame(samples, probs, "external", "e" * 64))
    b = r["binary_abnormality"]
    assert "six-class metrics do not apply" in r["task"]
    assert (b["sensitivity"]["successes"], b["specificity"]["successes"]) == (5, 0)
    assert r["exact_tuberculosis_recall"]["successes"] == 0


def test_cli_recompute_writes_reports(tmp_path):
    f = tmp_path / "predictions_val.csv"
    _val_predictions().to_csv(f, index=False)
    out = tmp_path / "out"
    assert ev.main(["--predictions", str(f), "--out", str(out)]) == 0
    report = json.loads((out / "metrics_val.json").read_text())
    assert report["model_id"] == "lungai-densenet121-6class"
    assert report["manifest_sha256_lf"] and report["result"]["n_evaluated"] == 1579
    assert (out / "summary_val.md").is_file()


def test_final_splits_require_explicit_confirmation():
    with pytest.raises(SystemExit):
        ev.main(["--split", "test", "--checkpoint", "x.h5"])
