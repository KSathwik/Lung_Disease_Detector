"""The run-selection rules of docs/data/run_selection_protocol.md."""

import numpy as np
import pandas as pd

from lungai.evaluation.compare_runs import common_set, decide, mcnemar_exact


def _rows(rows):
    d = pd.DataFrame(rows, columns=["image_id", "source_id", "pk", "lungai_label", "raw_md5", "split"])
    d["cache"] = None
    return d


def test_common_set_excludes_any_train_or_test_image_or_patient_of_either_build():
    a = _rows([("1", "nih", "nih|p1", "Normal", "m1", "val"),      # patient p1 is in b's train -> excluded
               ("2", "nih", "nih|p2", "Normal", "m2", "val"),      # kept
               ("3", "nih", "nih|p3", "Pneumonia", "m3", "test")])
    b = _rows([("4", "nih", "nih|p1", "Normal", "m4", "train"),
               ("5", "kermany", "kermany|x", "Pneumonia", "m3", "val"),  # same file as a's test image -> excluded
               ("6", "tbx", "tbx|h6", "Tuberculosis", "m6", "val"),     # kept
               ("2b", "nih", "nih|p2", "Normal", "m2", "val")])         # duplicate of a's image 2 -> counted once
    c = common_set(a, b)
    assert sorted(c.raw_md5) == ["m2", "m6"]
    assert sorted(c.y.tolist()) == [1, 5]  # Normal, Tuberculosis


def test_common_set_drops_images_labelled_differently_in_the_two_builds():
    a = _rows([("1", "nih", "nih|p1", "Normal", "m1", "val")])
    b = _rows([("1", "nih", "nih|p1", "Pneumonia", "m1", "val")])
    assert len(common_set(a, b)) == 0


def test_decision_rule():
    assert decide({"macro_f1": 0.70}, {"macro_f1": 0.72}) == "b"
    assert decide({"macro_f1": 0.72}, {"macro_f1": 0.70}) == "a"
    assert decide({"macro_f1": 0.70, "macro_roc_auc": 0.90}, {"macro_f1": 0.705, "macro_roc_auc": 0.905}) == "a"
    assert decide({"macro_f1": 0.70, "macro_roc_auc": 0.90}, {"macro_f1": 0.705, "macro_roc_auc": 0.915}) == "b"


def test_mcnemar_counts_discordant_pairs():
    a = np.array([True, True, False, False, True])
    b = np.array([True, False, True, False, False])
    r = mcnemar_exact(a, b)
    assert (r["a_right_b_wrong"], r["a_wrong_b_right"]) == (2, 1)
    assert 0 < r["p_value_exact"] <= 1
