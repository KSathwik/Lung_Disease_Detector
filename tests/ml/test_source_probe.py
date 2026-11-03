"""The source probe's scoring (no TensorFlow): separable sources score high, within-class rows reported."""

import numpy as np

from lungai.evaluation.source_probe import probe


def test_probe_detects_separable_sources_overall_and_within_class():
    rng = np.random.default_rng(0)
    def make(n):
        s = np.array(["a"] * n + ["b"] * n)
        y = np.array(["Normal", "Pneumonia"] * n)
        x = rng.normal(size=(2 * n, 5)) + (s == "b")[:, None] * 3.0  # source shifts every feature
        return x, s, y
    xtr, str_, _ = make(60)
    xte, ste, yte = make(40)
    r = probe(xtr, str_, xte, ste, yte)
    assert r["all"]["balanced_accuracy"] > 0.9 and r["all"]["chance"] == 0.5
    assert set(r) == {"all", "within:Normal", "within:Pneumonia"}
    assert r["within:Normal"]["sources"] == ["a", "b"]
