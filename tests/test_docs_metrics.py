"""Numbers in the documentation must match the generated evidence files."""

import json

import pytest

from lungai.config import REPO_ROOT
from lungai.data.checks import add_patient_groups, check_patient_leakage
from lungai.data.manifest import counts_by_split, load_manifest
from lungai.evaluation.historical import build


@pytest.fixture(scope="module")
def record():
    return build()


def _pct(x):
    return f"{100 * x:.2f}%"


def _read(rel):
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_committed_record_matches_a_fresh_recomputation(record):
    committed = json.loads(_read("artifacts/evaluation_reports/historical_record_verification.json"))
    fresh = {e["key"]: e["internal_test"]["recomputed"] for e in record["entries"]}
    for e in committed["entries"]:
        assert e["internal_test"]["recomputed"] == fresh[e["key"]]


def test_experiment_results_table_matches_record(record):
    doc = _read("docs/experiment_results.md")
    for e in record["entries"]:
        m = e["internal_test"]["recomputed"]
        ci = m["accuracy_ci95"]
        row = f"{_pct(m['accuracy'])} ({100 * ci['lower']:.2f}–{100 * ci['upper']:.2f})"
        assert row in doc, f"{e['key']}: {row}"
        assert _pct(m["macro_f1"]) in doc and _pct(m["weighted_f1"]) in doc
        b = e["external"]["binary_abnormality"]
        assert f"{b['tp']}/{b['tp'] + b['fn']}" in doc


def test_v5_history_quotes_the_canonical_record(record):
    canon = next(e for e in record["entries"] if e["canonical"])
    m = canon["internal_test"]["recomputed"]
    b = canon["external"]["binary_abnormality"]
    for rel in ("README.md", "docs/model_card.md"):
        doc = _read(rel)
        assert _pct(m["accuracy"]) in doc
        assert f"{b['tn']}/{b['tn'] + b['fp']}" in doc, "specificity must be stated with sensitivity"
    card = _read("docs/model_card.md")
    assert _pct(m["macro_f1"]) in card
    assert str(canon["internal_test"]["recorded_unverifiable"]["macro_roc_auc"]) in card


def test_generated_result_blocks_are_current():
    """The results blocks in the docs equal what render_docs produces from the current generated artifacts."""
    from lungai.evaluation import render_docs
    if not render_docs.METRICS.is_file():
        pytest.skip("no generated model_metrics.json")
    r = render_docs.load()
    if r["_run"] is None or r["_build"] is None:
        pytest.skip("run metadata or build report not on this machine (both are gitignored)")
    for rel, block in render_docs.TARGETS.items():
        doc = _read(rel).replace("\r\n", "\n")
        expected = render_docs.summary(r) if block == "summary" else render_docs.full(r)
        assert f"<!-- BEGIN GENERATED: {block} -->\n{expected}\n<!-- END GENERATED: {block} -->" in doc, rel
    assert render_docs.full(r) in _read("docs/results/current_model.md").replace("\r\n", "\n")


def test_split_counts_in_docs_match_manifest():
    df = load_manifest()
    counts = counts_by_split(df)
    grouped = add_patient_groups(df)
    leaked = check_patient_leakage(df)["by_derived_patient_group"]["test_images_sharing_key_with_train"]
    for rel in ("docs/data_splits.md",):  # the V5 split description (README now describes V6)
        doc = _read(rel)
        for split, c in counts.items():
            groups = grouped[grouped["split"] == split]["patient_group"].nunique()
            assert f"{c['images']:,}" in doc and f"{c['patient_ids']:,}" in doc and f"{groups:,}" in doc, (rel, split)
        assert f"{leaked}" in doc
