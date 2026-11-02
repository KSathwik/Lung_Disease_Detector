"""Verify the recorded (historical) evaluation results against their own evidence.

    python -m lungai.evaluation.historical

The repository holds aggregate result files from earlier experiments but no per-image
predictions and no checkpoints. This script recomputes every threshold metric from the
stored confusion matrices, checks it against the stored value, recomputes the external
binary assessment from the stored prediction breakdown, and marks what cannot be
verified (ROC-AUC, PR-AUC, calibration). Output:

    artifacts/evaluation_reports/historical_record_verification.json
    artifacts/evaluation_reports/historical_record_verification.md

These are *recorded* results, not results of the canonical evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from lungai import __version__
from lungai.config import CLASS_NAMES, MODEL_DISPLAY_NAME, MODEL_ID, MODEL_VERSION, REPO_ROOT, sha256_file
from lungai.evaluation.metrics import binary_abnormality_report, metrics_from_confusion, wilson_ci

OUT_DIR = REPO_ROOT / "artifacts" / "evaluation_reports"
TOL = 6e-5  # recorded values are rounded to 4 decimals

# Recorded experiments on the V5 manifest. "canonical" marks the run whose architecture and
# preprocessing are the canonical model's (lungai/configs/lungai_densenet121.json).
RECORDS: List[Dict[str, Any]] = [
    {"key": "canonical_v5", "label": "Canonical (DenseNet-121 + Gaussian low-pass preprocessing)", "canonical": True,
     "metrics": "experiments/densenet_frequency_v5/frequency_metrics.json",
     "external": "experiments/densenet_frequency_v5/frequency_montgomery.json"},
    {"key": "erm_baseline_v5", "label": "Historical: DenseNet-121 ERM baseline (no low-pass filter)", "canonical": False,
     "metrics": "experiments/results/densenet_v5_metrics.json",
     "external": "experiments/results/densenet_v5_montgomery.json"},
    {"key": "coral_v5", "label": "Historical: DenseNet-121 + Deep CORAL", "canonical": False,
     "metrics": "experiments/densenet_coral_v5/coral_metrics.json",
     "external": "experiments/densenet_coral_v5/coral_montgomery.json"},
    {"key": "dann_v5", "label": "Historical: DenseNet-121 + DANN", "canonical": False,
     "metrics": "experiments/densenet_dann_v5/dann_metrics.json",
     "external": "experiments/densenet_dann_v5/dann_montgomery.json"},
    {"key": "hybrid_v5", "label": "Historical: DenseNet-121 + low-pass + CORAL", "canonical": False,
     "metrics": "experiments/densenet_hybrid_v5/hybrid_metrics.json",
     "external": "experiments/densenet_hybrid_v5/hybrid_montgomery.json"},
]

# Validation-set evidence recorded for each paradigm (the only data model selection may use).
VALIDATION_EVIDENCE = {
    "canonical_v5": {
        "source": "experiments/densenet_frequency_v5/frequency_validation_grid.json",
        "what": "Filter chosen on validation: the ERM baseline checkpoint evaluated on val under 10 preprocessing "
                "filters; Gaussian sigma=1.0 had the highest val macro-F1 (0.8457 vs 0.7495 unfiltered). The trained "
                "model's own validation score was logged but not saved.",
    },
    "erm_baseline_v5": {"source": "experiments/densenet_v5/training_history.json",
                        "what": "Final-epoch val accuracy 0.7771 (epoch before: 0.8056)."},
    "coral_v5": {"source": "experiments/densenet_coral_v5/coral_lambda_search.json",
                 "what": "Selected lambda 0.01: val accuracy 0.8151, val macro-F1 0.7771."},
    "dann_v5": {"source": "experiments/densenet_dann_v5/dann_lambda_search.json",
                "what": "Selected lambda 0.1: val accuracy 0.7688, val macro-F1 0.7397."},
    "hybrid_v5": {"source": "experiments/densenet_hybrid_v5/hybrid_config.json",
                  "what": "No validation score recorded."},
}


def _load(rel: str) -> Optional[Dict[str, Any]]:
    p = REPO_ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _compare(name: str, recorded: Optional[float], recomputed: Optional[float]) -> Dict[str, Any]:
    if recorded is None:
        return {"metric": name, "recorded": None, "recomputed": recomputed, "status": "not recorded"}
    ok = recomputed is not None and abs(recorded - recomputed) <= TOL
    return {"metric": name, "recorded": recorded, "recomputed": round(recomputed, 6) if recomputed is not None else None,
            "status": "consistent" if ok else "MISMATCH"}


def verify_internal(rec: Dict[str, Any]) -> Dict[str, Any]:
    data = _load(rec["metrics"])
    if data is None:
        return {"status": "missing", "file": rec["metrics"]}
    order = list(data["per_class_metrics"].keys())
    if order != CLASS_NAMES:
        return {"status": "class order differs from canonical", "file": rec["metrics"], "order": order}
    m = metrics_from_confusion(data["confusion_matrix"], CLASS_NAMES)
    om = data["overall_metrics"]
    checks = [
        _compare("accuracy", om.get("accuracy"), m["accuracy"]),
        _compare("macro_precision", om.get("macro_precision"), m["macro_precision"]),
        _compare("macro_recall", om.get("macro_recall"), m["macro_recall"]),
        _compare("macro_f1", om.get("macro_f1"), m["macro_f1"]),
        _compare("weighted_f1", om.get("weighted_f1"), m["weighted_f1"]),
    ]
    for cls in CLASS_NAMES:
        pc, rc = data["per_class_metrics"][cls], m["per_class"][cls]
        checks += [
            _compare(f"{cls} precision", pc.get("precision"), rc["precision"]),
            _compare(f"{cls} recall", pc.get("recall"), rc["recall"]),
            _compare(f"{cls} f1", pc.get("f1_score"), rc["f1"]),
            _compare(f"{cls} specificity", pc.get("specificity"), rc["specificity"]),
        ]
        if pc.get("support") != rc["support"]:
            checks.append({"metric": f"{cls} support", "recorded": pc.get("support"), "recomputed": rc["support"], "status": "MISMATCH"})
    mismatches = [c for c in checks if c["status"] == "MISMATCH"]
    return {
        "status": "consistent" if not mismatches else "inconsistent",
        "file": rec["metrics"],
        "file_sha256": sha256_file(REPO_ROOT / rec["metrics"]),
        "split": "test",
        "n": m["n"],
        "recomputed": m,
        "recorded_unverifiable": {
            "macro_roc_auc": om.get("macro_roc_auc"),
            "macro_pr_auc": om.get("macro_pr_auc"),
            "calibration_confidence": data.get("calibration_confidence"),
            "reason": "needs per-image probabilities, which were never saved",
        },
        "checks": checks,
        "mismatches": mismatches,
    }


def verify_external(rec: Dict[str, Any]) -> Dict[str, Any]:
    data = _load(rec["external"])
    if data is None:
        return {"status": "missing", "file": rec["external"]}
    tb = data.get("tb_predictions_breakdown", {})
    nm = data.get("normal_predictions_breakdown", {})
    n_tb, n_norm = int(sum(tb.values())), int(sum(nm.values()))
    true_abn = [True] * n_tb + [False] * n_norm
    pred_abn = [k != "Normal" for k, v in tb.items() for _ in range(v)] + [k != "Normal" for k, v in nm.items() for _ in range(v)]
    binary = binary_abnormality_report(true_abn, pred_abn)
    exact_tb = int(tb.get("Tuberculosis", 0))
    exact_norm = int(nm.get("Normal", 0))
    counts_ok = (n_tb == data.get("tb_cases") and n_norm == data.get("normal_cases") and n_tb + n_norm == data.get("total_scans"))
    sens_ok = abs(binary["sensitivity"]["estimate"] - data.get("binary_abnormal_sensitivity", -1)) <= TOL
    return {
        "status": "consistent" if counts_ok and sens_ok else "inconsistent",
        "file": rec["external"],
        "cohort": "Montgomery County (external, binary TB vs normal)",
        "recorded_counts": {"total": data.get("total_scans"), "tb": data.get("tb_cases"), "normal": data.get("normal_cases")},
        "prediction_breakdown": {"tb_cases": tb, "normal_cases": nm},
        "binary_abnormality": binary,
        "exact_tuberculosis_recall": wilson_ci(exact_tb, n_tb),
        "exact_normal_recall": wilson_ci(exact_norm, n_norm),
        "interpretation": _interpret(binary),
        "verified_against_images": False,
    }


def _interpret(b: Dict[str, Any]) -> str:
    if b["fp"] and b["tn"] == 0 and b["fn"] == 0:
        return ("Degenerate: every scan, including every normal scan, was predicted abnormal. Sensitivity "
                f"{b['tp']}/{b['tp'] + b['fn']} carries no screening value with specificity 0/{b['tn'] + b['fp']}.")
    if b["tn"] == 0:
        return f"No normal scan was classified Normal (specificity 0/{b['fp']}); the model does not separate normal from TB scans."
    return "Read sensitivity and specificity together."


def build() -> Dict[str, Any]:
    entries = []
    for rec in RECORDS:
        entries.append({
            "key": rec["key"], "label": rec["label"], "canonical": rec["canonical"],
            "internal_test": verify_internal(rec),
            "external": verify_external(rec),
            "validation_evidence": VALIDATION_EVIDENCE[rec["key"]],
        })
    return {
        "report": "Verification of recorded (historical) evaluation results",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lungai_version": __version__,
        "canonical_model": {"model_id": MODEL_ID, "display_name": MODEL_DISPLAY_NAME, "version": MODEL_VERSION},
        "evidence_level": (
            "RECORDED: aggregate results from earlier runs. Threshold metrics are checked for internal consistency "
            "against the stored confusion matrices. They cannot be reproduced: no checkpoint, image or per-image "
            "prediction is in the repository. The internal test split has known patient leakage "
            "(see artifacts/data_validation)."
        ),
        "dataset_version": "V5",
        "class_order": CLASS_NAMES,
        "entries": entries,
    }


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:.2f}%"


def render_markdown(r: Dict[str, Any]) -> str:
    L = [
        "# Recorded evaluation results — verification",
        "",
        f"- Generated: {r['generated_at']} by `python -m lungai.evaluation.historical` (lungai {r['lungai_version']}). Do not edit by hand.",
        f"- Evidence level: {r['evidence_level']}",
        "",
        "## Internal test split (V5, N = 1,570) — recomputed from stored confusion matrices",
        "",
        "| Run | Accuracy (95% CI) | Macro P | Macro R | Macro F1 | Weighted F1 | Macro ROC-AUC (recorded, unverifiable) | Consistency |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for e in r["entries"]:
        it = e["internal_test"]
        if "recomputed" not in it:
            L.append(f"| {e['label']} | missing | | | | | | {it['status']} |")
            continue
        m = it["recomputed"]
        ci = m["accuracy_ci95"]
        auc = it["recorded_unverifiable"]["macro_roc_auc"]
        L.append(
            f"| {'**' + e['label'] + '**' if e['canonical'] else e['label']} | {_pct(m['accuracy'])} ({_pct(ci['lower'])}–{_pct(ci['upper'])}) "
            f"| {_pct(m['macro_precision'])} | {_pct(m['macro_recall'])} | {_pct(m['macro_f1'])} | {_pct(m['weighted_f1'])} "
            f"| {auc if auc is not None else '—'} | {it['status']} ({len(it['mismatches'])} mismatches / {len(it['checks'])} checks) |"
        )
    canon = next(e for e in r["entries"] if e["canonical"])
    L += ["", f"## Canonical run — per class (internal test)", "",
          "| Class | Support | Precision | Recall (95% CI) | Specificity | F1 |", "|---|---:|---:|---|---:|---:|"]
    for cls, v in canon["internal_test"]["recomputed"]["per_class"].items():
        ci = v["recall_ci95"]
        L.append(f"| {cls} | {v['support']} | {v['precision']:.4f} | {v['recall']:.4f} ({ci['lower']:.3f}–{ci['upper']:.3f}) | {v['specificity']:.4f} | {v['f1']:.4f} |")
    L += ["", "Confusion matrix (rows = true, columns = predicted, canonical class order):", "", "```"]
    for cls, row in zip(CLASS_NAMES, canon["internal_test"]["recomputed"]["confusion_matrix"]):
        L.append(f"{cls:<24} {row}")
    L += ["```", "", "## External cohort (Montgomery County, binary) — recomputed from stored prediction breakdowns", "",
          "| Run | N (TB / normal) | Sensitivity, predicted ≠ Normal (95% CI) | Specificity (95% CI) | Exact TB recall | Interpretation |",
          "|---|---|---|---|---|---|"]
    for e in r["entries"]:
        ex = e["external"]
        if "binary_abnormality" not in ex:
            L.append(f"| {e['label']} | missing | | | | |")
            continue
        b = ex["binary_abnormality"]
        s, sp, t = b["sensitivity"], b["specificity"], ex["exact_tuberculosis_recall"]
        L.append(
            f"| {e['label']} | {b['tp'] + b['fn'] + b['tn'] + b['fp']} ({b['tp'] + b['fn']} / {b['tn'] + b['fp']}) "
            f"| {s['successes']}/{s['n']} = {_pct(s['estimate'])} ({_pct(s['lower'])}–{_pct(s['upper'])}) "
            f"| {sp['successes']}/{sp['n']} = {_pct(sp['estimate'])} ({_pct(sp['lower'])}–{_pct(sp['upper'])}) "
            f"| {t['successes']}/{t['n']} | {ex['interpretation']} |"
        )
    L += ["", "## Validation-set evidence available for model selection", "", "| Run | Source | Evidence |", "|---|---|---|"]
    for e in r["entries"]:
        v = e["validation_evidence"]
        L.append(f"| {e['label']} | `{v['source']}` | {v['what']} |")
    L.append("")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    report = build()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "historical_record_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out / "historical_record_verification.md").write_text(render_markdown(report), encoding="utf-8")
    bad = [e["key"] for e in report["entries"] if e["internal_test"].get("status") != "consistent"]
    print(f"wrote {args.out} ; inconsistent records: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
