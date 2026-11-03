"""Compose the metrics the backend serves at /api/v1/model-metrics from generated evaluation reports.

    python -m lungai.evaluation.served_metrics --eval-prefix v6 --run runs/v6_densenet121_seed42

Reads artifacts/evaluation_reports/<prefix>_{val,test,external}/metrics_*.json and the run's
run_metadata.json; writes artifacts/evaluation_reports/model_metrics.json. Nothing is typed in by hand:
every number is copied from those files. The V5 record (historical_record_verification.json) is
attached as history, never as the current result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from lungai.config import (CHECKPOINT_FILENAME, CHECKPOINT_SHA256, CLASS_NAMES, MODEL_DISPLAY_NAME, MODEL_ID, MODEL_VERSION,
                           REPO_ROOT)

REPORTS = REPO_ROOT / "artifacts" / "evaluation_reports"
OUT = REPORTS / "model_metrics.json"
HISTORICAL = REPORTS / "historical_record_verification.json"

EXTERNAL_INTERPRETATION = (
    "Montgomery contains only normal and TB scans, so it tests whether the model separates abnormal from normal "
    "on an unseen hospital, not six-class accuracy. Read sensitivity together with specificity, and the exact "
    "Tuberculosis recall: a scan counted as 'abnormal' may have been given a wrong disease class."
)


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(eval_prefix: str, run_dir: Path, historical: Optional[Path] = HISTORICAL) -> Dict[str, Any]:
    val = _load(REPORTS / f"{eval_prefix}_val" / "metrics_val.json")
    test = _load(REPORTS / f"{eval_prefix}_test" / "metrics_test.json")
    ext = _load(REPORTS / f"{eval_prefix}_external" / "metrics_external.json")
    meta = _load(run_dir / "run_metadata.json")
    shas = {r["checkpoint_sha256"] for r in (val, test, ext)} | {meta["checkpoint_sha256"]}
    if len(shas) != 1:
        raise ValueError(f"evaluation reports and run metadata name different checkpoints: {sorted(shas)}")
    if test["class_order"] != CLASS_NAMES:
        raise ValueError("class order in the test report differs from the canonical class order")

    m = test["result"]["metrics"]
    per_class = {name: {**m["per_class"][name], "roc_auc": m["per_class_roc_auc"][name]} for name in CLASS_NAMES}
    vm = val["result"]["metrics"]
    sel = meta.get("selected") or {}
    out: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "python -m lungai.evaluation.served_metrics",
        # The configured version and file name when this is the configured checkpoint (a run is evaluated before it is
        # given a release version); otherwise the values recorded in the evaluation report.
        "canonical_model": {"model_id": MODEL_ID, "display_name": MODEL_DISPLAY_NAME,
                            "version": MODEL_VERSION if test["checkpoint_sha256"] == CHECKPOINT_SHA256 else test["model_version"],
                            "version_at_evaluation": test["model_version"],
                            "checkpoint_file": CHECKPOINT_FILENAME if test["checkpoint_sha256"] == CHECKPOINT_SHA256 else test["checkpoint_file"],
                            "checkpoint_sha256": test["checkpoint_sha256"],
                            "evaluated_as": f"runs/{meta['run_name']}/{test['checkpoint_file']}"},
        "evidence_level": (
            "REPRODUCED: computed by lungai.evaluation.evaluate_v6 from per-image predictions of this checkpoint "
            f"(saved under artifacts/evaluation_reports/{eval_prefix}_*). One training run (seed {meta['seed']}), "
            "CPU-only and time-boxed; "
            + ("run-to-run variation over further seeds of the same recipe and split is in "
               "artifacts/evaluation_reports/multi_seed_summary.json."
               if (REPORTS / "multi_seed_summary.json").is_file() else
               "no repeated seeds, so run-to-run variation is not measured.")),
        "dataset_version": test["dataset_version"],
        "build": test["build"],
        "split_fingerprint": test["split_fingerprint"],
        "class_order": CLASS_NAMES,
        "internal_test": {
            "split": "test", "n": m["n"], "consistency": "reproduced",
            "metrics": {
                "accuracy": m["accuracy"], "accuracy_ci95": m["accuracy_ci95"], "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"], "macro_recall": m["macro_recall"], "macro_precision": m["macro_precision"],
                "macro_roc_auc": m["macro_roc_auc_ovr"], "macro_pr_auc": m["macro_pr_auc"],
                "bootstrap_ci95": m["bootstrap_ci95"], "calibration": m["calibration"],
                "confusion_matrix": m["confusion_matrix"], "per_class": per_class,
            },
            "per_source": test["result"]["per_source"],
        },
        "external": {
            "cohort": "Montgomery County CXR (NLM)", "interpretation": EXTERNAL_INTERPRETATION,
            "n": ext["result"]["n_evaluated"],
            "binary_abnormality": ext["result"]["binary_abnormality"],
            "exact_tuberculosis_recall": ext["result"]["exact_tuberculosis_recall"],
            "prediction_breakdown": ext["result"]["prediction_breakdown"],
        },
        "validation_evidence": {
            "used_for": "checkpoint selection (best validation macro-F1, tie-break macro ROC-AUC)",
            "selected_epoch": sel.get("epoch"), "n": vm["n"], "accuracy": vm["accuracy"],
            "macro_f1": vm["macro_f1"], "macro_roc_auc": vm["macro_roc_auc_ovr"],
            "epochs_trained": len(meta.get("history", [])), "stop_reason": meta.get("stop_reason"),
        },
        "training": {"run_name": meta["run_name"], "seed": meta["seed"], "started_at": meta["started_at"],
                     "finished_at": meta["finished_at"], "train_images": meta.get("train_images"),
                     "environment": meta.get("environment")},
    }
    if historical is not None and historical.is_file():
        h = _load(historical)
        canon = next(e for e in h["entries"] if e["canonical"])
        out["historical"] = {
            "note": "Earlier model, kept for the record only. Its checkpoint is lost and its dataset (V5) had patient overlap "
                    "between training and test; these numbers are not comparable with the current model's.",
            "model": h["canonical_model"], "dataset_version": h["dataset_version"], "evidence_level": h["evidence_level"],
            "internal_test": {"n": canon["internal_test"]["n"], "metrics": canon["internal_test"]["recomputed"]},
            "external_binary_abnormality": canon["external"]["binary_abnormality"],
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-prefix", default="v6", help="evaluation folder prefix, e.g. v6 or v6_r2")
    ap.add_argument("--run", type=Path, required=True, help="training run folder containing run_metadata.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    data = build(args.eval_prefix, args.run if args.run.is_absolute() else REPO_ROOT / args.run)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} for checkpoint {data['canonical_model']['checkpoint_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
