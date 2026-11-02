"""Canonical evaluation entry point.

Run the checkpoint on a split and save per-image predictions plus metrics:

    python -m lungai.evaluation.evaluate --split val  --checkpoint models/densenet121_frequency_v5.h5
    python -m lungai.evaluation.evaluate --split test --checkpoint models/densenet121_frequency_v5.h5 --final
    python -m lungai.evaluation.evaluate --split external --checkpoint models/densenet121_frequency_v5.h5 --final

Recompute metrics from a saved predictions file (no TensorFlow, no images needed):

    python -m lungai.evaluation.evaluate --predictions artifacts/evaluation_reports/<run>/predictions_test.csv

``test`` and ``external`` require ``--final``: they are reported once and never used for tuning.
Outputs go to ``artifacts/evaluation_reports/<model_id>_<split>_<UTC timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from lungai import __version__
from lungai.config import (
    CLASS_IDS, CLASS_NAMES, CLASS_TO_INDEX, DATASET, EVALUATION, EXTERNAL_COHORT, MODEL_DISPLAY_NAME, MODEL_ID,
    MODEL_VERSION, NORMAL_CLASS, NUM_CLASSES, REPO_ROOT, data_root, manifest_path, sha256_file,
)
from lungai.data.manifest import load_manifest, resolve_image_path, split_frame
from lungai.evaluation.metrics import (
    EvaluationInputError, binary_abnormality_report, multiclass_report, wilson_ci,
)

logger = logging.getLogger("lungai.evaluation")
OUT_ROOT = REPO_ROOT / "artifacts" / "evaluation_reports"
PROB_COLUMNS = [f"prob_{cid}" for cid in CLASS_IDS]
FINAL_SPLITS = ("test", "external")


# ─── Sample lists ─────────────────────────────────────────────────────────────

def internal_samples(split: str, root: Path) -> pd.DataFrame:
    df = split_frame(load_manifest(), split)
    return pd.DataFrame({
        "image_id": df["image_id"],
        "path": [resolve_image_path(p, root) for p in df["image_path"]],
        "true_label": df["clinical_label"],
        "source_dataset": df["source_dataset"],
    })


def external_samples(root: Path) -> pd.DataFrame:
    meta = root / EXTERNAL_COHORT["metadata_csv"]
    if not meta.is_file():
        raise FileNotFoundError(f"External cohort metadata not found under the data root: {EXTERNAL_COHORT['metadata_csv']}")
    m = pd.read_csv(meta)
    img_dir = root / EXTERNAL_COHORT["image_dir"]
    normal = m[EXTERNAL_COHORT["finding_column"]] == EXTERNAL_COHORT["normal_value"]
    return pd.DataFrame({
        "image_id": m[EXTERNAL_COHORT["id_column"]].astype(str),
        "path": [img_dir / s for s in m[EXTERNAL_COHORT["id_column"]].astype(str)],
        "true_label": np.where(normal, EXTERNAL_COHORT["labels"]["normal"], EXTERNAL_COHORT["labels"]["abnormal"]),
        "source_dataset": "Montgomery",
    })


def require_files(samples: pd.DataFrame) -> None:
    missing = [str(p) for p in samples["path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(samples)} images are missing (first: {missing[0]}). "
            "Set LUNGAI_DATA_ROOT / --data-root to the folder containing data/raw and data/downloads. "
            "Evaluation never skips missing images silently."
        )


# ─── Predictions file ─────────────────────────────────────────────────────────

def predictions_frame(samples: pd.DataFrame, probs: np.ndarray, split: str, checkpoint_sha: str) -> pd.DataFrame:
    if probs.shape != (len(samples), NUM_CLASSES):
        raise EvaluationInputError(f"model output shape {probs.shape} does not match ({len(samples)}, {NUM_CLASSES})")
    out = pd.DataFrame({
        "image_id": samples["image_id"].values,
        "split": split,
        "source_dataset": samples["source_dataset"].values,
        "true_label": samples["true_label"].values,
    })
    for col, values in zip(PROB_COLUMNS, probs.T):
        out[col] = values
    out["pred_index"] = probs.argmax(axis=1)
    out["pred_label"] = [CLASS_NAMES[i] for i in out["pred_index"]]
    out["checkpoint_sha256"] = checkpoint_sha
    return out


def check_predictions(pred: pd.DataFrame) -> str:
    """Validate a predictions file against the canonical schema and, for internal splits, the manifest."""
    required = ["image_id", "split", "true_label", "pred_index", "checkpoint_sha256", *PROB_COLUMNS]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise EvaluationInputError(f"predictions file is missing columns {missing}")
    splits = pred["split"].unique().tolist()
    if len(splits) != 1:
        raise EvaluationInputError(f"predictions file must contain exactly one split, found {splits}")
    if pred["image_id"].duplicated().any():
        raise EvaluationInputError("predictions file has duplicate image_id rows")
    if pred["checkpoint_sha256"].nunique() != 1:
        raise EvaluationInputError("predictions file mixes outputs from more than one checkpoint")
    probs = pred[PROB_COLUMNS].to_numpy(dtype=float)
    if not np.array_equal(pred["pred_index"].to_numpy(), probs.argmax(axis=1)):
        raise EvaluationInputError("pred_index does not equal argmax of the stored probabilities")
    split = splits[0]
    if split in ("train", "val", "test"):
        ref = split_frame(load_manifest(), split).set_index("image_id")["clinical_label"]
        ids = set(pred["image_id"])
        if ids != set(ref.index):
            raise EvaluationInputError(
                f"sample identities do not match manifest split '{split}': {len(ids - set(ref.index))} unknown, "
                f"{len(set(ref.index) - ids)} missing"
            )
        wrong = (pred.set_index("image_id")["true_label"] != ref.reindex(pred["image_id"])).sum()
        if wrong:
            raise EvaluationInputError(f"{wrong} true labels disagree with the manifest")
    unknown = set(pred["true_label"]) - set(CLASS_NAMES)
    if unknown:
        raise EvaluationInputError(f"true labels outside the canonical class mapping: {sorted(unknown)}")
    return split


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(pred: pd.DataFrame) -> Dict[str, Any]:
    split = check_predictions(pred)
    probs = pred[PROB_COLUMNS].to_numpy(dtype=float)
    y_true = pred["true_label"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    result: Dict[str, Any] = {"split": split, "n_evaluated": int(len(pred)), "n_excluded": 0}
    if split == "external":
        true_abn = (pred["true_label"] != NORMAL_CLASS).to_numpy()
        pred_abn = (pred["pred_label"] != NORMAL_CLASS).to_numpy()
        p_abnormal = 1.0 - probs[:, CLASS_TO_INDEX[NORMAL_CLASS]]
        tb = pred["true_label"] == "Tuberculosis"
        result["task"] = "external binary abnormality (TB vs normal); six-class metrics do not apply"
        result["binary_abnormality"] = binary_abnormality_report(true_abn, pred_abn, p_abnormal)
        result["exact_tuberculosis_recall"] = wilson_ci(int((pred.loc[tb, "pred_label"] == "Tuberculosis").sum()), int(tb.sum()))
        result["prediction_breakdown"] = {
            lbl: {k: int(v) for k, v in pred.loc[pred["true_label"] == lbl, "pred_label"].value_counts().items()}
            for lbl in sorted(pred["true_label"].unique())
        }
    else:
        result["task"] = "six-class single-label classification"
        result["metrics"] = multiclass_report(
            y_true, probs, CLASS_NAMES, n_bootstrap=EVALUATION["bootstrap_resamples"],
            seed=EVALUATION["bootstrap_seed"], calibration_bins=EVALUATION["calibration_bins"],
        )
        if "source_dataset" in pred.columns:
            result["per_source"] = {
                src: {
                    "n": int(len(g)),
                    "accuracy": float((g["pred_label"] == g["true_label"]).mean()),
                    "recall_by_class": {c: float((gc["pred_label"] == c).mean()) for c, gc in g.groupby("true_label")},
                }
                for src, g in pred.groupby("source_dataset")
            }
    return result


def report_header(split: str, checkpoint_sha: str, checkpoint_name: Optional[str]) -> Dict[str, Any]:
    try:
        import tensorflow as tf  # noqa: F401
        tf_version = tf.__version__
    except Exception:
        tf_version = None
    return {
        "report": "LungAI canonical evaluation",
        "model_id": MODEL_ID,
        "model_display_name": MODEL_DISPLAY_NAME,
        "model_version": MODEL_VERSION,
        "checkpoint_file": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha,
        "dataset_version": DATASET["version"] if split != "external" else EXTERNAL_COHORT["name"],
        "manifest": DATASET["manifest"],
        "manifest_sha256_lf": sha256_file(manifest_path(), normalize_newlines=True),
        "split": split,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software": {"lungai": __version__, "python": platform.python_version(), "tensorflow": tf_version,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "class_order": CLASS_NAMES,
        "metric_definitions": "see lungai/evaluation/metrics.py and docs/evaluation.md",
    }


def render_summary(report: Dict[str, Any]) -> str:
    L = [f"# Evaluation — {report['model_display_name']} ({report['split']})", "",
         f"- Checkpoint: `{report['checkpoint_file']}` sha256 `{report['checkpoint_sha256']}`",
         f"- Dataset: {report['dataset_version']}, manifest sha256 (LF) `{report['manifest_sha256_lf']}`",
         f"- Evaluated: {report['evaluated_at']}; samples evaluated {report['result']['n_evaluated']}, excluded {report['result']['n_excluded']}",
         ""]
    r = report["result"]
    if "metrics" in r:
        m = r["metrics"]
        b = m.get("bootstrap_ci95", {})
        acc_ci = b.get("accuracy", {})
        L += ["| Metric | Value | 95% CI |", "|---|---:|---|",
              f"| Accuracy | {m['accuracy']:.4f} | {acc_ci.get('lower', float('nan')):.4f}–{acc_ci.get('upper', float('nan')):.4f} (bootstrap) |",
              f"| Macro F1 | {m['macro_f1']:.4f} | {b.get('macro_f1', {}).get('lower', float('nan')):.4f}–{b.get('macro_f1', {}).get('upper', float('nan')):.4f} |",
              f"| Weighted F1 | {m['weighted_f1']:.4f} | |",
              f"| Macro ROC-AUC (OvR) | {m['macro_roc_auc_ovr']} | |",
              f"| Macro PR-AUC | {m['macro_pr_auc']} | |",
              f"| ECE ({m['calibration']['n_bins']} bins) | {m['calibration']['ece']:.4f} | |", "",
              "| Class | Support | Precision | Recall | Specificity | F1 | ROC-AUC |", "|---|---:|---:|---:|---:|---:|---:|"]
        for c, v in m["per_class"].items():
            L.append(f"| {c} | {v['support']} | {v['precision']:.4f} | {v['recall']:.4f} | {v['specificity']:.4f} | {v['f1']:.4f} | {m['per_class_roc_auc'][c]} |")
    else:
        bb = r["binary_abnormality"]
        s, sp = bb["sensitivity"], bb["specificity"]
        L += [f"- Task: {r['task']}",
              f"- Sensitivity (predicted ≠ Normal among TB): {s['successes']}/{s['n']} (95% CI {s['lower']:.3f}–{s['upper']:.3f})",
              f"- Specificity (predicted Normal among normal): {sp['successes']}/{sp['n']} (95% CI {sp['lower']:.3f}–{sp['upper']:.3f})",
              f"- ROC-AUC of P(abnormal) = 1 − P(Normal): {bb.get('roc_auc_p_abnormal')}",
              f"- Exact Tuberculosis recall: {r['exact_tuberculosis_recall']['successes']}/{r['exact_tuberculosis_recall']['n']}"]
    L.append("")
    return "\n".join(L)


def write_outputs(out_dir: Path, report: Dict[str, Any], pred: Optional[pd.DataFrame]) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    if pred is not None:
        pred.to_csv(out_dir / f"predictions_{report['split']}.csv", index=False)
    (out_dir / f"metrics_{report['split']}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / f"summary_{report['split']}.md").write_text(render_summary(report), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--split", choices=["val", "test", "external"])
    src.add_argument("--predictions", type=Path, help="recompute metrics from a saved predictions CSV")
    ap.add_argument("--checkpoint", type=Path, help="Keras checkpoint (required with --split)")
    ap.add_argument("--allow-unverified-checkpoint", action="store_true",
                    help="evaluate a checkpoint whose SHA-256 differs from the canonical one (reported as such)")
    ap.add_argument("--final", action="store_true", help="confirm a one-off final evaluation on test/external")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.predictions:
        pred = pd.read_csv(args.predictions)
        result = compute_metrics(pred)
        header = report_header(result["split"], str(pred["checkpoint_sha256"].iloc[0]), None)
        header["recomputed_from"] = args.predictions.name
        report = {**header, "result": result}
        out = args.out or OUT_ROOT / f"{MODEL_ID}_{result['split']}_recomputed_{stamp}"
        write_outputs(out, report, None)
        logger.info("wrote %s", out)
        return 0

    if args.split in FINAL_SPLITS and not args.final:
        ap.error(f"--split {args.split} is a final evaluation; pass --final to confirm it is not being used for tuning")
    if not args.checkpoint:
        ap.error("--checkpoint is required with --split")

    from lungai.data.loader import predict_probabilities
    from lungai.models.densenet import load_checkpoint

    root = args.data_root.resolve() if args.data_root else data_root()
    try:
        samples = external_samples(root) if args.split == "external" else internal_samples(args.split, root)
        require_files(samples)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 2
    allow = args.allow_unverified_checkpoint or os.getenv("LUNGAI_ALLOW_UNVERIFIED_CHECKPOINT", "").lower() == "true"
    model, sha, canonical = load_checkpoint(args.checkpoint, allow_unverified=allow)
    logger.info("evaluating %s on %s (%d images), checkpoint canonical=%s", MODEL_ID, args.split, len(samples), canonical)
    probs = predict_probabilities(model, list(samples["path"]), args.batch_size)
    pred = predictions_frame(samples, probs, args.split, sha)
    result = compute_metrics(pred)
    header = report_header(args.split, sha, args.checkpoint.name)
    header["checkpoint_is_canonical"] = canonical
    report = {**header, "result": result}
    out = args.out or OUT_ROOT / f"{MODEL_ID}_{args.split}_{stamp}"
    write_outputs(out, report, pred)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
