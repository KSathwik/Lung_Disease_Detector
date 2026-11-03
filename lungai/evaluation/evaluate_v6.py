"""Evaluate a checkpoint on a dataset V6 build split (val, test, or the external cohort).

    python -m lungai.evaluation.evaluate_v6 --build data/derived/v6_build_001 --checkpoint runs/<name>/model.h5 --split val
    python -m lungai.evaluation.evaluate_v6 ... --split test --final
    python -m lungai.evaluation.evaluate_v6 ... --split external --final

Writes artifacts/evaluation_reports/<model_id>_v6_<split>_<UTC>/ with per-image predictions, metrics JSON and summary.
Uses the same metric definitions as lungai.evaluation.evaluate (lungai/evaluation/metrics.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from lungai import __version__
from lungai.config import CLASS_IDS, CLASS_NAMES, CLASS_TO_INDEX, EVALUATION, MODEL_DISPLAY_NAME, MODEL_ID, NORMAL_CLASS, REPO_ROOT, sha256_file
from lungai.data.v6 import split_fingerprint
from lungai.evaluation.evaluate import render_summary
from lungai.evaluation.metrics import binary_abnormality_report, multiclass_report, wilson_ci
from lungai.models.densenet import load_checkpoint
from lungai.training.train_v6 import load_split, predict_cached

logger = logging.getLogger("lungai.evaluation.v6")
OUT_ROOT = REPO_ROOT / "artifacts" / "evaluation_reports"


def evaluate(build: Path, checkpoint: Path, split: str) -> dict:
    df = load_split(build, split)
    model, sha, canonical = load_checkpoint(checkpoint, allow_unverified=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        probs = predict_cached(model, df["cache"].tolist(), pool)
    pred = df[["image_id", "source_id", "lungai_label"]].rename(columns={"lungai_label": "true_label"}).copy()
    for cid, col in zip(CLASS_IDS, probs.T):
        pred[f"prob_{cid}"] = col
    pred["pred_index"] = probs.argmax(1)
    pred["pred_label"] = [CLASS_NAMES[i] for i in pred["pred_index"]]
    pred["checkpoint_sha256"] = sha
    y = pred["true_label"].map(CLASS_TO_INDEX).to_numpy(np.int64)
    result = {"split": split, "n_evaluated": int(len(pred)), "n_excluded": 0}
    if split == "external":
        true_abn = (pred["true_label"] != NORMAL_CLASS).to_numpy()
        pred_abn = (pred["pred_label"] != NORMAL_CLASS).to_numpy()
        tb = pred["true_label"] == "Tuberculosis"
        result.update(task="external binary abnormality (TB vs normal); six-class metrics do not apply",
                      binary_abnormality=binary_abnormality_report(true_abn, pred_abn, 1.0 - probs[:, CLASS_TO_INDEX[NORMAL_CLASS]]),
                      exact_tuberculosis_recall=wilson_ci(int((pred.loc[tb, "pred_label"] == "Tuberculosis").sum()), int(tb.sum())),
                      prediction_breakdown={lab: {k: int(v) for k, v in g["pred_label"].value_counts().items()}
                                            for lab, g in pred.groupby("true_label")})
    else:
        result.update(task="six-class single-label classification",
                      metrics=multiclass_report(y, probs, CLASS_NAMES, n_bootstrap=EVALUATION["bootstrap_resamples"],
                                                seed=EVALUATION["bootstrap_seed"], calibration_bins=EVALUATION["calibration_bins"]),
                      per_source={src: {"n": int(len(g)), "accuracy": float((g["pred_label"] == g["true_label"]).mean()),
                                        "recall_by_class": {c: float((gc["pred_label"] == c).mean()) for c, gc in g.groupby("true_label")}}
                                  for src, g in pred.groupby("source_id")})
    manifest = pd.read_csv(build / "manifest_v6.csv", dtype=str, keep_default_na=False)
    import tensorflow as tf
    header = {
        "report": "LungAI canonical evaluation (dataset V6)", "model_id": MODEL_ID, "model_display_name": MODEL_DISPLAY_NAME,
        "model_version": "v6.0", "checkpoint_file": checkpoint.name, "checkpoint_sha256": sha,
        "dataset_version": "V6" if split != "external" else "Montgomery County (external)", "build": build.name,
        "manifest": f"{build.name}/manifest_v6.csv", "manifest_sha256_lf": sha256_file(build / "manifest_v6.csv"),
        "split_fingerprint": split_fingerprint(manifest[manifest["split"].isin(["train", "val", "test"])]),
        "split": split, "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software": {"lungai": __version__, "python": platform.python_version(), "tensorflow": tf.__version__, "numpy": np.__version__},
        "class_order": CLASS_NAMES, "metric_definitions": "lungai/evaluation/metrics.py; docs/evaluation.md",
    }
    return {"report": {**header, "result": result}, "predictions": pred}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test", "external"], required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.split in ("test", "external") and not args.final:
        ap.error("test/external are one-off final evaluations; pass --final")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = evaluate(args.build, args.checkpoint, args.split)
    d = args.out or OUT_ROOT / f"{MODEL_ID}_v6_{args.split}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    d.mkdir(parents=True, exist_ok=False)
    out["predictions"].to_csv(d / f"predictions_{args.split}.csv", index=False)
    (d / f"metrics_{args.split}.json").write_text(json.dumps(out["report"], indent=2) + "\n", encoding="utf-8")
    (d / f"summary_{args.split}.md").write_text(render_summary(out["report"]), encoding="utf-8")
    logger.info("wrote %s", d)
    print(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
