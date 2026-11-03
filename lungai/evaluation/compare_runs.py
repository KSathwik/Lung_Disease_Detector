"""Compare two V6 runs on a common selection set, per docs/data/run_selection_protocol.md.

    python -m lungai.evaluation.compare_runs \
        --a v6_build_001 runs/v6_densenet121_seed42 --b v6_build_002 runs/v6r2_densenet121_seed42

The common set = validation images of either build that are in neither build's train or test split
(by MD5 and by patient key). Never reads test or external predictions. Writes
artifacts/evaluation_reports/run_comparison.json (+ per-image predictions CSV next to it).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from lungai.config import CLASS_NAMES, CLASS_TO_INDEX, NUM_CLASSES, REPO_ROOT, sha256_file

OUT = REPO_ROOT / "artifacts" / "evaluation_reports" / "run_comparison.json"
SHARED_SOURCES = ["nih_cxr14", "kermany", "shenzhen", "cohen_covid"]
COLS = ["image_id", "source_id", "source_patient_id", "patient_id_status", "lungai_label", "raw_md5", "raw_sha256", "split"]


def _manifest(build: Path) -> pd.DataFrame:
    d = pd.read_csv(build / "manifest_v6.csv", dtype=str, keep_default_na=False, usecols=COLS)
    known = d.patient_id_status.isin(["genuine", "derived_from_filename"])
    d["pk"] = d.source_id + "|" + d.source_patient_id.where(known, d.raw_sha256)
    d["cache"] = [build / "cache" / f"{h}.npy" for h in d.raw_sha256]
    return d


def common_set(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([a, b])
    banned = both[both.split.isin(["train", "test"])]
    val = both[both.split == "val"]
    labels = val.groupby("raw_md5").lungai_label.nunique()
    val = val[~val.raw_md5.isin(labels[labels > 1].index)]  # rule 3: same label in both builds
    val = val.drop_duplicates("raw_md5")
    c = val[~val.raw_md5.isin(set(banned.raw_md5)) & ~val.pk.isin(set(banned.pk))].reset_index(drop=True)
    c["y"] = c.lungai_label.map(CLASS_TO_INDEX).astype(int)
    return c


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import f1_score, roc_auc_score, recall_score
    pred = p.argmax(1)
    present = sorted(set(y.tolist()))
    out = {"n": int(len(y)), "accuracy": float((pred == y).mean()),
           "macro_f1": float(f1_score(y, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)),
           "recall_by_class": {CLASS_NAMES[k]: float(v) for k, v in
                               zip(range(NUM_CLASSES), recall_score(y, pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0))}}
    if len(present) == NUM_CLASSES:
        out["macro_roc_auc"] = float(roc_auc_score(np.eye(NUM_CLASSES)[y], p, average="macro", multi_class="ovr"))
    return out


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, float]:
    from scipy.stats import binomtest
    b = int((correct_a & ~correct_b).sum())  # A right, B wrong
    c = int((~correct_a & correct_b).sum())  # A wrong, B right
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value_exact": float(p)}


def decide(ma: Dict[str, float], mb: Dict[str, float]) -> str:
    """Decision rule of the protocol: higher macro-F1; differences < 0.01 keep A unless B's ROC-AUC is ≥ 0.01 higher."""
    d = mb["macro_f1"] - ma["macro_f1"]
    if d >= 0.01:
        return "b"
    if d <= -0.01:
        return "a"
    return "b" if mb.get("macro_roc_auc", 0) - ma.get("macro_roc_auc", 0) >= 0.01 else "a"


def main(argv: Optional[List[str]] = None) -> int:
    import tensorflow as tf  # noqa: F401  (loaded only here)
    from lungai.models.densenet import load_checkpoint
    from lungai.training.train_v6 import predict_cached

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs=2, metavar=("BUILD", "RUN"), required=True)
    ap.add_argument("--b", nargs=2, metavar=("BUILD", "RUN"), required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    builds = {k: REPO_ROOT / "data" / "derived" / v[0] for k, v in (("a", args.a), ("b", args.b))}
    runs = {k: REPO_ROOT / v[1] for k, v in (("a", args.a), ("b", args.b))}
    ma_, mb_ = _manifest(builds["a"]), _manifest(builds["b"])
    c = common_set(ma_, mb_)
    missing = [str(p) for p in c.cache if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} cached images missing (first {missing[0]})")
    pool = ThreadPoolExecutor(max_workers=6)
    probs, res = {}, {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "protocol": "docs/data/run_selection_protocol.md", "n_common": int(len(c)),
                      "common_by_class": c.lungai_label.value_counts().to_dict(), "common_by_source": c.source_id.value_counts().to_dict()}
    for k in ("a", "b"):
        ckpt = runs[k] / "model.h5"
        model, _, _ = load_checkpoint(ckpt, allow_unverified=True)
        probs[k] = predict_cached(model, c.cache.tolist(), pool)
        res[k] = {"build": builds[k].name, "run": runs[k].name, "checkpoint_sha256": sha256_file(ckpt),
                  "all": _metrics(c.y.to_numpy(), probs[k]),
                  "shared_sources_only": _metrics(c.y.to_numpy()[c.source_id.isin(SHARED_SOURCES)], probs[k][c.source_id.isin(SHARED_SOURCES).to_numpy()])}
        del model
    pool.shutdown()
    y = c.y.to_numpy()
    res["mcnemar_all"] = mcnemar_exact(probs["a"].argmax(1) == y, probs["b"].argmax(1) == y)
    res["decision"] = decide(res["a"]["all"], res["b"]["all"])
    res["selected_run"] = res[res["decision"]]["run"]
    pred = c[["image_id", "source_id", "lungai_label"]].copy()
    for k in ("a", "b"):
        for i, name in enumerate(CLASS_NAMES):
            pred[f"{k}_prob_{i}"] = probs[k][:, i]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out.with_name("run_comparison_predictions.csv"), index=False)
    args.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_common", "decision", "selected_run", "mcnemar_all")}, indent=2))
    for k in ("a", "b"):
        print(k, res[k]["run"], {m: round(v, 4) for m, v in res[k]["all"].items() if isinstance(v, float)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
