"""Seed-to-seed variability of one training recipe, and the preprocessing ablation, from saved evaluation reports.

    python -m lungai.evaluation.aggregate_seeds --seeds v6_r2:runs/v6r2_densenet121_seed42 v6_r3:runs/... v6_r4:runs/... \
        [--ablation v6_r5:runs/...]

Each item is <evaluation prefix>:<run dir>. Reads <prefix>_{val,test,external}/metrics_*.json and run_metadata.json,
checks that all seed runs use the same build (split fingerprint), and writes
artifacts/evaluation_reports/multi_seed_summary.json with every run's values plus mean, sample SD, min and max.
The ablation run is reported separately and never pooled with the seeds.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from lungai.config import CLASS_NAMES, REPO_ROOT

REPORTS = REPO_ROOT / "artifacts" / "evaluation_reports"
OUT = REPORTS / "multi_seed_summary.json"


def run_values(prefix: str, run: Path) -> Dict:
    def load(split):
        return json.loads((REPORTS / f"{prefix}_{split}" / f"metrics_{split}.json").read_text(encoding="utf-8"))
    val, test, ext = load("val"), load("test"), load("external")
    meta = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    m, b = test["result"]["metrics"], ext["result"]["binary_abnormality"]
    v = {"test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"], "test_macro_roc_auc": m["macro_roc_auc_ovr"],
         "test_ece": m["calibration"]["ece"], "val_macro_f1": val["result"]["metrics"]["macro_f1"],
         "ext_sensitivity": b["sensitivity"]["estimate"], "ext_specificity": b["specificity"]["estimate"],
         "ext_auc": b["roc_auc_p_abnormal"], "ext_exact_tb_recall": ext["result"]["exact_tuberculosis_recall"]["estimate"]}
    v.update({f"test_f1:{c}": m["per_class"][c]["f1"] for c in CLASS_NAMES})
    return {"prefix": prefix, "run": run.name, "seed": meta["seed"], "build": meta["build"],
            "split_fingerprint": meta["split_fingerprint"], "checkpoint_sha256": meta["checkpoint_sha256"],
            "selected_epoch": (meta.get("selected") or {}).get("epoch"), "epochs": len(meta.get("history", [])),
            "stop_reason": meta.get("stop_reason"), "values": v}


def summarise(runs: List[Dict]) -> Dict:
    keys = runs[0]["values"].keys()
    out = {}
    for k in keys:
        a = np.array([r["values"][k] for r in runs], dtype=float)
        out[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
                  "min": float(a.min()), "max": float(a.max()), "n_runs": int(len(a))}
    return out


def _parse(item: str):
    prefix, run = item.split(":", 1)
    run = Path(run)
    return prefix, run if run.is_absolute() else REPO_ROOT / run


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", required=True, help="prefix:run_dir for each seed run of the same recipe")
    ap.add_argument("--ablation", nargs="*", default=[], help="prefix:run_dir for ablation runs (reported separately)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    seeds = [run_values(*_parse(s)) for s in args.seeds]
    fps = {r["split_fingerprint"] for r in seeds}
    if len(fps) != 1:
        raise ValueError(f"seed runs use different splits: {fps}")
    if len({r["seed"] for r in seeds}) != len(seeds):
        raise ValueError("two runs share a seed")
    res = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "build": seeds[0]["build"],
           "split_fingerprint": fps.pop(), "runs": seeds, "summary": summarise(seeds),
           "note": "SD is the sample standard deviation over seeds (ddof=1); with few seeds it is a rough estimate.",
           "ablation": [run_values(*_parse(a)) for a in args.ablation]}
    args.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for k in ("test_accuracy", "test_macro_f1", "test_macro_roc_auc", "ext_specificity", "ext_exact_tb_recall"):
        s = res["summary"][k]
        print(f"{k:22s} mean {s['mean']:.4f} sd {s['sd']} [{s['min']:.4f}, {s['max']:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
