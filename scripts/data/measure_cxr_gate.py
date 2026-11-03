"""How often does the backend's CXR gate reject real chest X-rays? Runs on a V6 build's test split and external cohort.

    python scripts/data/measure_cxr_gate.py --build data/derived/v6_build_001

Writes artifacts/app_checks/cxr_gate_rejection_<build>.json (counts per split, source and label; rejection reasons).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "backend"), str(REPO)]

import pandas as pd  # noqa: E402
from ml.cxr_gate import CXRGate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--heuristic-threshold", type=float, default=None,
                    help="override the configured heuristic threshold (e.g. 0.83 to measure the old setting on the same split)")
    args = ap.parse_args()
    gate = CXRGate(heuristic_threshold=args.heuristic_threshold)
    m = pd.read_csv(args.build / "manifest_v6.csv", dtype=str, keep_default_na=False)
    tot, rej, reasons = Counter(), Counter(), Counter()
    for _, r in m[m.split.isin(["test", "external"])].iterrows():
        ok, _score, msg = gate.validate((args.raw_root / r.raw_relpath).read_bytes())
        key = (r.split, r.source_id, r.lungai_label)
        tot[key] += 1
        if not ok:
            rej[key] += 1
            reasons[msg.split(":")[0][:60]] += 1
    out = {"build": args.build.name, "threshold": gate.threshold, "heuristic_threshold": gate.heuristic_threshold,
           "by_group": [{"split": k[0], "source": k[1], "label": k[2], "n": tot[k], "rejected": rej[k]} for k in sorted(tot)],
           "total": {s: {"n": sum(v for k, v in tot.items() if k[0] == s), "rejected": sum(v for k, v in rej.items() if k[0] == s)}
                     for s in ("test", "external")},
           "reasons": dict(reasons)}
    path = args.out or REPO / "artifacts" / "app_checks" / f"cxr_gate_rejection_{args.build.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["total"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
