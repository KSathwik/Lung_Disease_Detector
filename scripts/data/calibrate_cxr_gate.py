"""Choose the threshold of the CXR gate's heuristic fallback on VALIDATION radiographs.

    python scripts/data/calibrate_cxr_gate.py --build data/derived/v6_build_002 [--target-pass 0.98]

The heuristic score (backend/ml/cxr_gate.py::_anatomical_profile_score) is on a different scale from the
trained gate model's probability, so it needs its own threshold. This script scores every validation image
of a V6 build that passes stage 1, takes the (1 - target_pass) quantile as the threshold, and reports the
scores of the synthetic non-radiograph images used by backend/tests/test_cxr_gate.py. No real non-radiograph
set is available, so the rejection side is only checked on those synthetic images.

Writes artifacts/app_checks/cxr_gate_calibration_<build>.json. Test and external splits are not read.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "backend"), str(REPO)]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from ml.cxr_gate import CXRGate  # noqa: E402


def _negatives() -> dict:
    from tests import test_cxr_gate as t  # synthetic negatives shared with the API tests
    return {"car_color": t._make_car_color_bytes(), "car_grayscale": t._make_car_grayscale_bytes(), "dog": t._make_dog_bytes(),
            "selfie": t._make_selfie_bytes(), "landscape": t._make_landscape_bytes(), "screenshot": t._make_screenshot_bytes(),
            "ct": t._get_ct_scan_bytes()}


def score(gate: CXRGate, data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    ok1, _ = gate.validate_stage1_heuristics(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return ok1, gate._anatomical_profile_score(gray)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--target-pass", type=float, default=0.98)
    args = ap.parse_args()
    gate = CXRGate()
    m = pd.read_csv(args.build / "manifest_v6.csv", dtype=str, keep_default_na=False)
    val = m[m.split == "val"]
    rows = []
    for _, r in val.iterrows():
        ok1, s = score(gate, (args.raw_root / r.raw_relpath).read_bytes())
        rows.append({"source": r.source_id, "label": r.lungai_label, "stage1": ok1, "score": s})
    d = pd.DataFrame(rows)
    passed1 = d[d.stage1]
    thr_quantile = float(np.floor(passed1.score.quantile(1 - args.target_pass) * 1000) / 1000)  # round down to 3 d.p.
    neg = {k: dict(zip(("stage1", "score"), score(gate, v))) for k, v in _negatives().items()}
    # Rule used: the lowest threshold that still rejects every available non-radiograph (0.01 margin above the highest
    # negative that passes stage 1); the quantile rule alone lets negatives through when the score ranges overlap.
    neg_max = max((v["score"] for v in neg.values() if v["stage1"]), default=0.0)
    thr = max(thr_quantile, float(np.ceil((neg_max + 0.01) * 1000) / 1000))
    grid = [round(x, 2) for x in np.arange(0.50, 0.86, 0.02)]
    out = {
        "build": args.build.name, "split": "val", "n": len(d), "stage1_rejected": int((~d.stage1).sum()),
        "rule": "max(quantile threshold for target pass rate, highest stage-1-passing negative score + 0.01)",
        "target_pass_rate": args.target_pass, "threshold_from_quantile": thr_quantile,
        "highest_negative_score_passing_stage1": round(neg_max, 4), "heuristic_threshold": thr,
        "val_pass_rate_by_threshold": {str(t): round(float(((d.stage1) & (d.score >= t)).mean()), 4) for t in grid},
        "val_pass_rate_at_threshold": float(((d.stage1) & (d.score >= thr)).mean()),
        "val_pass_rate_at_0.83": float(((d.stage1) & (d.score >= 0.83)).mean()),
        "val_score_quantiles": {str(q): round(float(passed1.score.quantile(q)), 4) for q in (0.01, 0.02, 0.05, 0.25, 0.5, 0.75)},
        "by_source": {s: {"n": int(len(g)), "pass_at_threshold": round(float(((g.stage1) & (g.score >= thr)).mean()), 4)}
                      for s, g in d.groupby("source")},
        "synthetic_negatives": {k: {"stage1_pass": bool(v["stage1"]), "score": round(float(v["score"]), 4),
                                    "rejected_at_threshold": (not v["stage1"]) or v["score"] < thr} for k, v in neg.items()},
    }
    path = REPO / "artifacts" / "app_checks" / f"cxr_gate_calibration_{args.build.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
