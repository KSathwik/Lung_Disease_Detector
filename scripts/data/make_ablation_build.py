"""Preprocessing ablation: the same build (manifest, split, labels) with a different cached image transform.

    python scripts/data/make_ablation_build.py --build data/derived/v6_build_002 --variant no_filters

Copies manifest_v6.csv and v6_build_report.json unchanged (so the split fingerprint is identical) and writes
cache/<raw_sha256>.npy for every train/val/test/external row with the variant transform. Training and
evaluation then run on the new folder exactly as on the original build. Variants:

    no_filters   decode -> RGB -> Lanczos-4 resize to 224x224 (no Gaussian blur, no CLAHE, no low-pass)

The ablation model is a research artefact; the application always serves the canonical preprocessing.
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lungai import preprocessing as pp  # noqa: E402


def no_filters(bgr: np.ndarray) -> np.ndarray:
    img = bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return pp.resize_display(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


VARIANTS = {"no_filters": no_filters}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--raw-root", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = args.out or args.build.with_name(f"{args.build.name}_{args.variant}")
    if out.exists():
        raise FileExistsError(f"{out} exists; not overwriting")
    (out / "cache").mkdir(parents=True)
    shutil.copy2(args.build / "manifest_v6.csv", out / "manifest_v6.csv")
    report = json.loads((args.build / "v6_build_report.json").read_text(encoding="utf-8"))
    report["ablation"] = {"base_build": args.build.name, "variant": args.variant,
                          "transform": VARIANTS[args.variant].__doc__ or "decode -> RGB -> Lanczos-4 resize 224 (no filters)",
                          "note": "manifest and split copied unchanged; only cache/*.npy differ"}
    (out / "v6_build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    m = pd.read_csv(out / "manifest_v6.csv", dtype=str, keep_default_na=False)
    rows = m[m.split.isin(["train", "val", "test", "external"])].drop_duplicates("raw_sha256")
    fn = VARIANTS[args.variant]

    def one(r):
        img = cv2.imdecode(np.fromfile(str(args.raw_root / r.raw_relpath), np.uint8), cv2.IMREAD_COLOR)
        np.save(out / "cache" / f"{r.raw_sha256}.npy", fn(img))

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, rows.itertuples()))
    print(f"wrote {len(rows)} cached images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
