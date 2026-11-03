"""Check which historical V5 images have been recovered, by content (md5), wherever they were extracted.

    python scripts/data/verify_recovery.py --search <root>/raw [--out data/derived/v5_recovery]

Hashes every image file under --search and matches it to the md5_hash column of the V5 manifest.
Filenames and folder layout do not matter, so renamed or re-packaged copies are found if their bytes
are identical. Writes (never touching the manifest or the raw files):

    <out>/recovery_report.json   per-source recovered / missing counts, unmatched files
    <out>/path_map.csv           image_id -> recovered file path (relative to --search)

Re-encoded or resized copies (e.g. the downscaled VinDr derivative vs the official DICOMs) cannot match by md5;
they are reported as missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lungai.config import REPO_ROOT  # noqa: E402
from lungai.data.manifest import load_manifest  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(search: Path) -> pd.DataFrame:
    files = [p for p in search.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT]
    with ThreadPoolExecutor(max_workers=8) as ex:
        hashes = list(ex.map(md5_file, files))
    return pd.DataFrame({"path": [str(p.relative_to(search)).replace("\\", "/") for p in files], "md5": hashes})


def reconcile(manifest: pd.DataFrame, found: pd.DataFrame) -> dict:
    first = found.drop_duplicates("md5").set_index("md5")["path"]
    m = manifest.assign(recovered_path=manifest["md5_hash"].map(first))
    per_source = {}
    for src, g in m.groupby("source_dataset"):
        per_source[src] = {"expected": int(len(g)), "recovered": int(g["recovered_path"].notna().sum()),
                           "missing": int(g["recovered_path"].isna().sum()),
                           "missing_by_split": {k: int(v) for k, v in g[g["recovered_path"].isna()]["split"].value_counts().items()}}
    unmatched = found[~found["md5"].isin(set(manifest["md5_hash"]))]
    return {
        "per_source": per_source,
        "expected_total": int(len(m)),
        "recovered_total": int(m["recovered_path"].notna().sum()),
        "files_scanned": int(len(found)),
        "files_not_in_manifest": int(len(unmatched)),
        "duplicate_files_same_content": int(found["md5"].duplicated().sum()),
        "path_map": m[["image_id", "source_dataset", "split", "recovered_path"]],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "derived" / "v5_recovery")
    args = ap.parse_args(argv)
    if not args.search.is_dir():
        ap.error(f"--search folder not found: {args.search}")
    r = reconcile(load_manifest(), scan(args.search))
    args.out.mkdir(parents=True, exist_ok=True)
    r.pop("path_map").to_csv(args.out / "path_map.csv", index=False)
    r["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (args.out / "recovery_report.json").write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: r[k] for k in ("expected_total", "recovered_total", "files_scanned", "files_not_in_manifest")}, indent=2))
    for src, v in r["per_source"].items():
        print(f"{src:<24} {v['recovered']:>6,} / {v['expected']:>6,} recovered")
    return 0 if r["recovered_total"] == r["expected_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
