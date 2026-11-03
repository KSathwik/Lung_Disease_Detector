"""Find external-cohort images (e.g. Montgomery) that also appear in candidate training sources.

    python scripts/data/check_external_overlap.py --external <root>/raw/montgomery --candidates <root>/raw/rahman_tb [<more folders>]

Two tests per pair of images:
1. exact content (md5) — catches byte-identical copies;
2. perceptual hash (64-bit DCT pHash of a 32x32 grayscale thumbnail) — catches resized or re-encoded copies.
   Pairs within --max-distance bits are verified by normalised correlation of 64x64 thumbnails (>= --min-corr).

Writes <out>/external_overlap.json and lists every confirmed match. Exit code 1 if any overlap is found.
A match means the "external" image must be removed from training (or the cohort is not external).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _gray(path: Path) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return img


def phash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    bits = (dct > np.median(dct[1:].ravel())).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def thumb(gray: np.ndarray) -> np.ndarray:
    t = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n else t


def describe(path: Path) -> Optional[Tuple[str, str, int, np.ndarray]]:
    g = _gray(path)
    if g is None:
        return None
    return str(path), hashlib.md5(path.read_bytes()).hexdigest(), phash(g), thumb(g)


def collect(folders: List[Path]) -> list:
    files = [p for d in folders for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT]
    with ThreadPoolExecutor(max_workers=8) as ex:
        return [d for d in ex.map(describe, files) if d is not None]


def find_overlap(external: list, candidates: list, max_distance: int, min_corr: float) -> list:
    by_md5 = {}
    for c in candidates:
        by_md5.setdefault(c[1], []).append(c[0])
    cand_hash = np.array([c[2] for c in candidates], dtype=np.uint64)
    matches = []
    for path, md5, ph, th in external:
        for cp in by_md5.get(md5, []):
            matches.append({"external": path, "candidate": cp, "method": "md5", "hamming": 0, "correlation": 1.0})
        if not len(cand_hash):
            continue
        x = np.bitwise_xor(cand_hash, np.uint64(ph))
        dist = np.array([bin(int(v)).count("1") for v in x])
        for j in np.nonzero(dist <= max_distance)[0]:
            if candidates[j][1] == md5:
                continue
            corr = float(np.dot(th, candidates[j][3]))
            if corr >= min_corr:
                matches.append({"external": path, "candidate": candidates[j][0], "method": "phash+correlation",
                                "hamming": int(dist[j]), "correlation": round(corr, 4)})
    return matches


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, nargs="+", required=True)
    ap.add_argument("--max-distance", type=int, default=10)
    ap.add_argument("--min-corr", type=float, default=0.97)
    ap.add_argument("--out", type=Path, default=Path("data/derived/external_overlap"))
    args = ap.parse_args(argv)
    for d in [args.external, *args.candidates]:
        if not d.is_dir():
            ap.error(f"folder not found: {d}")
    ext, cand = collect([args.external]), collect(args.candidates)
    matches = find_overlap(ext, cand, args.max_distance, args.min_corr)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "external_images": len(ext), "candidate_images": len(cand),
        "external_images_with_a_match": len({m["external"] for m in matches}),
        "thresholds": {"phash_max_hamming": args.max_distance, "min_thumbnail_correlation": args.min_corr},
        "matches": matches,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "external_overlap.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"external={len(ext)} candidates={len(cand)} external images found in candidates={report['external_images_with_a_match']}")
    return 1 if matches else 0


if __name__ == "__main__":
    sys.exit(main())
