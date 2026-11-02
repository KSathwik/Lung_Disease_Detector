"""Dataset integrity checks: patient leakage, duplicates, files and labels.

Each check returns a dict with ``status`` ("pass", "fail", "warn" or "blocked") and the
evidence behind it, so the validation report can be written without interpretation.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from lungai.data.manifest import VALID_SPLITS, resolve_image_path

# Kermany (Guangzhou paediatric) filenames carry a patient token: "person123_bacteria_456.jpeg" or
# "IM-0115-0001.jpeg". The original train/ and test/ folders numbered persons independently, so the
# folder prefix the V2 ingestion kept ("train_" / "test_") is part of the key.
_KERMANY_TOKEN = re.compile(r"(person\d+)|IM-(\d+)", re.IGNORECASE)
KERMANY_SOURCES = ("Existing_Pneumonia", "Existing_Normal")
# Sources whose manifest patient_id is one synthetic id per image; patient identity is unknown.
SYNTHETIC_ID_PREFIXES = ("existing_pt_", "tbx11k_pt_")


def _basename(raw_path: str) -> str:
    return re.split(r"[\\/]", raw_path)[-1]


def derived_patient_group(row: pd.Series) -> str:
    """Best available patient identity for a manifest row.

    Uses the Kermany filename token where one exists, otherwise the manifest patient_id.
    """
    if row["source_dataset"] in KERMANY_SOURCES:
        name = _basename(row["image_path"])
        m = _KERMANY_TOKEN.search(name)
        if m:
            token = (m.group(1) or "IM" + m.group(2)).lower()
            return f"{row['source_dataset']}:{name.split('_')[0].lower()}:{token}"
    return str(row["patient_id"])


def add_patient_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["patient_group"] = out.apply(derived_patient_group, axis=1)
    return out


def _cross_split(df: pd.DataFrame, key: str) -> Dict[str, Any]:
    splits_per_key = df.groupby(key)["split"].agg(lambda s: frozenset(s))
    spanning = splits_per_key[splits_per_key.map(len) > 1]
    pairs: Dict[str, int] = defaultdict(int)
    for s in spanning:
        pairs["+".join(sorted(s))] += 1
    in_span = df[key].isin(spanning.index)
    test_in_train = df[(df["split"] == "test") & df[key].isin(splits_per_key[splits_per_key.map(lambda s: "train" in s)].index)]
    val_in_train = df[(df["split"] == "val") & df[key].isin(splits_per_key[splits_per_key.map(lambda s: "train" in s)].index)]
    return {
        "keys_spanning_splits": int(len(spanning)),
        "images_in_spanning_keys": int(in_span.sum()),
        "split_patterns": dict(sorted(pairs.items())),
        "test_images_sharing_key_with_train": int(len(test_in_train)),
        "val_images_sharing_key_with_train": int(len(val_in_train)),
        "test_images_sharing_key_with_train_by_source": {
            k: int(v) for k, v in test_in_train["source_dataset"].value_counts().sort_index().items()
        },
    }


def check_patient_leakage(df: pd.DataFrame) -> Dict[str, Any]:
    """Patient overlap across splits, by manifest patient_id and by derived patient group."""
    by_id = _cross_split(df, "patient_id")
    grouped = add_patient_groups(df)
    by_group = _cross_split(grouped, "patient_group")
    synthetic = df["patient_id"].astype(str).str.startswith(SYNTHETIC_ID_PREFIXES)
    leak = by_group["keys_spanning_splits"] > 0 or by_id["keys_spanning_splits"] > 0
    return {
        "status": "fail" if leak else ("warn" if synthetic.any() else "pass"),
        "by_manifest_patient_id": by_id,
        "by_derived_patient_group": by_group,
        "rows_with_synthetic_per_image_patient_id": int(synthetic.sum()),
        "synthetic_id_rows_by_source": {k: int(v) for k, v in df[synthetic]["source_dataset"].value_counts().sort_index().items()},
        "unique_manifest_patient_ids": int(df["patient_id"].nunique()),
        "unique_derived_patient_groups": int(grouped["patient_group"].nunique()),
        "note": (
            "Manifest patient_id is a synthetic per-image id for Existing_* and TBX11K rows, so the manifest-id check "
            "is uninformative for them. Derived groups recover Kermany patients from filenames; COVID-QU and Shenzhen "
            "(Existing_COVID-19, Existing_Tuberculosis) and TBX11K patient identity cannot be recovered from the manifest."
        ),
    }


def check_exact_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    dup_any = df[df["md5_hash"].duplicated(keep=False)]
    splits_per_hash = dup_any.groupby("md5_hash")["split"].nunique()
    cross = int((splits_per_hash > 1).sum())
    return {
        "status": "fail" if cross else ("warn" if len(dup_any) else "pass"),
        "duplicate_md5_groups": int(dup_any["md5_hash"].nunique()),
        "duplicate_md5_groups_spanning_splits": cross,
        "method": "manifest md5_hash (content hash recorded at ingestion; re-verified by check_files when images are present)",
    }


def check_perceptual_duplicates(df: pd.DataFrame, max_distance: int = 0) -> Dict[str, Any]:
    """Pairs with identical (distance 0) dHash, reported by split pattern and label agreement."""
    if "dhash" not in df.columns:
        return {"status": "blocked", "reason": "manifest has no dhash column"}
    groups = df.groupby(df["dhash"].astype("uint64"))
    pairs: List[Dict[str, Any]] = []
    for _, g in groups:
        if len(g) < 2:
            continue
        rows = g.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                pairs.append({
                    "image_id_a": a["image_id"], "image_id_b": b["image_id"],
                    "splits": "+".join(sorted({a["split"], b["split"]})),
                    "labels_agree": a["clinical_label"] == b["clinical_label"],
                })
    cross = [p for p in pairs if "+" in p["splits"]]
    return {
        "status": "warn" if cross else "pass",
        "identical_dhash_pairs": len(pairs),
        "identical_dhash_pairs_spanning_splits": len(cross),
        "pairs": pairs,
        "method": f"64-bit dHash from manifest, Hamming distance <= {max_distance}; candidates for visual review, not proof of duplication",
    }


def check_label_source_confounding(df: pd.DataFrame) -> Dict[str, Any]:
    tab = pd.crosstab(df["clinical_label"], df["source_dataset"])
    share = (tab.T / tab.sum(axis=1)).T
    single_source = [c for c in tab.index if (tab.loc[c] > 0).sum() == 1]
    dominant = {c: {"source": share.loc[c].idxmax(), "share": round(float(share.loc[c].max()), 3)} for c in tab.index}
    return {
        "status": "warn" if single_source or any(v["share"] >= 0.8 for v in dominant.values()) else "pass",
        "classes_from_a_single_source": single_source,
        "dominant_source_per_class": dominant,
        "counts": {c: {s: int(v) for s, v in tab.loc[c].items() if v} for c in tab.index},
    }


def check_proxy_labels(df: pd.DataFrame) -> Dict[str, Any]:
    if "original_finding" not in df.columns:
        return {"status": "blocked", "reason": "manifest has no original_finding column"}
    proxy = df[(df["source_dataset"] == "TBX11K") & (df["clinical_label"] == "Pneumonia")]
    pneu = int((df["clinical_label"] == "Pneumonia").sum())
    return {
        "status": "warn" if len(proxy) else "pass",
        "tbx11k_rows_labelled_pneumonia": int(len(proxy)),
        "original_findings": {k: int(v) for k, v in proxy["original_finding"].value_counts().items()},
        "share_of_pneumonia_class": round(len(proxy) / pneu, 3) if pneu else None,
        "note": "TBX11K 'sick but not TB' (abnormal, cause unspecified) was mapped to Pneumonia during V2 ingestion.",
    }


def check_placeholder_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    cols = ["width", "height", "mean_intensity", "std_intensity"]
    if not all(c in df.columns for c in cols):
        return {"status": "blocked", "reason": "metadata columns missing"}
    placeholder = (df["width"] == 224) & (df["height"] == 224) & (df["mean_intensity"] == 128.0) & (df["std_intensity"] == 50.0)
    return {
        "status": "warn" if placeholder.any() else "pass",
        "rows_with_default_metadata": int(placeholder.sum()),
        "note": "width/height/mean/std columns for these rows are defaults written by rebuild_unified_dataset_v4.py, not measurements.",
    }


def check_absolute_paths(df: pd.DataFrame) -> Dict[str, Any]:
    absolute = df["image_path"].astype(str).str.match(r"^[A-Za-z]:[\\/]|^/")
    return {
        "status": "warn" if absolute.any() else "pass",
        "rows_with_absolute_paths": int(absolute.sum()),
        "handling": "lungai.data.manifest.normalize_manifest_path relocates them under the data root",
    }


def check_files(df: pd.DataFrame, root: Path, verify_md5: bool = True, decode: bool = True,
                limit: Optional[int] = None) -> Dict[str, Any]:
    """Check that image files exist, match their recorded md5 and decode. Blocked when no image is present."""
    from lungai.preprocessing import ImageDecodeError, read_image

    rows = df if limit is None else df.head(limit)
    missing: List[str] = []
    md5_mismatch: List[str] = []
    corrupt: List[str] = []
    checked = 0
    for r in rows.itertuples(index=False):
        p = resolve_image_path(r.image_path, root)
        if not p.is_file():
            missing.append(r.image_id)
            continue
        checked += 1
        if verify_md5 and hashlib.md5(p.read_bytes()).hexdigest() != r.md5_hash:
            md5_mismatch.append(r.image_id)
        if decode:
            try:
                read_image(p)
            except (ImageDecodeError, FileNotFoundError):
                corrupt.append(r.image_id)
    if checked == 0:
        return {"status": "blocked", "reason": "no manifest image found under the configured data root (set LUNGAI_DATA_ROOT or --data-root)", "rows_checked": int(len(rows))}
    bad = missing or md5_mismatch or corrupt
    return {
        "status": "fail" if bad else "pass",
        "rows_checked": int(len(rows)),
        "files_found": checked,
        "missing": len(missing), "missing_examples": missing[:20],
        "md5_mismatch": len(md5_mismatch), "md5_mismatch_examples": md5_mismatch[:20],
        "undecodable": len(corrupt), "undecodable_examples": corrupt[:20],
    }


def check_external_isolation(df: pd.DataFrame, external_ids: Iterable[str] = ()) -> Dict[str, Any]:
    in_source = df["source_dataset"].astype(str).str.contains("montgomery", case=False)
    in_path = df["image_path"].astype(str).str.contains("montgomery", case=False)
    names = {_basename(p).lower() for p in df["image_path"].astype(str)}
    by_name = sorted(n for n in (e.lower() for e in external_ids) if n in names)
    leak = bool(in_source.any() or in_path.any() or by_name)
    return {
        "status": "fail" if leak else "pass",
        "rows_with_montgomery_source": int(in_source.sum()),
        "rows_with_montgomery_in_path": int(in_path.sum()),
        "external_filenames_in_manifest": by_name,
        "note": "Name/path level only; content-level isolation needs the images (md5 against the external cohort).",
    }


def summarize_status(checks: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, str]]:
    statuses = {k: v.get("status", "blocked") for k, v in checks.items()}
    order = ["fail", "warn", "blocked", "pass"]
    overall = next((s for s in order if s in statuses.values()), "pass")
    return overall, statuses


__all__ = [
    "VALID_SPLITS", "add_patient_groups", "check_absolute_paths", "check_exact_duplicates", "check_external_isolation",
    "check_files", "check_label_source_confounding", "check_patient_leakage", "check_perceptual_duplicates",
    "check_placeholder_metadata", "check_proxy_labels", "derived_patient_group", "summarize_status",
]
