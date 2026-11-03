"""Load and validate the canonical dataset manifest."""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional

import pandas as pd

from lungai.config import CLASS_NAMES, DATASET_V5, data_root, manifest_path, sha256_file

REQUIRED_COLUMNS: List[str] = [
    "image_id", "image_path", "patient_id", "source_dataset", "clinical_label", "split", "md5_hash",
]
VALID_SPLITS = ("train", "val", "test")
_MD5 = re.compile(r"^[0-9a-f]{32}$")


class ManifestError(ValueError):
    """Raised when the manifest fails a structural check that makes it unusable."""


def load_manifest(path: Optional[Path] = None, verify_checksum: bool = True) -> pd.DataFrame:
    """Load the manifest and run the structural checks every consumer relies on.

    Raises ManifestError on a missing file, checksum mismatch, missing columns, unknown
    labels or splits, or duplicate image ids/paths.
    """
    path = Path(path) if path else manifest_path()
    if not path.is_file():
        raise ManifestError(f"Manifest not found: {path}")
    if verify_checksum:
        actual = sha256_file(path, normalize_newlines=True)
        if actual != DATASET_V5["manifest_sha256_lf"]:
            raise ManifestError(
                f"Manifest checksum mismatch for {path.name}: expected {DATASET_V5['manifest_sha256_lf']}, got {actual}. "
                "The dataset definition changed; record a new dataset version instead of editing V5."
            )
    df = pd.read_csv(path, dtype={"md5_hash": str, "patient_id": str, "image_id": str})
    errors = schema_errors(df)
    if errors:
        raise ManifestError("Manifest failed schema validation:\n- " + "\n- ".join(errors))
    return df


def schema_errors(df: pd.DataFrame) -> List[str]:
    """Return a list of structural problems (empty when the manifest is usable)."""
    errors: List[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    for col in REQUIRED_COLUMNS:
        n = int(df[col].isna().sum())
        if n:
            errors.append(f"{n} empty values in '{col}'")
    bad_labels = sorted(set(df["clinical_label"].dropna()) - set(CLASS_NAMES))
    if bad_labels:
        errors.append(f"labels not in the canonical class mapping: {bad_labels}")
    bad_splits = sorted(set(df["split"].dropna()) - set(VALID_SPLITS))
    if bad_splits:
        errors.append(f"unknown split names: {bad_splits}")
    for col in ("image_id", "image_path"):
        dup = int(df[col].duplicated().sum())
        if dup:
            errors.append(f"{dup} duplicate values in '{col}'")
    bad_md5 = int((~df["md5_hash"].astype(str).str.fullmatch(_MD5)).sum())
    if bad_md5:
        errors.append(f"{bad_md5} malformed md5_hash values")
    return errors


def normalize_manifest_path(raw: str) -> str:
    """Manifest path -> forward-slash path relative to the data root.

    Historical rows store Windows paths, 211 of them absolute paths from the original
    author's machine (``<drive>:\\...\\data\\raw\\...``). Those are cut at the ``data``
    folder so every row resolves against one configurable data root.
    """
    parts = list(PureWindowsPath(raw).parts) if ("\\" in raw or re.match(r"^[A-Za-z]:", raw)) else list(Path(raw).parts)
    if parts and (PureWindowsPath(raw).is_absolute() or Path(raw).is_absolute()):
        for i in range(len(parts) - 1):
            if parts[i].lower() == "data" and parts[i + 1].lower() in ("raw", "downloads", "processed"):
                parts = parts[i:]
                break
        else:
            raise ManifestError(f"Absolute manifest path outside a data/ folder cannot be relocated: {raw}")
    return "/".join(p for p in parts if p not in ("", "\\", "/"))


def resolve_image_path(raw: str, root: Optional[Path] = None) -> Path:
    return (root or data_root()) / normalize_manifest_path(raw)


def counts_by_split(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split in VALID_SPLITS:
        sub = df[df["split"] == split]
        out[split] = {"images": int(len(sub)), "patient_ids": int(sub["patient_id"].nunique())}
    return out


def class_distribution(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    tab = pd.crosstab(df["clinical_label"], df["split"]).reindex(index=CLASS_NAMES, columns=list(VALID_SPLITS), fill_value=0)
    return {cls: {s: int(tab.loc[cls, s]) for s in VALID_SPLITS} for cls in CLASS_NAMES}


def split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in VALID_SPLITS:
        raise ManifestError(f"Unknown split '{split}'. Expected one of {VALID_SPLITS}.")
    return df[df["split"] == split].reset_index(drop=True)
