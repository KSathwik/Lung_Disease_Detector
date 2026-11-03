"""Dataset V6 (proposed): manifest schema, duplicate-aware grouping, split assignment and fingerprints.

Source adapters (scripts/data/build_v6_manifest.py) produce rows in V6_COLUMNS; this module turns them into a
split manifest. Nothing here invents metadata: unknown patient identity stays 'unknown'.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from lungai.config import CLASS_NAMES

V6_COLUMNS: List[str] = [
    "image_id",            # unique, '<source_id>:<source_image_id>'
    "source_id",           # registry id (scripts/data/sources.json)
    "source_version",      # publisher version / release string
    "source_image_id",     # original filename or id, verbatim
    "source_patient_id",   # verbatim publisher patient id, or empty
    "patient_id_status",   # genuine | derived_from_filename | unknown
    "source_label_raw",    # verbatim publisher label(s)
    "source_label_set",    # all findings present, '|'-separated, verbatim names
    "lungai_label",        # canonical class name, or empty when excluded
    "label_rule_id",       # rule in lungai/configs/dataset_v6_proposal.json that produced the label
    "exclusion_reason",    # empty when included
    "role",                # development | external
    "view_position",       # as published, or empty
    "age", "sex",          # as published, or empty (never imputed)
    "raw_relpath",         # path under the raw root, forward slashes
    "raw_md5", "raw_sha256", "phash",
    "width", "height",     # measured from the file
    "patient_group",       # grouping key used for splitting (after duplicate merging)
    "split",               # train | val | test | external | excluded
]
PATIENT_ID_STATUSES = ("genuine", "derived_from_filename", "unknown")


def validate_rows(df: pd.DataFrame) -> List[str]:
    errs: List[str] = []
    missing = [c for c in V6_COLUMNS if c not in df.columns and c not in ("patient_group", "split")]
    if missing:
        return [f"missing columns: {missing}"]
    if df["image_id"].duplicated().any():
        errs.append(f"{int(df['image_id'].duplicated().sum())} duplicate image_id values")
    bad = set(df["patient_id_status"]) - set(PATIENT_ID_STATUSES)
    if bad:
        errs.append(f"invalid patient_id_status values: {sorted(bad)}")
    inc = df["exclusion_reason"].fillna("") == ""
    bad_labels = set(df.loc[inc, "lungai_label"]) - set(CLASS_NAMES)
    if bad_labels:
        errs.append(f"included rows with non-canonical labels: {sorted(bad_labels)}")
    if (inc & (df["label_rule_id"].fillna("") == "")).any():
        errs.append("included rows without a label_rule_id")
    genuine_without_id = (df["patient_id_status"] != "unknown") & (df["source_patient_id"].fillna("") == "")
    if genuine_without_id.any():
        errs.append(f"{int(genuine_without_id.sum())} rows claim a patient id status but have no source_patient_id")
    return errs


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def assign_groups(df: pd.DataFrame, duplicate_pairs: Iterable[Tuple[str, str]] = ()) -> pd.Series:
    """Patient group per image: patient id where known, else the image itself; merged across duplicates."""
    base = np.where(df["patient_id_status"] == "unknown", "img:" + df["image_id"],
                    "pt:" + df["source_id"] + ":" + df["source_patient_id"].astype(str))
    base = pd.Series(base, index=df.index)
    uf = _UnionFind(base.unique())
    for h in ("raw_md5",):
        if h in df.columns:
            for _, g in df[df[h].fillna("") != ""].groupby(h):
                keys = base[g.index].unique()
                for k in keys[1:]:
                    uf.union(keys[0], k)
    by_id = dict(zip(df["image_id"], base))
    for a, b in duplicate_pairs:
        if a in by_id and b in by_id:
            uf.union(by_id[a], by_id[b])
    return base.map(uf.find).rename("patient_group")


def assign_splits(df: pd.DataFrame, fractions: Dict[str, float], seed: int) -> pd.Series:
    """Grouped split: whole groups go to one split, stratified by each group's majority (label, source)."""
    names = ("train", "val", "test")
    rng = np.random.default_rng(seed)
    dev = df[(df["role"] == "development") & (df["exclusion_reason"].fillna("") == "")]
    out = pd.Series("excluded", index=df.index, name="split")
    out[df["role"] == "external"] = "external"
    stratum = dev.groupby("patient_group").apply(lambda g: f"{g['lungai_label'].mode().iloc[0]}|{g['source_id'].mode().iloc[0]}")
    size = dev.groupby("patient_group").size()
    assignment: Dict[str, str] = {}
    for s in sorted(stratum.unique()):
        groups = np.array(sorted(stratum[stratum == s].index))
        rng.shuffle(groups)
        total = int(size[groups].sum())
        filled = {n: 0 for n in names}
        for g in groups:
            target = max(names, key=lambda n: (fractions[n] * total - filled[n], -names.index(n)))
            assignment[g] = target
            filled[target] += int(size[g])
    out[dev.index] = dev["patient_group"].map(assignment)
    return out


def split_fingerprint(df: pd.DataFrame, id_col: str = "image_id", split_col: str = "split") -> str:
    """SHA-256 over sorted 'image_id<TAB>split' lines: identifies a partition independently of row order."""
    lines = sorted(f"{i}\t{s}" for i, s in zip(df[id_col].astype(str), df[split_col].astype(str)))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def leakage_summary(df: pd.DataFrame) -> Dict[str, int]:
    dev = df[df["split"].isin(["train", "val", "test"])]
    per_group = dev.groupby("patient_group")["split"].nunique()
    md5_split = dev[dev["raw_md5"].fillna("") != ""].groupby("raw_md5")["split"].nunique()
    ext_md5 = set(df.loc[df["split"] == "external", "raw_md5"].dropna()) - {""}
    return {
        "groups_spanning_splits": int((per_group > 1).sum()),
        "md5_spanning_splits": int((md5_split > 1).sum()),
        "external_md5_in_development": int(dev["raw_md5"].isin(ext_md5).sum()),
    }
