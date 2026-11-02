"""Canonical V5 manifest: checksum, schema, counts and known integrity findings."""

import pandas as pd
import pytest

from lungai.config import CLASS_NAMES, DATASET
from lungai.data import checks
from lungai.data.manifest import (
    ManifestError, class_distribution, counts_by_split, load_manifest, normalize_manifest_path, schema_errors,
)


@pytest.fixture(scope="module")
def df():
    return load_manifest()


def test_checksum_and_schema(df):
    assert schema_errors(df) == []
    assert len(df) == 10547


def test_partition_counts(df):
    c = counts_by_split(df)
    assert {k: v["images"] for k, v in c.items()} == {"train": 7398, "val": 1579, "test": 1570}
    assert df["patient_id"].nunique() == 10270  # ID strings; see test_patient_ids_are_partly_synthetic


def test_class_distribution(df):
    dist = class_distribution(df)
    assert list(dist) == CLASS_NAMES
    assert dist["Pulmonary Nodule / Mass"] == {"train": 529, "val": 117, "test": 108}
    assert sum(sum(v.values()) for v in dist.values()) == 10547


def test_known_patient_leakage_is_detected(df):
    """Regression pin on a known defect of V5 (fix requires a new dataset version and retraining)."""
    r = checks.check_patient_leakage(df)
    assert r["status"] == "fail"
    assert r["by_manifest_patient_id"]["keys_spanning_splits"] == 0
    g = r["by_derived_patient_group"]
    assert g["test_images_sharing_key_with_train"] == 154
    assert g["test_images_sharing_key_with_train_by_source"] == {"Existing_Normal": 44, "Existing_Pneumonia": 110}


def test_patient_ids_are_partly_synthetic(df):
    r = checks.check_patient_leakage(df)
    assert r["rows_with_synthetic_per_image_patient_id"] == 8478


def test_no_exact_duplicates_and_external_cohort_isolated(df):
    assert checks.check_exact_duplicates(df)["status"] == "pass"
    assert checks.check_external_isolation(df)["status"] == "pass"


def test_checksum_mismatch_is_refused(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("image_id\nx\n")
    with pytest.raises(ManifestError, match="checksum"):
        load_manifest(p)


def test_schema_errors_are_reported():
    bad = pd.DataFrame({"image_id": ["a", "a"], "image_path": ["p1", "p2"], "patient_id": ["1", "2"],
                        "source_dataset": ["s", "s"], "clinical_label": ["Normal", "Lung Cancer"],
                        "split": ["train", "holdout"], "md5_hash": ["0" * 32, "xyz"]})
    errs = " | ".join(schema_errors(bad))
    assert "Lung Cancer" in errs and "holdout" in errs and "duplicate values in 'image_id'" in errs and "malformed md5" in errs


@pytest.mark.parametrize("raw,expected", [
    ("data\\raw\\COVID-19\\COVID-1.png", "data/raw/COVID-19/COVID-1.png"),
    ("data/downloads/jsrt/a.png", "data/downloads/jsrt/a.png"),
    ("d:\\Sathwik\\lung_disease_detector\\Lung_Disease_Detector-main\\data\\raw\\NIH_ChestX-ray14\\0001.png",
     "data/raw/NIH_ChestX-ray14/0001.png"),
])
def test_paths_resolve_under_one_data_root(raw, expected):
    assert normalize_manifest_path(raw) == expected


def test_image_checks_block_without_data(df, tmp_path):
    r = checks.check_files(df.head(5), tmp_path)
    assert r["status"] == "blocked"


def test_image_checks_detect_missing_and_corrupt(tmp_path):
    import cv2
    import hashlib
    import numpy as np
    good = tmp_path / "data" / "raw" / "good.png"
    good.parent.mkdir(parents=True)
    cv2.imwrite(str(good), np.full((64, 64, 3), 100, np.uint8))
    bad = tmp_path / "data" / "raw" / "bad.png"
    bad.write_bytes(b"garbage")
    rows = pd.DataFrame({
        "image_id": ["good", "bad", "gone"],
        "image_path": ["data\\raw\\good.png", "data\\raw\\bad.png", "data\\raw\\gone.png"],
        "md5_hash": [hashlib.md5(good.read_bytes()).hexdigest(), hashlib.md5(b"garbage").hexdigest(), "0" * 32],
    })
    r = checks.check_files(rows, tmp_path)
    assert r["status"] == "fail" and r["missing"] == 1 and r["undecodable"] == 1 and r["md5_mismatch"] == 0


def test_dataset_config_points_at_this_manifest():
    assert DATASET["manifest"] == "data/manifests/unified_manifest_v5.csv"
