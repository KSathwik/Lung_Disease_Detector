"""Grouped split generation for the next dataset version."""

import pandas as pd

from lungai.data.checks import add_patient_groups, check_patient_leakage
from lungai.data.manifest import load_manifest
from lungai.data.splits import grouped_stratified_split


def test_grouped_split_is_deterministic_and_leak_free():
    df = load_manifest()
    a = grouped_stratified_split(df, seed=42)
    b = grouped_stratified_split(df, seed=42)
    assert a.equals(b)
    resplit = df.assign(split=a)
    r = check_patient_leakage(resplit)
    assert r["by_derived_patient_group"]["keys_spanning_splits"] == 0
    assert r["by_manifest_patient_id"]["keys_spanning_splits"] == 0
    frac = a.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.01 and abs(frac["val"] - 0.15) < 0.01 and abs(frac["test"] - 0.15) < 0.01


def test_different_seed_gives_different_split():
    df = load_manifest()
    assert not grouped_stratified_split(df, seed=1).equals(grouped_stratified_split(df, seed=2))


def test_groups_never_split_on_toy_data():
    toy = pd.DataFrame({
        "image_path": [f"data\\raw\\Pneumonia\\train_person{p}_bacteria_{i}.jpeg" for p in range(20) for i in range(3)],
        "patient_id": [f"existing_pt_{p}_{i}" for p in range(20) for i in range(3)],
        "source_dataset": "Existing_Pneumonia",
        "clinical_label": "Pneumonia",
    })
    split = grouped_stratified_split(toy, seed=0)
    groups = add_patient_groups(toy).assign(split=split).groupby("patient_group")["split"].nunique()
    assert (groups == 1).all()
