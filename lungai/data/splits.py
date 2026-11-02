"""Deterministic patient-grouped, class-stratified split generation.

Not used to produce V5 (V5's split is preserved as recorded). This is the procedure for
the next dataset version: it assigns whole patient groups to a split so no patient can
appear in more than one partition.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from lungai.data.checks import add_patient_groups


def grouped_stratified_split(
    df: pd.DataFrame,
    fractions: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    label_col: str = "clinical_label",
) -> pd.Series:
    """Return a 'split' Series (train/val/test) aligned with ``df``.

    Groups are formed with ``derived_patient_group``; a group's stratum is its majority
    label. Within each stratum the groups are shuffled with ``seed`` and filled greedily
    into the split furthest below its target image count, which keeps image fractions
    close to ``fractions`` even when groups have different sizes.
    """
    if abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) <= 0:
        raise ValueError(f"fractions must be positive and sum to 1, got {fractions}")
    names = ("train", "val", "test")
    grouped = add_patient_groups(df)
    rng = np.random.default_rng(seed)
    assignment: Dict[str, str] = {}

    group_label = grouped.groupby("patient_group")[label_col].agg(lambda s: s.value_counts().index[0])
    group_size = grouped.groupby("patient_group").size()
    for label in sorted(group_label.unique()):
        groups = np.array(sorted(group_label[group_label == label].index))
        rng.shuffle(groups)
        total = int(group_size[groups].sum())
        filled = {n: 0 for n in names}
        for g in groups:
            deficit = {n: fractions[i] * total - filled[n] for i, n in enumerate(names)}
            target = max(names, key=lambda n: (deficit[n], -names.index(n)))
            assignment[g] = target
            filled[target] += int(group_size[g])

    return grouped["patient_group"].map(assignment).rename("split")
