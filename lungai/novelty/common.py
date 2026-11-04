"""Shared constants, sample definitions, scores, bootstrap and the access log of the novelty protocol.

No TensorFlow. Every setting here is copied from docs/publication/NOVELTY_PROTOCOL.md (commit f427c04);
the section of the protocol is named next to each constant.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lungai.config import CLASS_NAMES, NORMAL_CLASS, REPO_ROOT

PROTOCOL_COMMIT = "f427c04"
PROTOCOL_PATH = REPO_ROOT / "docs" / "publication" / "NOVELTY_PROTOCOL.md"
OUT_DIR = REPO_ROOT / "artifacts" / "novelty"
ACCESS_DIR = OUT_DIR / "access"
ACCESS_LOG = OUT_DIR / "access_log.jsonl"

BUILDS = {"main": REPO_ROOT / "data" / "derived" / "v6_build_002",
          "nofilter": REPO_ROOT / "data" / "derived" / "v6_build_002_no_filters"}  # section 3.1
SERVED_CHECKPOINT = REPO_ROOT / "models" / "densenet121_v6r2_seed42.h5"  # section 4.3
SERVED_SHA256 = "7992b8da10ff97f90207dc730c3163ba968a4d6094e1bff263377612ee0597c6"

SOURCES = ["cohen_covid", "covid_radiography", "kermany", "nih_cxr14", "shenzhen", "tbx11k"]  # six development sources
EXTERNAL_SOURCE = "montgomery"

SAMPLING_SEED = 0  # section 3.3
BOOTSTRAP_SEED = 1  # sections 4.6, 7, 8, 9
BOOTSTRAP_B = 1000
MIN_CELL = 20  # section 3.3 minimum cell rule


@dataclass(frozen=True)
class SampleSet:
    """A set of manifest rows whose features are cached: a capped probe sample or a whole split."""
    key: str
    split: str
    cap: Optional[int]
    expected_n: int


# Section 3.3 (probe samples), section 9 (C2-lite train cap 400), section 7 (whole splits for C3/C4)
SAMPLE_SETS: Dict[str, SampleSet] = {s.key: s for s in [
    SampleSet("train150", "train", 150, 1952),
    SampleSet("val100", "val", 100, 1019),
    SampleSet("test100", "test", 100, 1033),
    SampleSet("train400", "train", 400, 4358),
    SampleSet("train", "train", None, 10957),
    SampleSet("val", "val", None, 2348),
    SampleSet("test", "test", None, 2350),
    SampleSet("external", "external", None, 138),
]}

# Section 3.3: the within-source disease sample and within-Normal sources that the counts imply (asserted at run time)
EXPECTED_WITHIN_SOURCE = {
    "cohen_covid": ["COVID-19", "Pneumonia"],
    "kermany": ["Normal", "Pneumonia"],
    "nih_cxr14": ["Normal", "Pleural Effusion", "Pneumonia", "Pulmonary Nodule / Mass"],
    "shenzhen": ["Normal", "Tuberculosis"],
    "tbx11k": ["Normal", "Tuberculosis"],
}
# Section 3.2 counts of the main build: (source, class) -> (train, val, test); used by the tests and as a reference
PROTOCOL_COUNTS: Dict[Tuple[str, str], Tuple[int, int, int]] = {
    ("cohen_covid", "COVID-19"): (341, 60, 77), ("cohen_covid", "Normal"): (11, 3, 4),
    ("cohen_covid", "Pneumonia"): (133, 29, 29), ("cohen_covid", "Tuberculosis"): (8, 2, 1),
    ("covid_radiography", "COVID-19"): (1742, 384, 374), ("kermany", "Normal"): (377, 81, 81),
    ("kermany", "Pneumonia"): (1471, 316, 317), ("nih_cxr14", "Normal"): (836, 181, 183),
    ("nih_cxr14", "Pleural Effusion"): (2099, 457, 444), ("nih_cxr14", "Pneumonia"): (225, 49, 48),
    ("nih_cxr14", "Pulmonary Nodule / Mass"): (2110, 441, 449), ("shenzhen", "Normal"): (228, 49, 49),
    ("shenzhen", "Tuberculosis"): (235, 51, 50), ("tbx11k", "Normal"): (700, 150, 150),
    ("tbx11k", "Tuberculosis"): (441, 95, 94),
}
EXPECTED_WITHIN_NORMAL = ["kermany", "nih_cxr14", "shenzhen", "tbx11k"]
EXPECTED_C_D = 0.45


class ProtocolError(RuntimeError):
    """Raised when a realised quantity differs from what the protocol fixes; the run stops (section 3.3)."""


class AccessError(RuntimeError):
    """Raised when a once-only split (test, Montgomery) was already scored and --force-rerun was not given."""


# --------------------------------------------------------------------------- provenance
def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def provenance() -> Dict[str, Any]:
    """Code commit, protocol commit, whether lungai/novelty has uncommitted changes, and a UTC timestamp."""
    dirty = _git("status", "--porcelain", "--", "lungai/novelty")
    return {"code_commit": _git("rev-parse", "HEAD"),
            "code_uncommitted_changes_in_lungai_novelty": bool(dirty) if dirty is not None else None,
            "protocol_commit": PROTOCOL_COMMIT,
            "protocol_last_commit": _git("log", "-1", "--format=%H", "--", "docs/publication/NOVELTY_PROTOCOL.md"),
            "generated_at": now()}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    tmp.replace(path)


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# --------------------------------------------------------------------------- access log (section 11)
def claim_access(name: str, output: Path, force_rerun: bool = False, access_dir: Optional[Path] = None,
                 log_path: Optional[Path] = None) -> Dict[str, Any]:
    """Record that a once-only metric computation (test or Montgomery) is about to run.

    "Access" means computing a metric on the split (section 11). The marker is written BEFORE the metric is computed,
    so a crash after this call still counts as an access; re-running then needs ``force_rerun=True``, which is recorded
    in the marker and in the log and must be described in the protocol's section 13.
    """
    access_dir = access_dir or ACCESS_DIR
    log_path = log_path or ACCESS_LOG
    marker = access_dir / f"{name}.json"
    previous = read_json(marker) if marker.is_file() else None
    if previous is not None and not force_rerun:
        raise AccessError(f"'{name}' was already accessed at {previous.get('accessed_at')} (marker {marker}). "
                          "Each test/Montgomery computation runs once (protocol section 11); pass --force-rerun only "
                          "after logging the reason in section 13.")
    rec = {"access": name, "accessed_at": now(), "output": str(output), "force_rerun": bool(force_rerun),
           "previous_access": previous, **{k: v for k, v in provenance().items() if k != "generated_at"}}
    write_json(marker, rec)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in rec.items() if k != "previous_access"}) + "\n")
    return rec


# --------------------------------------------------------------------------- samples
def load_manifest(build: Path) -> pd.DataFrame:
    m = pd.read_csv(build / "manifest_v6.csv", dtype=str, keep_default_na=False)
    m["cache"] = [build / "cache" / f"{h}.npy" for h in m["raw_sha256"]]
    return m


def select(manifest: pd.DataFrame, sset: SampleSet, seed: int = SAMPLING_SEED) -> pd.DataFrame:
    """Rows of a sample set: the source_probe._sample rule (per (source, class) cell, min(n, cap), seed) or a whole split."""
    from lungai.evaluation.source_probe import _sample
    rows = manifest[manifest["split"] == sset.split]
    if sset.cap is None:
        return rows.reset_index(drop=True)
    return _sample(rows, sset.cap, seed)


def check_count(df: pd.DataFrame, sset: SampleSet) -> None:
    if len(df) != sset.expected_n:
        cells = df.groupby(["source_id", "lungai_label"]).size().to_dict()
        raise ProtocolError(f"sample '{sset.key}' has {len(df)} images, the protocol states {sset.expected_n}; the run "
                            f"stops and the difference is logged under section 13. Realised cells: {cells}")


def duplicates_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Byte-identical images appearing more than once in a sample are reported, not removed (section 3.1)."""
    dup = df["raw_sha256"].duplicated(keep=False)
    col = "source_id" if "source_id" in df.columns else "source"  # manifest rows or a feature index
    return {"rows_with_shared_hash": int(dup.sum()), "distinct_hashes_shared": int(df.loc[dup, "raw_sha256"].nunique()),
            "by_source": df.loc[dup, col].value_counts().to_dict()}


# --------------------------------------------------------------------------- minimum cell rule (section 3.3)
def qualifying_cells(scored_src: Sequence[str], scored_y: Sequence[str], train_src: Sequence[str],
                     train_y: Sequence[str], min_cell: int = MIN_CELL) -> Dict[str, List[str]]:
    """Sources with >= 2 (source, class) cells of >= min_cell images in the scored set and in the training set."""
    sc = pd.Series(list(zip(scored_src, scored_y))).value_counts()
    tc = pd.Series(list(zip(train_src, train_y))).value_counts()
    out: Dict[str, List[str]] = {}
    for (s, c), n in sc.items():
        if n >= min_cell and tc.get((s, c), 0) >= min_cell:
            out.setdefault(s, []).append(c)
    return {s: sorted(cs) for s, cs in sorted(out.items()) if len(cs) >= 2}


def within_normal_sources(scored_src, scored_y, train_src, train_y, min_cell: int = MIN_CELL) -> List[str]:
    sc = pd.Series([s for s, y in zip(scored_src, scored_y) if y == NORMAL_CLASS]).value_counts()
    tc = pd.Series([s for s, y in zip(train_src, train_y) if y == NORMAL_CLASS]).value_counts()
    return sorted(s for s, n in sc.items() if n >= min_cell and tc.get(s, 0) >= min_cell)


def chance_disease(cells: Dict[str, List[str]]) -> float:
    """c_D = mean over the within-source sources of 1/k_s (section 4.5)."""
    return float(np.mean([1.0 / len(c) for c in cells.values()]))


# --------------------------------------------------------------------------- fast scores on integer codes
def encode(values: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    idx = {v: i for i, v in enumerate(vocabulary)}
    return np.array([idx[v] for v in values], dtype=np.int64)


def balanced_accuracy(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Mean recall over the classes present in y (as sklearn.balanced_accuracy_score); other predictions are errors."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    support = cm.sum(1)
    present = support > 0
    return float((np.diag(cm)[present] / support[present]).mean())


def macro_f1(y: np.ndarray, p: np.ndarray, labels: Sequence[int], k: int) -> float:
    """Macro-F1 over ``labels`` (as sklearn f1_score(labels=..., average='macro', zero_division=0))."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    lab = np.asarray(labels)
    tp = np.diag(cm)[lab].astype(float)
    support = cm.sum(1)[lab]
    predicted = cm.sum(0)[lab]
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(lab)), where=denom > 0)
    return float(f1.mean())


def recall_per_class(y: np.ndarray, p: np.ndarray, labels: Sequence[int]) -> Dict[int, Optional[float]]:
    return {int(c): (float((p[y == c] == c).mean()) if (y == c).any() else None) for c in labels}


def roc_auc_binary(t: np.ndarray, score: np.ndarray) -> Optional[float]:
    """ROC-AUC by average ranks (ties handled as sklearn.roc_auc_score)."""
    from scipy.stats import rankdata
    t = np.asarray(t, bool)
    npos, nneg = int(t.sum()), int((~t).sum())
    if npos == 0 or nneg == 0:
        return None
    r = rankdata(score)
    return float((r[t].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def macro_roc_auc(y: np.ndarray, probs: np.ndarray) -> Optional[float]:
    """One-vs-rest macro ROC-AUC over the columns of probs; None if any class is absent (as metrics.ranking_metrics)."""
    vals = [roc_auc_binary(y == c, probs[:, c]) for c in range(probs.shape[1])]
    return None if any(v is None for v in vals) else float(np.mean(vals))


def within_source_f1(y: np.ndarray, p: np.ndarray, src: np.ndarray, cells: Dict[int, List[int]], k: int) -> float:
    """F1_ws (section 4.5): per source, macro-F1 over its qualifying classes on its images of those classes (labels
    restricted to those classes, any other prediction is an error); then the unweighted mean over the sources."""
    vals = []
    # PROTOCOL NOTE: "on that source's images" is read with section 3.3's "within-source disease sample": images of
    # excluded cells (e.g. cohen Normal, cohen Tuberculosis) are not scored; predictions outside the source's classes
    # still count as errors.
    for s, labs in cells.items():
        m = (src == s) & np.isin(y, labs)
        vals.append(macro_f1(y[m], p[m], labs, k))
    return float(np.mean(vals))


def bootstrap_indices(n: int, b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """(b, n) resample indices; the same matrix is used for every variant/iteration so differences are paired."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n, n) for _ in range(b)])


def percentile_ci(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"lower": None, "upper": None, "resamples": 0}
    return {"lower": float(np.percentile(v, 2.5)), "upper": float(np.percentile(v, 97.5)), "resamples": int(len(v))}


def class_codes() -> Tuple[List[str], Dict[str, int]]:
    return list(CLASS_NAMES), {c: i for i, c in enumerate(CLASS_NAMES)}


def logistic(C: float, max_iter: int):
    """The protocol's linear head: multinomial (lbfgs default), class_weight balanced, random_state 0."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced", random_state=0)


def full_proba(clf, X: np.ndarray, k: int) -> np.ndarray:
    """Predicted probabilities over all k classes (columns of classes absent from training are 0)."""
    p = clf.predict_proba(X)
    out = np.zeros((len(X), k))
    out[:, np.asarray(clf.classes_, dtype=int)] = p
    return out
