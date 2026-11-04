"""C4: leave-one-source-out (LOSO) linear heads on frozen full-image features (protocol section 8, section 11 step 5).

    python -m lungai.novelty.loso --part main         # C selection on validation, held-out test matrices, once (marker)
    python -m lungai.novelty.loso --part montgomery   # all-six-source heads on Montgomery, once (marker)

Primary matrix: ImageNet features. Secondary matrix: run-2 backbone features, named "linear-head leave-one-source-out on
frozen features"; that backbone saw all six sources in training, so these numbers are never called held-out or
external. Head: StandardScaler (fitted on the fold's training features) + LogisticRegression(C, class_weight balanced,
max_iter=1000, random_state=0); C from {0.01, 0.1, 1.0} by macro-F1 on the validation images of the non-held-out sources
over the classes that occur in them (ties: smaller C). Training uses the train split of the other five sources; the
held-out source's test split is scored over its evaluated classes (>= 20 test images there and >= 50 train images in the
other sources; a source needs >= 2 evaluated classes). In-distribution reference: the same recipe on all six sources.
Binary view: 1 - P(Normal), for held-out sources with >= 20 Normal and >= 20 abnormal test images. Bootstrap B = 1000,
95%, unit = held-out test image, seed 1. Writes artifacts/novelty/c4_loso.json. Every number carries the
patient-identity caveat.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from lungai.config import NORMAL_CLASS
from lungai.novelty.common import (OUT_DIR, SOURCES, ProtocolError, balanced_accuracy, bootstrap_indices, claim_access,
                                   class_codes, full_proba, logistic, macro_f1, percentile_ci, provenance, read_json,
                                   recall_per_class, roc_auc_binary, write_json)

C4_JSON = OUT_DIR / "c4_loso.json"
EXTRACTORS = {"imagenet": "ImageNet DenseNet-121 features (primary matrix)",
              "served": "linear-head leave-one-source-out on frozen features (secondary matrix; the run-2 backbone saw "
                        "all six sources in training: not held-out, not external)"}
C_GRID = (0.01, 0.1, 1.0)
HEAD_MAX_ITER = 1000
MIN_TEST = 20
MIN_TRAIN_OTHER = 50
MIN_BINARY = 20
EXPECTED_INCLUDED = {"cohen_covid": ["COVID-19", "Pneumonia"], "kermany": ["Normal", "Pneumonia"],
                     "nih_cxr14": ["Normal", "Pneumonia"], "shenzhen": ["Normal", "Tuberculosis"],
                     "tbx11k": ["Normal", "Tuberculosis"]}
EXPECTED_BINARY = ["kermany", "nih_cxr14", "shenzhen", "tbx11k"]
CAVEAT = ("Patient-identity caveat: separation is proven only for rows with known patient identity; 4,130 TBX11K and "
          "COVID-19 Radiography rows are grouped per image, and Kermany identities are derived from file names.")


def inclusion(train_idx: pd.DataFrame, test_idx: pd.DataFrame, sources: Sequence[str] = SOURCES) -> Dict[str, Any]:
    """Section 8 inclusion rule. Returns per source: evaluated classes, excluded classes (with reasons), included flag,
    and whether the binary view applies (>= 20 Normal and >= 20 abnormal test images)."""
    out: Dict[str, Any] = {}
    for h in sources:
        te = test_idx[test_idx["source"] == h]["label"].value_counts()
        tr_other = train_idx[train_idx["source"] != h]["label"].value_counts()
        evaluated, excluded = [], {}
        for c, n in sorted(te.items()):
            n_tr = int(tr_other.get(c, 0))
            if n >= MIN_TEST and n_tr >= MIN_TRAIN_OTHER:
                evaluated.append(c)
            else:
                excluded[c] = f"{int(n)} test images in {h}, {n_tr} train images in the other sources"
        n_norm = int(te.get(NORMAL_CLASS, 0))
        n_abn = int(te.sum() - n_norm)
        out[h] = {"evaluated_classes": evaluated, "excluded_classes": excluded, "included": len(evaluated) >= 2,
                  "binary_view": n_norm >= MIN_BINARY and n_abn >= MIN_BINARY,
                  "test_normal": n_norm, "test_abnormal": n_abn}
    return out


def select_C(ztr: np.ndarray, ytr: np.ndarray, zva: np.ndarray, yva: np.ndarray, labels: List[int]) -> Dict[str, Any]:
    scores = {}
    for C in C_GRID:
        h = logistic(C, HEAD_MAX_ITER).fit(ztr, ytr)
        scores[C] = macro_f1(yva, full_proba(h, zva, 6).argmax(1), labels, 6)
    best = max(C_GRID, key=lambda c: (scores[c], -c))  # ties go to the smaller C
    return {"C": best, "val_macro_f1": {str(c): v for c, v in scores.items()}, "labels": labels}


def fit_head(xtr, ytr, xva, yva):
    """Scaler fitted on the training features; C selected on validation over the classes present in yva."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(xtr)
    ztr, zva = sc.transform(xtr), sc.transform(xva)
    labels = sorted(set(yva.tolist()))
    missing = sorted(set(labels) - set(ytr.tolist()))
    if missing:
        # PROTOCOL NOTE: section 8 scores C over the classes occurring in the validation images; with the section 3.2
        # counts every such class also has training images. If not, the run stops rather than choosing silently.
        raise ProtocolError(f"validation classes {missing} have no training image in this fold")
    sel = select_C(ztr, ytr, zva, yva, labels)
    return sc, logistic(sel["C"], HEAD_MAX_ITER).fit(ztr, ytr), sel


def fold_metrics(y: np.ndarray, probs: np.ndarray, labels: List[int], normal: int) -> Dict[str, Any]:
    """Macro-F1 and balanced accuracy over the evaluated classes (images of those classes only; any other prediction
    is an error) and per-class recall."""
    m = np.isin(y, labels)
    pred = probs.argmax(1)
    return {"macro_f1": macro_f1(y[m], pred[m], labels, 6), "balanced_accuracy": balanced_accuracy(y[m], pred[m], 6),
            "recall": recall_per_class(y[m], pred[m], labels), "n": int(m.sum())}


def binary_metrics(y: np.ndarray, probs: np.ndarray, normal: int) -> Dict[str, Any]:
    t = y != normal
    score = 1.0 - probs[:, normal]
    pred = score >= 0.5  # PROTOCOL NOTE: "balanced accuracy at 0.5"; a score of exactly 0.5 is called abnormal
    sens = float(pred[t].mean()) if t.any() else None
    spec = float((~pred[~t]).mean()) if (~t).any() else None
    return {"roc_auc": roc_auc_binary(t, score), "balanced_accuracy": None if sens is None or spec is None else (sens + spec) / 2,
            "sensitivity": sens, "specificity": spec, "n_normal": int((~t).sum()), "n_abnormal": int(t.sum())}


def _ci_block(boot, fn, keys) -> Dict[str, Any]:
    draws = [fn(idx) for idx in boot]
    return {k: percentile_ci([d[k] for d in draws]) for k in keys}


def run_matrix(ext: str, data: Dict[str, Any]) -> Dict[str, Any]:
    classes, cidx = class_codes()
    normal = cidx[NORMAL_CLASS]
    (xtr, itr), (xva, iva), (xte, ite) = data["train"], data["val"], data["test"]
    enc = lambda idx: np.array([cidx[c] for c in idx["label"]], dtype=np.int64)  # noqa: E731
    ytr, yva, yte = enc(itr), enc(iva), enc(ite)
    inc = inclusion(itr, ite)
    included = {h: v["evaluated_classes"] for h, v in inc.items() if v["included"]}
    if included != EXPECTED_INCLUDED or sorted(h for h, v in inc.items() if v["binary_view"]) != EXPECTED_BINARY:
        raise ProtocolError(f"inclusion differs from section 8: {inc}")
    sc_ref, ref, ref_sel = fit_head(xtr, ytr, xva, yva)
    # PROTOCOL NOTE: the in-distribution reference uses "the same head recipe"; its C is selected the same way, on the
    # validation images of all six sources (all classes present there).
    folds = {}
    for h in SOURCES:
        if not inc[h]["included"]:
            folds[h] = {"included": False, **inc[h]}
            continue
        mtr = (itr["source"] != h).to_numpy()
        mva = (iva["source"] != h).to_numpy()
        mte = (ite["source"] == h).to_numpy()
        sc, head, sel = fit_head(xtr[mtr], ytr[mtr], xva[mva], yva[mva])
        p_lo = full_proba(head, sc.transform(xte[mte]), 6)
        p_ref = full_proba(ref, sc_ref.transform(xte[mte]), 6)
        y_h = yte[mte]
        labels = [cidx[c] for c in inc[h]["evaluated_classes"]]
        lo, rf = fold_metrics(y_h, p_lo, labels, normal), fold_metrics(y_h, p_ref, labels, normal)
        boot = bootstrap_indices(len(y_h))
        both = lambda idx: {**{f"loso_{k}": v for k, v in fold_metrics(y_h[idx], p_lo[idx], labels, normal).items()  # noqa: E731
                               if k in ("macro_f1", "balanced_accuracy")},
                            **{f"ref_{k}": v for k, v in fold_metrics(y_h[idx], p_ref[idx], labels, normal).items()
                               if k in ("macro_f1", "balanced_accuracy")}}
        draws = [both(idx) for idx in boot]
        ci = {k: percentile_ci([d[k] for d in draws]) for k in draws[0]}
        ci["gap_macro_f1"] = percentile_ci([d["ref_macro_f1"] - d["loso_macro_f1"] for d in draws])
        ci["gap_balanced_accuracy"] = percentile_ci([d["ref_balanced_accuracy"] - d["loso_balanced_accuracy"] for d in draws])
        fold = {"included": True, **inc[h], "train_images": int(mtr.sum()), "val_images_for_C": int(mva.sum()),
                "classes_in_training": sorted(classes[c] for c in set(ytr[mtr].tolist())), "C_selection": sel,
                "loso": {**lo, "recall": {classes[c]: v for c, v in lo["recall"].items()}},
                "reference": {**rf, "recall": {classes[c]: v for c, v in rf["recall"].items()}},
                "gap_macro_f1": rf["macro_f1"] - lo["macro_f1"],
                "gap_balanced_accuracy": rf["balanced_accuracy"] - lo["balanced_accuracy"], "ci95": ci}
        if inc[h]["binary_view"]:
            bv = binary_metrics(y_h, p_lo, normal)
            bci = _ci_block(boot, lambda idx: binary_metrics(y_h[idx], p_lo[idx], normal), ["roc_auc", "balanced_accuracy"])
            fold["binary"] = {**bv, "ci95": bci, "abnormal_definition": "any non-Normal label, including classes absent "
                              "from training"}
        folds[h] = fold
    return {"status": "complete", "label": EXTRACTORS[ext], "reference_C_selection": ref_sel, "folds": folds,
            "bootstrap": {"B": 1000, "seed": 1, "unit": "held-out test image"}, "caveat": CAVEAT, **provenance()}


def run_montgomery(ext: str, data: Dict[str, Any]) -> Dict[str, Any]:
    from lungai.evaluation.metrics import binary_abnormality_report
    classes, cidx = class_codes()
    normal = cidx[NORMAL_CLASS]
    (xtr, itr), (xva, iva), (xex, iex) = data["train"], data["val"], data["external"]
    enc = lambda idx: np.array([cidx[c] for c in idx["label"]], dtype=np.int64)  # noqa: E731
    sc, head, sel = fit_head(xtr, enc(itr), xva, enc(iva))
    p = full_proba(head, sc.transform(xex), 6)
    t = (iex["label"] != NORMAL_CLASS).to_numpy()
    pred = p.argmax(1)
    rep = binary_abnormality_report(t, pred != normal, 1.0 - p[:, normal])
    rep["exact_tuberculosis_recall"] = float((pred[t] == cidx["Tuberculosis"]).mean()) if t.any() else None
    return {"label": EXTRACTORS[ext], "C_selection": sel, "n": int(len(iex)), **rep,
            "note": "Montgomery was already used for runs 1-5; it is not untouched. " + CAVEAT}


def main(argv: Optional[List[str]] = None) -> int:
    from lungai.novelty.features import is_cached, load_features
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["main", "montgomery"], required=True)
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args(argv)
    doc = read_json(C4_JSON)
    doc.setdefault("protocol", "docs/publication/NOVELTY_PROTOCOL.md")
    sets = ("train", "val", "test") if args.part == "main" else ("train", "val", "external")
    avail = [e for e in EXTRACTORS if all(is_cached(e, "full", s) for s in sets)]
    if not avail:
        print("no extractor has its full-split feature caches (lungai.novelty.features --stage c3c4)")
        return 2
    if args.part == "montgomery" and any(doc.get("matrices", {}).get(e, {}).get("status") != "complete" for e in avail):
        print("run --part main first (section 11 step 5)")
        return 2
    claim_access(f"c4_{args.part}", C4_JSON, args.force_rerun)
    key = "matrices" if args.part == "main" else "montgomery"
    block = doc.setdefault(key, {})
    for e in EXTRACTORS:  # ImageNet first, then the trained backbone (section 11)
        if e not in avail:
            block[e] = {"status": "not run", "reason": "feature caches missing"}
            continue
        data = {s: load_features(e, "full", s) for s in sets}
        print(f"C4 {args.part}: {e}", flush=True)
        block[e] = run_matrix(e, data) if args.part == "main" else run_montgomery(e, data)
    doc["last_written"] = provenance()
    write_json(C4_JSON, doc)
    print(f"wrote {C4_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
