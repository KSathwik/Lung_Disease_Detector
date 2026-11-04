"""C3: INLP source erasure on frozen full-image features, with a LEACE check (protocol section 7, section 11 step 4).

    python -m lungai.novelty.inlp_eval --part val          # validation curve, k* and C choice (repeatable)
    python -m lungai.novelty.inlp_eval --part test         # test table, once (access marker)
    python -m lungai.novelty.inlp_eval --part montgomery   # k = 0 and k = k* heads on Montgomery, once (access marker)

Features: features_<extractor>_full_{train,val,test,external}.npy (lungai.novelty.features --stage c3c4), extractors
``imagenet`` (primary) and ``served`` (secondary, "trained backbone, not a held-out test"). StandardScaler fitted on all
train images; INLP in the standardised space (lungai.novelty.inlp). Disease head: LogisticRegression(C, class_weight
balanced, max_iter=1000, random_state=0) on projected train features; C from {0.01, 0.1, 1.0} by validation six-class
macro-F1 on unprojected features, kept for every k. Metrics per k on the full val/test splits: source balanced accuracy
(fresh classifier), F1_ws (minimum cell rule on full-split counts), F1_all, accuracy, macro ROC-AUC; test changes versus
k = 0 with paired bootstrap 95% intervals (B = 1000, unit = test image, seed 1). Writes artifacts/novelty/c3_inlp.json
and the fitted heads to artifacts/novelty/c3_models_<extractor>.pkl (so test and Montgomery reuse the val-part fits).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from lungai.config import NORMAL_CLASS
from lungai.novelty.common import (OUT_DIR, SOURCES, ProtocolError, balanced_accuracy, bootstrap_indices,
                                   claim_access, class_codes, encode, full_proba, logistic, macro_f1, macro_roc_auc,
                                   percentile_ci, provenance, qualifying_cells, read_json, within_source_f1, write_json)
from lungai.novelty.inlp import INLP_MAX_ITER, inlp, leace_fit, projection_at, summary

C3_JSON = OUT_DIR / "c3_inlp.json"
EXTRACTORS = {"imagenet": "ImageNet DenseNet-121 (primary; no project training)",
              "served": "run-2 backbone: trained backbone, not a held-out test"}
C_GRID = (0.01, 0.1, 1.0)
HEAD_MAX_ITER = 1000


def models_path(ext: str):
    return OUT_DIR / f"c3_models_{ext}.pkl"


def _load(ext: str) -> Optional[Dict[str, Any]]:
    from lungai.novelty.features import is_cached, load_features
    if not all(is_cached(ext, "full", s) for s in ("train", "val", "test", "external")):
        return None
    return {s: load_features(ext, "full", s) for s in ("train", "val", "test", "external")}


def disease_head(C: float):
    return logistic(C, HEAD_MAX_ITER)


def disease_scores(y: np.ndarray, probs: np.ndarray, src: np.ndarray, cells: Dict[int, List[int]]) -> Dict[str, Any]:
    pred = probs.argmax(1)
    return {"F1_ws": within_source_f1(y, pred, src, cells, 6), "F1_all": macro_f1(y, pred, list(range(6)), 6),
            "accuracy": float((pred == y).mean()), "macro_roc_auc": macro_roc_auc(y, probs)}


def _coded(idx, classes):
    return encode(idx["source"], SOURCES), encode(idx["label"], classes)


def _cells(idx_scored, idx_train, classes) -> Dict[str, Any]:
    named = qualifying_cells(idx_scored["source"], idx_scored["label"], idx_train["source"], idx_train["label"])
    sidx = {s: i for i, s in enumerate(SOURCES)}
    cidx = {c: i for i, c in enumerate(classes)}
    return {"named": named, "coded": {sidx[s]: [cidx[c] for c in cs] for s, cs in named.items()}}


def part_val(ext: str, data: Dict[str, Any]) -> Dict[str, Any]:
    from sklearn.preprocessing import StandardScaler
    classes, _ = class_codes()
    (xtr, itr), (xva, iva) = data["train"], data["val"]
    sc = StandardScaler().fit(xtr)
    ztr, zva = sc.transform(xtr), sc.transform(xva)
    s_tr, y_tr = _coded(itr, classes)
    s_va, y_va = _coded(iva, classes)
    cells = _cells(iva, itr, classes)
    # C chosen once on unprojected features by validation six-class macro-F1
    c_scores = {}
    for C in C_GRID:
        h = disease_head(C).fit(ztr, y_tr)
        c_scores[C] = macro_f1(y_va, full_proba(h, zva, 6).argmax(1), list(range(6)), 6)
    # PROTOCOL NOTE: section 7 does not state a tie rule for C; ties go to the smaller C (as section 8 states for C4).
    C = max(C_GRID, key=lambda c: (c_scores[c], -c))
    res = inlp(ztr, s_tr, zva, s_va)
    heads, curve = [], []
    for k, P in enumerate(res.projections):
        h = disease_head(C).fit(ztr @ P, y_tr)
        heads.append(h)
        curve.append({"k": k, "removed_dims": res.removed_dims[k], "val_source_ba": res.val_source_ba[k],
                      **{f"val_{m}": v for m, v in disease_scores(y_va, full_proba(h, zva @ P, 6), s_va, cells["coded"]).items()}})
    er = leace_fit(ztr, s_tr)
    l_src = logistic(1.0, INLP_MAX_ITER).fit(er.transform(ztr), s_tr)
    l_head = disease_head(C).fit(er.transform(ztr), y_tr)
    leace = {"rank": er.rank, "val_source_ba": balanced_accuracy(s_va, l_src.predict(er.transform(zva)), 6),
             **{f"val_{m}": v for m, v in disease_scores(y_va, full_proba(l_head, er.transform(zva), 6), s_va,
                                                         cells["coded"]).items()}}
    with open(models_path(ext), "wb") as f:
        pickle.dump({"scaler": sc, "C": C, "basis": res.basis, "removed_dims": res.removed_dims, "source_classifiers": res.source_classifiers,
                     "disease_heads": heads, "leace": er, "leace_source": l_src, "leace_head": l_head,
                     "k_star": res.k_star, "provenance": provenance()}, f)
    return {"status": "complete", "label": EXTRACTORS[ext], "n_train": int(len(itr)), "n_val": int(len(iva)),
            "C_scores_val_macro_f1": {str(c): v for c, v in c_scores.items()}, "C": C, "inlp": summary(res),
            "within_source_cells_val": cells["named"], "curve": curve, "leace": leace, **provenance()}


def _load_models(ext: str) -> Dict[str, Any]:
    with open(models_path(ext), "rb") as f:
        mdl = pickle.load(f)
    mdl["projections"] = [projection_at(mdl["basis"], mdl["removed_dims"], k) for k in range(mdl["k_star"] + 1)]
    return mdl


def part_test(ext: str, data: Dict[str, Any]) -> Dict[str, Any]:
    classes, _ = class_codes()
    mdl = _load_models(ext)
    (xte, ite), (_, itr) = data["test"], data["train"]
    zte = mdl["scaler"].transform(xte)
    s_te, y_te = _coded(ite, classes)
    cells = _cells(ite, itr, classes)
    boot = bootstrap_indices(len(y_te))
    probs = [full_proba(h, zte @ P, 6) for h, P in zip(mdl["disease_heads"], mdl["projections"])]
    ze = mdl["leace"].transform(zte)
    p_leace = full_proba(mdl["leace_head"], ze, 6)

    def scorer(p):
        return lambda idx: disease_scores(y_te[idx], p[idx], s_te[idx], cells["coded"])

    s0 = scorer(probs[0])
    base_draws = [s0(idx) for idx in boot]
    rows = []
    for k, (P, clf, p) in enumerate(zip(mdl["projections"], mdl["source_classifiers"], probs)):
        row = {"k": k, "test_source_ba": balanced_accuracy(s_te, encode_pred(clf, zte @ P), 6),
               **{f"test_{m}": v for m, v in disease_scores(y_te, p, s_te, cells["coded"]).items()}}
        if k > 0:
            sk = scorer(p)
            draws = [sk(idx) for idx in boot]
            row["change_vs_k0"] = {m: {"estimate": _sub(row[f"test_{m}"], rows[0][f"test_{m}"]),
                                       "ci95": percentile_ci([_sub(d[m], b[m]) for d, b in zip(draws, base_draws)])}
                                   for m in ("F1_ws", "F1_all", "accuracy", "macro_roc_auc")}
        rows.append(row)
    leace = {"test_source_ba": balanced_accuracy(s_te, encode_pred(mdl["leace_source"], ze), 6),
             **{f"test_{m}": v for m, v in disease_scores(y_te, p_leace, s_te, cells["coded"]).items()}}
    ld = [scorer(p_leace)(idx) for idx in boot]
    leace["change_vs_k0"] = {m: {"estimate": _sub(leace[f"test_{m}"], rows[0][f"test_{m}"]),
                                 "ci95": percentile_ci([_sub(d[m], b[m]) for d, b in zip(ld, base_draws)])}
                             for m in ("F1_ws", "F1_all", "accuracy", "macro_roc_auc")}
    return {"status": "complete", "label": EXTRACTORS[ext], "n_test": int(len(ite)), "k_star": mdl["k_star"],
            "within_source_cells_test": cells["named"], "table": rows, "leace_check": leace,
            "bootstrap": {"B": int(boot.shape[0]), "seed": 1, "unit": "test image"}, **provenance()}


def _sub(a, b):
    return None if a is None or b is None else a - b


def encode_pred(clf, z: np.ndarray) -> np.ndarray:
    return np.asarray(clf.predict(z), dtype=np.int64)


def part_montgomery(ext: str, data: Dict[str, Any]) -> Dict[str, Any]:
    from lungai.evaluation.metrics import binary_abnormality_report
    classes, cidx = class_codes()
    mdl = _load_models(ext)
    xex, iex = data["external"]
    z = mdl["scaler"].transform(xex)
    t = (iex["label"] != NORMAL_CLASS).to_numpy()
    out = {"n": int(len(iex)), "labels": iex["label"].value_counts().to_dict(),
           "note": "Montgomery was already used for runs 1-5; it is not untouched. Not used for any choice."}
    for k in sorted({0, mdl["k_star"]}):
        p = full_proba(mdl["disease_heads"][k], z @ mdl["projections"][k], 6)
        pred = p.argmax(1)
        rep = binary_abnormality_report(t, pred != cidx[NORMAL_CLASS], 1.0 - p[:, cidx[NORMAL_CLASS]])
        rep["exact_tuberculosis_recall"] = float((pred[t] == cidx["Tuberculosis"]).mean()) if t.any() else None
        out["k_star" if k == mdl["k_star"] and k != 0 else "k0"] = {"k": k, **rep}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["val", "test", "montgomery"], required=True)
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args(argv)
    doc = read_json(C3_JSON)
    doc.setdefault("protocol", "docs/publication/NOVELTY_PROTOCOL.md")
    exts = doc.setdefault("extractors", {})
    loaded = {e: _load(e) for e in EXTRACTORS}
    if args.part != "val":
        need = [e for e in EXTRACTORS if loaded[e] is not None]
        if not need or any(exts.get(e, {}).get("val", {}).get("status") != "complete" or not models_path(e).is_file()
                           for e in need):
            print("run --part val for every available extractor first (val curve before test, section 11)")
            return 2
        if args.part == "montgomery" and any(exts.get(e, {}).get("test", {}).get("status") != "complete" for e in need):
            print("the test table must be written before Montgomery (section 7)")
            return 2
        claim_access(f"c3_{args.part}", C3_JSON, args.force_rerun)
    fn = {"val": part_val, "test": part_test, "montgomery": part_montgomery}[args.part]
    for e in EXTRACTORS:
        if loaded[e] is None:
            exts.setdefault(e, {})[args.part] = {"status": "not run", "reason": "feature caches missing"}
            continue
        print(f"C3 {args.part}: {e}", flush=True)
        exts.setdefault(e, {})[args.part] = fn(e, loaded[e])
    if args.part == "test":
        for e in EXTRACTORS:
            v, cur = exts.get(e, {}).get("val", {}), exts.get(e, {}).get(args.part, {})
            if v.get("inlp") and cur.get("k_star") is not None and cur["k_star"] != v["inlp"]["k_star"]:
                raise ProtocolError(f"{e}: k* of the stored models differs from the val curve")
    doc["last_written"] = provenance()
    write_json(C3_JSON, doc)
    print(f"wrote {C3_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
