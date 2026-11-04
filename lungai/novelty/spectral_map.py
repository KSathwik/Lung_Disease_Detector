"""C1: spectral map of source and disease information in frozen DenseNet-121 features (protocol sections 4-6, 11 steps 1-3).

    python -m lungai.novelty.spectral_map --stage c1-main --split val      # step 1, validation first
    python -m lungai.novelty.spectral_map --stage c1-main --split test     # step 1, test once (access marker)
    python -m lungai.novelty.spectral_map --stage c1-served --split test   # step 2
    python -m lungai.novelty.spectral_map --stage c1-nofilter --split test # step 3

Per variant: StandardScaler + LogisticRegression(C=1, max_iter=2000, class_weight=balanced, random_state=0) source and
disease probes fitted on the train150 sample of that variant, scored on val100 or test100. Metrics: source balanced
accuracy BA_src (chance 1/6), within-source disease macro-F1 F1_ws (primary, minimum cell rule), six-class macro-F1
F1_all, within-Normal source balanced accuracy BA_norm, S*, D*, SDS = S* - D*, band-stop differences dS, dD and band-pass
retention; percentile bootstrap (B = 1000, 95%, unit = scored image, seed 1, the same indices for every variant).
Reads the feature caches of lungai.novelty.features; writes artifacts/novelty/c1_spectral_map.json (merged per
stage/split). A test split is scored once: a second run is refused unless --force-rerun is given.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lungai.config import NORMAL_CLASS
from lungai.novelty.common import (EXPECTED_C_D, EXPECTED_WITHIN_NORMAL, EXPECTED_WITHIN_SOURCE, OUT_DIR, SOURCES,
                                   ProtocolError, balanced_accuracy, bootstrap_indices, chance_disease, claim_access,
                                   class_codes, duplicates_report, encode, logistic, macro_f1, percentile_ci, provenance,
                                   qualifying_cells, read_json, within_normal_sources, within_source_f1, write_json)
from lungai.novelty.spectral import BANDS, PASS_VARIANTS, VARIANTS

C1_JSON = OUT_DIR / "c1_spectral_map.json"
STAGES: Dict[str, Dict[str, Any]] = {  # section 4.2 table
    "c1-main": {"extractor": "imagenet", "build": "main", "variants": VARIANTS, "splits": ["val", "test"]},
    "c1-served": {"extractor": "served", "build": "main", "variants": PASS_VARIANTS, "splits": ["test"]},
    "c1-nofilter": {"extractor": "imagenet", "build": "nofilter", "variants": PASS_VARIANTS, "splits": ["test"]},
}
SPLIT_SET = {"val": "val100", "test": "test100"}
TRAIN_SET = "train150"
CHANCE_SRC = 1.0 / 6.0
RETENTION_MIN_DENOMINATOR = 0.05
UNCERTAINTY_NOTE = ("Percentile bootstrap over scored images (B = 1000, seed 1, same indices for every variant). Probes "
                    "are fitted once on the fixed train sample: intervals describe sample variation of the scored "
                    "images, not training or seed variation.")


def fit_probe(x_train: np.ndarray, labels: np.ndarray):
    """Section 4.4 probe: StandardScaler fitted on the train sample + multinomial logistic regression."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(x_train)
    return sc, logistic(1.0, 2000).fit(sc.transform(x_train), labels)


def point_metrics(src: np.ndarray, src_pred: np.ndarray, y: np.ndarray, y_pred: np.ndarray,
                  cells: Dict[int, List[int]], normal_sources: Sequence[int], normal_code: int) -> Dict[str, float]:
    """BA_src, F1_ws, F1_all, BA_norm on integer-coded arrays (6 sources, 6 classes)."""
    m = (y == normal_code) & np.isin(src, normal_sources)
    return {"BA_src": balanced_accuracy(src, src_pred, 6),
            "F1_ws": within_source_f1(y, y_pred, src, cells, 6),
            "F1_all": macro_f1(y, y_pred, list(range(6)), 6),
            "BA_norm": balanced_accuracy(src[m], src_pred[m], 6)}


def derived(pm: Dict[str, Dict[str, float]], c_d: float) -> Dict[str, Any]:
    """S*, D*, SDS per variant; dS, dD per band (stop variants); retention per band (pass variants)."""
    out: Dict[str, Any] = {"variants": {}, "band_stop": {}, "band_pass_retention": {}}
    for v, m in pm.items():
        s = (m["BA_src"] - CHANCE_SRC) / (1 - CHANCE_SRC)
        d = (m["F1_ws"] - c_d) / (1 - c_d)
        out["variants"][v] = {"S_star": s, "D_star": d, "SDS": s - d}
    if "full" in pm:
        full = pm["full"]
        for b in BANDS:
            if f"stop_{b}" in pm:
                st = pm[f"stop_{b}"]
                out["band_stop"][b] = {"dS": full["BA_src"] - st["BA_src"], "dD": full["F1_ws"] - st["F1_ws"]}
            if f"pass_{b}" in pm:
                sf, df = out["variants"]["full"]["S_star"], out["variants"]["full"]["D_star"]
                sp, dp = out["variants"][f"pass_{b}"]["S_star"], out["variants"][f"pass_{b}"]["D_star"]
                out["band_pass_retention"][b] = {
                    "S": sp / sf if sf > RETENTION_MIN_DENOMINATOR else "not defined",
                    "D": dp / df if df > RETENTION_MIN_DENOMINATOR else "not defined"}
    return out


def score_variants(preds: Dict[str, Tuple[np.ndarray, np.ndarray]], src: np.ndarray, y: np.ndarray,
                   cells: Dict[int, List[int]], normal_sources: Sequence[int], normal_code: int, c_d: float,
                   boot: np.ndarray) -> Dict[str, Any]:
    """Point estimates and paired percentile intervals for every variant and every derived quantity.

    preds: variant -> (source predictions, disease predictions), integer codes aligned with src/y.
    boot: (B, n) resample indices shared by all variants.
    """
    pm = {v: point_metrics(src, sp, y, yp, cells, normal_sources, normal_code) for v, (sp, yp) in preds.items()}
    pd_ = derived(pm, c_d)
    draws_m: Dict[str, Dict[str, List[float]]] = {v: {k: [] for k in pm[v]} for v in pm}
    draws_d: Dict[str, Dict[str, List[float]]] = {v: {"S_star": [], "D_star": [], "SDS": []} for v in pm}
    draws_b: Dict[str, Dict[str, List[float]]] = {b: {"dS": [], "dD": []} for b in pd_["band_stop"]}
    for idx in boot:
        bm = {v: point_metrics(src[idx], sp[idx], y[idx], yp[idx], cells, normal_sources, normal_code)
              for v, (sp, yp) in preds.items()}
        bd = derived(bm, c_d)
        for v in pm:
            for k, val in bm[v].items():
                draws_m[v][k].append(val)
            for k, val in bd["variants"][v].items():
                draws_d[v][k].append(val)
        for b, d in bd["band_stop"].items():
            draws_b[b]["dS"].append(d["dS"])
            draws_b[b]["dD"].append(d["dD"])
    res: Dict[str, Any] = {"variants": {}, "band_stop": {}, "band_pass_retention": pd_["band_pass_retention"]}
    for v in pm:
        res["variants"][v] = {k: {"estimate": pm[v][k], "ci95": percentile_ci(draws_m[v][k])} for k in pm[v]}
        res["variants"][v].update({k: {"estimate": pd_["variants"][v][k], "ci95": percentile_ci(draws_d[v][k])}
                                   for k in ("S_star", "D_star", "SDS")})
    for b, d in pd_["band_stop"].items():
        res["band_stop"][b] = {k: {"estimate": d[k], "ci95": percentile_ci(draws_b[b][k])} for k in ("dS", "dD")}
    return res


def hypotheses(res: Dict[str, Any]) -> Dict[str, Any]:
    """H1a and H1c of section 6, reported whether they hold or not (H1b is descriptive)."""
    v = res["variants"]
    out: Dict[str, Any] = {}
    if "full" in v:
        f = v["full"]["BA_src"]
        out["H1a_full_BA_src_above_chance"] = {"point": f["estimate"] > CHANCE_SRC,
                                               "ci_lower_above_chance": (f["ci95"]["lower"] or 0) > CHANCE_SRC}
    passes = {k: x["BA_norm"]["estimate"] for k, x in v.items() if k.startswith("pass_")}
    if passes:
        out["H1c_BA_norm_above_0.25_in_a_pass_variant"] = {"point": any(x > 0.25 for x in passes.values()),
                                                           "variants_above": sorted(k for k, x in passes.items() if x > 0.25)}
    out["H1b"] = "descriptive: band profiles of BA_src and F1_ws are reported; no band was predicted"
    return out


def run_stage(stage: str, split: str) -> Dict[str, Any]:
    from lungai.novelty.features import feature_path, index_path, load_features
    spec = STAGES[stage]
    ext, build, variants = spec["extractor"], spec["build"], spec["variants"]
    sets = [TRAIN_SET, SPLIT_SET[split]]
    missing = [str(p) for v in variants for s in sets for p in (feature_path(ext, v, s, build), index_path(ext, v, s, build))
               if not p.is_file()]
    if missing:
        return {"status": "not run", "missing_feature_files": missing}
    classes, cidx = class_codes()
    sidx = {s: i for i, s in enumerate(SOURCES)}
    feats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    ref_tr = ref_sc = None
    for v in variants:
        xtr, itr = load_features(ext, v, TRAIN_SET, build)
        xsc, isc = load_features(ext, v, SPLIT_SET[split], build)
        if ref_tr is None:
            ref_tr, ref_sc = itr, isc
        elif not (itr["image_id"].equals(ref_tr["image_id"]) and isc["image_id"].equals(ref_sc["image_id"])):
            raise ProtocolError(f"variant {v} is not row-aligned with the other variants (pairing, section 3.3)")
        feats[v] = (xtr, xsc)
    cells_named = qualifying_cells(ref_sc["source"], ref_sc["label"], ref_tr["source"], ref_tr["label"])
    normal_named = within_normal_sources(ref_sc["source"], ref_sc["label"], ref_tr["source"], ref_tr["label"])
    c_d = chance_disease(cells_named)
    if cells_named != EXPECTED_WITHIN_SOURCE or normal_named != EXPECTED_WITHIN_NORMAL or abs(c_d - EXPECTED_C_D) > 1e-12:
        raise ProtocolError(f"within-source sample differs from section 3.3: cells {cells_named}, normal {normal_named}, "
                            f"c_D {c_d}")
    src_tr, y_tr = ref_tr["source"].to_numpy(), ref_tr["label"].to_numpy()
    src = encode(ref_sc["source"], SOURCES)
    y = encode(ref_sc["label"], classes)
    cells = {sidx[s]: [cidx[c] for c in cs] for s, cs in cells_named.items()}
    normal_sources = [sidx[s] for s in normal_named]
    preds = {}
    for v in variants:
        xtr, xsc = feats[v]
        sc_s, clf_s = fit_probe(xtr, src_tr)
        sc_d, clf_d = fit_probe(xtr, y_tr)
        preds[v] = (encode(clf_s.predict(sc_s.transform(xsc)), SOURCES), encode(clf_d.predict(sc_d.transform(xsc)), classes))
    boot = bootstrap_indices(len(y))
    res = score_variants(preds, src, y, cells, normal_sources, cidx[NORMAL_CLASS], c_d, boot)
    return {"status": "complete", "extractor": ext, "build": build, "scored_split": split,
            "train_set": TRAIN_SET, "scored_set": SPLIT_SET[split], "n_train": int(len(ref_tr)), "n_scored": int(len(ref_sc)),
            "train_cells": {f"{s}|{c}": int(n) for (s, c), n in ref_tr.groupby(["source", "label"]).size().items()},
            "scored_cells": {f"{s}|{c}": int(n) for (s, c), n in ref_sc.groupby(["source", "label"]).size().items()},
            "duplicates_train": duplicates_report(ref_tr), "duplicates_scored": duplicates_report(ref_sc),
            "within_source_cells": cells_named, "within_source_excluded": _excluded(ref_sc, cells_named),
            "within_normal_sources": normal_named, "c_D": c_d, "chance_source": CHANCE_SRC,
            "chance_within_normal": 1.0 / len(normal_named), "probe": "StandardScaler + LogisticRegression(C=1.0, "
            "max_iter=2000, class_weight='balanced', random_state=0), lbfgs multinomial", "bootstrap": {
                "B": int(boot.shape[0]), "seed": 1, "unit": "scored image", "interval": "95% percentile"},
            "uncertainty_note": UNCERTAINTY_NOTE, **res, "hypotheses": hypotheses(res), **provenance()}


def _excluded(scored, cells: Dict[str, List[str]]) -> List[str]:
    out = []
    for (s, c), n in scored.groupby(["source", "label"]).size().items():
        if c not in cells.get(s, []):
            out.append(f"{s} {c} ({int(n)} images)")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--force-rerun", action="store_true", help="score a test split again (must be logged in section 13)")
    args = ap.parse_args(argv)
    if args.split not in STAGES[args.stage]["splits"]:
        ap.error(f"{args.stage} scores only {STAGES[args.stage]['splits']} (protocol section 11)")
    data = read_json(C1_JSON)
    data.setdefault("protocol", "docs/publication/NOVELTY_PROTOCOL.md")
    timing = read_json(OUT_DIR / "timing.json")
    if timing:
        data["timing"] = timing
    stages = data.setdefault("stages", {})
    if args.split == "test":
        if args.stage == "c1-main" and (stages.get("c1-main", {}).get("val", {}).get("status") != "complete"):
            print("c1-main validation results must be written before the test split (protocol section 11)")
            return 2
        from lungai.novelty.features import feature_path
        spec = STAGES[args.stage]
        if not all(feature_path(spec["extractor"], v, s, spec["build"]).is_file()
                   for v in spec["variants"] for s in (TRAIN_SET, "test100")):
            stages.setdefault(args.stage, {})["test"] = {"status": "not run", "reason": "feature caches incomplete"}
            write_json(C1_JSON, data)
            print(f"{args.stage} test: not run (feature caches incomplete); no access recorded")
            return 1
        claim_access(f"{args.stage}_test", C1_JSON, args.force_rerun)
    result = run_stage(args.stage, args.split)
    stages.setdefault(args.stage, {})[args.split] = result
    data.update({"last_written": provenance()})
    write_json(C1_JSON, data)
    print(f"{args.stage} {args.split}: {result['status']}; wrote {C1_JSON}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
