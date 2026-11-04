"""C2-lite (exploratory linear head on band-stopped ImageNet features) and the section 10 C2 go/no-go rule.

    python -m lungai.novelty.c2_lite --part select     # criterion 1 from the C1-main validation results; prints b*
    python -m lungai.novelty.features --stage c2-lite --band <b*>
    python -m lungai.novelty.features --stage c2-control --band <b*>   # only if some band satisfies criterion 1
    python -m lungai.novelty.c2_lite --part val        # C2-lite on validation (+ matched-energy control)
    python -m lungai.novelty.c2_lite --part test       # C2-lite test table, once, exploratory (access marker)
    python -m lungai.novelty.c2_lite --part decision   # section 10 GO / NO-GO from validation numbers only

All selection uses validation data (val100, ImageNet extractor, main cache). Condition 4 (run 5 and the M.Tech
deliverables finished, and the student approving one night of CPU time before 22:00) is never set by this script: it is
reported as "requires student approval", so the script can at most report "GO pending condition 4".
Writes artifacts/novelty/c2_decision.json.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lungai.config import NORMAL_CLASS
from lungai.novelty.common import (EXPECTED_C_D, OUT_DIR, SOURCES, ProtocolError, bootstrap_indices, chance_disease,
                                   claim_access, class_codes, encode, provenance, qualifying_cells, read_json,
                                   within_normal_sources, write_json)
from lungai.novelty.spectral import BANDS

C2_JSON = OUT_DIR / "c2_decision.json"
C1_JSON = OUT_DIR / "c1_spectral_map.json"
DS_MIN = 0.05  # criterion 1 and 3
DD_MAX = 0.02  # criterion 1 and 3
DD_UPPER_CI_MAX = 0.03  # criterion 1
CONTROL_SEEDS = (1, 2, 3)
CONDITION_4 = "requires student approval"


# --------------------------------------------------------------------------- the rule (pure functions)
def criterion_1(band_stop: Dict[str, Any]) -> Dict[str, Any]:
    """band_stop: C1-main validation ``band_stop`` block (band -> {dS, dD: {estimate, ci95}})."""
    per_band = {}
    for b in BANDS:
        if b not in band_stop:
            continue
        ds, dd = band_stop[b]["dS"], band_stop[b]["dD"]
        ok = (ds["estimate"] >= DS_MIN and ds["ci95"]["lower"] is not None and ds["ci95"]["lower"] > 0
              and dd["estimate"] <= DD_MAX and dd["ci95"]["upper"] is not None and dd["ci95"]["upper"] < DD_UPPER_CI_MAX)
        per_band[b] = {"dS": ds["estimate"], "dS_ci_lower": ds["ci95"]["lower"], "dD": dd["estimate"],
                       "dD_ci_upper": dd["ci95"]["upper"], "satisfied": bool(ok)}
    if not per_band:
        raise ProtocolError("no band-stop results in the C1-main validation block")
    sat = [b for b, v in per_band.items() if v["satisfied"]]
    pool = sat or list(per_band)
    b_star = max(pool, key=lambda b: (per_band[b]["dS"], -BANDS.index(b)))  # largest dS; tie -> lower band (not reached in practice)
    return {"per_band": per_band, "satisfied": bool(sat), "bands_satisfying": sat, "b_star": b_star,
            "b_star_rule": "largest dS among bands satisfying criterion 1" if sat else
            "no band satisfies criterion 1: largest dS overall (decision is NO-GO)"}


def criterion_2(c1: Dict[str, Any], control: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not c1["satisfied"]:
        return {"satisfied": False, "evaluated": False, "reason": "control not run: no band satisfies criterion 1"}
    if not control or control.get("status") != "complete":
        return {"satisfied": False, "evaluated": False, "reason": "control results missing"}
    lower = c1["per_band"][c1["b_star"]]["dS_ci_lower"]
    return {"satisfied": bool(control["mean"] < lower), "evaluated": True, "control_mean_dS": control["mean"],
            "control_min_dS": control["min"], "control_max_dS": control["max"], "dS_b_star_ci_lower": lower}


def criterion_3(lite_val: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not lite_val or lite_val.get("status") != "complete":
        return {"satisfied": False, "evaluated": False, "reason": "C2-lite validation results missing"}
    ds, dd = lite_val["dS_lite"]["estimate"], lite_val["dD_lite"]["estimate"]
    # PROTOCOL NOTE: criterion 3 states no interval condition; it is applied to the validation point estimates.
    return {"satisfied": bool(ds >= DS_MIN and dd <= DD_MAX), "evaluated": True, "dS_lite": ds, "dD_lite": dd}


def go_nogo(band_stop: Dict[str, Any], control: Optional[Dict[str, Any]], lite_val: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 10. Conditions 1-3 from validation numbers; condition 4 is never auto-true."""
    c1 = criterion_1(band_stop)
    c2 = criterion_2(c1, control)
    c3 = criterion_3(lite_val)
    c4 = {"satisfied": None, "status": CONDITION_4,
          "text": "Run 5 and all M.Tech deliverables (handoff steps 1-5) are finished, and the student approves one "
                  "night of CPU time before 22:00."}
    first_three = c1["satisfied"] and c2["satisfied"] and c3["satisfied"]
    decision = "GO pending condition 4 (requires student approval)" if first_three else "NO-GO"
    return {"b_star": c1["b_star"], "condition_1": c1, "condition_2": c2, "condition_3": c3, "condition_4": c4,
            "conditions_1_to_3_hold": bool(first_three), "decision": decision,
            "if_no_go": "report C1, C3, C4 and C2-lite (exploratory); list C2 as future work"}


# --------------------------------------------------------------------------- data parts
def _preds(ext: str, variant: str, train_set: str, scored_set: str):
    from lungai.novelty.features import load_features
    from lungai.novelty.spectral_map import fit_probe
    classes, _ = class_codes()
    xtr, itr = load_features(ext, variant, train_set)
    xsc, isc = load_features(ext, variant, scored_set)
    sc_s, clf_s = fit_probe(xtr, itr["source"].to_numpy())
    sc_d, clf_d = fit_probe(xtr, itr["label"].to_numpy())
    return itr, isc, (encode(clf_s.predict(sc_s.transform(xsc)), SOURCES),
                      encode(clf_d.predict(sc_d.transform(xsc)), classes))


def _setup(itr, isc) -> Tuple[Any, ...]:
    classes, cidx = class_codes()
    sidx = {s: i for i, s in enumerate(SOURCES)}
    cells_named = qualifying_cells(isc["source"], isc["label"], itr["source"], itr["label"])
    normal = within_normal_sources(isc["source"], isc["label"], itr["source"], itr["label"])
    c_d = chance_disease(cells_named)
    if abs(c_d - EXPECTED_C_D) > 1e-12:
        raise ProtocolError(f"c_D {c_d} differs from section 4.5 (cells {cells_named})")
    cells = {sidx[s]: [cidx[c] for c in cs] for s, cs in cells_named.items()}
    return (encode(isc["source"], SOURCES), encode(isc["label"], classes), cells, [sidx[s] for s in normal],
            cidx[NORMAL_CLASS], c_d, cells_named)


def lite_table(band: str, scored_set: str) -> Dict[str, Any]:
    """C2-lite: probes fitted on the cap-400 train sample, full versus stop_<band>, on ``scored_set``."""
    from lungai.novelty.spectral_map import score_variants
    preds, ref = {}, None
    for v in ("full", f"stop_{band}"):
        itr, isc, p = _preds("imagenet", v, "train400", scored_set)
        if ref is not None and not isc["image_id"].equals(ref[1]["image_id"]):
            raise ProtocolError("full and band-stop scored samples are not row-aligned")
        ref = ref or (itr, isc)
        preds[v] = p
    src, y, cells, normal, ncode, c_d, cells_named = _setup(*ref)
    res = score_variants(preds, src, y, cells, normal, ncode, c_d, bootstrap_indices(len(y)))
    bs = res["band_stop"][band]
    return {"status": "complete", "band": band, "train_set": "train400", "scored_set": scored_set,
            "n_train": int(len(ref[0])), "n_scored": int(len(ref[1])), "within_source_cells": cells_named,
            "dS_lite": bs["dS"], "dD_lite": bs["dD"], "variants": res["variants"],
            "bootstrap": {"B": 1000, "seed": 1, "unit": "scored image"}, **provenance()}


def control_table(band: str) -> Dict[str, Any]:
    """Matched-energy random band-stop control: dS = BA_src(full) - BA_src(ctrl_seed) on val100, probes on train150."""
    from lungai.novelty.common import balanced_accuracy
    meta = read_json(OUT_DIR / f"control_{band}.json")
    _, isc, (sp_full, _) = _preds("imagenet", "full", "train150", "val100")
    src = encode(isc["source"], SOURCES)
    ba_full = balanced_accuracy(src, sp_full, 6)
    vals = {}
    for seed in CONTROL_SEEDS:
        _, isc_c, (sp, _) = _preds("imagenet", f"ctrl{seed}_{band}", "train150", "val100")
        if not isc_c["image_id"].equals(isc["image_id"]):
            raise ProtocolError("control and full validation samples are not row-aligned")
        vals[str(seed)] = ba_full - balanced_accuracy(src, sp, 6)
    v = list(vals.values())
    return {"status": "complete", "band": band, "q": meta.get("q"), "mask_seeds": list(CONTROL_SEEDS),
            "BA_src_full": ba_full, "dS_by_seed": vals, "mean": float(np.mean(v)), "min": float(min(v)),
            "max": float(max(v)), **provenance()}


def main(argv: Optional[List[str]] = None) -> int:
    from lungai.novelty.features import is_cached
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["select", "val", "test", "decision"], required=True)
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args(argv)
    c1 = read_json(C1_JSON).get("stages", {}).get("c1-main", {}).get("val", {})
    if c1.get("status") != "complete":
        print("C1-main validation results are required first (protocol section 9)")
        return 2
    doc = read_json(C2_JSON)
    doc.setdefault("protocol", "docs/publication/NOVELTY_PROTOCOL.md")
    crit1 = criterion_1(c1["band_stop"])
    band = crit1["b_star"]
    doc["criterion_1"] = crit1
    if args.part == "select":
        doc["last_written"] = provenance()
        write_json(C2_JSON, doc)
        print(f"b* = {band} ({crit1['b_star_rule']}); control needed: {crit1['satisfied']}")
        return 0
    if args.part == "val":
        need = [("full", "train400"), (f"stop_{band}", "train400"), ("full", "val100"), (f"stop_{band}", "val100")]
        if crit1["satisfied"]:
            need += [(f"ctrl{s}_{band}", t) for s in CONTROL_SEEDS for t in ("train150", "val100")] + [("full", "train150")]
        miss = [f"{v}/{s}" for v, s in need if not is_cached("imagenet", v, s)]
        if miss:
            print(f"missing feature caches: {miss} (run lungai.novelty.features --stage c2-lite / c2-control --band {band})")
            return 2
        doc["c2_lite_val"] = lite_table(band, "val100")
        doc["control"] = control_table(band) if crit1["satisfied"] else {
            "status": "not run", "reason": "no band satisfies criterion 1 (section 9)"}
    elif args.part == "test":
        if doc.get("c2_lite_val", {}).get("status") != "complete":
            print("validation decision first (section 11 step 6)")
            return 2
        if not all(is_cached("imagenet", v, "test100") for v in ("full", f"stop_{band}")):
            print("test100 feature caches missing")
            return 2
        claim_access("c2lite_test", C2_JSON, args.force_rerun)
        doc["c2_lite_test"] = {**lite_table(band, "test100"), "label": "exploratory"}
    else:
        doc["decision"] = {**go_nogo(c1["band_stop"], doc.get("control"), doc.get("c2_lite_val")), **provenance(),
                           "data": "validation only (C1-main val100, ImageNet; C2-lite val100); no new data access"}
        print(f"decision: {doc['decision']['decision']}")
    doc["last_written"] = provenance()
    write_json(C2_JSON, doc)
    print(f"wrote {C2_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
