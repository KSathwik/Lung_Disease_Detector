"""Figures and generated macros for the novelty extension, from artifacts/novelty/*.json only.

    python -m lungai.novelty.render_novelty            # figures into docs/figures/ (PNG 300 dpi + PDF)
    python -m lungai.evaluation.render_docs            # merges novelty_macros() into results_macros.tex / results.json

Figures: fig_spectral_map (C1), fig_inlp (C3), fig_loso_matrix (C4); Okabe-Ito colours and the cividis map
(colour-blind safe), the style of lungai.evaluation.make_figures. A figure whose JSON is missing is skipped.
Macros (letters only, no digits): Spec... (C1), Inlp... (C3), Loso... (C4), Bs... (C2-lite and the C2 decision);
``novelty_macros()`` returns {} when no novelty JSON exists, so current outputs are unchanged.
No TensorFlow; matplotlib is imported only when figures are drawn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from lungai.novelty.common import OUT_DIR, read_json
from lungai.novelty.spectral import BANDS

C1_JSON = OUT_DIR / "c1_spectral_map.json"
C3_JSON = OUT_DIR / "c3_inlp.json"
C4_JSON = OUT_DIR / "c4_loso.json"
C2_JSON = OUT_DIR / "c2_decision.json"
NOVELTY_JSONS = (C1_JSON, C3_JSON, C4_JSON, C2_JSON)

STAGE_TAG = {"c1-main": "Main", "c1-served": "Served", "c1-nofilter": "Nofilter"}
SPLIT_TAG = {"val": "Val", "test": "Test"}
EXT_TAG = {"imagenet": "Img", "served": "Mdl"}  # as the Probe... macros of render_docs
METRIC_TAG = {"BA_src": "SrcBA", "F1_ws": "FWs", "F1_all": "FAll", "BA_norm": "NormBA",
              "S_star": "SStar", "D_star": "DStar", "SDS": "SDS"}
PERCENT_METRICS = {"BA_src", "F1_ws", "F1_all", "BA_norm"}


def _p1(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}"


def _f3(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def variant_tag(v: str) -> str:
    if v == "full":
        return "Full"
    kind, _, b = v.partition("_")
    return kind.capitalize() + b


def _with_ci(out: Dict[str, str], key: str, block: Dict[str, Any], fmt) -> None:
    out[key] = fmt(block["estimate"])
    ci = block.get("ci95") or {}
    out[key + "Lo"], out[key + "Hi"] = fmt(ci.get("lower")), fmt(ci.get("upper"))


def spec_macros(c1: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    t = c1.get("timing")
    if t:
        out["SpecSecPerImage"] = f"{t['seconds_per_image']:.3f}"
        out["SpecTimingImages"] = str(t["images"])
        for stage, tag in (("c1-main", "Main"), ("c1-served", "Served"), ("c1-nofilter", "Nofilter"), ("c3c4", "Feat")):
            if stage in t.get("projected_hours", {}):
                out[f"SpecProjHours{tag}"] = f"{t['projected_hours'][stage]:.1f}"
    for stage, splits in c1.get("stages", {}).items():
        st = STAGE_TAG[stage]
        for split, r in splits.items():
            if r.get("status") != "complete":
                continue
            pre = f"Spec{st}{SPLIT_TAG[split]}"
            out.update({f"{pre}N": f"{r['n_scored']:,}", f"{pre}NTrain": f"{r['n_train']:,}",
                        f"{pre}ChanceD": f"{r['c_D']:.2f}", f"{pre}NWithinSources": str(len(r["within_source_cells"]))})
            for v, mets in r["variants"].items():
                for m, block in mets.items():
                    _with_ci(out, f"{pre}{variant_tag(v)}{METRIC_TAG[m]}", block, _p1 if m in PERCENT_METRICS else _f3)
            for b, d in r.get("band_stop", {}).items():
                _with_ci(out, f"{pre}Stop{b}DeltaS", d["dS"], _p1)
                _with_ci(out, f"{pre}Stop{b}DeltaD", d["dD"], _p1)
            for b, d in r.get("band_pass_retention", {}).items():
                for k, tag in (("S", "RetS"), ("D", "RetD")):
                    out[f"{pre}Pass{b}{tag}"] = d[k] if isinstance(d[k], str) else f"{d[k]:.2f}"
    return out


def _metric_keys():
    return (("source_ba", "SrcBA", _p1), ("F1_ws", "FWs", _p1), ("F1_all", "FAll", _p1), ("accuracy", "Acc", _p1),
            ("macro_roc_auc", "AUC", _f3))


def inlp_macros(c3: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ext, parts in c3.get("extractors", {}).items():
        pre = f"Inlp{EXT_TAG[ext]}"
        val = parts.get("val", {})
        if val.get("status") == "complete":
            ks = val["inlp"]["k_star"]
            out.update({f"{pre}KStar": str(ks), f"{pre}StoppedBy": val["inlp"]["stopped_by"], f"{pre}C": str(val["C"]),
                        f"{pre}RemovedDims": str(val["inlp"]["removed_dims"][ks]), f"{pre}NTrain": f"{val['n_train']:,}",
                        f"{pre}NVal": f"{val['n_val']:,}"})
            for k, tag in ((0, "KZero"), (ks, "KStar")):
                row = val["curve"][k]
                for m, mt, fmt in _metric_keys():
                    key = "val_source_ba" if m == "source_ba" else f"val_{m}"
                    out[f"{pre}Val{mt}{tag}"] = fmt(row.get(key))
            for m, mt, fmt in _metric_keys():
                out[f"{pre}LeaceVal{mt}"] = fmt(val["leace"].get(f"val_{m}"))
        test = parts.get("test", {})
        if test.get("status") == "complete":
            ks = test["k_star"]
            out[f"{pre}NTest"] = f"{test['n_test']:,}"
            for k, tag in ((0, "KZero"), (ks, "KStar")):
                row = test["table"][k]
                for m, mt, fmt in _metric_keys():
                    out[f"{pre}Test{mt}{tag}"] = fmt(row.get(f"test_{m}"))
            for label, row in (("KStar", test["table"][ks]), ("Leace", test["leace_check"])):
                for m, block in (row.get("change_vs_k0") or {}).items():
                    mt = dict((a, b) for a, b, _ in _metric_keys())[m]
                    _with_ci(out, f"{pre}TestChange{mt}{label}", block, _f3 if m == "macro_roc_auc" else _p1)
            for m, mt, fmt in _metric_keys():
                out[f"{pre}LeaceTest{mt}"] = fmt(test["leace_check"].get(f"test_{m}"))
        mont = parts.get("montgomery", {})
        for key, tag in (("k0", "KZero"), ("k_star", "KStar")):
            if key in mont:
                out.update(_binary_macros(f"{pre}Ext", tag, mont[key]))
    return out


def _binary_macros(pre: str, tag: str, rep: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for name, short in (("sensitivity", "Sens"), ("specificity", "Spec")):
        c = rep[name]
        out.update({f"{pre}{short}{tag}": _p1(c["estimate"]), f"{pre}{short}{tag}Lo": _p1(c["lower"]),
                    f"{pre}{short}{tag}Hi": _p1(c["upper"]), f"{pre}{short}{tag}K": str(c["successes"]),
                    f"{pre}{short}{tag}N": str(c["n"])})
    if rep.get("roc_auc_p_abnormal") is not None:
        out[f"{pre}AUC{tag}"] = _f3(rep["roc_auc_p_abnormal"])
    return out


def loso_macros(c4: Dict[str, Any]) -> Dict[str, str]:
    from lungai.evaluation.render_docs import SOURCE_MACRO
    out: Dict[str, str] = {}
    for ext, mat in c4.get("matrices", {}).items():
        if mat.get("status") != "complete":
            continue
        pre = f"Loso{EXT_TAG[ext]}"
        out[f"{pre}NIncluded"] = str(sum(1 for f in mat["folds"].values() if f.get("included")))
        for src, f in mat["folds"].items():
            if not f.get("included"):
                continue
            k = pre + SOURCE_MACRO.get(src, "".join(w.capitalize() for w in src.split("_")))
            ci = f["ci95"]
            out.update({f"{k}N": str(f["loso"]["n"]), f"{k}C": str(f["C_selection"]["C"]),
                        f"{k}MacroFOne": _p1(f["loso"]["macro_f1"]), f"{k}MacroFOneLo": _p1(ci["loso_macro_f1"]["lower"]),
                        f"{k}MacroFOneHi": _p1(ci["loso_macro_f1"]["upper"]), f"{k}BalAcc": _p1(f["loso"]["balanced_accuracy"]),
                        f"{k}RefMacroFOne": _p1(f["reference"]["macro_f1"]), f"{k}RefBalAcc": _p1(f["reference"]["balanced_accuracy"]),
                        f"{k}Gap": _p1(f["gap_macro_f1"]), f"{k}GapLo": _p1(ci["gap_macro_f1"]["lower"]),
                        f"{k}GapHi": _p1(ci["gap_macro_f1"]["upper"])})
            if "binary" in f:
                b = f["binary"]
                out.update({f"{k}BinAUC": _f3(b["roc_auc"]), f"{k}BinAUCLo": _f3(b["ci95"]["roc_auc"]["lower"]),
                            f"{k}BinAUCHi": _f3(b["ci95"]["roc_auc"]["upper"]), f"{k}BinBalAcc": _p1(b["balanced_accuracy"])})
    for ext, rep in c4.get("montgomery", {}).items():
        if "sensitivity" in rep:
            out.update(_binary_macros(f"Loso{EXT_TAG[ext]}Ext", "", rep))
    return out


def bs_macros(c2: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    c1 = c2.get("criterion_1")
    if c1:
        out.update({"BsBand": c1["b_star"], "BsNBandsSatisfying": str(len(c1["bands_satisfying"]))})
    for key, tag in (("c2_lite_val", "Val"), ("c2_lite_test", "Test")):
        r = c2.get(key, {})
        if r.get("status") == "complete":
            _with_ci(out, f"BsLite{tag}DeltaS", r["dS_lite"], _p1)
            _with_ci(out, f"BsLite{tag}DeltaD", r["dD_lite"], _p1)
            out[f"BsLite{tag}NTrain"] = f"{r['n_train']:,}"
    ctl = c2.get("control", {})
    if ctl.get("status") == "complete":
        out.update({"BsControlQ": f"{ctl['q']:.3f}", "BsControlMean": _p1(ctl["mean"]), "BsControlMin": _p1(ctl["min"]),
                    "BsControlMax": _p1(ctl["max"])})
    d = c2.get("decision")
    if d:
        out["BsDecision"] = d["decision"]
        for i, word in enumerate(("One", "Two", "Three"), start=1):
            out[f"BsCondition{word}"] = "holds" if d[f"condition_{i}"]["satisfied"] else "does not hold"
        out["BsConditionFour"] = d["condition_4"]["status"]
    return out


def novelty_macros(out_dir: Optional[Path] = None) -> Dict[str, str]:
    """Flat macro values from the novelty JSONs that exist ({} when none exists)."""
    d = out_dir or OUT_DIR
    out: Dict[str, str] = {}
    for name, fn in (("c1_spectral_map.json", spec_macros), ("c3_inlp.json", inlp_macros), ("c4_loso.json", loso_macros),
                     ("c2_decision.json", bs_macros)):
        data = read_json(d / name)
        if data:
            out.update(fn(data))
    bad = [k for k in out if any(ch.isdigit() for ch in k) or not k.isalnum()]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad[:5]}")
    return out


# --------------------------------------------------------------------------- figures
def _pick(splits: Dict[str, Any]) -> Optional[str]:
    for s in ("test", "val"):
        if splits.get(s, {}).get("status") == "complete":
            return s
    return None


def fig_spectral_map(c1: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from lungai.evaluation.make_figures import GREY, OI, save, setup_style
    stages = c1.get("stages", {})
    main = stages.get("c1-main", {})
    split = _pick(main)
    if split is None:
        return []
    setup_style()
    r = main[split]
    x = np.arange(len(BANDS))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    styles = [("c1-main", "o-", "ImageNet, main cache"), ("c1-served", "s--", "Run-2 backbone"),
              ("c1-nofilter", "^:", "ImageNet, no-filter cache")]
    for metric, colour, mlabel in (("BA_src", OI["blue"], "Source BA"), ("F1_ws", OI["verm"], "Within-source F1")):
        for stage, ls, slabel in styles:
            rr = stages.get(stage, {}).get(split if stage == "c1-main" else "test", {})
            if rr.get("status") != "complete":
                continue
            v = [rr["variants"][f"pass_{b}"][metric] for b in BANDS]
            est = np.array([q["estimate"] for q in v])
            lo = np.array([q["ci95"]["lower"] for q in v], float)
            hi = np.array([q["ci95"]["upper"] for q in v], float)
            ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt=ls, color=colour, ms=4, lw=1, capsize=2,
                        label=f"{mlabel} ({slabel})" if stage != "c1-main" else mlabel)
        ax.axhline(r["variants"]["full"][metric]["estimate"], color=colour, lw=0.8, ls="-.", alpha=0.7)
    ax.axhline(1 / 6, color=GREY, lw=0.8, ls=":")
    ax.set_xticks(x, [f"pass {b}" for b in BANDS])
    ax.set_ylabel("Score"); ax.set_title(f"(a) Band-pass ({split})"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax = axes[1]
    if r.get("band_stop"):
        w = 0.38
        for off, key, colour, lab in ((-w / 2, "dS", OI["blue"], "dS (source BA lost)"),
                                      (w / 2, "dD", OI["verm"], "dD (within-source F1 lost)")):
            est = np.array([r["band_stop"][b][key]["estimate"] for b in BANDS])
            lo = np.array([r["band_stop"][b][key]["ci95"]["lower"] for b in BANDS], float)
            hi = np.array([r["band_stop"][b][key]["ci95"]["upper"] for b in BANDS], float)
            ax.bar(x + off, est, w, color=colour, label=lab)
            ax.errorbar(x + off, est, yerr=[est - lo, hi - est], fmt="none", ecolor="black", lw=0.8, capsize=2)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x, [f"stop {b}" for b in BANDS])
        ax.set_ylabel("Full minus band-stop"); ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"(b) Band-stop differences ({split})")
    fig.tight_layout()
    return save(fig, "fig_spectral_map")


def fig_inlp(c3: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt
    from lungai.evaluation.make_figures import GREY, OI, save, setup_style
    exts = {e: p for e, p in c3.get("extractors", {}).items() if p.get("val", {}).get("status") == "complete"}
    if not exts:
        return []
    setup_style()
    fig, axes = plt.subplots(1, len(exts), figsize=(3.6 * len(exts), 3.0), squeeze=False)
    names = {"imagenet": "ImageNet features", "served": "Run-2 backbone (trained, not held-out)"}
    for ax, (e, p) in zip(axes[0], exts.items()):
        cv = p["val"]["curve"]
        k = [r["k"] for r in cv]
        ax.plot(k, [r["val_source_ba"] for r in cv], "o-", color=OI["blue"], ms=3, label="Source BA (val)")
        ax.plot(k, [r["val_F1_ws"] for r in cv], "s-", color=OI["verm"], ms=3, label="Within-source F1 (val)")
        t = p.get("test", {})
        if t.get("status") == "complete":
            ax.plot(k, [r["test_source_ba"] for r in t["table"]], "o--", color=OI["sky"], ms=3, label="Source BA (test)")
            ax.plot(k, [r["test_F1_ws"] for r in t["table"]], "s--", color=OI["orange"], ms=3, label="Within-source F1 (test)")
        ax.axhline(1 / 6, color=GREY, lw=0.8, ls=":")
        ax.axvline(p["val"]["inlp"]["k_star"], color=GREY, lw=0.8, ls="--")
        ax.set_xlabel("INLP iteration k"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.02); ax.set_title(names.get(e, e))
        ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    return save(fig, "fig_inlp")


def fig_loso_matrix(c4: Dict[str, Any]) -> List[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from lungai.evaluation.make_figures import save, setup_style, src_label
    mats = {e: m for e, m in c4.get("matrices", {}).items() if m.get("status") == "complete"}
    if not mats:
        return []
    setup_style()
    fig, axes = plt.subplots(1, len(mats), figsize=(3.6 * len(mats), 3.2), squeeze=False)
    names = {"imagenet": "ImageNet features", "served": "Run-2 backbone (saw all sources)"}
    cols = ["LOSO F1", "Reference F1", "Gap"]
    for ax, (e, m) in zip(axes[0], mats.items()):
        rows = [s for s, f in m["folds"].items() if f.get("included")]
        vals = np.array([[m["folds"][s]["loso"]["macro_f1"], m["folds"][s]["reference"]["macro_f1"],
                          m["folds"][s]["gap_macro_f1"]] for s in rows])
        ax.imshow(vals, cmap="cividis", vmin=min(0.0, vals.min()), vmax=1.0, aspect="auto")
        for i in range(len(rows)):
            for j in range(3):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if vals[i, j] < 0.55 else "black")
        ax.set_xticks(range(3), cols); ax.set_yticks(range(len(rows)), [src_label(s) for s in rows])
        ax.grid(False); ax.set_title(names.get(e, e))
    fig.tight_layout()
    return save(fig, "fig_loso_matrix")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    for path, fn in ((C1_JSON, fig_spectral_map), (C3_JSON, fig_inlp), (C4_JSON, fig_loso_matrix)):
        data = read_json(path)
        if not data or not fn(data):
            print(f"{fn.__name__}: skipped (no complete results in {path.name})")
    print(f"{len(novelty_macros())} novelty macros available to render_docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
