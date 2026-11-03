"""Generate the manuscript / thesis figures from evaluation artifacts.

    python -m lungai.evaluation.make_figures --eval-prefix v6 \
        --run runs/v6_densenet121_seed42 --build data/derived/v6_build_001

Reads only (never modifies): artifacts/evaluation_reports/<prefix>_{val,test,external}/,
runs/<run>/run_metadata.json, <build>/v6_build_report.json and manifest_v6.csv,
lungai/configs/lungai_densenet121.json. Writes PNG (300 dpi) and PDF per figure into
docs/figures/. All numbers come from those files; only labels are typed here.
TensorFlow is not imported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs" / "figures"

# Okabe-Ito colour-blind-safe palette
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7",
}
CLASS_COLORS = [OI["verm"], OI["blue"], OI["sky"], OI["orange"], OI["purple"], OI["green"]]
SOURCE_COLORS = [OI["blue"], OI["orange"], OI["green"], OI["purple"], OI["sky"], OI["verm"]]
GREY = "#6B6B6B"
LIGHT = "#F2F2F2"

SOURCE_LABELS = {
    "nih_cxr14": "NIH ChestX-ray14", "kermany": "Kermany (paediatric)", "cohen_covid": "Cohen COVID-19",
    "shenzhen": "Shenzhen", "montgomery": "Montgomery", "tbx11k": "TBX11K",
    "covid_radiography": "COVID-19 Radiography",
}


def setup_style() -> None:
    serif = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "serif", "font.serif": serif, "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "savefig.dpi": 300, "figure.dpi": 100, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, stem: str) -> List[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.08)
        paths.append(p)
    plt.close(fig)
    print("wrote", ", ".join(str(p.relative_to(REPO)) for p in paths))
    return paths


def src_label(s: str) -> str:
    return SOURCE_LABELS.get(s, s)


# --------------------------------------------------------------------------- inputs
class Inputs:
    def __init__(self, prefix: str, run: Path, build: Path):
        ev = REPO / "artifacts" / "evaluation_reports"
        self.cfg = json.load(open(REPO / "lungai" / "configs" / "lungai_densenet121.json", encoding="utf-8"))
        self.classes: List[str] = [c["name"] for c in sorted(self.cfg["classes"], key=lambda c: c["index"])]
        self.class_ids: List[str] = [c["id"] for c in sorted(self.cfg["classes"], key=lambda c: c["index"])]
        self.metrics: Dict[str, dict] = {}
        self.preds: Dict[str, pd.DataFrame] = {}
        for split in ("val", "test", "external"):
            d = ev / f"{prefix}_{split}"
            mp, pp = d / f"metrics_{split}.json", d / f"predictions_{split}.csv"
            if mp.is_file():
                self.metrics[split] = json.load(open(mp, encoding="utf-8"))
            if pp.is_file():
                self.preds[split] = pd.read_csv(pp)
        run = run if run.is_absolute() else REPO / run
        build = build if build.is_absolute() else REPO / build
        self.run_meta = json.load(open(run / "run_metadata.json", encoding="utf-8"))
        self.build_dir = build
        self.report = json.load(open(build / "v6_build_report.json", encoding="utf-8"))
        self.manifest = pd.read_csv(build / "manifest_v6.csv", low_memory=False)


# --------------------------------------------------------------------------- result figures
def fig_training_curves(inp: Inputs):
    h = pd.DataFrame(inp.run_meta["history"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.plot(h["epoch"], h["train_loss"], "o-", color=OI["verm"], label="Training loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss"); ax.set_title("(a) Training loss")
    ax = axes[1]
    ax.plot(h["epoch"], h["val_accuracy"], "o-", color=OI["blue"], label="Accuracy")
    ax.plot(h["epoch"], h["val_macro_f1"], "s--", color=OI["green"], label="Macro-F1")
    ax.plot(h["epoch"], h["val_macro_roc_auc"], "^:", color=OI["orange"], label="Macro ROC-AUC")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation score"); ax.set_title("(b) Validation metrics")
    # phase boundary
    ph = h["phase"].tolist()
    for a in axes:
        a.set_xticks(h["epoch"])
        for i in range(1, len(ph)):
            if ph[i] != ph[i - 1]:
                a.axvline(h["epoch"].iloc[i] - 0.5, color=GREY, lw=0.8, ls="--")
        lo_, hi_ = a.get_ylim()
        a.set_ylim(lo_, hi_ + 0.08 * (hi_ - lo_))
        top = a.get_ylim()[1]
        first_ft = h.index[h["phase"] != ph[0]]
        if len(first_ft):
            b = h["epoch"].iloc[first_ft[0]] - 0.5
            a.text((h["epoch"].iloc[0] + b) / 2, top, ph[0].replace("_", " "), ha="center", va="top", fontsize=8, color=GREY)
            a.text((b + h["epoch"].iloc[-1]) / 2 + 0.25, top, ph[-1].replace("_", " "), ha="center", va="top", fontsize=8, color=GREY)
    sel = inp.run_meta.get("selected", {})
    if "epoch" in sel:
        axes[1].plot([sel["epoch"]], [sel["val_macro_f1"]], marker="*", ms=13, mfc="none", mec="black", ls="none",
                     label="Selected epoch")
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout()
    return save(fig, "fig_training_curves")


def _cm(inp: Inputs):
    return np.array(inp.metrics["test"]["result"]["metrics"]["confusion_matrix"])


def fig_confusion_matrix(inp: Inputs):
    cm = _cm(inp)
    rown = cm / cm.sum(1, keepdims=True).clip(min=1)
    short = [c.replace("Pulmonary Nodule / Mass", "Nodule / Mass").replace("Pleural Effusion", "Pleural\nEffusion") for c in inp.classes]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    for ax, data, title, fmt in ((axes[0], cm, "(a) Counts", "d"), (axes[1], rown, "(b) Row-normalised (recall)", ".2f")):
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
        ax.grid(False)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(short, rotation=35, ha="right"); ax.set_yticklabels(short)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(True)
        for i in range(6):
            for j in range(6):
                v = data[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=9,
                        color="white" if v > 0.55 * data.max() else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    return save(fig, "fig_confusion_matrix")


def fig_roc_curves(inp: Inputs):
    from sklearn.metrics import auc, roc_curve
    df = inp.preds["test"]
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    styles = ["-", "--", "-.", ":", "-", "--"]
    for k, (name, cid) in enumerate(zip(inp.classes, inp.class_ids)):
        y = (df["true_label"] == name).astype(int).values
        s = df[f"prob_{cid}"].values
        if y.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, styles[k], color=CLASS_COLORS[k], lw=1.8, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.9, ls=":")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    return save(fig, "fig_roc_curves")


def fig_per_class_metrics(inp: Inputs):
    pc = inp.metrics["test"]["result"]["metrics"]["per_class"]
    names = inp.classes
    prec = np.array([pc[n]["precision"] for n in names])
    rec = np.array([pc[n]["recall"] for n in names])
    f1 = np.array([pc[n]["f1"] for n in names])
    lo = rec - np.array([pc[n]["recall_ci95"]["lower"] for n in names])
    hi = np.array([pc[n]["recall_ci95"]["upper"] for n in names]) - rec
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(x - w, prec, w, color=OI["blue"], label="Precision")
    ax.bar(x, rec, w, color=OI["orange"], label="Recall (95% Wilson CI)", yerr=[lo, hi], capsize=3,
           error_kw={"lw": 1.0, "ecolor": "black"})
    ax.bar(x + w, f1, w, color=OI["green"], label="F1")
    sup = [pc[n]["support"] for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n.replace('Pulmonary Nodule / Mass', 'Nodule / Mass')}\n(n = {s})" for n, s in zip(names, sup)])
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score (test split)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    fig.tight_layout()
    return save(fig, "fig_per_class_metrics")


def fig_per_source(inp: Inputs):
    ps = inp.metrics["test"]["result"]["per_source"]
    from math import sqrt
    srcs = sorted(ps, key=lambda s: -ps[s]["n"])
    acc = np.array([ps[s]["accuracy"] for s in srcs]); n = np.array([ps[s]["n"] for s in srcs])
    z = 1.96
    centre = (acc + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo, hi = acc - (centre - half), (centre + half) - acc
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(srcs))
    ax.bar(x, acc, color=SOURCE_COLORS[:len(srcs)], yerr=[lo, hi], capsize=3, error_kw={"lw": 1.0})
    for i, a in enumerate(acc):
        ax.text(i, 0.03, f"{a:.3f}", ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{src_label(s)}\n(n = {k})" for s, k in zip(srcs, n)])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy (test split)")
    overall = inp.metrics["test"]["result"]["metrics"]["accuracy"]
    ax.axhline(overall, color=GREY, ls="--", lw=1.0)
    ax.text(-0.45, overall + 0.012, f"overall {overall:.3f}", ha="left", va="bottom", fontsize=8.5, color=GREY)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, "fig_per_source")


def fig_external_breakdown(inp: Inputs):
    br = inp.metrics["external"]["result"]["prediction_breakdown"]
    groups = [g for g in ("Normal", "Tuberculosis") if g in br]
    glabel = {"Normal": "Normal scans", "Tuberculosis": "TB scans"}
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(groups))[::-1]
    left = np.zeros(len(groups))
    totals = np.array([sum(br[g].values()) for g in groups], dtype=float)
    for k, c in enumerate(inp.classes):
        vals = np.array([br[g].get(c, 0) for g in groups], dtype=float)
        if vals.sum() == 0:
            continue
        ax.barh(y, vals / totals * 100, left=left, color=CLASS_COLORS[k], edgecolor="white", label=c, height=0.55)
        for yi, v, l, t in zip(y, vals, left, totals):
            if v / t * 100 >= 7:
                ax.text(l + v / t * 50, yi, f"{int(v)}", ha="center", va="center", fontsize=9,
                        color="black" if c in ("Pleural Effusion", "Pneumonia") else "white")
        left += vals / totals * 100
    ax.set_yticks(y)
    ax.set_yticklabels([f"{glabel[g]}\n(n = {int(t)})" for g, t in zip(groups, totals)])
    ax.set_xlim(0, 100); ax.set_xlabel("Share of scans assigned to each predicted class (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=8.5)
    ax.set_title("Montgomery external cohort (cell values are image counts)", fontsize=10)
    fig.tight_layout()
    return save(fig, "fig_external_breakdown")


def fig_calibration(inp: Inputs):
    cal = inp.metrics["test"]["result"]["metrics"]["calibration"]
    bins = [b for b in cal["bins"] if b["count"] > 0]
    conf = np.array([b["mean_confidence"] for b in bins]); acc = np.array([b["accuracy"] for b in bins])
    cnt = np.array([b["count"] for b in bins])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.6, 5.4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.0, label="Perfect calibration")
    ax.plot(conf, acc, "o-", color=OI["blue"], lw=1.6, label=f"Model (ECE = {cal['ece']:.3f})")
    ax.set_ylabel("Observed accuracy"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 1.03)
    ax.legend(loc="upper left", frameon=False)
    ax2.bar(conf, cnt, width=0.04, color=OI["sky"], edgecolor="black", lw=0.4)
    ax2.set_xlabel("Mean predicted confidence (top-1)"); ax2.set_ylabel("Images")
    fig.tight_layout()
    return save(fig, "fig_calibration")


def fig_dataset_composition(inp: Inputs):
    m = inp.manifest
    m = m[m["split"].isin(["train", "val", "test", "external"])]
    splits = [s for s in ("train", "val", "test", "external") if (m["split"] == s).any()]
    classes = [c for c in inp.classes if (m["lungai_label"] == c).any()]
    sources = sorted(m["source_id"].unique(), key=lambda s: -(m["source_id"] == s).sum())
    fig, axes = plt.subplots(1, len(splits), figsize=(2.6 * len(splits) + 0.6, 3.9), sharey=False)
    for ax, sp in zip(axes, splits):
        d = m[m["split"] == sp]
        tab = d.groupby(["lungai_label", "source_id"]).size().unstack(fill_value=0)
        cls = [c for c in classes if c in tab.index]
        bottom = np.zeros(len(cls))
        for k, s in enumerate(sources):
            if s not in tab.columns:
                continue
            v = np.array([tab.loc[c, s] for c in cls], dtype=float)
            ax.bar(range(len(cls)), v, bottom=bottom, color=SOURCE_COLORS[k % 6], edgecolor="white", lw=0.4,
                   label=src_label(s))
            bottom += v
        ax.set_xticks(range(len(cls)))
        ax.set_xticklabels([c.replace("Pulmonary Nodule / Mass", "Nodule/Mass").replace("Pleural Effusion", "Pleural eff.") for c in cls],
                           rotation=60, ha="right", fontsize=8.5)
        ax.set_title(f"{sp.capitalize()} (n = {int(bottom.sum())})", fontsize=10)
        ax.set_ylim(0, bottom.max() * 1.06)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Images")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save(fig, "fig_dataset_composition")


# --------------------------------------------------------------------------- diagram helpers
def new_canvas(w_in: float, h_in: float, w: float = 100, h: Optional[float] = None):
    h = h if h is not None else w * h_in / w_in
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.grid(False)
    return fig, ax


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#333333", fs=9.5, bold=False, lw=1.1, ls="-", color="black",
        va="center", ha="center", rounding=0.8):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}", fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    tx = x + w / 2 if ha == "center" else x + 1.0
    ty = y + h / 2 if va == "center" else y + h - 1.0
    ax.text(tx, ty, text, ha=ha, va=va, fontsize=fs, fontweight="bold" if bold else "normal", color=color,
            linespacing=1.25)
    return p


def arrow(ax, p0, p1, color="#333333", lw=1.3, ls="-", style="-|>", rad=0.0, text=None, fs=8.5, toff=(0, 1.2)):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls,
                                                      connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
                                                      mutation_scale=11))
    if text:
        ax.text((p0[0] + p1[0]) / 2 + toff[0], (p0[1] + p1[1]) / 2 + toff[1], text, ha="center", va="bottom",
                fontsize=fs, color=color, style="italic")


def poly_arrow(ax, pts, **kw):
    color = kw.get("color", "#333333"); lw = kw.get("lw", 1.3)
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], **kw)


SOURCE_NAMES = {"nih_cxr14": "NIH", "kermany": "Kermany", "cohen_covid": "Cohen", "shenzhen": "Shenzhen",
                "montgomery": "Montgomery", "tbx11k": "TBX11K", "covid_radiography": "COVID-19 Rad."}


def _source_lines(inp) -> str:
    """Sources of the build actually used, three per line, in brackets."""
    names = [SOURCE_NAMES.get(s, s) for s in inp.report["sources_used"]]
    rows = [", ".join(names[i:i + 3]) for i in range(0, len(names), 3)]
    return "(" + ",\n".join(rows) + ")"


def fig_pipeline(inp: Inputs):
    fig, ax = new_canvas(7.6, 5.0, 100, 66)
    stages = [
        ("1  Acquisition", "Official downloads\n" + _source_lines(inp) + "\nProvenance and checksums\nlogged", "#E8F1FA"),
        ("2  Dataset build", "Source labels mapped to\nsix classes; exclusions\nPer source/class caps\nPatient-grouped split\n(70/15/15); duplicates\nmerged; leakage checks", "#E8F1FA"),
        ("3  Training", "DenseNet-121, ImageNet\nweights; head then\nfine-tune phases\nClass-weighted loss\nSeed " + str(inp.run_meta["seed"]) + "; train/val only\nBest val. macro-F1", "#FDF3E0"),
        ("4  Evaluation", "Validation during\ndevelopment\nTest and external run\nonce, after selection\nWilson and bootstrap CIs\nCalibration, per source", "#FDF3E0"),
        ("5  Served metrics", "served_metrics writes\nmodel_metrics.json\nincl. checkpoint SHA-256\nand split fingerprint\nHistorical record kept\nseparately", "#E6F4EE"),
        ("6  Application", "FastAPI + React\n/predict, /model,\n/model-metrics, /health\nCXR gate, Grad-CAM,\nSQLite history,\nMetrics page", "#E6F4EE"),
    ]
    bw, bh, gx = 28, 24, 5.5
    xs = [2, 2 + bw + gx, 2 + 2 * (bw + gx)]
    ys = [38, 4]
    for i, (t, body, fc) in enumerate(stages):
        r, c = divmod(i, 3)
        x, y = xs[c], ys[r]
        box(ax, x, y, bw, bh, "", fc=fc)
        ax.text(x + bw / 2, y + bh - 2.2, t, ha="center", va="top", fontsize=10.5, fontweight="bold")
        ax.text(x + bw / 2, y + bh - 6.8, body, ha="center", va="top", fontsize=8.8, linespacing=1.3)
    for c in range(2):
        arrow(ax, (xs[c] + bw, ys[0] + bh / 2), (xs[c + 1], ys[0] + bh / 2))
        arrow(ax, (xs[c] + bw, ys[1] + bh / 2), (xs[c + 1], ys[1] + bh / 2))
    mid = (ys[0] + ys[1] + bh) / 2
    poly_arrow(ax, [(xs[2] + bw / 2, ys[0]), (xs[2] + bw / 2, mid), (xs[0] + bw / 2, mid), (xs[0] + bw / 2, ys[1] + bh)])
    ax.text(50, mid + 1.0, "built dataset, then trained and frozen checkpoint", ha="center", va="bottom", fontsize=8.5, style="italic", color=GREY)
    return save(fig, "fig_pipeline")


def fig_system_architecture(inp: Inputs):
    fig, ax = new_canvas(7.8, 5.6, 100, 72)
    box(ax, 2, 50, 26, 20, "", fc="#E8F1FA")
    ax.text(15, 68, "React frontend", ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.text(15, 63.5, "Analyze\nHistory\nModel Performance\nPatients\nservices/api.js", ha="center", va="top", fontsize=9, linespacing=1.3)
    box(ax, 36, 6, 62, 64, "", fc="#FAFAFA", ec=GREY, ls="--")
    ax.text(67, 68.3, "FastAPI backend (/api/v1)", ha="center", va="top", fontsize=10.5, fontweight="bold")
    box(ax, 39, 52, 56, 9, "Routes: /predict   /model   /model-metrics   /health\n(+ /predictions, /patients, /reports)", fc="#FFFFFF", fs=9)
    box(ax, 39, 36, 26, 11, "CXR gate\nstage 1: image heuristics\nstage 2: trained model, else\nheuristic fallback", fc="#FDF3E0", fs=8.4)
    box(ax, 69, 36, 26, 11, "InferenceEngine\nlungai.preprocessing,\nSHA-256-checked checkpoint", fc="#FDF3E0", fs=8.4)
    box(ax, 69, 21, 26, 10, "Grad-CAM\nlast conv. block (relu)\noverlay on display image", fc="#FDF3E0", fs=8.4)
    box(ax, 39, 21, 26, 10, "Prediction record\nprobabilities, model ID,\ncheckpoint hash", fc="#FFFFFF", fs=8.4)
    box(ax, 39, 9, 26, 8, "SQLite\npatients, scans,\npredictions, reports", fc="#E6F4EE", fs=8.4)
    box(ax, 69, 9, 24, 8, "Model store\n" + inp.cfg["model"]["checkpoint"]["filename"], fc="#E6F4EE", fs=8.4)
    box(ax, 2, 9, 28, 12, "model_metrics.json\n(artifacts/evaluation_reports)\nserved by /model-metrics", fc="#E6F4EE", fs=8.4)
    arrow(ax, (28, 62), (39, 58), style="<|-|>")
    ax.text(15, 48.3, "HTTP /api/v1, X-API-Key header", ha="center", va="top", fontsize=8, style="italic", color=GREY)
    arrow(ax, (52, 52), (52, 47))
    arrow(ax, (65, 41.5), (69, 41.5))
    arrow(ax, (82, 36), (82, 31))
    arrow(ax, (69, 26), (65, 26))
    arrow(ax, (52, 21), (52, 17))
    poly_arrow(ax, [(93, 13), (96.5, 13), (96.5, 41.5), (95, 41.5)], color=OI["green"])
    poly_arrow(ax, [(30, 15), (33.5, 15), (33.5, 55.5), (39, 55.5)], color=OI["green"])
    return save(fig, "fig_system_architecture")


def fig_dataflow(inp: Inputs):
    fig, ax = new_canvas(7.8, 4.4, 100, 57)
    steps = [
        ("1  Upload", "Analyze page\nPOST /predict\n(image file)"),
        ("2  Validation", "Image type, size limit\n(10 MB), non-empty"),
        ("3  CXR gate", "Stage 1 heuristics;\nstage 2 model or\nheuristic fallback\n(on the raw bytes)"),
        ("4  Preprocessing", "InferenceEngine:\nblur, CLAHE, low-pass,\nresize 224x224,\nnormalise"),
        ("5  Model", "DenseNet-121\nsix softmax\nprobabilities"),
        ("6  Grad-CAM", "Heat map of the\npredicted class\n(overlay)"),
        ("7  Store", "SQLite: scan,\nprobabilities, model ID,\ncheckpoint hash"),
        ("8  Response", "JSON to the UI:\nprediction, scores,\nheat map"),
    ]
    bw, bh, gx = 20.5, 19.5, 5.2
    xs = [2 + i * (bw + gx) for i in range(4)]
    ys = [27, 3]
    for i, (t, body) in enumerate(steps):
        r, c = divmod(i, 4)
        x, y = xs[c], ys[r]
        fc = "#E8F1FA" if i in (0, 7) else ("#FDF3E0" if i in (2, 3, 4, 5) else "#E6F4EE")
        box(ax, x, y, bw, bh, "", fc=fc)
        ax.text(x + bw / 2, y + bh - 1.8, t, ha="center", va="top", fontsize=10, fontweight="bold")
        ax.text(x + bw / 2, y + bh - 6.3, body, ha="center", va="top", fontsize=8.6, linespacing=1.3)
    for c in range(3):
        arrow(ax, (xs[c] + bw, ys[0] + bh / 2), (xs[c + 1], ys[0] + bh / 2))
        arrow(ax, (xs[c] + bw, ys[1] + bh / 2), (xs[c + 1], ys[1] + bh / 2))
    mid = (ys[0] + ys[1] + bh) / 2
    poly_arrow(ax, [(xs[3] + bw / 2, ys[0]), (xs[3] + bw / 2, mid), (xs[0] + bw / 2, mid), (xs[0] + bw / 2, ys[1] + bh)])
    # rejection branch
    rx0 = xs[1] + bw / 2 - 6
    rw = (xs[2] + bw / 2 + 6) - rx0
    box(ax, rx0, 50.5, rw, 5, "Rejected: HTTP 400 / 413 / 422 with reason", fc="#FBEBE3", ec=OI["verm"], fs=8.8, ls="--", color=OI["verm"])
    arrow(ax, (xs[1] + bw / 2, ys[0] + bh), (xs[1] + bw / 2, 50.5), color=OI["verm"], ls="--", lw=1.0)
    arrow(ax, (xs[2] + bw / 2, ys[0] + bh), (xs[2] + bw / 2, 50.5), color=OI["verm"], ls="--", lw=1.0)
    return save(fig, "fig_dataflow")


def fig_model_architecture(inp: Inputs):
    cfg = inp.cfg
    head = cfg["model"]["head"]
    fig, ax = new_canvas(7.8, 4.9, 100, 62.8)
    layers = [
        ("Input\n224x224x3", "#FFFFFF"),
        ("DenseNet-\n121 backbone\n(ImageNet\nweights)", "#E8F1FA"),
        ("Global\naverage\npooling", "#FDF3E0"),
        ("Batch\nnormal-\nisation", "#FDF3E0"),
        (f"Dense({head['dense_units']})\nReLU", "#FDF3E0"),
        (f"Dropout\n({head['dropout']})", "#FDF3E0"),
        ("Dense(6)\nsoftmax", "#E6F4EE"),
    ]
    n = len(layers)
    bw, gx = 12.4, 1.8
    x0, y, bh = 2, 36, 20
    xb = lambda i: x0 + i * (bw + gx)
    for i, (t, fc) in enumerate(layers):
        box(ax, xb(i), y, bw, bh, t, fc=fc, fs=8.8)
        if i < n - 1:
            arrow(ax, (xb(i) + bw, y + bh / 2), (xb(i) + bw + gx, y + bh / 2))
    ax.text(50, 59.5, "Six-class DenseNet-121 classifier (single-label, mutually exclusive classes)", ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.text(xb(6) + bw, 33.5, "Outputs:\n" + "\n".join(inp.classes), ha="right", va="top", fontsize=7.6, linespacing=1.25, color=GREY)
    ax.plot([xb(2), xb(5) + bw], [34.5, 34.5], color="#333333", lw=1.0)
    ax.text((xb(2) + xb(5) + bw) / 2, 33.0, "classification head (trained from scratch)", ha="center", va="top", fontsize=8.8)
    # phases: epoch budgets and rates come from the run's own metadata
    rm = inp.run_meta
    ph = rm.get("config", {}).get("training", {}).get("phases", cfg["training"]["phases"])
    p1, p2 = ph[0], ph[1]
    n1 = sum(1 for r in rm["history"] if r["phase"] == p1["name"])
    max2 = rm.get("phase2_max_epochs", p2["epochs"])
    pat = rm.get("early_stopping_patience")
    pat_txt = f"; early-stopping patience {pat}" if pat else ""
    box(ax, 2, 2.5, 46, 14, f"Phase 1: {p1['name']}  ({n1} epochs)\nBackbone frozen; Adam, lr = {p1['learning_rate']:g}\nOnly the head is trained",
        fc="#F7F7F7", fs=8.8, ls="--")
    box(ax, 52, 2.5, 46, 14, f"Phase 2: {p2['name'].replace('_', '-')}  (up to {max2} epochs)\nLast {p2['unfreeze_last_n_layers']} backbone layers unfrozen; Adam, lr = {p2['learning_rate']:g}\n{pat_txt[2:].capitalize()}",
        fc="#F7F7F7", fs=8.8, ls="--")
    arrow(ax, (48, 9.5), (52, 9.5))
    ax.text(2, 26, f"Class-weighted cross-entropy, batch size {cfg['training']['batch_size']}, seed {cfg['training']['seed']};\n"
            "checkpoint = best validation macro-F1", ha="left", va="center", fontsize=8.8, style="italic")
    return save(fig, "fig_model_architecture")


def fig_preprocessing(inp: Inputs):
    sys.path.insert(0, str(REPO))
    import cv2
    from lungai import preprocessing as P
    m = inp.manifest
    t = m[(m["split"] == "test") & (m["source_id"] == "shenzhen")].sort_values("image_id")
    if t.empty:
        raise RuntimeError("no Shenzhen test image in the manifest")
    row = t.iloc[len(t) // 2]
    path = REPO / "data" / "raw" / row["raw_relpath"]
    bgr = P.read_image(path)
    rgb0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(rgb0, (P.PRE_BLUR_KSIZE, P.PRE_BLUR_KSIZE), P.PRE_BLUR_SIGMA)
    lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = P._clahe().apply(np.ascontiguousarray(lab[:, :, 0], dtype=np.uint8))
    clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    low = P.gaussian_lowpass(clahe)
    final = P.resize_display(low)
    assert np.array_equal(low, P.clean_image(bgr)), "step-by-step result differs from lungai.preprocessing.clean_image"
    panels = [
        (rgb0, f"(a) Original\n{rgb0.shape[1]}x{rgb0.shape[0]} px"),
        (blur, f"(b) Gaussian blur\n{P.PRE_BLUR_KSIZE}x{P.PRE_BLUR_KSIZE}, sigma = {P.PRE_BLUR_SIGMA:g}"),
        (clahe, f"(c) CLAHE on LAB L\nclip {P.CLAHE_CLIP:g}, tiles {P.CLAHE_TILES[0]}x{P.CLAHE_TILES[1]}"),
        (low, f"(d) Gaussian low-pass\nsigma = {P.LOWPASS_SIGMA:g}"),
        (final, f"(e) Lanczos resize\n{final.shape[1]}x{final.shape[0]} px"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(10.0, 2.9))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray"); ax.set_title(title, fontsize=9); ax.axis("off")
    fig.text(0.5, 0.01, "Image: Shenzhen CXR set (NLM), Jaeger et al. 2014", ha="center", va="bottom", fontsize=8, color=GREY, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save(fig, "fig_preprocessing")


def fig_gradcam_examples(inp: Inputs):
    """Grad-CAM from the served model, one test image per class.

    Selection rule fixed before any heatmap was viewed: for each class, the correctly classified test image whose
    top-class probability is the median among that class's correct predictions (ties: smallest image_id). Uses the
    backend's InferenceEngine, so the overlays are exactly what the application shows. Requires the evaluated
    checkpoint to be the one installed in models/ (checked by SHA-256)."""
    import base64
    import cv2
    sys.path[:0] = [str(REPO / "backend"), str(REPO)]
    from ml.inference import InferenceEngine
    engine = InferenceEngine()
    engine.initialize()
    want = inp.metrics["test"]["checkpoint_sha256"]
    if not engine.is_ready or engine.model_sha256 != want:
        raise RuntimeError(f"installed checkpoint {engine.model_sha256} is not the evaluated one {want}; install it first")
    p = inp.preds["test"].copy()
    prob_cols = [f"prob_{c}" for c in inp.class_ids]
    p["conf"] = p[prob_cols].max(axis=1)
    p = p[p["pred_label"] == p["true_label"]]
    paths = inp.manifest.set_index("image_id")["raw_relpath"]
    picks = []
    for c in inp.classes:
        g = p[p["true_label"] == c].sort_values(["conf", "image_id"]).reset_index(drop=True)
        if len(g):
            picks.append(g.iloc[(len(g) - 1) // 2])

    def _decode(uri: str) -> np.ndarray:
        buf = np.frombuffer(base64.b64decode(uri.split(",", 1)[1]), np.uint8)
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, len(picks), figsize=(1.75 * len(picks), 4.0))
    for j, row in enumerate(picks):
        data = (REPO / "data" / "raw" / paths[row["image_id"]]).read_bytes()
        norm, display = engine.preprocessor.preprocess_with_display(data)
        idx = inp.classes.index(row["pred_label"])
        cam = engine.generate_gradcam_overlay(norm, display, idx)
        axes[0, j].imshow(display); axes[1, j].imshow(_decode(cam["overlay_image"]))
        axes[0, j].set_title(f"{row['true_label'].replace('Pulmonary ', '')}\np = {row['conf']:.2f}", fontsize=8)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.text(0.5, 0.01, "Top: preprocessed input. Bottom: Grad-CAM for the predicted class (model attention, not radiological "
             "reasoning). One correctly classified test image per class, median confidence.", ha="center", va="bottom",
             fontsize=7, color=GREY, style="italic", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return save(fig, "fig_gradcam_examples")


FIGS = [
    ("fig_pipeline", fig_pipeline), ("fig_system_architecture", fig_system_architecture),
    ("fig_dataflow", fig_dataflow), ("fig_model_architecture", fig_model_architecture),
    ("fig_preprocessing", fig_preprocessing), ("fig_dataset_composition", fig_dataset_composition),
    ("fig_training_curves", fig_training_curves), ("fig_confusion_matrix", fig_confusion_matrix),
    ("fig_roc_curves", fig_roc_curves), ("fig_per_class_metrics", fig_per_class_metrics),
    ("fig_per_source", fig_per_source), ("fig_external_breakdown", fig_external_breakdown),
    ("fig_calibration", fig_calibration),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-prefix", default="v6")
    ap.add_argument("--run", default="runs/v6_densenet121_seed42")
    ap.add_argument("--build", default="data/derived/v6_build_001")
    ap.add_argument("--only", nargs="*", help="figure stems to build (default: all)")
    ap.add_argument("--gradcam", action="store_true",
                    help="also build fig_gradcam_examples (loads TensorFlow and the installed checkpoint)")
    args = ap.parse_args(argv)
    setup_style()
    inp = Inputs(args.eval_prefix, Path(args.run), Path(args.build))
    if args.gradcam:
        fig_gradcam_examples(inp)
    for stem, fn in FIGS:
        if args.only and stem not in args.only:
            continue
        fn(inp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
