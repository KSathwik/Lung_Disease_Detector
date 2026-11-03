"""Raw -> derived: build the proposed V6 manifest from raw source folders. Does not download anything.

    python scripts/data/build_v6_manifest.py --raw-root <root>/raw --out <root>/derived/v6 [--sources nih_cxr14 shenzhen ...]

Reads lungai/configs/dataset_v6_proposal.json (labelling rules, seed, fractions) and each source's raw folder
(<raw-root>/<source_id>/, as laid out by the publisher). Writes, never overwriting an existing output folder:

    manifest_v6.csv        every candidate image, included or excluded, with its rule id or exclusion reason
    v6_build_report.json   counts, leakage summary, split fingerprint, config and source fingerprints

Adapters marked VERIFIED_LAYOUT follow the publisher's documented file layout; the others check their expected
columns and stop with an error if the files differ, instead of guessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lungai.config import CLASS_NAMES, PACKAGE_DIR, sha256_file  # noqa: E402
from lungai.data.v6 import V6_COLUMNS, assign_groups, assign_splits, leakage_summary, split_fingerprint, validate_rows  # noqa: E402

PROPOSAL = PACKAGE_DIR / "configs" / "dataset_v6_proposal.json"
IMAGE_EXT = {".png", ".jpg", ".jpeg"}


class AdapterError(RuntimeError):
    """Raised when raw files do not match the layout an adapter expects."""


def _row(source_id: str, rel: str, **kw) -> dict:
    r = {c: "" for c in V6_COLUMNS if c not in ("patient_group", "split")}
    r.update(source_id=source_id, raw_relpath=rel, source_image_id=Path(rel).name, role="development",
             image_id=f"{source_id}:{Path(rel).name}")
    r.update(kw)
    return r


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise AdapterError(f"expected {what} at {path}")
    return path


# ─── Adapters ─────────────────────────────────────────────────────────────────

def adapt_nih(root: Path) -> List[dict]:
    """VERIFIED_LAYOUT: Data_Entry_2017_v2020.csv + images*/ PNGs (any depth)."""
    meta = pd.read_csv(_require(root / "Data_Entry_2017_v2020.csv", "NIH metadata CSV"))
    sex_col = "Patient Sex" if "Patient Sex" in meta.columns else "Patient Gender"
    need = {"Image Index", "Finding Labels", "Patient ID", "View Position", "Patient Age", sex_col}
    if not need <= set(meta.columns):
        raise AdapterError(f"NIH CSV columns differ; need {sorted(need)}, found {list(meta.columns)}")
    files = {p.name: p for p in root.rglob("*.png")}
    rows, seen_normal_patients = [], set()
    for r in meta.to_dict("records"):
        p = files.get(r["Image Index"])
        if p is None:
            continue  # image tarball not downloaded
        raw = str(r["Finding Labels"])
        labels = raw.split("|")
        pid = str(r["Patient ID"])
        rule, lab, excl = "", "", ""
        if labels == ["No Finding"]:
            if pid in seen_normal_patients:
                excl = "nih_normal: one image per patient"
            else:
                rule, lab = "nih_normal", "Normal"
                seen_normal_patients.add(pid)
        elif labels == ["Effusion"]:
            rule, lab = "nih_effusion", "Pleural Effusion"
        elif set(labels) <= {"Nodule", "Mass"}:
            rule, lab = "nih_nodule_mass", "Pulmonary Nodule / Mass"
        elif labels == ["Pneumonia"]:
            rule, lab = "nih_pneumonia", "Pneumonia"
        else:
            excl = "nih: other or multiple findings"
        rows.append(_row("nih_cxr14", p.relative_to(root.parent).as_posix(), source_version="Data_Entry_2017_v2020",
                         source_patient_id=pid, patient_id_status="genuine", source_label_raw=raw,
                         source_label_set=raw, lungai_label=lab, label_rule_id=rule, exclusion_reason=excl,
                         view_position=r["View Position"], age=r["Patient Age"], sex=r[sex_col]))
    return rows


def _adapt_nlm(source_id: str, root: Path, role: str) -> List[dict]:
    """VERIFIED_LAYOUT (Montgomery index observed 2026-11-02): CXR_png/<PREFIX>_<nnnn>_<0|1>.png."""
    img_dir = _require(root / "CXR_png", "CXR_png folder")
    rows = []
    prefix = "mont" if source_id == "montgomery" else "shz"
    for p in sorted(img_dir.glob("*.png")):
        m = re.match(r"^([A-Z]+CXR)_(\d{4})_([01])\.png$", p.name)
        if not m:
            raise AdapterError(f"unexpected {source_id} filename {p.name}")
        tb = m.group(3) == "1"
        rows.append(_row(source_id, p.relative_to(root.parent).as_posix(), source_version="NLM public release",
                         source_patient_id=m.group(2), patient_id_status="genuine",
                         source_label_raw="abnormal (TB manifestation)" if tb else "normal",
                         source_label_set="TB" if tb else "normal",
                         lungai_label="Tuberculosis" if tb else "Normal",
                         label_rule_id=f"{prefix}_{'tb' if tb else 'normal'}", role=role))
    return rows


def adapt_montgomery(root: Path) -> List[dict]:
    return _adapt_nlm("montgomery", root, "external")


def adapt_shenzhen(root: Path) -> List[dict]:
    return _adapt_nlm("shenzhen", root, "development")


def adapt_kermany(root: Path) -> List[dict]:
    """VERIFIED_LAYOUT (Mendeley v2 / Kaggle): .../chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}/*.jpeg."""
    files = [p for p in root.rglob("*.jpeg") if p.parent.name in ("NORMAL", "PNEUMONIA")]
    if not files:
        raise AdapterError("no NORMAL/PNEUMONIA jpeg folders found")
    tok = re.compile(r"(person\d+)|IM-(\d+)", re.IGNORECASE)
    rows = []
    for p in sorted(files):
        folder = p.parent.parent.name  # original train/test/val folder: person numbers are scoped by it
        m = tok.search(p.name)
        pid = f"{folder}:{(m.group(1) or 'IM' + m.group(2)).lower()}" if m else ""
        normal = p.parent.name == "NORMAL"
        rows.append(_row("kermany", p.relative_to(root.parent).as_posix(), source_version="Mendeley v2",
                         image_id=f"kermany:{folder}/{p.parent.name}/{p.name}",
                         source_patient_id=pid, patient_id_status="derived_from_filename" if pid else "unknown",
                         source_label_raw=p.parent.name, source_label_set=p.parent.name,
                         lungai_label="Normal" if normal else "Pneumonia",
                         label_rule_id="kermany_normal" if normal else "kermany_pneumonia", view_position="AP"))
    return rows


def adapt_vindr(root: Path) -> List[dict]:
    """Expected (PhysioNet v1.0.0): annotations/image_labels_train.csv with image_id, rad_id and one column per label;
    images converted from DICOM to PNG under png/ (conversion is a separate, documented step)."""
    lab = pd.read_csv(_require(root / "annotations" / "image_labels_train.csv", "VinDr image labels"))
    need = {"image_id", "rad_id", "Pleural effusion", "Nodule/Mass", "No finding", "Pneumonia", "Tuberculosis",
            "Consolidation", "Lung tumor"}
    if not need <= set(lab.columns):
        raise AdapterError(f"VinDr label columns differ; need {sorted(need)}")
    label_cols = [c for c in lab.columns if c not in ("image_id", "rad_id")]
    vote = lab.groupby("image_id")[label_cols].mean()  # fraction of radiologists marking each label
    pngs = {p.stem: p for p in (root / "png").rglob("*.png")} if (root / "png").exists() else {}
    rows = []
    for iid, v in vote.iterrows():
        majority = {c for c in label_cols if v[c] > 0.5}
        excl_eff = majority & {"Nodule/Mass", "Lung tumor", "Pneumonia", "Tuberculosis", "Consolidation"}
        excl_nod = majority & {"Pleural effusion", "Pneumonia", "Tuberculosis", "Consolidation"}
        rule, label, excl = "", "", ""
        if "Pleural effusion" in majority and not excl_eff:
            rule, label = "vindr_effusion", "Pleural Effusion"
        elif "Nodule/Mass" in majority and not excl_nod:
            rule, label = "vindr_nodule_mass", "Pulmonary Nodule / Mass"
        elif v.get("No finding", 0) == 1.0:
            rule, label = "vindr_normal", "Normal"
        else:
            excl = "vindr: no rule matched (co-occurring findings or no majority)"
        p = pngs.get(iid)
        if p is None:
            excl = excl or "vindr: image not converted/present"
        rows.append(_row("vindr_cxr", p.relative_to(root.parent).as_posix() if p else "", source_version="1.0.0",
                         image_id=f"vindr_cxr:{iid}", source_image_id=iid, patient_id_status="unknown",
                         source_label_raw=json.dumps({c: round(float(v[c]), 3) for c in label_cols if v[c] > 0}),
                         source_label_set="|".join(sorted(majority)), lungai_label=label, label_rule_id=rule,
                         exclusion_reason=excl))
    return rows


def adapt_cohen(root: Path) -> List[dict]:
    """Cohen et al. COVID-19 image collection: metadata.csv + images/. Column names checked; fails if they differ."""
    meta = pd.read_csv(_require(root / "metadata.csv", "Cohen metadata.csv"))
    need = {"patientid", "finding", "view", "modality", "folder", "filename", "license", "sex", "age"}
    if not need <= set(meta.columns):
        raise AdapterError(f"Cohen metadata columns differ; need {sorted(need)}")
    rows = []
    for r in meta.to_dict("records"):
        rel = f"{r['folder']}/{r['filename']}"
        p = root / rel
        finding = str(r["finding"])
        view = str(r["view"])
        frontal = view.upper().startswith(("PA", "AP"))
        rule, lab, excl = "", "", ""
        if str(r["modality"]) != "X-ray" or not frontal:
            excl = "cohen: not a frontal X-ray"
        elif "COVID-19" in finding:
            rule, lab = "cohen_covid", "COVID-19"
        elif finding.startswith("Pneumonia"):
            rule, lab = "cohen_pneumonia", "Pneumonia"
        elif finding == "Tuberculosis":
            rule, lab = "cohen_tb", "Tuberculosis"
        elif finding == "No Finding":
            rule, lab = "cohen_normal", "Normal"
        else:
            excl = "cohen: finding outside the six classes"
        if not p.is_file():
            excl = excl or "cohen: image file missing"
        rows.append(_row("cohen_covid", p.relative_to(root.parent).as_posix() if p.is_file() else "",
                         source_version="git master", image_id=f"cohen_covid:{r['filename']}",
                         source_image_id=str(r["filename"]), source_patient_id=str(r["patientid"]),
                         patient_id_status="genuine", source_label_raw=finding, source_label_set=finding,
                         lungai_label=lab, label_rule_id=rule, exclusion_reason=excl, view_position=view,
                         age="" if pd.isna(r["age"]) else r["age"], sex="" if pd.isna(r["sex"]) else r["sex"]))
    return rows


def adapt_tbx11k(root: Path) -> List[dict]:
    """TBX11K (official zip): TBX11K/imgs/{health,sick,tb}/ + TBX11K/annotations/json/*.json (COCO).
    TB images are labelled Tuberculosis only if every TB box is ActiveTuberculosis;
    any ObsoletePulmonaryTuberculosis (latent TB per the authors) excludes the image."""
    base = root / "TBX11K" if (root / "TBX11K").is_dir() else root
    img_root = _require(base / "imgs", "TBX11K imgs/ folder")
    cats_by_file: Dict[str, set] = {}
    for js in sorted((base / "annotations" / "json").glob("*.json")):
        d = json.loads(js.read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in d.get("categories", [])}
        files = {i["id"]: i["file_name"] for i in d.get("images", [])}
        for a in d.get("annotations", []):
            fn = Path(files.get(a["image_id"], "")).name
            if fn:
                cats_by_file.setdefault(fn, set()).add(names.get(a["category_id"], str(a["category_id"])))
    if not cats_by_file:
        raise AdapterError("no TBX11K COCO annotations found under annotations/json")
    rows = []
    for folder, kind in (("health", "healthy"), ("sick", "sick_but_non_tb"), ("tb", "tb")):
        d = img_root / folder
        if not d.is_dir():
            raise AdapterError(f"TBX11K folder missing: {d}")
        for p in sorted(d.glob("*.png")):
            rule, lab, excl, cats = "", "", "", ""
            if kind == "healthy":
                rule, lab = "tbx_healthy", "Normal"
            elif kind == "sick_but_non_tb":
                excl = "tbx_sick_non_tb: cause unspecified, not used as Pneumonia"
            else:
                c = cats_by_file.get(p.name, set())
                cats = "|".join(sorted(c))
                if not c:
                    excl = "tbx: TB image without box annotation"
                elif c == {"ActiveTuberculosis"}:
                    rule, lab = "tbx_active_tb", "Tuberculosis"
                else:
                    excl = "tbx_latent: latent (ObsoletePulmonaryTuberculosis) or unknown TB annotation"
            rows.append(_row("tbx11k", p.relative_to(root.parent).as_posix(), source_version="TBX11K (official, train+val)",
                             image_id=f"tbx11k:{folder}/{p.name}", patient_id_status="unknown",
                             source_label_raw=kind, source_label_set=cats or kind,
                             lungai_label=lab, label_rule_id=rule, exclusion_reason=excl))
    return rows


def adapt_covid_radiography(root: Path) -> List[dict]:
    """COVID-19 Radiography Database (Kaggle, authors' release): <root>/**/COVID/images/*.png. Only the COVID class is used."""
    dirs = [d for d in root.rglob("COVID") if (d / "images").is_dir()]
    if len(dirs) != 1:
        raise AdapterError(f"expected exactly one COVID/images folder under {root}, found {len(dirs)}")
    rows = []
    for p in sorted((dirs[0] / "images").glob("*.png")):
        rows.append(_row("covid_radiography", p.relative_to(root.parent).as_posix(), source_version="Kaggle release",
                         image_id=f"covid_radiography:{p.name}", patient_id_status="unknown",
                         source_label_raw="COVID", source_label_set="COVID", lungai_label="COVID-19", label_rule_id="covid"))
    return rows


def _pending(name: str) -> Callable[[Path], List[dict]]:
    def f(root: Path) -> List[dict]:
        raise AdapterError(f"the {name} adapter is not implemented yet: inspect the official files first "
                           "(see docs/data/data_recovery_plan.md, step 'adapters')")
    return f


ADAPTERS: Dict[str, Callable[[Path], List[dict]]] = {
    "nih_cxr14": adapt_nih, "montgomery": adapt_montgomery, "shenzhen": adapt_shenzhen,
    "kermany": adapt_kermany, "vindr_cxr": adapt_vindr, "cohen_covid": adapt_cohen, "tbx11k": adapt_tbx11k,
    "covid_radiography": adapt_covid_radiography, "jsrt": _pending("JSRT"),
}


# ─── Sampling caps ────────────────────────────────────────────────────────────

def apply_caps(df: pd.DataFrame, caps: Dict[str, Dict[str, int]], seed: int) -> pd.DataFrame:
    """Per (source, class) cap, applied by sampling whole patient groups (single images when identity is unknown)."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    inc = (df["exclusion_reason"] == "") & (df["role"] == "development")
    df["_k"] = np.where(df["patient_id_status"] == "unknown", "img:" + df["image_id"],
                        "pt:" + df["source_patient_id"].astype(str))
    for (src, lab), g in df[inc].groupby(["source_id", "lungai_label"]):
        cap = (caps.get(src) or {}).get(lab)
        if cap is None or len(g) <= cap:
            continue
        groups = np.array(sorted(g["_k"].unique()))
        rng.shuffle(groups)
        sizes = g.groupby("_k").size()
        keep, n = set(), 0
        for k in groups:
            if n + sizes[k] > cap:
                continue
            keep.add(k)
            n += int(sizes[k])
            if n >= cap:
                break
        drop = g.index[~g["_k"].isin(keep)]
        df.loc[drop, "exclusion_reason"] = f"not sampled: cap {cap} for {src}/{lab}"
    return df.drop(columns="_k")


# ─── Measurements and cache ───────────────────────────────────────────────────

def _phash_hex(gray: np.ndarray) -> str:
    small = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8]
    bits = (small > np.median(small.ravel()[1:])).ravel()
    return "%016x" % int("".join("1" if b else "0" for b in bits), 2)


def measure(raw_root: Path, rel: str, cache_dir: Optional[Path] = None) -> dict:
    """Content hashes, size, pHash, 64x64 thumbnail; optionally caches the canonical 224x224 preprocessed image."""
    from lungai import preprocessing as pp
    data = (raw_root / rel).read_bytes()
    out = {"raw_md5": hashlib.md5(data).hexdigest(), "raw_sha256": hashlib.sha256(data).hexdigest()}
    color = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if color is None or min(color.shape[:2]) < pp.MIN_DIMENSION:
        out["exclusion_reason"] = "undecodable or too small image"
        return out
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    t = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    out.update(phash=_phash_hex(gray), height=gray.shape[0], width=gray.shape[1], _thumb=t / (np.linalg.norm(t) or 1.0))
    if cache_dir is not None:
        np.save(cache_dir / f"{out['raw_sha256']}.npy", pp.resize_display(pp.clean_image(color)))
    return out


def near_duplicate_pairs(df: pd.DataFrame, thumbs: Dict[str, np.ndarray], max_hamming: int, min_corr: float) -> List[tuple]:
    m = df[df["phash"].fillna("") != ""]
    ids = m["image_id"].tolist()
    h = np.array([int(x, 16) for x in m["phash"]], dtype=np.uint64)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    pairs = []
    for i in range(len(h) - 1):
        x = np.bitwise_xor(h[i + 1:], h[i])
        d = sum(lut[((x >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.uint8)] for k in range(8))
        for j in np.nonzero(d <= max_hamming)[0]:
            a, b = ids[i], ids[i + 1 + j]
            if float(np.dot(thumbs[a], thumbs[b])) >= min_corr:
                pairs.append((a, b))
    return pairs


def build(raw_root: Path, sources: List[str], cfg: dict, cache_dir: Optional[Path] = None, workers: int = 8) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor
    rows: List[dict] = []
    for sid in sources:
        if not cfg["sources"].get(sid, {}).get("include", False) and sid not in cfg["external_cohorts"]:
            continue
        rows += ADAPTERS[sid](raw_root / sid)
    df = pd.DataFrame(rows, columns=[c for c in V6_COLUMNS if c not in ("patient_group", "split")])
    if "sampling" in cfg:
        df = apply_caps(df, cfg["sampling"]["caps"], cfg["sampling"]["seed"])
    todo = df.index[(df["raw_relpath"] != "") & ((df["exclusion_reason"] == "") | (df["role"] == "external"))]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda i: measure(raw_root, df.at[i, "raw_relpath"], cache_dir), todo))
    thumbs = {}
    for i, res in zip(todo, results):
        t = res.pop("_thumb", None)
        if t is not None:
            thumbs[df.at[i, "image_id"]] = t
        for k, v in res.items():
            if k != "exclusion_reason" or not df.at[i, "exclusion_reason"]:
                df.at[i, k] = v
    errs = validate_rows(df)
    if errs:
        raise AdapterError("V6 rows failed validation:\n- " + "\n- ".join(errs))
    nd = cfg.get("near_duplicate", {"phash_max_hamming": 6, "min_thumbnail_correlation": 0.97})
    pairs = near_duplicate_pairs(df.loc[todo], thumbs, nd["phash_max_hamming"], nd["min_thumbnail_correlation"])
    # Development images that duplicate an external-cohort image are excluded outright
    ext_ids = set(df.loc[df["role"] == "external", "image_id"])
    ext_md5 = set(df.loc[df["role"] == "external", "raw_md5"]) - {""}
    tainted = {b if a in ext_ids else a for a, b in pairs if (a in ext_ids) != (b in ext_ids)}
    tainted |= set(df.loc[(df["role"] == "development") & df["raw_md5"].isin(ext_md5), "image_id"])
    df.loc[df["image_id"].isin(tainted) & (df["role"] == "development"), "exclusion_reason"] = "duplicates an external-cohort image"
    df["patient_group"] = assign_groups(df, duplicate_pairs=pairs)
    df["split"] = assign_splits(df, cfg["split_fractions"], cfg["seed"])
    df.attrs["near_duplicate_pairs"] = pairs
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", default=list(ADAPTERS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error(f"{args.out} exists; V6 builds never overwrite a previous build")
    cfg = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True)
    cache = args.out / "cache"
    cache.mkdir()
    df = build(args.raw_root, args.sources, cfg, cache_dir=cache, workers=args.workers)
    df.to_csv(args.out / "manifest_v6.csv", index=False, lineterminator="\n")
    dev = df[df["split"].isin(["train", "val", "test"])]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "proposal_sha256": sha256_file(PROPOSAL, normalize_newlines=True),
        "manifest_sha256": sha256_file(args.out / "manifest_v6.csv"),
        "split_fingerprint": split_fingerprint(dev),
        "counts": {s: {lab: int(n) for lab, n in g["lungai_label"].value_counts().items()} for s, g in df.groupby("split")},
        "by_source_split": {f"{a}|{b}": int(n) for (a, b), n in df.groupby(["source_id", "split"]).size().items()},
        "exclusions": {k: int(v) for k, v in df.loc[df["exclusion_reason"] != "", "exclusion_reason"].value_counts().items()},
        "patient_identity": {k: int(v) for k, v in dev["patient_id_status"].value_counts().items()},
        "leakage": leakage_summary(df),
        "near_duplicate_pairs": len(df.attrs.get("near_duplicate_pairs", [])),
        "sources_used": sorted(df["source_id"].unique().tolist()),
        "cache": "cache/<raw_sha256>.npy = canonical lungai.preprocessing output (224x224x3 uint8, before normalisation)",
        "caveat": "groups_spanning_splits = 0 proves separation only for rows with known patient identity; "
                  "rows with patient_id_status 'unknown' are grouped per image (plus exact duplicates).",
        "class_order": CLASS_NAMES,
    }
    (args.out / "v6_build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "leakage", "split_fingerprint")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
