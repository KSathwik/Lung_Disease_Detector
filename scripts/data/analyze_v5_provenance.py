"""Reconstruct the composition and provenance of the historical V5 dataset from manifests only.

    python scripts/data/analyze_v5_provenance.py

Reads data/manifests/unified_manifest_v5.csv (never modified) and the V1 manifest
experiments/data/unified_manifest.csv, which kept the original source labels that V5 dropped.
Writes artifacts/data_quality/v5_provenance.{json,md}. No images are needed.

VinDr class ids in the V1 record are interpreted with the VinBigData competition label order
(0 Aortic enlargement ... 8 Nodule/Mass, 10 Pleural effusion ... 13 Pulmonary fibrosis, 14 No finding).
That mapping is an assumption until checked against the source annotation file.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lungai.config import REPO_ROOT, sha256_file  # noqa: E402
from lungai.data.checks import add_patient_groups  # noqa: E402
from lungai.data.manifest import load_manifest, manifest_path  # noqa: E402

V1_PATH = REPO_ROOT / "experiments" / "data" / "unified_manifest.csv"
OUT = REPO_ROOT / "artifacts" / "data_quality"
VINDR_CLASSES = {
    0: "Aortic enlargement", 1: "Atelectasis", 2: "Calcification", 3: "Cardiomegaly", 4: "Consolidation",
    5: "ILD", 6: "Infiltration", 7: "Lung Opacity", 8: "Nodule/Mass", 9: "Other lesion", 10: "Pleural effusion",
    11: "Pleural thickening", 12: "Pneumothorax", 13: "Pulmonary fibrosis", 14: "No finding",
}
SOURCE_REGISTRY = {
    "Existing_COVID-19": "covid_radiography", "Existing_Normal": "kermany", "Existing_Pneumonia": "kermany",
    "Existing_Tuberculosis": "rahman_tb", "TBX11K": "tbx11k", "VinBigData_VinDr_CXR": "vindr_cxr",
    "NIH_ChestX-ray14": "nih_cxr14", "JSRT": "jsrt",
}
# How patient identity is represented per source in V5 (established from manifest content and ingestion code)
PATIENT_ID_STATUS = {
    "NIH_ChestX-ray14": "genuine (NIH Patient ID, from filename prefix)",
    "Existing_Normal": "recoverable from filename token (Kermany personNNN / IM-NNNN); manifest uses synthetic per-image IDs",
    "Existing_Pneumonia": "recoverable from filename token (Kermany personNNN); manifest uses synthetic per-image IDs",
    "Existing_COVID-19": "unknown (archive renamed files COVID-<n>.png; upstream IDs only via source URLs in the archive metadata)",
    "Existing_Tuberculosis": "unknown (archive renamed files Tuberculosis-<n>.png)",
    "TBX11K": "unknown (dataset publishes no patient IDs); manifest uses synthetic per-image IDs",
    "VinBigData_VinDr_CXR": "unknown (VinDr-CXR is de-identified per image); manifest 'vinbigdata_pt_N' grouping has undocumented origin",
    "JSRT": "one image per case (per JSRT documentation, to confirm); manifest uses per-case IDs",
}


def _basename(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1]


def near_duplicates(df: pd.DataFrame, max_distance: int) -> list:
    """Pairs of images whose 64-bit dHash differ in <= max_distance bits (exhaustive, vectorised)."""
    h = df["dhash"].astype("uint64").to_numpy()
    pairs = []
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    for i in range(len(h) - 1):
        x = np.bitwise_xor(h[i + 1:], h[i])
        d = sum(lut[((x >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.uint8)] for k in range(8))
        for j in np.nonzero(d <= max_distance)[0]:
            a, b = df.iloc[i], df.iloc[i + 1 + j]
            pairs.append({"a": a["image_id"], "b": b["image_id"], "hamming": int(d[j]),
                          "splits": "+".join(sorted({a["split"], b["split"]})),
                          "sources": "+".join(sorted({a["source_dataset"], b["source_dataset"]})),
                          "labels_agree": bool(a["clinical_label"] == b["clinical_label"])})
    return pairs


def analyze() -> dict:
    v5 = load_manifest()
    v1 = pd.read_csv(V1_PATH)
    v5["file"] = _basename(v5["image_path"])
    v1["file"] = _basename(v1["image_path"])
    grouped = add_patient_groups(v5)

    per_source = {}
    for src, g in v5.groupby("source_dataset"):
        per_source[src] = {
            "registry_id": SOURCE_REGISTRY.get(src),
            "images": int(len(g)),
            "by_label": {k: int(v) for k, v in g["clinical_label"].value_counts().items()},
            "by_split": {k: int(v) for k, v in g["split"].value_counts().items()},
            "manifest_patient_ids": int(g["patient_id"].nunique()),
            "derived_patient_groups": int(grouped.loc[g.index, "patient_group"].nunique()),
            "patient_identity": PATIENT_ID_STATUS.get(src, "unknown"),
            "filename_examples": g["file"].head(3).tolist(),
            "absolute_paths": int(g["image_path"].astype(str).str.match(r"^[A-Za-z]:[\\/]").sum()),
        }

    # VinDr: recover multi-finding labels from the V1 record
    vd = v5[v5.source_dataset == "VinBigData_VinDr_CXR"].merge(
        v1[v1.source_dataset == "VinBigData_VinDr_CXR"][["file", "original_label"]], on="file", how="left")
    ids = vd["original_label"].dropna().astype(str).str.split(",").map(lambda xs: sorted({int(x) for x in xs}))
    vd.loc[ids.index, "n_findings"] = ids.map(len)
    vindr = {
        "v5_rows": int(len(vd)), "matched_to_v1_record": int(ids.shape[0]),
        "distinct_findings_per_image": {int(k): int(v) for k, v in vd["n_findings"].value_counts().sort_index().items()},
        "images_with_2plus_findings": int((vd["n_findings"] >= 2).sum()),
        "effusion_images_also_nodule_mass": int(sum(8 in x for x, lab in zip(ids, vd.loc[ids.index, "clinical_label"]) if lab == "Pleural Effusion")),
        "co_finding_frequency_by_label": {
            lab: {VINDR_CLASSES[c]: int(sum(c in x for x in ids[vd.loc[ids.index, "clinical_label"] == lab]))
                  for c in range(14)}
            for lab in ("Pleural Effusion", "Pulmonary Nodule / Mass")
        },
        "ingestion_rule": "experiments/rebuild_unified_dataset_v2.py: 'Pleural_effusion' in cats and len(cats) <= 3 (WBF COCO categories); effusion takes priority over Nodule/Mass",
        "class_id_mapping": "assumed VinBigData competition order (to verify)",
    }

    tb = v5[v5.source_dataset == "TBX11K"].merge(
        v1[v1.source_dataset == "TBX11K"][["file", "original_label"]], on="file", how="left")
    tbx11k = {
        "by_label_and_v1_original_label": {
            lab: {k: int(v) for k, v in g["original_label"].fillna("not_in_v1_record").value_counts().items()}
            for lab, g in tb.groupby("clinical_label")
        },
        "note": "Rows absent from the V1 record were added in V2+ from data.csv image_type ('healthy' -> Normal, 'sick_but_no_tb' -> Pneumonia).",
    }

    js = v5[v5.source_dataset == "JSRT"].merge(v1[v1.source_dataset == "JSRT"][["file", "original_label"]], on="file", how="left")
    jsrt = {lab: {k: int(v) for k, v in g["original_label"].str.split("_").str[0].value_counts().items()}
            for lab, g in js.groupby("clinical_label")}

    near = near_duplicates(v5, max_distance=2)
    by_distance = {d: [p for p in near if p["hamming"] == d] for d in range(3)}
    return {
        "report": "V5 provenance reconstruction (manifest-only)",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": "data/manifests/unified_manifest_v5.csv",
        "manifest_sha256_lf": sha256_file(manifest_path(), normalize_newlines=True),
        "v1_manifest": "experiments/data/unified_manifest.csv",
        "v1_manifest_sha256_lf": sha256_file(V1_PATH, normalize_newlines=True),
        "total_images": int(len(v5)),
        "per_source": per_source,
        "vindr_multi_finding": vindr,
        "tbx11k_label_origin": tbx11k,
        "jsrt_label_origin": jsrt,
        "near_duplicates_dhash": {
            "by_hamming_distance": {
                str(d): {"pairs": len(ps), "cross_split": sum("+" in p["splits"] for p in ps),
                         "cross_source": sum("+" in p["sources"] for p in ps),
                         "label_conflicts": sum(not p["labels_agree"] for p in ps)}
                for d, ps in by_distance.items()
            },
            "pairs_distance_le1": by_distance[0] + by_distance[1],
            "note": ("The ingestion dHash algorithm is undocumented and a 64-bit dHash discriminates chest X-rays poorly "
                     "(their coarse structure is similar), so pair counts grow quickly with distance. These are review "
                     "candidates only; V6 needs a stronger near-duplicate method computed from the images."),
        },
    }


def render(r: dict) -> str:
    L = ["# V5 provenance reconstruction", "",
         f"Generated {r['generated_at']} by `python scripts/data/analyze_v5_provenance.py` from the manifests only. Do not edit by hand.", "",
         "| Source (manifest) | Registry id | Images | Labels | Manifest patient IDs | Derived groups | Patient identity |",
         "|---|---|---:|---|---:|---:|---|"]
    for s, v in r["per_source"].items():
        labels = ", ".join(f"{k} {n}" for k, n in v["by_label"].items())
        L.append(f"| {s} | {v['registry_id']} | {v['images']:,} | {labels} | {v['manifest_patient_ids']:,} | {v['derived_patient_groups']:,} | {v['patient_identity']} |")
    vd = r["vindr_multi_finding"]
    L += ["", "## VinDr: findings per image (V1 record)", "",
          f"- {vd['images_with_2plus_findings']:,} of {vd['v5_rows']:,} V5 VinDr images carry 2 or more distinct findings.",
          f"- {vd['effusion_images_also_nodule_mass']} images labelled Pleural Effusion also carry Nodule/Mass.",
          f"- Distinct findings per image: {vd['distinct_findings_per_image']}",
          f"- Rule used: {vd['ingestion_rule']}. Class-id mapping: {vd['class_id_mapping']}.", ""]
    L += ["## TBX11K: V5 label vs original category", ""]
    for lab, d in r["tbx11k_label_origin"]["by_label_and_v1_original_label"].items():
        L.append(f"- {lab}: {d}")
    L += ["", f"## JSRT: V5 label vs original category", ""] + [f"- {k}: {v}" for k, v in r["jsrt_label_origin"].items()]
    nd = r["near_duplicates_dhash"]
    L += ["", "## Near-duplicate candidates (manifest dHash)", "",
          "| Hamming distance | Pairs | Across splits | Across sources | Label conflicts |", "|---:|---:|---:|---:|---:|"]
    for d, v in nd["by_hamming_distance"].items():
        L.append(f"| {d} | {v['pairs']:,} | {v['cross_split']:,} | {v['cross_source']:,} | {v['label_conflicts']:,} |")
    L += ["", nd["note"], ""]
    return "\n".join(L)


def main() -> int:
    r = analyze()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v5_provenance.json").write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    (OUT / "v5_provenance.md").write_text(render(r), encoding="utf-8")
    print(render(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
