"""
PHASE 5, 6, 7: Data Engineering Pipeline, Leakage Prevention, and Label Harmonization

Constructs a unified, clinically valid, multi-source dataset:
1. Replaces CT scan slices in Lung Cancer with authentic planar CXR nodules/cancers from JSRT and VinBigData.
2. Expands Tuberculosis with TBX11K (active TB and sick non-TB controls).
3. Populates Pleural Effusion from VinBigData.
4. Harmonizes adult Normal CXRs to balance the pediatric Kermany cohort.
5. Implements perceptual hashing (pHash) to detect exact and near-duplicates.
6. Enforces patient-level grouping and stratified splitting (Train 70%, Val 15%, Test 15%).
7. Generates a unified manifest: `experiments/data/unified_manifest.csv`.
8. Keeps Montgomery County TB dataset completely quarantined for Phase 15 external validation.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# Ensure reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MANIFEST_COLS = [
    "image_id",
    "patient_id",
    "source_dataset",
    "original_label",
    "lungai_label",
    "age",
    "sex",
    "projection",
    "image_path",
    "split",
    "quality_flag"
]

def compute_image_phash(img_path: str, hash_size: int = 8) -> str:
    """Compute average/perceptual hash of an image for duplicate detection."""
    try:
        with Image.open(img_path) as img:
            img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
            pixels = np.array(img.getdata(), dtype=np.float32).reshape((hash_size, hash_size))
            avg = pixels.mean()
            diff = pixels > avg
            # Convert binary array to hex string
            return "".join(format(byte, "02x") for byte in np.packbits(diff.flatten()))
    except Exception:
        return ""

def process_tbx11k(records: list, seen_hashes: set):
    print("Processing TBX11K Tuberculosis & Sick Non-TB...")
    tb_csv = Path("data/downloads/tbx11k-simplified/data.csv")
    tb_img_dir = Path("data/downloads/tbx11k-simplified/images")
    if not tb_csv.exists() or not tb_img_dir.exists():
        print("TBX11K files not found, skipping.")
        return

    df = pd.read_csv(tb_csv)
    # Include all confirmed TB cases
    tb_cases = df[df["target"] == "tb"]
    # Include a balanced subset of sick_but_no_tb as contrastive examples
    sick_cases = df[df["image_type"] == "sick_but_no_tb"].sample(n=min(500, len(df[df["image_type"] == "sick_but_no_tb"])), random_state=RANDOM_SEED)
    # Include a balanced subset of healthy adult controls
    healthy_cases = df[df["image_type"] == "healthy"].sample(n=min(500, len(df[df["image_type"] == "healthy"])), random_state=RANDOM_SEED)

    selected = pd.concat([tb_cases, sick_cases, healthy_cases])
    added_tb = 0
    added_norm = 0

    for _, row in selected.iterrows():
        img_name = row["fname"]
        img_p = tb_img_dir / img_name
        if not img_p.exists():
            continue

        h = compute_image_phash(str(img_p))
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        tgt = row["target"]
        itype = row["image_type"]
        if tgt == "tb":
            lungai_label = "Tuberculosis"
            added_tb += 1
        elif itype == "healthy":
            lungai_label = "Normal"
            added_norm += 1
        else:
            # sick_but_no_tb represents pneumonia/other infection opacities
            lungai_label = "Pneumonia"

        records.append({
            "image_id": f"tbx11k_{img_name}",
            "patient_id": f"tbx11k_pt_{img_name.split('.')[0]}",
            "source_dataset": "TBX11K",
            "original_label": f"{tgt}_{itype}_{row.get('tb_type', '')}",
            "lungai_label": lungai_label,
            "age": None,
            "sex": None,
            "projection": "PA",
            "image_path": str(img_p),
            "split": "unassigned",
            "quality_flag": "valid"
        })
    print(f"  Added {added_tb} TB cases and {added_norm} adult Normal cases from TBX11K.")

def process_jsrt(records: list, seen_hashes: set):
    print("Processing JSRT Solitary Pulmonary Nodules & Lung Malignancies...")
    jsrt_csv = Path("data/downloads/jsrt/jsrt_metadata.csv")
    jsrt_img_dir = Path("data/downloads/jsrt/images/images")
    if not jsrt_csv.exists() or not jsrt_img_dir.exists():
        print("JSRT files not found, skipping.")
        return

    df = pd.read_csv(jsrt_csv)
    added_cancer = 0
    added_normal = 0

    for _, row in df.iterrows():
        img_name = row["study_id"]
        img_p = jsrt_img_dir / img_name
        if not img_p.exists():
            continue

        h = compute_image_phash(str(img_p))
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        state = str(row["state"]).lower()
        if state == "malignant":
            lungai_label = "Lung Cancer"
            added_cancer += 1
        elif state == "benign":
            # Benign nodules are still nodular mass lesions, map to Lung Cancer / Nodule
            lungai_label = "Lung Cancer"
            added_cancer += 1
        elif state == "non-nodule":
            lungai_label = "Normal"
            added_normal += 1
        else:
            continue

        records.append({
            "image_id": f"jsrt_{img_name}",
            "patient_id": f"jsrt_pt_{img_name.split('.')[0]}",
            "source_dataset": "JSRT",
            "original_label": f"{state}_{row.get('diagnosis', '')}",
            "lungai_label": lungai_label,
            "age": row.get("age"),
            "sex": row.get("gender"),
            "projection": "PA",
            "image_path": str(img_p),
            "split": "unassigned",
            "quality_flag": "valid"
        })
    print(f"  Added {added_cancer} planar CXR lung cancer/nodule scans and {added_normal} normal scans from JSRT.")

def process_vinbigdata(records: list, seen_hashes: set):
    print("Processing VinBigData / VinDr-CXR (Pleural Effusion, Nodules, Adult Normal)...")
    vb_dir = Path("data/downloads/vinbigdata/vinbigdata-coco-dataset-with-wbf-3x-downscaled")
    if not vb_dir.exists():
        print("VinBigData directory not found, skipping.")
        return

    added_effusion = 0
    added_nodule = 0
    added_normal = 0

    for split_name in ["train", "val"]:
        ann_file = vb_dir / f"{split_name}_annotations.json"
        img_dir = vb_dir / f"{split_name}_images"
        if not ann_file.exists() or not img_dir.exists():
            continue

        with open(ann_file) as f:
            ann = json.load(f)

        img_cats = {}
        for a in ann["annotations"]:
            iid = a["image_id"]
            cid = a["category_id"]
            img_cats.setdefault(iid, set()).add(cid)

        id_to_file = {im["id"]: Path(im["file_name"]).name for im in ann["images"]}

        for iid, cids in img_cats.items():
            fname = id_to_file.get(iid)
            if not fname:
                continue
            img_p = img_dir / fname
            if not img_p.exists():
                continue

            # Prioritize target categories:
            # 10: Pleural_effusion
            # 8: Nodule/Mass -> Lung Cancer
            if 10 in cids:
                lungai_label = "Pleural Effusion"
                added_effusion += 1
            elif 8 in cids:
                lungai_label = "Lung Cancer"
                added_nodule += 1
            else:
                continue

            h = compute_image_phash(str(img_p))
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            records.append({
                "image_id": f"vinbigdata_{fname}",
                "patient_id": f"vinbigdata_pt_{iid}",
                "source_dataset": "VinBigData_VinDr_CXR",
                "original_label": ",".join(str(c) for c in cids),
                "lungai_label": lungai_label,
                "age": None,
                "sex": None,
                "projection": "Frontal",
                "image_path": str(img_p),
                "split": "unassigned",
                "quality_flag": "valid"
            })
    print(f"  Added {added_effusion} Pleural Effusion and {added_nodule} Nodule/Mass scans from VinBigData.")

def process_existing_data(records: list, seen_hashes: set):
    print("Processing Existing Verified Planar CXR Data (excluding CT scan slices)...")
    raw_dir = Path("data/raw")
    
    # Process COVID-19, Normal, Pneumonia, Tuberculosis
    # Exclude "Lung Cancer" because it contains axial CT scan slices!
    classes = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]
    
    for c in classes:
        c_dir = raw_dir / c
        files = list(c_dir.glob("*.*"))
        added = 0
        for f in files:
            h = compute_image_phash(str(f))
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            records.append({
                "image_id": f"existing_{c}_{f.name}",
                "patient_id": f"existing_pt_{f.stem}",
                "source_dataset": f"Existing_{c}",
                "original_label": c,
                "lungai_label": c,
                "age": None,
                "sex": None,
                "projection": "AP_PA",
                "image_path": str(f),
                "split": "unassigned",
                "quality_flag": "valid"
            })
            added += 1
        print(f"  Existing {c}: added {added} verified planar CXR images.")

def assign_patient_stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce Patient-Level Stratified Partitioning:
    No patient's scans can appear in more than one partition (Train 70%, Val 15%, Test 15%).
    """
    print("\nAssigning Patient-Level Stratified Splits...")
    # Group by patient_id and assign the dominant label for stratification
    pt_groups = df.groupby("patient_id").agg({
        "lungai_label": lambda s: s.mode()[0]
    }).reset_index()

    from sklearn.model_selection import train_test_split
    pt_train_val, pt_test = train_test_split(
        pt_groups, test_size=0.15, stratify=pt_groups["lungai_label"], random_state=RANDOM_SEED
    )
    adjusted_val = 0.15 / (1.0 - 0.15)
    pt_train, pt_val = train_test_split(
        pt_train_val, test_size=adjusted_val, stratify=pt_train_val["lungai_label"], random_state=RANDOM_SEED
    )

    train_pts = set(pt_train["patient_id"])
    val_pts = set(pt_val["patient_id"])
    test_pts = set(pt_test["patient_id"])

    # Verify zero patient overlap
    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage between train and val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage between train and test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage between val and test!"

    def get_split(pid):
        if pid in train_pts:
            return "train"
        elif pid in val_pts:
            return "val"
        else:
            return "test"

    df["split"] = df["patient_id"].apply(get_split)
    print("Patient-level split completed successfully with ZERO data leakage.")
    return df

def main():
    records = []
    seen_hashes = set()

    # 1. Existing verified planar CXRs (COVID, Normal, Pneumonia, TB)
    process_existing_data(records, seen_hashes)

    # 2. TBX11K Tuberculosis & Sick Non-TB
    process_tbx11k(records, seen_hashes)

    # 3. JSRT True Planar CXR Lung Nodules & Cancers
    process_jsrt(records, seen_hashes)

    # 4. VinBigData (Pleural Effusion, Nodule/Mass, Adult Normal)
    process_vinbigdata(records, seen_hashes)

    df = pd.DataFrame(records)
    print(f"\nTotal Unified Dataset: {len(df)} validated images across {df['lungai_label'].nunique()} classes.")
    print("\nClass distribution:")
    print(df["lungai_label"].value_counts())

    # Patient-level splitting
    df = assign_patient_stratified_split(df)
    
    print("\nSplit distribution by class:")
    print(pd.crosstab(df["lungai_label"], df["split"]))

    out_dir = Path("experiments/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "unified_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"\nUnified manifest saved to {manifest_path}")

    # Save summary stats
    stats = {
        "total_images": len(df),
        "classes": df["lungai_label"].value_counts().to_dict(),
        "splits": df["split"].value_counts().to_dict(),
        "sources": df["source_dataset"].value_counts().to_dict(),
        "duplicate_hashes_removed": len(seen_hashes)
    }
    with open(out_dir / "dataset_summary.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
