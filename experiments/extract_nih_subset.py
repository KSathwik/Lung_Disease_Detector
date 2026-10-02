"""
Staged Acquisition and Quality Verification of NIH ChestX-ray14 Single-Finding Cohort
Under the Phase 2B Controlled Protocol (No-Blind-Download Rule)

Target Classes:
1. Effusion (Isolated) -> Pleural Effusion
2. Nodule & Mass (Isolated) -> Pulmonary Nodule / Mass
3. No Finding (Isolated, Single View per Patient) -> Normal
4. Pneumonia (Isolated) -> Pneumonia
"""

import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import cv2
from PIL import Image

METADATA_PATH = "data/downloads/nih/Data_Entry_2017_v2020.csv"
OUTPUT_DIR = "data/raw/NIH_ChestX-ray14"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(METADATA_PATH)

# In batch 001 (approx first 4,999 rows)
sub = df.iloc[:4999].copy()

# Filter single-finding candidates
eff_df = sub[sub["Finding Labels"] == "Effusion"].copy()
nod_df = sub[sub["Finding Labels"] == "Nodule"].copy()
mass_df = sub[sub["Finding Labels"] == "Mass"].copy()
pne_df = sub[sub["Finding Labels"] == "Pneumonia"].copy()

# Sample 200 normal controls (1 scan per patient)
norm_candidates = sub[sub["Finding Labels"] == "No Finding"].drop_duplicates(subset=["Patient ID"]).head(200).copy()

target_files = set(eff_df["Image Index"]).union(
    set(nod_df["Image Index"]),
    set(mass_df["Image Index"]),
    set(pne_df["Image Index"]),
    set(norm_candidates["Image Index"])
)

print(f"Total target NIH images to extract: {len(target_files)}")
print(f"  Effusion: {len(eff_df)}")
print(f"  Nodule: {len(nod_df)}")
print(f"  Mass: {len(mass_df)}")
print(f"  Pneumonia: {len(pne_df)}")
print(f"  Normal: {len(norm_candidates)}")

# Stream from NIH Box static archive
url = "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

extracted = 0
print("Opening stream to NIH Box images_001.tar.gz...")
with urllib.request.urlopen(req, timeout=30) as resp:
    with tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            fname = os.path.basename(member.name)
            if fname in target_files:
                f = tar.extractfile(member)
                dest = os.path.join(OUTPUT_DIR, fname)
                with open(dest, "wb") as out_f:
                    out_f.write(f.read())
                extracted += 1
                if extracted % 50 == 0 or extracted == len(target_files):
                    print(f"Extracted {extracted}/{len(target_files)} images...")
                if extracted >= len(target_files):
                    break

print(f"Staged extraction complete! Total extracted: {extracted}")
