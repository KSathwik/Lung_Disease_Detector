"""Acquisition, recovery, external-overlap and V6-build scripts, on small synthetic fixtures (no downloads)."""

import importlib.util
import json
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

from lungai.config import REPO_ROOT
from lungai.data.v6 import assign_groups, assign_splits, leakage_summary, split_fingerprint

SCRIPTS = REPO_ROOT / "scripts" / "data"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cxr(seed, size=256):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (size, size), dtype=np.uint8), (0, 0), 6)
    cv2.ellipse(img, (size // 3, size // 2), (size // 8, size // 3), 0, 0, 360, int(rng.integers(90, 200)), -1)
    return img


def _write(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# ─── Registry and acquisition ─────────────────────────────────────────────────

def test_registry_entries_are_complete_and_https():
    reg = json.loads((SCRIPTS / "sources.json").read_text(encoding="utf-8"))["sources"]
    ids = [s["id"] for s in reg]
    assert len(ids) == len(set(ids))
    for s in reg:
        for k in ("name", "official_url", "citation", "license", "access", "acquisition", "verified"):
            assert s.get(k), (s["id"], k)
        assert s["acquisition"] in ("direct", "kaggle_cli", "manual", "nlm_index", "gdrive", "git")
        for f in s.get("files", []):
            assert f["url"].startswith("https://")
    assert next(s for s in reg if s["id"] == "montgomery")["role"] == "external_evaluation_only"


def test_acquire_is_dry_run_by_default(tmp_path, capsys):
    acq = _load("acquire")
    assert acq.main(["nih_cxr14", "--root", str(tmp_path)]) == 0
    assert "Dry run only" in capsys.readouterr().out
    assert not (tmp_path / "raw").exists()


def test_acquire_never_fetches_manual_sources(tmp_path):
    acq = _load("acquire")
    assert acq.main(["vindr_cxr", "--root", str(tmp_path), "--execute"]) == 3
    assert not any((tmp_path / "raw").rglob("*.*")) if (tmp_path / "raw").exists() else True


def test_download_refuses_http_and_overwrite(tmp_path):
    acq = _load("acquire")
    with pytest.raises(ValueError, match="non-HTTPS"):
        acq.download("http://example.org/x", tmp_path / "x")
    existing = tmp_path / "y"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        acq.download("https://example.org/y", existing)
    assert existing.read_bytes() == b"keep"


# ─── Recovery by content ──────────────────────────────────────────────────────

def test_recovery_matches_by_content_not_name(tmp_path):
    rec = _load("verify_recovery")
    a, b = tmp_path / "renamed_a.png", tmp_path / "sub" / "b.png"
    _write(a, _cxr(1))
    _write(b, _cxr(2))
    manifest = pd.DataFrame({"image_id": ["A", "B", "C"], "source_dataset": ["s1", "s1", "s2"], "split": ["train", "test", "val"],
                             "md5_hash": [rec.md5_file(a), rec.md5_file(b), "0" * 32]})
    r = rec.reconcile(manifest, rec.scan(tmp_path))
    assert r["recovered_total"] == 2 and r["per_source"]["s2"]["missing"] == 1
    assert set(r["path_map"].dropna()["recovered_path"]) == {"renamed_a.png", "sub/b.png"}


# ─── External overlap ─────────────────────────────────────────────────────────

def test_external_overlap_finds_exact_and_resized_copies(tmp_path):
    ov = _load("check_external_overlap")
    ext, cand = tmp_path / "ext", tmp_path / "cand"
    img1, img2, other = _cxr(10, 512), _cxr(11, 512), _cxr(12, 512)
    _write(ext / "MCUCXR_0001_1.png", img1)
    _write(ext / "MCUCXR_0002_0.png", img2)
    _write(cand / "Tuberculosis-1.png", img1)                                  # byte-identical copy
    _write(cand / "Tuberculosis-2.png", cv2.resize(img2, (256, 256)))          # resized copy
    _write(cand / "Tuberculosis-3.png", other)                                 # unrelated
    matches = ov.find_overlap(ov.collect([ext]), ov.collect([cand]), max_distance=10, min_corr=0.97)
    found = {(m["external"].split("\\")[-1].split("/")[-1], m["candidate"].split("\\")[-1].split("/")[-1], m["method"]) for m in matches}
    assert ("MCUCXR_0001_1.png", "Tuberculosis-1.png", "md5") in found
    assert ("MCUCXR_0002_0.png", "Tuberculosis-2.png", "phash+correlation") in found
    assert not any("Tuberculosis-3" in m["candidate"] for m in matches)


# ─── V6 build ─────────────────────────────────────────────────────────────────

def _fixture_raw(tmp_path):
    raw = tmp_path / "raw"
    for i in range(12):
        _write(raw / "montgomery" / "CXR_png" / f"MCUCXR_{i:04d}_{i % 2}.png", _cxr(100 + i))
        _write(raw / "shenzhen" / "CXR_png" / f"CHNCXR_{i:04d}_{i % 2}.png", _cxr(200 + i))
    for folder in ("train", "test"):
        for p in range(10):
            for k in range(2):
                cls, name = ("NORMAL", f"IM-{p:04d}-000{k}.jpeg") if p % 2 else ("PNEUMONIA", f"person{p}_bacteria_{k}.jpeg")
                _write(raw / "kermany" / "chest_xray" / folder / cls / name, _cxr(300 + p * 10 + k + (0 if folder == "train" else 500)))
    rows = []
    for i in range(30):
        name = f"{i:08d}_000.png"
        _write(raw / "nih_cxr14" / "images" / name, _cxr(400 + i))
        rows.append({"Image Index": name, "Finding Labels": ["No Finding", "Effusion", "Nodule|Mass", "Effusion|Mass", "Pneumonia"][i % 5],
                     "Follow-up #": 0, "Patient ID": i // 2, "Patient Age": 50, "Patient Sex": "M", "View Position": "PA"})
    pd.DataFrame(rows).to_csv(raw / "nih_cxr14" / "Data_Entry_2017_v2020.csv", index=False)
    return raw


def test_v6_build_rules_grouping_and_quarantine(tmp_path):
    b6 = _load("build_v6_manifest")
    raw = _fixture_raw(tmp_path)
    cfg = json.loads(b6.PROPOSAL.read_text(encoding="utf-8"))
    cfg["sources"]["kermany"]["include"] = True
    df = b6.build(raw, ["montgomery", "shenzhen", "kermany", "nih_cxr14"], cfg)

    assert set(df.loc[df.source_id == "montgomery", "split"]) == {"external"}
    nih = df[df.source_id == "nih_cxr14"]
    assert (nih.loc[nih.source_label_raw == "Effusion|Mass", "split"] == "excluded").all()
    assert set(nih.loc[nih.exclusion_reason == "", "label_rule_id"]) <= {"nih_normal", "nih_effusion", "nih_nodule_mass", "nih_pneumonia"}
    k = df[df.source_id == "kermany"]
    assert set(k.patient_id_status) == {"derived_from_filename"}
    leak = leakage_summary(df)
    assert leak == {"groups_spanning_splits": 0, "md5_spanning_splits": 0, "external_md5_in_development": 0}
    # Kermany patients are scoped by their original folder: train/person0 and test/person0 are different people
    assert k[k.source_image_id.str.startswith("person0_")]["source_patient_id"].nunique() == 2


def test_v6_duplicates_are_forced_into_one_group(tmp_path):
    df = pd.DataFrame({
        "image_id": ["a", "b", "c", "d"], "source_id": ["s1", "s2", "s1", "s1"], "source_patient_id": ["", "", "p1", "p1"],
        "patient_id_status": ["unknown", "unknown", "genuine", "genuine"], "raw_md5": ["x", "x", "y", "z"],
        "lungai_label": ["Normal"] * 4, "exclusion_reason": [""] * 4, "role": ["development"] * 4,
    })
    g = assign_groups(df, duplicate_pairs=[("b", "c")])
    assert g.nunique() == 1  # a=b by md5, b~c by near-duplicate, c=d by patient


def test_v6_split_is_deterministic_and_fingerprinted():
    n = 300
    df = pd.DataFrame({"image_id": [f"i{i}" for i in range(n)], "source_id": "s", "patient_group": [f"g{i // 3}" for i in range(n)],
                       "lungai_label": ["Normal", "Pneumonia"] * (n // 2), "exclusion_reason": "", "role": "development"})
    a = assign_splits(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    b = assign_splits(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    assert a.equals(b)
    assert (df.assign(split=a).groupby("patient_group")["split"].nunique() == 1).all()
    fp = split_fingerprint(df.assign(split=a))
    assert fp == split_fingerprint(df.assign(split=a).sample(frac=1, random_state=0)) and len(fp) == 64


def test_v6_pending_adapters_fail_loudly(tmp_path):
    b6 = _load("build_v6_manifest")
    with pytest.raises(b6.AdapterError, match="not implemented"):
        b6.ADAPTERS["jsrt"](tmp_path)


def test_v6_proposal_encodes_the_required_label_policies():
    cfg = json.loads((REPO_ROOT / "lungai/configs/dataset_v6_proposal.json").read_text(encoding="utf-8"))
    tbx = cfg["sources"]["tbx11k"]
    assert {e["id"] for e in tbx["exclude"]} >= {"tbx_sick_non_tb", "tbx_latent"}
    assert not any(r["label"] == "Pneumonia" for r in tbx["rules"])
    assert cfg["sources"]["rahman_tb"]["include"] is False
    assert cfg["external_cohorts"] == ["montgomery"] and cfg["sources"]["montgomery"]["include"] is False


def test_v6_caps_sample_whole_patients(tmp_path):
    b6 = _load("build_v6_manifest")
    df = pd.DataFrame({"image_id": [f"i{i}" for i in range(40)], "source_id": "nih_cxr14", "lungai_label": "Pleural Effusion",
                       "source_patient_id": [str(i // 4) for i in range(40)], "patient_id_status": "genuine",
                       "exclusion_reason": "", "role": "development"})
    out = b6.apply_caps(df, {"nih_cxr14": {"Pleural Effusion": 10}}, seed=1)
    kept = out[out.exclusion_reason == ""]
    assert len(kept) == 8  # two whole patients of 4 images fit under the cap of 10
    assert kept.groupby("source_patient_id").size().eq(4).all()
    again = b6.apply_caps(df, {"nih_cxr14": {"Pleural Effusion": 10}}, seed=1)
    assert out.equals(again)


def test_v6_development_copy_of_external_image_is_excluded(tmp_path):
    b6 = _load("build_v6_manifest")
    raw = tmp_path / "raw"
    img = _cxr(7, 512)
    _write(raw / "montgomery" / "CXR_png" / "MCUCXR_0001_1.png", img)
    _write(raw / "shenzhen" / "CXR_png" / "CHNCXR_0001_1.png", cv2.resize(img, (300, 300)))  # resized copy
    for i in range(2, 8):
        _write(raw / "shenzhen" / "CXR_png" / f"CHNCXR_{i:04d}_{i % 2}.png", _cxr(50 + i))
    cfg = json.loads(b6.PROPOSAL.read_text(encoding="utf-8"))
    df = b6.build(raw, ["montgomery", "shenzhen"], cfg, cache_dir=None, workers=2)
    row = df[df.image_id == "shenzhen:CHNCXR_0001_1.png"].iloc[0]
    assert row.exclusion_reason == "duplicates an external-cohort image" and row.split == "excluded"


def test_v6_cohen_and_tbx11k_adapters(tmp_path):
    b6 = _load("build_v6_manifest")
    raw = tmp_path / "raw"
    c = raw / "cohen_covid"
    for i, n in enumerate(["a.png", "b.png", "c.png", "d.png"]):
        _write(c / "images" / n, _cxr(60 + i))
    pd.DataFrame({"patientid": [1, 1, 2, 3], "finding": ["Pneumonia/Viral/COVID-19", "Pneumonia/Viral/COVID-19", "Pneumonia/Bacterial/Streptococcus", "Pneumonia/Viral/COVID-19"],
                  "view": ["PA", "L", "AP Supine", "PA"], "modality": ["X-ray", "X-ray", "X-ray", "CT"], "folder": "images",
                  "filename": ["a.png", "b.png", "c.png", "d.png"], "license": "CC BY", "sex": "M", "age": 50}).to_csv(c / "metadata.csv", index=False)
    rows = pd.DataFrame(b6.adapt_cohen(c))
    assert rows.set_index("source_image_id")["lungai_label"].to_dict() == {"a.png": "COVID-19", "b.png": "", "c.png": "Pneumonia", "d.png": ""}
    t = raw / "tbx11k" / "TBX11K"
    for folder, n in [("health", "h1.png"), ("sick", "s1.png"), ("tb", "tb1.png"), ("tb", "tb2.png")]:
        _write(t / "imgs" / folder / n, _cxr(len(n) + len(folder)))
    (t / "annotations" / "json").mkdir(parents=True)
    coco = {"categories": [{"id": 1, "name": "ActiveTuberculosis"}, {"id": 2, "name": "ObsoletePulmonaryTuberculosis"}],
            "images": [{"id": 1, "file_name": "tb/tb1.png"}, {"id": 2, "file_name": "tb/tb2.png"}],
            "annotations": [{"image_id": 1, "category_id": 1}, {"image_id": 2, "category_id": 1}, {"image_id": 2, "category_id": 2}]}
    (t / "annotations" / "json" / "TBX11K_train.json").write_text(json.dumps(coco))
    r = pd.DataFrame(b6.adapt_tbx11k(raw / "tbx11k")).set_index("source_image_id")
    assert r.loc["h1.png", "lungai_label"] == "Normal" and r.loc["tb1.png", "lungai_label"] == "Tuberculosis"
    assert r.loc["s1.png", "lungai_label"] == "" and "sick" in r.loc["s1.png", "exclusion_reason"]
    assert r.loc["tb2.png", "lungai_label"] == "" and "latent" in r.loc["tb2.png", "exclusion_reason"]
