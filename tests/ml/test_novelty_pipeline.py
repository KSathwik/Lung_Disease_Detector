"""End-to-end run of the novelty CLIs (C1, C3, C4, C2-lite, decision, macros) on synthetic feature caches.

No TensorFlow and no image: feature files with the protocol's sample sizes are written to a temporary directory and
every module's output paths are redirected there.
"""

import numpy as np
import pandas as pd
import pytest

from lungai.novelty import c2_lite, common, features, inlp_eval, loso, render_novelty, spectral_map
from lungai.novelty.common import PROTOCOL_COUNTS as COUNTS, SAMPLE_SETS, AccessError
from lungai.novelty.spectral import PASS_VARIANTS, VARIANTS

D = 24
SRC = sorted({s for s, _ in COUNTS})
CLS = ["COVID-19", "Normal", "Pleural Effusion", "Pneumonia", "Pulmonary Nodule / Mass", "Tuberculosis"]


def _rows(set_key):
    s = SAMPLE_SETS[set_key]
    if s.split == "external":
        return pd.DataFrame([("montgomery", "Normal")] * 80 + [("montgomery", "Tuberculosis")] * 58, columns=["source", "label"])
    col = {"train": 0, "val": 1, "test": 2}[s.split]
    rows = [(src, c) for (src, c), n in COUNTS.items() for _ in range(min(n[col], s.cap) if s.cap else n[col])]
    return pd.DataFrame(rows, columns=["source", "label"])


def _write(out, ext, variant, set_key, rng, strength):
    idx = _rows(set_key)
    idx.insert(0, "image_id", [f"{set_key}:{i}" for i in range(len(idx))])
    idx["raw_sha256"] = idx["image_id"]
    x = rng.normal(size=(len(idx), D))
    si = idx["source"].map({s: i for i, s in enumerate(SRC)}).fillna(0).astype(int).to_numpy()
    ci = idx["label"].map({c: i for i, c in enumerate(CLS)}).to_numpy()
    x[np.arange(len(x)), si] += 2.5 * strength
    x[np.arange(len(x)), 8 + ci] += 5.0
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"features_{ext}_{variant}_{set_key}.npy", x.astype(np.float32))
    idx.to_csv(out / f"features_{ext}_{variant}_{set_key}.csv", index=False)


@pytest.fixture(scope="module")
def novelty_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("novelty")
    rng = np.random.default_rng(0)
    for ext in ("imagenet", "served"):
        for s in ("train", "val", "test", "external"):
            _write(out, ext, "full", s, rng, 1.0)
        for v in VARIANTS:
            for s in ("train150", "val100", "test100", "train400"):
                _write(out, ext, v, s, rng, 0.2 if v == "stop_C" else 1.0)
    return out


@pytest.fixture()
def redirected(novelty_dir, monkeypatch):
    for mod in (common, features, inlp_eval, c2_lite):
        monkeypatch.setattr(mod, "OUT_DIR", novelty_dir)
    monkeypatch.setattr(common, "ACCESS_DIR", novelty_dir / "access")
    monkeypatch.setattr(common, "ACCESS_LOG", novelty_dir / "access_log.jsonl")
    monkeypatch.setattr(spectral_map, "C1_JSON", novelty_dir / "c1_spectral_map.json")
    monkeypatch.setattr(spectral_map, "OUT_DIR", novelty_dir)
    monkeypatch.setattr(c2_lite, "C1_JSON", novelty_dir / "c1_spectral_map.json")
    monkeypatch.setattr(c2_lite, "C2_JSON", novelty_dir / "c2_decision.json")
    monkeypatch.setattr(inlp_eval, "C3_JSON", novelty_dir / "c3_inlp.json")
    monkeypatch.setattr(loso, "C4_JSON", novelty_dir / "c4_loso.json")
    return novelty_dir


def test_pipeline_in_protocol_order(redirected, tmp_path, monkeypatch):
    d = redirected
    assert spectral_map.main(["--stage", "c1-main", "--split", "val"]) == 0
    assert spectral_map.main(["--stage", "c1-main", "--split", "test"]) == 0
    with pytest.raises(AccessError):
        spectral_map.main(["--stage", "c1-main", "--split", "test"])
    assert spectral_map.main(["--stage", "c1-served", "--split", "test"]) == 0
    assert spectral_map.main(["--stage", "c1-nofilter", "--split", "test"]) == 1  # no no-filter caches: not run
    c1 = common.read_json(d / "c1_spectral_map.json")["stages"]
    v = c1["c1-main"]["val"]
    assert v["n_scored"] == 1019 and v["n_train"] == 1952 and v["c_D"] == pytest.approx(0.45)
    assert set(v["variants"]) == set(VARIANTS) and set(c1["c1-served"]["test"]["variants"]) == set(PASS_VARIANTS)
    assert v["band_stop"]["C"]["dS"]["estimate"] > 0.3  # the weakened band-stop variant loses source information
    assert c1["c1-nofilter"]["test"]["status"] == "not run"

    assert inlp_eval.main(["--part", "test"]) == 2  # validation curve first
    assert inlp_eval.main(["--part", "val"]) == 0
    assert inlp_eval.main(["--part", "test"]) == 0
    assert inlp_eval.main(["--part", "montgomery"]) == 0
    with pytest.raises(AccessError):
        inlp_eval.main(["--part", "montgomery"])
    c3 = common.read_json(d / "c3_inlp.json")["extractors"]["imagenet"]
    ks = c3["val"]["inlp"]["k_star"]
    assert c3["val"]["curve"][ks]["val_source_ba"] <= 1 / 6 + 0.05 and c3["test"]["k_star"] == ks
    assert "k0" in c3["montgomery"] and c3["montgomery"]["n"] == 138

    assert loso.main(["--part", "main"]) == 0
    assert loso.main(["--part", "montgomery"]) == 0
    c4 = common.read_json(d / "c4_loso.json")
    folds = c4["matrices"]["imagenet"]["folds"]
    assert sorted(h for h, f in folds.items() if f["included"]) == sorted(loso.EXPECTED_INCLUDED)
    assert folds["nih_cxr14"]["C_selection"]["labels"] == [0, 1, 3, 5]  # Effusion, Nodule/Mass neither scored nor used
    assert "binary" in folds["kermany"] and "binary" not in folds["cohen_covid"]

    assert c2_lite.main(["--part", "select"]) == 0
    band = common.read_json(d / "c2_decision.json")["criterion_1"]["b_star"]
    assert band == "C"
    assert c2_lite.main(["--part", "val"]) == 2  # control features missing (criterion 1 holds here)
    for seed in (1, 2, 3):
        for s in ("train150", "val100"):
            _write(d, "imagenet", f"ctrl{seed}_C", s, np.random.default_rng(seed), 1.0)
    common.write_json(d / "control_C.json", {"band": "C", "q": 0.1})
    assert c2_lite.main(["--part", "val"]) == 0
    assert c2_lite.main(["--part", "test"]) == 0
    assert c2_lite.main(["--part", "decision"]) == 0
    dec = common.read_json(d / "c2_decision.json")["decision"]
    assert dec["condition_4"]["status"] == "requires student approval" and dec["decision"] != "GO"

    macros = render_novelty.novelty_macros(d)
    assert macros["BsBand"] == "C" and "SpecMainTestFullSrcBA" in macros and "LosoImgKermanyGap" in macros
    assert "InlpImgKStar" in macros and "InlpImgExtSensKZero" in macros and all(k.isalpha() for k in macros)

    # figures (into a temporary folder, never docs/figures)
    from lungai.evaluation import make_figures
    monkeypatch.setattr(make_figures, "OUT_DIR", tmp_path)
    monkeypatch.setattr(make_figures, "REPO", tmp_path)
    for name, fn in (("c1_spectral_map.json", render_novelty.fig_spectral_map), ("c3_inlp.json", render_novelty.fig_inlp),
                     ("c4_loso.json", render_novelty.fig_loso_matrix)):
        paths = fn(common.read_json(d / name))
        assert [p.suffix for p in paths] == [".png", ".pdf"] and all(p.is_file() for p in paths)
