"""Novelty extension (protocol C1-C4) on synthetic data only: no TensorFlow, no model, no real image."""

import json

import numpy as np
import pandas as pd
import pytest

from lungai.novelty import c2_lite
from lungai.novelty.common import (EXPECTED_C_D, PROTOCOL_COUNTS, EXPECTED_WITHIN_NORMAL, EXPECTED_WITHIN_SOURCE, AccessError,
                                   balanced_accuracy, bootstrap_indices, chance_disease, claim_access, macro_f1,
                                   macro_roc_auc, qualifying_cells, within_normal_sources)
from lungai.novelty.inlp import STOP_THRESHOLD, inlp, leace_fit, nullspace_projection, projection_at
from lungai.novelty.loso import EXPECTED_BINARY, EXPECTED_INCLUDED, inclusion
from lungai.novelty.render_novelty import novelty_macros, spec_macros
from lungai.novelty.spectral import (BANDS, EDGES, N, VARIANTS, apply_band, band_masks, high_weight,
                                     nondc_energy_fraction, radial_frequency, random_pair_mask, variant_mask)
from lungai.novelty.spectral_map import CHANCE_SRC, derived, score_variants

COUNTS = PROTOCOL_COUNTS


def _index(col, cap=None):
    rows = [(s, c) for (s, c), n in COUNTS.items() for _ in range(min(n[col], cap) if cap else n[col])]
    return pd.DataFrame(rows, columns=["source", "label"])


# --------------------------------------------------------------------------- spectral masks
def test_band_masks_partition_unity_and_are_conjugate_symmetric():
    m = band_masks()
    total = sum(m[b] for b in BANDS)
    assert np.allclose(total, 1.0, atol=1e-12)
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    for b in BANDS:
        assert m[b].min() >= 0 and m[b].max() <= 1
        assert np.array_equal(m[b], m[b][(-i) % N, (-j) % N])


def test_dc_belongs_to_band_a_and_every_variant_keeps_dc():
    m = band_masks()
    assert m["A"][0, 0] == 1.0 and all(m[b][0, 0] == 0.0 for b in BANDS[1:])
    for v in VARIANTS[1:]:
        assert variant_mask(v)[0, 0] == 1.0
    assert variant_mask("full") is None
    off_dc = np.ones((N, N), bool)
    off_dc[0, 0] = False
    for b in BANDS:  # pass_b + stop_b = 1 away from DC (both keep DC)
        assert np.allclose((variant_mask(f"pass_{b}") + variant_mask(f"stop_{b}"))[off_dc], 1.0)


def test_nyquist_and_corners_are_in_band_f_and_edges_are_half_weight():
    r = radial_frequency()
    assert r[0, N // 2] == pytest.approx(0.5) and r.max() == pytest.approx(np.sqrt(0.5))
    m = band_masks()
    assert m["F"][0, N // 2] == 1.0 and m["F"][N // 2, N // 2] == 1.0
    for e in EDGES:
        assert high_weight(np.array([e]), e)[0] == pytest.approx(0.5)
        assert high_weight(np.array([e * 2 ** -0.25 * 0.999]), e)[0] == 0.0
        assert high_weight(np.array([e * 2 ** 0.25 * 1.001]), e)[0] == 1.0
    assert high_weight(np.array([0.0]), EDGES[0])[0] == 0.0
    with pytest.raises(ValueError):
        variant_mask("pass_G")


def test_energy_fraction_of_a_pure_tone():
    yy, xx = np.mgrid[0:N, 0:N]
    tone = np.clip(128 + 100 * np.cos(2 * np.pi * 0.3 * xx), 0, 255).astype(np.uint8)  # r = 0.3: band F
    m = band_masks()
    assert nondc_energy_fraction(tone, m["F"]) > 0.99
    assert nondc_energy_fraction(tone, m["A"]) < 0.01
    assert sum(nondc_energy_fraction(tone, m[b]) for b in BANDS) == pytest.approx(1.0)
    assert nondc_energy_fraction(np.full((N, N, 3), 77, np.uint8), m["C"]) == 0.0


def test_random_pair_mask_is_symmetric_binary_and_removes_about_q():
    a, b = random_pair_mask(0.3, 1), random_pair_mask(0.3, 1)
    assert np.array_equal(a, b) and not np.array_equal(a, random_pair_mask(0.3, 2))
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    assert np.array_equal(a, a[(-i) % N, (-j) % N])
    assert set(np.unique(a)) <= {0.0, 1.0} and a[0, 0] == 1.0
    assert abs((1 - a).mean() - 0.3) < 0.01


# --------------------------------------------------------------------------- band application
def test_apply_band_keeps_dtype_shape_mean_and_clips():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (N, N, 3), dtype=np.uint8)
    assert apply_band(img, None) is img
    assert np.array_equal(apply_band(img, np.ones((N, N))), img)
    const = np.full((N, N, 3), 200, np.uint8)
    for v in VARIANTS[1:]:
        assert np.array_equal(apply_band(const, variant_mask(v)), const)  # DC kept, nothing else
    out = apply_band(img, variant_mask("pass_F"))
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert abs(out.astype(float).mean() - img.astype(float).mean()) < 3.0
    checker = ((np.indices((N, N)).sum(0) % 2) * 255).astype(np.uint8)
    hp = apply_band(checker, variant_mask("stop_A"))
    assert hp.min() == 0 and hp.max() == 255  # values outside [0, 255] are clipped, not wrapped
    # channels are filtered independently: same as filtering each 2-D channel
    m = variant_mask("pass_C")
    assert np.array_equal(apply_band(img, m)[..., 1], apply_band(np.ascontiguousarray(img[..., 1]), m))
    with pytest.raises(ValueError):
        apply_band(img.astype(np.float32), m)


def test_low_frequency_image_survives_band_a_and_vanishes_in_band_f():
    yy, xx = np.mgrid[0:N, 0:N]
    smooth = (128 + 60 * np.cos(2 * np.pi * xx / N)).astype(np.uint8)  # r = 1/224: band A
    assert np.abs(apply_band(smooth, variant_mask("pass_A")).astype(int) - smooth).max() <= 1
    assert np.abs(apply_band(smooth, variant_mask("pass_F")).astype(int) - int(round(smooth.mean()))).max() <= 1


# --------------------------------------------------------------------------- scores, minimum cell rule, SDS
def test_fast_scores_match_sklearn():
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 6, 400), rng.integers(0, 6, 400)
    assert balanced_accuracy(y, p, 6) == pytest.approx(balanced_accuracy_score(y, p))
    assert macro_f1(y, p, [1, 3], 6) == pytest.approx(f1_score(y, p, labels=[1, 3], average="macro", zero_division=0))
    probs = rng.dirichlet(np.ones(6), 400)
    assert macro_roc_auc(y, probs) == pytest.approx(roc_auc_score(np.eye(6)[y], probs, average="macro", multi_class="ovr"))


def test_minimum_cell_rule_reproduces_section_3_3():
    tr, te = _index(0, 150), _index(2, 100)
    cells = qualifying_cells(te["source"], te["label"], tr["source"], tr["label"])
    assert cells == EXPECTED_WITHIN_SOURCE
    assert within_normal_sources(te["source"], te["label"], tr["source"], tr["label"]) == EXPECTED_WITHIN_NORMAL
    assert chance_disease(cells) == pytest.approx(EXPECTED_C_D)
    assert len(tr) == 1952 and len(te) == 1033 and len(_index(1, 100)) == 1019 and len(_index(0, 400)) == 4358


def test_sds_arithmetic():
    pm = {"full": {"BA_src": 1.0, "F1_ws": 0.45, "F1_all": 0.5, "BA_norm": 1.0},
          "stop_A": {"BA_src": CHANCE_SRC, "F1_ws": 1.0, "F1_all": 0.5, "BA_norm": 0.25},
          "pass_A": {"BA_src": 0.5, "F1_ws": 0.725, "F1_all": 0.5, "BA_norm": 0.5}}
    d = derived(pm, 0.45)
    assert d["variants"]["full"] == pytest.approx({"S_star": 1.0, "D_star": 0.0, "SDS": 1.0})
    assert d["variants"]["stop_A"] == pytest.approx({"S_star": 0.0, "D_star": 1.0, "SDS": -1.0})
    assert d["variants"]["pass_A"]["S_star"] == pytest.approx(0.4) and d["variants"]["pass_A"]["D_star"] == pytest.approx(0.5)
    assert d["band_stop"]["A"] == pytest.approx({"dS": 1 - CHANCE_SRC, "dD": -0.55})
    assert d["band_pass_retention"]["A"]["S"] == pytest.approx(0.4)
    assert d["band_pass_retention"]["A"]["D"] == "not defined"  # D*(full) = 0 <= 0.05


def test_score_variants_perfect_predictions_and_paired_bootstrap():
    rng = np.random.default_rng(2)
    src = np.repeat(np.arange(6), 40)
    y = rng.integers(0, 2, len(src)) * 5  # classes 0 and 5 in every source
    cells = {s: [0, 5] for s in range(6)}
    noisy = np.where(rng.random(len(src)) < 0.5, src, (src + 1) % 6)
    res = score_variants({"full": (src, y), "stop_B": (noisy, y)}, src, y, cells, [0, 1], 0, 0.5,
                         bootstrap_indices(len(src), b=50))
    f = res["variants"]["full"]
    assert f["BA_src"]["estimate"] == 1.0 and f["F1_ws"]["estimate"] == 1.0 and f["S_star"]["estimate"] == 1.0
    assert f["BA_src"]["ci95"]["lower"] == 1.0
    ds = res["band_stop"]["B"]["dS"]
    assert ds["estimate"] > 0.3 and ds["ci95"]["lower"] > 0 and res["band_stop"]["B"]["dD"]["estimate"] == 0.0
    assert np.array_equal(bootstrap_indices(10, b=3), bootstrap_indices(10, b=3))


# --------------------------------------------------------------------------- INLP and LEACE
def _gaussian(n, rng, d=20):
    s = rng.integers(0, 6, n)
    y = rng.integers(0, 2, n)
    means = np.zeros((6, d))
    means[np.arange(6), np.arange(6)] = 3.0  # each source shifts its own direction
    x = rng.normal(size=(n, d)) + means[s]
    x[:, 10] += 2.0 * y  # disease direction orthogonal to the source directions
    return x, s, y


def test_inlp_brings_source_to_chance_with_orthogonal_idempotent_projection():
    rng = np.random.default_rng(3)
    xtr, str_, ytr = _gaussian(1200, rng)
    xva, sva, yva = _gaussian(600, rng)
    res = inlp(xtr, str_, xva, sva)
    assert res.val_source_ba[0] > 0.6
    assert res.stopped_by == "threshold" and res.val_source_ba[-1] <= STOP_THRESHOLD and 0 < res.k_star <= 20
    assert res.removed_dims == sorted(res.removed_dims)
    for k, P in enumerate(res.projections):
        assert np.allclose(P @ P, P, atol=1e-10) and np.allclose(P, P.T, atol=1e-12)
        assert np.allclose(P, projection_at(res.basis, res.removed_dims, k), atol=1e-10)
    P = res.projections[-1]
    assert np.allclose(res.basis @ P, 0, atol=1e-10)
    # the disease direction (orthogonal to the source shifts) survives
    from sklearn.linear_model import LogisticRegression
    acc = LogisticRegression(max_iter=1000).fit(xtr @ P, ytr).score(xva @ P, yva)
    assert acc > 0.75


def test_inlp_respects_iteration_cap():
    rng = np.random.default_rng(4)
    xtr, str_, _ = _gaussian(600, rng)
    xva, sva, _ = _gaussian(300, rng)
    res = inlp(xtr, str_, xva, sva, cap=1, threshold=0.0)
    assert res.k_star == 1 and res.stopped_by == "iteration cap" and len(res.projections) == 2


def test_nullspace_projection_empty_basis_is_identity():
    assert np.array_equal(nullspace_projection(np.zeros((0, 4)), 4), np.eye(4))


def test_leace_removes_linear_source_information():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(5)
    x, s, y = _gaussian(1500, rng)
    er = leace_fit(x, s)
    xe = er.transform(x)
    z = np.eye(6)[s]
    cross = (xe - xe.mean(0)).T @ (z - z.mean(0)) / len(x)
    assert np.abs(cross).max() < 1e-8 and er.rank == 5
    assert np.allclose(er.transform(xe), xe, atol=1e-8)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(xe, s)
    assert balanced_accuracy_score(s, clf.predict(xe)) < 1 / 6 + 0.08
    assert LogisticRegression(max_iter=1000).fit(xe, y).score(xe, y) > 0.75


# --------------------------------------------------------------------------- LOSO inclusion
def test_loso_inclusion_rule_matches_section_8():
    inc = inclusion(_index(0), _index(2))
    assert {h: v["evaluated_classes"] for h, v in inc.items() if v["included"]} == EXPECTED_INCLUDED
    assert sorted(h for h, v in inc.items() if v["binary_view"]) == EXPECTED_BINARY
    assert not inc["covid_radiography"]["included"]
    assert set(inc["cohen_covid"]["excluded_classes"]) == {"Normal", "Tuberculosis"}
    assert set(inc["nih_cxr14"]["excluded_classes"]) == {"Pleural Effusion", "Pulmonary Nodule / Mass"}


# --------------------------------------------------------------------------- go / no-go
def _band_stop(values):
    return {b: {"dS": {"estimate": ds, "ci95": {"lower": lo, "upper": ds + 0.02}},
                "dD": {"estimate": dd, "ci95": {"lower": dd - 0.01, "upper": hi}}}
            for b, (ds, lo, dd, hi) in values.items()}


def test_go_nogo_no_band_satisfies_criterion_one():
    bs = _band_stop({b: (0.03, 0.01, 0.0, 0.01) for b in BANDS} | {"D": (0.04, 0.02, 0.0, 0.01)})
    d = c2_lite.go_nogo(bs, None, None)
    assert d["decision"] == "NO-GO" and d["b_star"] == "D" and not d["condition_1"]["satisfied"]
    assert d["condition_2"]["evaluated"] is False
    assert d["condition_4"]["status"] == "requires student approval" and d["condition_4"]["satisfied"] is None


def test_go_nogo_all_three_conditions_still_need_student_approval():
    bs = _band_stop({b: (0.0, -0.01, 0.0, 0.01) for b in BANDS}
                    | {"C": (0.08, 0.04, 0.01, 0.025), "E": (0.12, 0.06, 0.03, 0.04), "F": (0.10, 0.05, 0.0, 0.02)})
    lite = {"status": "complete", "dS_lite": {"estimate": 0.06}, "dD_lite": {"estimate": 0.01}}
    ctl = {"status": "complete", "mean": 0.01, "min": 0.0, "max": 0.02}
    d = c2_lite.go_nogo(bs, ctl, lite)
    assert d["condition_1"]["bands_satisfying"] == ["C", "F"] and d["b_star"] == "F"  # E fails dD
    assert d["conditions_1_to_3_hold"] and d["decision"].startswith("GO pending condition 4")
    assert d["decision"] != "GO"
    bad_ctl = dict(ctl, mean=0.06)  # control mean not below the CI lower bound of dS(b*) = 0.05
    assert c2_lite.go_nogo(bs, bad_ctl, lite)["decision"] == "NO-GO"
    bad_lite = dict(lite, dD_lite={"estimate": 0.03})
    assert c2_lite.go_nogo(bs, ctl, bad_lite)["decision"] == "NO-GO"
    # criterion 1 needs the dS lower bound above 0 and the dD upper bound below 0.03
    assert not c2_lite.criterion_1(_band_stop({"A": (0.06, 0.0, 0.0, 0.01)}))["satisfied"]
    assert not c2_lite.criterion_1(_band_stop({"A": (0.06, 0.01, 0.0, 0.03)}))["satisfied"]


def test_go_nogo_on_synthetic_json(tmp_path):
    bs = _band_stop({b: (0.0, -0.01, 0.0, 0.01) for b in BANDS})
    p = tmp_path / "c1.json"
    p.write_text(json.dumps({"stages": {"c1-main": {"val": {"status": "complete", "band_stop": bs}}}}), encoding="utf-8")
    c1 = json.loads(p.read_text(encoding="utf-8"))["stages"]["c1-main"]["val"]
    assert c2_lite.go_nogo(c1["band_stop"], None, None)["decision"] == "NO-GO"


# --------------------------------------------------------------------------- access log
def test_access_marker_refuses_second_run_unless_forced(tmp_path):
    out = tmp_path / "x.json"
    rec = claim_access("c1-main_test", out, access_dir=tmp_path / "access", log_path=tmp_path / "log.jsonl")
    assert rec["force_rerun"] is False and (tmp_path / "access" / "c1-main_test.json").is_file()
    with pytest.raises(AccessError):
        claim_access("c1-main_test", out, access_dir=tmp_path / "access", log_path=tmp_path / "log.jsonl")
    forced = claim_access("c1-main_test", out, True, access_dir=tmp_path / "access", log_path=tmp_path / "log.jsonl")
    assert forced["force_rerun"] is True and forced["previous_access"]["accessed_at"] == rec["accessed_at"]
    assert len((tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_spectral_map_refuses_second_test_run(tmp_path, monkeypatch):
    from lungai.novelty import common, spectral_map
    monkeypatch.setattr(common, "ACCESS_DIR", tmp_path / "access")
    monkeypatch.setattr(common, "ACCESS_LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr(spectral_map, "C1_JSON", tmp_path / "c1.json")
    monkeypatch.setattr(spectral_map, "run_stage", lambda stage, split: {"status": "complete"})
    import lungai.novelty.features as feats
    monkeypatch.setattr(feats, "feature_path", lambda *a, **k: tmp_path / "exists.npy")
    (tmp_path / "exists.npy").write_bytes(b"")
    assert spectral_map.main(["--stage", "c1-served", "--split", "test"]) == 0
    with pytest.raises(AccessError):
        spectral_map.main(["--stage", "c1-served", "--split", "test"])
    assert spectral_map.main(["--stage", "c1-served", "--split", "test", "--force-rerun"]) == 0
    assert spectral_map.main(["--stage", "c1-main", "--split", "test"]) == 2  # validation first


# --------------------------------------------------------------------------- macros
def test_novelty_macros_empty_without_json_and_letters_only(tmp_path):
    assert novelty_macros(tmp_path) == {}
    block = lambda e: {"estimate": e, "ci95": {"lower": e - 0.01, "upper": e + 0.01}}  # noqa: E731
    r = {"status": "complete", "n_scored": 1019, "n_train": 1952, "c_D": 0.45, "within_source_cells": EXPECTED_WITHIN_SOURCE,
         "variants": {v: {m: block(0.5) for m in ("BA_src", "F1_ws", "F1_all", "BA_norm", "S_star", "D_star", "SDS")}
                      for v in VARIANTS},
         "band_stop": {b: {"dS": block(0.1), "dD": block(0.0)} for b in BANDS},
         "band_pass_retention": {b: {"S": 0.5, "D": "not defined"} for b in BANDS}}
    (tmp_path / "c1_spectral_map.json").write_text(json.dumps({"stages": {"c1-main": {"val": r}}}), encoding="utf-8")
    out = novelty_macros(tmp_path)
    assert out["SpecMainValFullSrcBA"] == "50.0" and out["SpecMainValStopCDeltaSLo"] == "9.0"
    assert out["SpecMainValPassARetD"] == "not defined" and out == spec_macros({"stages": {"c1-main": {"val": r}}})
    assert all(k.isalpha() for k in out)
