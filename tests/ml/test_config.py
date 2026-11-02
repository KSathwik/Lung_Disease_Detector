"""The canonical config is the single source of truth for identity and class order."""

import json

import pytest

from lungai import config
from lungai.config import CLASS_IDS, CLASS_NAMES, CLASS_TO_INDEX, REPO_ROOT, ConfigError

CANONICAL_ORDER = ["COVID-19", "Normal", "Pleural Effusion", "Pneumonia", "Pulmonary Nodule / Mass", "Tuberculosis"]


def test_class_order_is_fixed():
    assert CLASS_NAMES == CANONICAL_ORDER
    assert CLASS_TO_INDEX == {n: i for i, n in enumerate(CANONICAL_ORDER)}
    assert len(set(CLASS_IDS)) == 6


@pytest.mark.parametrize("artifact", [
    "experiments/densenet_frequency_v5/class_mapping.json",
    "experiments/densenet_v5/class_mapping.json",
])
def test_class_order_matches_the_recorded_training_mapping(artifact):
    """The checkpoint's output index i means CLASS_NAMES[i]; the training run's mapping must agree."""
    recorded = json.loads((REPO_ROOT / artifact).read_text(encoding="utf-8"))
    class_to_idx = recorded.get("class_to_idx", recorded)  # the baseline run saved the bare mapping
    assert class_to_idx == CLASS_TO_INDEX
    if "classes" in recorded:
        assert recorded["classes"] == CLASS_NAMES


def test_class_order_matches_recorded_metric_files():
    for rel in ("experiments/densenet_frequency_v5/frequency_metrics.json", "experiments/results/densenet_v5_metrics.json"):
        data = json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))
        assert list(data["per_class_metrics"]) == CLASS_NAMES


def test_training_and_inference_share_one_preprocessing_config():
    pre = config.PREPROCESSING
    assert pre["resize"]["size"] == [224, 224]
    assert pre["frequency_filter"] == {"type": "gaussian_lowpass", "sigma": 1.0, "kernel_size_rule": "2*ceil(3*sigma)+1"}
    assert config.MODEL["input_shape"] == [224, 224, 3]
    recorded = json.loads((REPO_ROOT / "experiments/densenet_frequency_v5/frequency_preprocessing_config.json").read_text())
    assert recorded["filter_parameters"] == {"sigma": pre["frequency_filter"]["sigma"]}
    assert recorded["clahe_parameters"] == {"clip_limit": pre["clahe"]["clip_limit"], "tile_grid_size": pre["clahe"]["tile_grid_size"]}


def test_training_config_matches_the_recorded_run():
    recorded = json.loads((REPO_ROOT / "experiments/densenet_v5/training_config.json").read_text())
    t = config.TRAINING
    assert t["seed"] == recorded["random_seed"] and t["batch_size"] == recorded["batch_size"]
    assert [p["epochs"] for p in t["phases"]] == [recorded["phase1"]["epochs"], recorded["phase2"]["epochs"]]
    assert [p["learning_rate"] for p in t["phases"]] == [recorded["phase1"]["learning_rate"], recorded["phase2"]["learning_rate"]]
    assert t["phases"][1]["unfreeze_last_n_layers"] == recorded["phase2"]["fine_tune_layers"]


def test_invalid_config_is_rejected():
    bad = json.loads(config.CONFIG_PATH.read_text(encoding="utf-8"))
    bad["classes"][1]["index"] = 5
    with pytest.raises(ConfigError):
        config._validate(bad)
    bad = json.loads(config.CONFIG_PATH.read_text(encoding="utf-8"))
    bad["classes"] = bad["classes"][:5]
    with pytest.raises(ConfigError):
        config._validate(bad)


def test_model_identity_is_path_free():
    ident = config.model_identity()
    assert ident["model_id"] == "lungai-densenet121-6class"
    assert [c["name"] for c in ident["classes"]] == CLASS_NAMES
    text = json.dumps(ident, ensure_ascii=False)
    assert "\\" not in text and ":/" not in text
