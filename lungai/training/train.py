"""Canonical training entry point for the LungAI DenseNet-121 classifier.

    python -m lungai.training.train --dry-run          # check config, manifest and every train/val image; no training
    python -m lungai.training.train --run-name my_run  # train (two phases), select on validation, save checkpoint

Uses only the train split for gradient updates and the val split for checkpoint selection.
The test split and the external cohort are never loaded here.

Outputs (new directory, never overwritten): ``runs/<run-name>/``
    model.h5                 selected checkpoint (best validation macro-F1)
    run_metadata.json        config snapshot, versions, manifest hash, history, selection, checkpoint SHA-256
    val_predictions.csv      per-image validation predictions of the selected checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from lungai import __version__
from lungai.config import (
    CLASS_NAMES, CLASS_TO_INDEX, CONFIG, DATASET, MODEL_ID, NUM_CLASSES, REPO_ROOT, TRAINING, data_root,
    manifest_path, sha256_file,
)
from lungai.data.manifest import load_manifest, resolve_image_path, split_frame

logger = logging.getLogger("lungai.training")
RUNS_DIR = REPO_ROOT / "runs"


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def class_weights(train_labels: pd.Series) -> Dict[int, float]:
    """Balanced weights n / (k * n_c), keyed by canonical class index. Fails if a class has no training images."""
    counts = train_labels.value_counts()
    absent = [c for c in CLASS_NAMES if counts.get(c, 0) == 0]
    if absent:
        raise ValueError(f"classes with no training images: {absent}")
    n = len(train_labels)
    return {CLASS_TO_INDEX[c]: n / (NUM_CLASSES * int(counts[c])) for c in CLASS_NAMES}


def resolve_split(split: str, root: Path) -> pd.DataFrame:
    df = split_frame(load_manifest(), split)
    df["path"] = [resolve_image_path(p, root) for p in df["image_path"]]
    df["y"] = df["clinical_label"].map(CLASS_TO_INDEX)
    return df


def check_images(df: pd.DataFrame, split: str) -> None:
    missing = [p for p in df["path"] if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(df)} {split} images are missing (first: {missing[0]}). Set LUNGAI_DATA_ROOT or "
            "--data-root to the folder containing data/raw and data/downloads. Training never skips missing images."
        )


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def train(run_dir: Path, root: Path) -> Dict[str, Any]:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import f1_score, roc_auc_score

    from lungai.data.loader import load_batch, predict_probabilities
    from lungai.models.densenet import build_model

    seed = TRAINING["seed"]
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    train_df, val_df = resolve_split("train", root), resolve_split("val", root)
    check_images(train_df, "train")
    check_images(val_df, "val")
    weights = class_weights(train_df["clinical_label"])
    batch, steps = TRAINING["batch_size"], TRAINING["steps_per_epoch"]
    paths = train_df["path"].tolist()
    labels = train_df["y"].to_numpy(dtype=np.int32)
    sample_w = np.array([weights[int(y)] for y in labels], dtype=np.float32)

    model, base = build_model(weights="imagenet")
    order = rng.permutation(len(paths))
    cursor = 0
    history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_path = run_dir / "model.h5"
    pool = ThreadPoolExecutor(max_workers=6)

    def next_batch():
        nonlocal order, cursor
        if cursor + batch > len(order):
            order, cursor = rng.permutation(len(paths)), 0
        idx = order[cursor:cursor + batch]
        cursor += batch
        x = load_batch([paths[i] for i in idx], rng=rng, pool=pool)
        return x, labels[idx], sample_w[idx]

    epoch_no = 0
    for phase in TRAINING["phases"]:
        if phase["backbone_trainable"]:
            base.trainable = True
            for layer in base.layers[:-phase["unfreeze_last_n_layers"]]:
                layer.trainable = False
        else:
            base.trainable = False
        optimizer = keras.optimizers.Adam(learning_rate=phase["learning_rate"])

        @tf.function
        def step(x, y, w):
            with tf.GradientTape() as tape:
                p = model(x, training=True)
                loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y, p) * w)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        for e in range(phase["epochs"]):
            epoch_no += 1
            t0, losses = time.time(), []
            for s in range(steps):
                losses.append(float(step(*next_batch())))
                if (s + 1) % 50 == 0:
                    logger.info("%s epoch %d step %d/%d loss %.4f", phase["name"], epoch_no, s + 1, steps, np.mean(losses))
            val_probs = predict_probabilities(model, val_df["path"].tolist(), batch)
            y_val = val_df["y"].to_numpy()
            val_pred = val_probs.argmax(axis=1)
            rec = {
                "epoch": epoch_no, "phase": phase["name"], "train_loss": float(np.mean(losses)),
                "val_accuracy": float((val_pred == y_val).mean()),
                "val_macro_f1": float(f1_score(y_val, val_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)),
                "val_macro_roc_auc": float(roc_auc_score(np.eye(NUM_CLASSES)[y_val], val_probs, average="macro", multi_class="ovr")),
                "seconds": round(time.time() - t0, 1),
            }
            history.append(rec)
            logger.info("epoch %d: %s", epoch_no, rec)
            key = (rec["val_macro_f1"], rec["val_macro_roc_auc"])
            if best is None or key > (best["val_macro_f1"], best["val_macro_roc_auc"]):
                best = dict(rec)
                model.save(best_path)
                pd.DataFrame({"image_id": val_df["image_id"], "true_label": val_df["clinical_label"],
                              **{f"prob_{i}": val_probs[:, i] for i in range(NUM_CLASSES)},
                              "pred_index": val_pred}).to_csv(run_dir / "val_predictions.csv", index=False)
    pool.shutdown()
    return {"history": history, "selected": best, "class_weights": {CLASS_NAMES[k]: v for k, v in weights.items()},
            "checkpoint_sha256": sha256_file(best_path)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=None, help="output folder name under runs/ (default: model id + UTC time)")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="validate config, manifest and image files, then exit")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = args.data_root.resolve() if args.data_root else data_root()
    train_df, val_df = resolve_split("train", root), resolve_split("val", root)
    weights = class_weights(train_df["clinical_label"])
    logger.info("manifest %s: train=%d val=%d; class weights %s", DATASET["version"], len(train_df), len(val_df),
                {CLASS_NAMES[k]: round(v, 4) for k, v in weights.items()})
    try:
        check_images(train_df, "train")
        check_images(val_df, "val")
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 2
    if args.dry_run:
        logger.info("dry run OK: config valid, manifest checksum matches, all %d train/val images present",
                    len(train_df) + len(val_df))
        return 0

    run_name = args.run_name or f"{MODEL_ID}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite or silently reuse an existing run
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = train(run_dir, root)

    import tensorflow as tf
    meta = {
        "model_id": MODEL_ID, "run_name": run_name, "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "lungai_version": __version__,
        "manifest": DATASET["manifest"], "manifest_sha256_lf": sha256_file(manifest_path(), normalize_newlines=True),
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "gpus": [d.name for d in tf.config.list_physical_devices("GPU")]},
        "nondeterminism": "GPU kernels and threaded data loading can make runs differ slightly even with fixed seeds.",
        "config": CONFIG,
        **result,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("selected epoch %s, checkpoint sha256 %s, outputs in %s", result["selected"]["epoch"],
                result["checkpoint_sha256"], run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
