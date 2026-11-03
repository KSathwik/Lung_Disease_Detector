"""Train the canonical DenseNet-121 on a dataset V6 build (cached, preprocessed images). Resumable and time-boxed.

    python -m lungai.training.train_v6 --build data/derived/v6_build_001 --run-name v6_seed42 [--max-hours 6]

Re-running the same command after an interruption resumes from the last completed epoch
(runs/<name>/last.weights.h5 + state.json; optimiser moments restart, which is recorded).

Uses train for gradients and val for checkpoint selection (best validation macro-F1, tie: macro ROC-AUC).
Never reads test or external rows. Writes runs/<name>/{model.h5,last.weights.h5,state.json,val_predictions.csv,run_metadata.json}.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from lungai import __version__
from lungai.config import CLASS_NAMES, CLASS_TO_INDEX, CONFIG, MODEL_ID, NUM_CLASSES, REPO_ROOT, TRAINING, sha256_file
from lungai.data.v6 import split_fingerprint
from lungai.preprocessing import augment, normalize
from lungai.training.train import class_weights, git_commit, set_seeds

logger = logging.getLogger("lungai.training.v6")
RUNS_DIR = REPO_ROOT / "runs"


def load_split(build: Path, split: str) -> pd.DataFrame:
    m = pd.read_csv(build / "manifest_v6.csv", dtype=str, keep_default_na=False)
    df = m[m["split"] == split].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"split '{split}' is empty in {build}")
    df["cache"] = [build / "cache" / f"{h}.npy" for h in df["raw_sha256"]]
    missing = [str(p) for p in df["cache"] if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} cached images missing for split {split} (first {missing[0]})")
    df["y"] = df["lungai_label"].map(CLASS_TO_INDEX).astype(int)
    return df


def cached_batch(paths: List[Path], rng: Optional[np.random.Generator], pool: ThreadPoolExecutor) -> np.ndarray:
    arrs = list(pool.map(lambda p: normalize(np.load(p)), paths))
    if rng is not None:
        arrs = [augment(a, rng) for a in arrs]
    return np.stack(arrs).astype(np.float32)


def predict_cached(model, paths: List[Path], pool: ThreadPoolExecutor, batch: int = 32) -> np.ndarray:
    # Eager model calls on CPU took minutes per batch and exhausted RAM (both 2026-11-03 runs stalled here),
    # so inference runs as one compiled graph; the last batch is zero-padded to keep a single input shape.
    import tensorflow as tf
    infer = tf.function(lambda x: model(x, training=False))
    out = []
    for i in range(0, len(paths), batch):
        x = cached_batch(paths[i:i + batch], None, pool)
        n = len(x)
        if n < batch:
            x = np.concatenate([x, np.zeros((batch - n, *x.shape[1:]), x.dtype)])
        out.append(infer(tf.constant(x)).numpy()[:n])
    return np.concatenate(out).astype(np.float64)


def train(build: Path, run_dir: Path, max_hours: float, phase2_epochs: int, patience: int, seed: int) -> Dict[str, Any]:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import f1_score, roc_auc_score
    from lungai.models.densenet import build_model

    set_seeds(seed)
    tr, va = load_split(build, "train"), load_split(build, "val")
    weights = class_weights(tr["lungai_label"])
    batch = TRAINING["batch_size"]
    steps = int(np.ceil(len(tr) / batch))
    paths, labels = tr["cache"].tolist(), tr["y"].to_numpy(np.int32)
    sample_w = np.array([weights[int(y)] for y in labels], np.float32)
    phases = [dict(TRAINING["phases"][0]), dict(TRAINING["phases"][1], epochs=phase2_epochs)]

    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "completed_epochs": 0, "history": [], "best": None, "resumed_from": [], "bad_epochs": 0}
    model, base = build_model(weights="imagenet")
    if state["completed_epochs"]:
        model.load_weights(run_dir / "last.weights.h5")
        state["resumed_from"].append({"epoch": state["completed_epochs"], "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                      "note": "weights restored; optimizer moments re-initialised"})
        logger.info("resuming after epoch %d", state["completed_epochs"])

    deadline = time.time() + max_hours * 3600
    pool = ThreadPoolExecutor(max_workers=6)
    epoch_no = 0
    stop_reason = "completed all epochs"
    for phase in phases:
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

        for _ in range(phase["epochs"]):
            epoch_no += 1
            if epoch_no <= state["completed_epochs"]:
                continue
            if time.time() > deadline:
                stop_reason = f"time budget of {max_hours} h reached"
                break
            if phase["backbone_trainable"] and state["bad_epochs"] >= patience:
                stop_reason = f"early stopping: no validation macro-F1 gain for {patience} epochs"
                break
            rng = np.random.default_rng(seed + epoch_no)  # per-epoch stream keeps resumed runs reproducible
            order = rng.permutation(len(paths))
            t0, losses = time.time(), []
            for s in range(steps):
                idx = order[s * batch:(s + 1) * batch]
                x = cached_batch([paths[i] for i in idx], rng, pool)
                losses.append(float(step(x, labels[idx], sample_w[idx])))
                if (s + 1) % 25 == 0:
                    logger.info("%s epoch %d step %d/%d loss %.4f (%.1fs)", phase["name"], epoch_no, s + 1, steps,
                                np.mean(losses), time.time() - t0)
            vp = predict_cached(model, va["cache"].tolist(), pool, batch)
            yv = va["y"].to_numpy()
            pred = vp.argmax(1)
            rec = {"epoch": epoch_no, "phase": phase["name"], "train_loss": float(np.mean(losses)),
                   "val_accuracy": float((pred == yv).mean()),
                   "val_macro_f1": float(f1_score(yv, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)),
                   "val_macro_roc_auc": float(roc_auc_score(np.eye(NUM_CLASSES)[yv], vp, average="macro", multi_class="ovr")),
                   "seconds": round(time.time() - t0, 1)}
            state["history"].append(rec)
            logger.info("epoch %d: %s", epoch_no, rec)
            best = state["best"]
            if best is None or (rec["val_macro_f1"], rec["val_macro_roc_auc"]) > (best["val_macro_f1"], best["val_macro_roc_auc"]):
                state["best"], state["bad_epochs"] = dict(rec), 0
                model.save(run_dir / "model.h5")
                pd.DataFrame({"image_id": va["image_id"], "true_label": va["lungai_label"],
                              **{f"prob_{i}": vp[:, i] for i in range(NUM_CLASSES)}, "pred_index": pred}
                             ).to_csv(run_dir / "val_predictions.csv", index=False)
            elif phase["backbone_trainable"]:
                state["bad_epochs"] += 1
            model.save_weights(run_dir / "last.weights.h5")
            state["completed_epochs"] = epoch_no
            state_path.write_text(json.dumps(state, indent=2))
        else:
            continue
        break
    pool.shutdown()
    state["stop_reason"] = stop_reason
    state_path.write_text(json.dumps(state, indent=2))
    return {"history": state["history"], "selected": state["best"], "stop_reason": stop_reason,
            "resumed_from": state["resumed_from"], "class_weights": {CLASS_NAMES[k]: v for k, v in weights.items()},
            "train_images": len(tr), "val_images": len(va), "steps_per_epoch": steps,
            "phase2_max_epochs": phase2_epochs, "early_stopping_patience": patience,
            "checkpoint_sha256": sha256_file(run_dir / "model.h5")}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--phase2-epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=TRAINING["seed"],
                    help="weights init, data order and augmentation (default: config); the split is fixed by the build")
    args = ap.parse_args(argv)
    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", encoding="utf-8")])
    report = json.loads((args.build / "v6_build_report.json").read_text(encoding="utf-8"))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = train(args.build, run_dir, args.max_hours, args.phase2_epochs, args.patience, args.seed)
    import tensorflow as tf
    manifest = pd.read_csv(args.build / "manifest_v6.csv", dtype=str, keep_default_na=False)
    dev = manifest[manifest["split"].isin(["train", "val", "test"])]
    meta = {
        "model_id": MODEL_ID, "run_name": args.run_name, "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
        "lungai_version": __version__, "dataset_version": "V6", "build": args.build.name,
        "manifest_sha256": sha256_file(args.build / "manifest_v6.csv"), "split_fingerprint": split_fingerprint(dev),
        "split_fingerprint_matches_build_report": split_fingerprint(dev) == report["split_fingerprint"],
        "seed": args.seed,
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
                        "note": "CPU-only (TensorFlow 2.16 has no native-Windows GPU support)"},
        "config": CONFIG, **result,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("done: %s; selected epoch %s; checkpoint sha256 %s", result["stop_reason"],
                (result["selected"] or {}).get("epoch"), result["checkpoint_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
