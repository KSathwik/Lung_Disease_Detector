"""Source-identifiability probe (a control for source shortcuts).

    python -m lungai.evaluation.source_probe --build data/derived/v6_build_002 [--checkpoint runs/<run>/model.h5]

Extracts 1024-d global-average-pooled DenseNet-121 features from the cached images and fits a multinomial
logistic regression that predicts the SOURCE dataset (not the disease) on training images, then scores it on
test images. Two feature extractors:
  - imagenet: DenseNet-121 with ImageNet weights only (nothing learned from this project);
  - model:    the trained checkpoint's backbone (what the classifier actually uses), when --checkpoint is given.
Reported: balanced accuracy over sources (all test images), and within each class that has ≥ 2 sources
(e.g. "among Normal images, can the source be told apart?"). High within-class source accuracy means a
classifier could separate classes by source style rather than disease. Writes
artifacts/data_quality/source_probe_<build>.json.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from lungai.config import REPO_ROOT

OUT_DIR = REPO_ROOT / "artifacts" / "data_quality"


def _sample(df: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    return (df.groupby(["source_id", "lungai_label"], group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_group), random_state=seed)).reset_index(drop=True))


def _features(extractor, paths: List[Path], pool: ThreadPoolExecutor, batch: int = 32) -> np.ndarray:
    import tensorflow as tf
    from lungai.training.train_v6 import cached_batch
    fn = tf.function(lambda x: extractor(x, training=False))
    out = []
    for i in range(0, len(paths), batch):
        x = cached_batch(paths[i:i + batch], None, pool)
        n = len(x)
        if n < batch:
            x = np.concatenate([x, np.zeros((batch - n, *x.shape[1:]), x.dtype)])
        out.append(fn(tf.constant(x)).numpy()[:n])
    return np.concatenate(out)


def probe(train_x, train_s, test_x, test_s, test_y) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(train_x)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(sc.transform(train_x), train_s)
    pred = clf.predict(sc.transform(test_x))
    res = {"all": {"n": int(len(test_s)), "n_sources": int(len(set(test_s))),
                   "balanced_accuracy": float(balanced_accuracy_score(test_s, pred)), "chance": 1.0 / len(set(test_s))}}
    for c in sorted(set(test_y)):
        m = test_y == c
        srcs = set(test_s[m])
        if len(srcs) >= 2:
            res[f"within:{c}"] = {"n": int(m.sum()), "sources": sorted(srcs),
                                  "balanced_accuracy": float(balanced_accuracy_score(test_s[m], pred[m])),
                                  "chance": 1.0 / len(srcs)}
    return res


def main(argv: Optional[List[str]] = None) -> int:
    from tensorflow import keras
    from lungai.models.densenet import load_checkpoint
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--per-group-train", type=int, default=150, help="max train images per (source, class)")
    ap.add_argument("--per-group-test", type=int, default=100, help="max test images per (source, class)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    m = pd.read_csv(args.build / "manifest_v6.csv", dtype=str, keep_default_na=False)
    m["cache"] = [args.build / "cache" / f"{h}.npy" for h in m.raw_sha256]
    tr = _sample(m[m.split == "train"], args.per_group_train, args.seed)
    te = _sample(m[m.split == "test"], args.per_group_test, args.seed)
    extractors = {"imagenet": keras.applications.DenseNet121(include_top=False, weights="imagenet", pooling="avg",
                                                             input_shape=(224, 224, 3))}
    if args.checkpoint:
        model, _, _ = load_checkpoint(args.checkpoint, allow_unverified=True)
        base = next(l for l in model.layers if "densenet121" in l.name)
        extractors["model"] = keras.Sequential([base, keras.layers.GlobalAveragePooling2D()])
    pool = ThreadPoolExecutor(4)
    result = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "build": args.build.name,
              "checkpoint": str(args.checkpoint) if args.checkpoint else None, "n_train": len(tr), "n_test": len(te),
              "train_by_source": tr.source_id.value_counts().to_dict(), "test_by_source": te.source_id.value_counts().to_dict()}
    for name, ex in extractors.items():
        ftr = _features(ex, tr.cache.tolist(), pool)
        fte = _features(ex, te.cache.tolist(), pool)
        result[name] = probe(ftr, tr.source_id.to_numpy(), fte, te.source_id.to_numpy(), te.lungai_label.to_numpy())
    pool.shutdown()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"source_probe_{args.build.name}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k in ("imagenet", "model")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
