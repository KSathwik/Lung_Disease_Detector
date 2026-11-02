"""Batch loading for training and evaluation (canonical preprocessing, training-only augmentation)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from lungai.preprocessing import augment, preprocess_file

_POOL_WORKERS = 6


def load_batch(paths: Sequence[Path], rng: Optional[np.random.Generator] = None,
               pool: Optional[ThreadPoolExecutor] = None) -> np.ndarray:
    """Preprocess images into a float32 batch. Augments only when ``rng`` is given (training).

    Any unreadable image raises; a missing file never becomes a blank tensor.
    """
    own = pool is None
    pool = pool or ThreadPoolExecutor(max_workers=_POOL_WORKERS)
    try:
        arrays: List[np.ndarray] = list(pool.map(preprocess_file, paths))
    finally:
        if own:
            pool.shutdown()
    if rng is not None:
        # augmentation draws are made sequentially so a given seed always yields the same batch
        arrays = [augment(a, rng) for a in arrays]
    return np.stack(arrays, axis=0).astype(np.float32)


def predict_probabilities(model, paths: Sequence[Path], batch_size: int = 32) -> np.ndarray:
    """Softmax outputs (N, num_classes) for images in the given order, with inference-mode layers."""
    out = []
    with ThreadPoolExecutor(max_workers=_POOL_WORKERS) as pool:
        for i in range(0, len(paths), batch_size):
            x = load_batch(paths[i:i + batch_size], pool=pool)
            out.append(np.asarray(model(x, training=False)))
    return np.concatenate(out, axis=0).astype(np.float64) if out else np.zeros((0, 0))
