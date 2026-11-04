"""Extract and cache 1024-d GAP features of the cached 224 x 224 images, optionally band-filtered (protocol 4.3, 7, 9).

    python -m lungai.novelty.features --timing --limit 200                      # step 0: seconds/image, no cache files
    python -m lungai.novelty.features --stage c3c4                              # step 0: whole splits, full, both extractors
    python -m lungai.novelty.features --stage c1-main                           # 13 variants x train150/val100/test100
    python -m lungai.novelty.features --stage c1-served                         # full + 6 pass, train150/test100
    python -m lungai.novelty.features --stage c1-nofilter                       # no-filter build, full + 6 pass
    python -m lungai.novelty.features --stage c2-lite --band C                  # full + stop_C on the cap-400 train sample
    python -m lungai.novelty.features --stage c2-control --band C               # q on val100, 3 random masks (seeds 1-3)

Extractors: ``imagenet`` = keras DenseNet121(include_top=False, weights="imagenet", pooling="avg"); ``served`` = the
``densenet121`` layer of models/densenet121_v6r2_seed42.h5 (SHA-256 checked) + GlobalAveragePooling2D, as in
lungai.evaluation.source_probe. Input: cached uint8 -> (band filter) -> lungai.preprocessing.normalize.
Writes artifacts/novelty[/nofilter]/features_<extractor>_<variant>_<set>.npy (float32) and a .csv index
(image_id, source, label, raw_sha256). Resumable: finished files are skipped, unfinished ones resume from 512-image
chunks. Extracting features is not an access (section 11); no metric is computed here.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lungai.novelty.common import (BUILDS, OUT_DIR, SAMPLE_SETS, SERVED_CHECKPOINT, SERVED_SHA256, ProtocolError,
                                   check_count, load_manifest, now, provenance, read_json, select, write_json)
from lungai.novelty.spectral import BANDS, PASS_VARIANTS, VARIANTS, band_masks, nondc_energy_fraction, \
    random_pair_mask, variant_mask

BATCH = 16
CHUNK = 512
CONTROL_SEEDS = (1, 2, 3)  # section 9 mask seeds
TIMING_JSON = OUT_DIR / "timing.json"

# Section 4.2 table and sections 7, 9: (extractor, build, variants, sets)
STAGES: Dict[str, List[Tuple[str, str, List[str], List[str]]]] = {
    "c3c4": [("imagenet", "main", ["full"], ["train", "val", "test", "external"]),
             ("served", "main", ["full"], ["train", "val", "test", "external"])],
    "c1-main": [("imagenet", "main", VARIANTS, ["train150", "val100", "test100"])],
    "c1-served": [("served", "main", PASS_VARIANTS, ["train150", "test100"])],
    "c1-nofilter": [("imagenet", "nofilter", PASS_VARIANTS, ["train150", "test100"])],
}
# forward passes per stage (projection only; the protocol's own table: 52,052 / 20,895 / 20,895)
STAGE_PASSES = {"c3c4": 2 * 15793, "c1-main": 52052, "c1-served": 20895, "c1-nofilter": 20895, "c2-lite": 8716,
                "c2-control": 3 * (1952 + 1019)}


# PROTOCOL NOTE: section 7 names one file artifacts/novelty/features_<extractor>.npy for all 15,793 images; the cache
# is stored per variant and set instead (features_<extractor>_full_{train,val,test,external}.npy hold the same rows).
def cache_dir(build: str) -> Path:
    return OUT_DIR if build == "main" else OUT_DIR / build


def feature_path(extractor: str, variant: str, set_key: str, build: str = "main") -> Path:
    return cache_dir(build) / f"features_{extractor}_{variant}_{set_key}.npy"


def index_path(extractor: str, variant: str, set_key: str, build: str = "main") -> Path:
    return feature_path(extractor, variant, set_key, build).with_suffix(".csv")


def is_cached(extractor: str, variant: str, set_key: str, build: str = "main") -> bool:
    return feature_path(extractor, variant, set_key, build).is_file() and index_path(extractor, variant, set_key, build).is_file()


def load_features(extractor: str, variant: str, set_key: str, build: str = "main") -> Tuple[np.ndarray, pd.DataFrame]:
    x = np.load(feature_path(extractor, variant, set_key, build))
    idx = pd.read_csv(index_path(extractor, variant, set_key, build), dtype=str, keep_default_na=False)
    if len(idx) != len(x):
        raise ProtocolError(f"feature/index length mismatch for {extractor} {variant} {set_key} {build}")
    return x, idx


def sample_rows(build: str, set_key: str) -> pd.DataFrame:
    """The protocol's rows for a sample set, with the realised count asserted; the no-filter build must sample the same
    raw_sha256 list as the main build (section 3.3)."""
    sset = SAMPLE_SETS[set_key]
    df = select(load_manifest(BUILDS[build]), sset)
    check_count(df, sset)
    if build != "main":
        ref = select(load_manifest(BUILDS["main"]), sset)
        if df["raw_sha256"].tolist() != ref["raw_sha256"].tolist():
            raise ProtocolError(f"{build} build samples a different hash list than main for {set_key}")
    missing = [str(p) for p in df["cache"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} cached images missing for {build}/{set_key} (first {missing[0]})")
    return df


def build_extractor(name: str):
    from tensorflow import keras
    if name == "imagenet":
        return keras.applications.DenseNet121(include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))
    if name == "served":
        from lungai.models.densenet import load_checkpoint
        model, digest, _ = load_checkpoint(SERVED_CHECKPOINT, allow_unverified=True)
        if digest != SERVED_SHA256:
            raise ProtocolError(f"served checkpoint SHA-256 {digest} differs from the protocol's {SERVED_SHA256}")
        base = next(l for l in model.layers if "densenet121" in l.name)
        return keras.Sequential([base, keras.layers.GlobalAveragePooling2D()])
    raise ValueError(f"unknown extractor {name!r}")


def _batch(paths: List[Path], mask: Optional[np.ndarray], pool: ThreadPoolExecutor) -> np.ndarray:
    from lungai.novelty.spectral import apply_band
    from lungai.preprocessing import normalize
    return np.stack(list(pool.map(lambda p: normalize(apply_band(np.load(p), mask)), paths))).astype(np.float32)


def extract(fn, paths: List[Path], mask: Optional[np.ndarray], pool: ThreadPoolExecutor, batch: int = BATCH) -> np.ndarray:
    """Features for paths with a compiled extractor ``fn``; the last batch is zero-padded (as source_probe._features)."""
    import tensorflow as tf
    out = []
    for i in range(0, len(paths), batch):
        x = _batch(paths[i:i + batch], mask, pool)
        n = len(x)
        if n < batch:
            x = np.concatenate([x, np.zeros((batch - n, *x.shape[1:]), x.dtype)])
        out.append(fn(tf.constant(x)).numpy()[:n])
    return np.concatenate(out).astype(np.float32)


def write_cached(fn, rows: pd.DataFrame, mask: Optional[np.ndarray], out: Path, pool: ThreadPoolExecutor) -> None:
    """Resumable extraction into ``out`` (chunks of CHUNK images in <out>.parts/ until complete)."""
    parts = out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in rows["cache"]]
    chunks = []
    for c0 in range(0, len(paths), CHUNK):
        cp = parts / f"chunk_{c0:06d}.npy"
        if not cp.is_file():
            t0 = time.time()
            arr = extract(fn, paths[c0:c0 + CHUNK], mask, pool)
            tmp = parts / f"chunk_{c0:06d}.tmp.npy"
            np.save(tmp, arr)
            tmp.replace(cp)
            print(f"  {out.name}: {min(c0 + CHUNK, len(paths))}/{len(paths)} ({(time.time() - t0) / len(arr):.3f} s/img)", flush=True)
        chunks.append(cp)
    x = np.concatenate([np.load(c) for c in chunks])
    pd.DataFrame({"image_id": rows["image_id"], "source": rows["source_id"], "label": rows["lungai_label"],
                  "raw_sha256": rows["raw_sha256"]}).to_csv(out.with_suffix(".csv"), index=False)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, x)
    tmp.replace(out)
    shutil.rmtree(parts, ignore_errors=True)


def control_q(band: str) -> float:
    """q (section 9): mean over val-sample images of the fraction of non-DC energy that M_band weights (main build)."""
    rows = sample_rows("main", "val100")
    m = band_masks()[band]
    return float(np.mean([nondc_energy_fraction(np.load(p), m) for p in rows["cache"]]))


def control_meta_path(band: str) -> Path:
    return OUT_DIR / f"control_{band}.json"


def run_jobs(jobs: List[Tuple[str, str, str, str, Optional[np.ndarray]]], threads: int) -> None:
    """jobs: (extractor, build, variant, set_key, mask). Skips files that exist."""
    import tensorflow as tf
    todo = [j for j in jobs if not is_cached(j[0], j[2], j[3], j[1])]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} feature files already cached; {len(todo)} to extract", flush=True)
    pool = ThreadPoolExecutor(threads)
    current, fn = None, None
    for extractor, build, variant, set_key, mask in todo:
        if extractor != current:
            model = build_extractor(extractor)
            fn = tf.function(lambda x, m=model: m(x, training=False))
            current = extractor
        rows = sample_rows(build, set_key)
        out = feature_path(extractor, variant, set_key, build)
        print(f"{now()} {extractor} {build} {variant} {set_key}: {len(rows)} images", flush=True)
        write_cached(fn, rows, mask, out, pool)
    pool.shutdown()


def timing(limit: int, extractor: str, variant: str) -> Dict:
    """Time ``limit`` images of the train150 sample (no cache written); projected hours per stage (section 4.2)."""
    import tensorflow as tf
    rows = sample_rows("main", "train150").head(limit)
    model = build_extractor(extractor)
    fn = tf.function(lambda x: model(x, training=False))
    pool = ThreadPoolExecutor(2)
    extract(fn, [Path(p) for p in rows["cache"][:BATCH]], variant_mask(variant), pool)  # graph tracing, not timed
    t0 = time.time()
    extract(fn, [Path(p) for p in rows["cache"]], variant_mask(variant), pool)
    sec = (time.time() - t0) / len(rows)
    pool.shutdown()
    return {"images": int(len(rows)), "extractor": extractor, "variant": variant, "batch": BATCH,
            "seconds_per_image": sec, "projected_hours": {k: round(v * sec / 3600, 2) for k, v in STAGE_PASSES.items()},
            "note": "timing does not change any setting (protocol 4.2)", **provenance()}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=sorted(STAGES) + ["c2-lite", "c2-control"])
    ap.add_argument("--band", choices=BANDS, help="b* for c2-lite / c2-control")
    ap.add_argument("--timing", action="store_true", help="time --limit images and print seconds/image (no cache files)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--extractor", choices=["imagenet", "served"], default="imagenet", help="for --timing")
    ap.add_argument("--variant", default="stop_C", help="for --timing (a filtered variant includes the FFT cost)")
    ap.add_argument("--threads", type=int, default=2, help="image-loading threads")
    ap.add_argument("--tf-threads", type=int, default=0, help="TensorFlow intra-op threads (0 = TensorFlow default)")
    args = ap.parse_args(argv)
    if args.tf_threads:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(args.tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    if args.timing:
        res = timing(args.limit, args.extractor, args.variant)
        print(f"{res['seconds_per_image']:.4f} seconds/image over {res['images']} images; projected hours "
              f"{res['projected_hours']}")
        if args.limit == 200:  # the protocol's timing run is recorded for the C1 JSON
            write_json(TIMING_JSON, res)
            print(f"wrote {TIMING_JSON}")
        return 0
    if not args.stage:
        ap.error("--stage or --timing is required")
    if args.stage in ("c2-lite", "c2-control") and not args.band:
        ap.error(f"--stage {args.stage} needs --band (b* from lungai.novelty.c2_lite --part select)")
    jobs: List[Tuple[str, str, str, str, Optional[np.ndarray]]] = []
    if args.stage in STAGES:
        for extractor, build, variants, sets in STAGES[args.stage]:
            for set_key in sets:
                for v in variants:
                    jobs.append((extractor, build, v, set_key, variant_mask(v)))
    elif args.stage == "c2-lite":
        for v in ("full", f"stop_{args.band}"):
            jobs.append(("imagenet", "main", v, "train400", variant_mask(v)))
    else:
        meta_path = control_meta_path(args.band)
        meta = read_json(meta_path)
        if not meta:
            q = control_q(args.band)
            meta = {"band": args.band, "q": q, "mask_seeds": list(CONTROL_SEEDS), "q_definition":
                    "mean over val100 images of the fraction of non-DC sum |F|^2 weighted by M_band (all channels)",
                    **provenance()}
            write_json(meta_path, meta)
            print(f"q = {q:.6f} for band {args.band}; wrote {meta_path}")
        for seed in CONTROL_SEEDS:
            m = random_pair_mask(meta["q"], seed)
            for set_key in ("train150", "val100"):
                jobs.append(("imagenet", "main", f"ctrl{seed}_{args.band}", set_key, m))
    run_jobs(jobs, args.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
