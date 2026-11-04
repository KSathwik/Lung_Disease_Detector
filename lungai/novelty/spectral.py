"""Radial Fourier band masks on the 224 x 224 grid and their application to cached uint8 images (protocol 4.1).

    from lungai.novelty.spectral import variant_mask, apply_band
    out = apply_band(np.load(cache_file), variant_mask("stop_C"))

Grid: fx, fy = fftfreq(224); r = sqrt(fx^2 + fy^2) (corners up to 0.707 belong to band F). Six octave bands A-F with
edges 1/64, 1/32, 1/16, 1/8, 1/4 and a raised-cosine transition of +/-0.25 octave at each edge:
t = clip((log2(r/e) + 0.25) / 0.5, 0, 1), h_e = 0.5 (1 - cos(pi t)); M_A = 1 - h_e1, M_k = h_e(k-1) (1 - h_ek) for B-E,
M_F = h_e5; the six masks sum to 1 everywhere and r = 0 belongs to band A. Variants: ``full``, ``pass_<b>`` (DC plus
band b) and ``stop_<b>`` (everything except band b, DC kept). Each channel is transformed independently, the inverse is
real (the masks are conjugate-symmetric), clipped to [0, 255] and rounded to uint8. No contrast re-scaling.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

N = 224
BANDS: List[str] = ["A", "B", "C", "D", "E", "F"]
EDGES: List[float] = [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4]  # e1..e5 (section 4.1)
HALF_WIDTH_OCTAVES = 0.25
VARIANTS: List[str] = ["full"] + [f"pass_{b}" for b in BANDS] + [f"stop_{b}" for b in BANDS]  # section 4.2, 13 variants
PASS_VARIANTS: List[str] = ["full"] + [f"pass_{b}" for b in BANDS]


def radial_frequency(n: int = N) -> np.ndarray:
    f = np.fft.fftfreq(n)
    return np.sqrt(f[None, :] ** 2 + f[:, None] ** 2)


def high_weight(r: np.ndarray, edge: float) -> np.ndarray:
    """h_e(r): 0 below 0.84 e, 1 above 1.19 e, raised cosine between; 0 at r = 0."""
    with np.errstate(divide="ignore"):
        lg = np.where(r > 0, np.log2(np.where(r > 0, r, 1.0) / edge), -np.inf)
    t = np.clip((lg + HALF_WIDTH_OCTAVES) / (2 * HALF_WIDTH_OCTAVES), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * t))


@lru_cache(maxsize=4)
def band_masks(n: int = N) -> Dict[str, np.ndarray]:
    """The six band masks M_A..M_F (float64, n x n, fftfreq order). They partition unity."""
    r = radial_frequency(n)
    h = [high_weight(r, e) for e in EDGES]
    masks = {"A": 1.0 - h[0]}
    for k in range(1, 5):
        masks[BANDS[k]] = h[k - 1] * (1.0 - h[k])
    masks["F"] = h[4]
    for m in masks.values():
        m.setflags(write=False)
    return masks


def variant_mask(variant: str, n: int = N) -> Optional[np.ndarray]:
    """Mask of a variant (None for ``full``). The DC coefficient is always kept (weight 1)."""
    if variant == "full":
        return None
    kind, _, band = variant.partition("_")
    if band not in BANDS or kind not in ("pass", "stop"):
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    m = band_masks(n)[band].copy()
    if kind == "stop":
        m = 1.0 - m
    m[0, 0] = 1.0  # DC rule (section 4.1)
    return m


def apply_band(img_uint8: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Filter an (H, W) or (H, W, C) uint8 image with a frequency mask; returns uint8 of the same shape.

    Per channel: FFT, multiply by the mask, inverse FFT, real part, clip to [0, 255], round to nearest (uint8).
    ``mask=None`` returns the image unchanged.
    """
    if mask is None:
        return img_uint8
    a = np.asarray(img_uint8)
    if a.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got {a.dtype}")
    two_d = a.ndim == 2
    x = a[..., None] if two_d else a
    if x.shape[:2] != mask.shape:
        raise ValueError(f"image {x.shape[:2]} and mask {mask.shape} differ")
    F = np.fft.fft2(x.astype(np.float64), axes=(0, 1))
    y = np.fft.ifft2(F * mask[:, :, None], axes=(0, 1)).real
    out = np.rint(np.clip(y, 0.0, 255.0)).astype(np.uint8)
    return out[..., 0] if two_d else out


def nondc_energy_fraction(img_uint8: np.ndarray, band_mask: np.ndarray) -> float:
    """Fraction of the non-DC spectral energy sum |F|^2 that ``band_mask`` weights (all channels pooled) (section 9).

    Returns 0.0 for a constant image (no non-DC energy).
    """
    a = np.asarray(img_uint8, dtype=np.float64)
    if a.ndim == 2:
        a = a[..., None]
    P = np.abs(np.fft.fft2(a, axes=(0, 1))) ** 2
    w = np.ones(band_mask.shape)
    w[0, 0] = 0.0  # exclude DC
    total = float((P * w[:, :, None]).sum())
    if total == 0.0:
        return 0.0
    return float((P * (w * band_mask)[:, :, None]).sum() / total)


def random_pair_mask(q: float, seed: int, n: int = N) -> np.ndarray:
    """Matched-energy random band-stop control (section 9): each conjugate-symmetric frequency pair (f, -f) is removed
    independently with probability q (0/1 mask, the same draw for both members of a pair); DC is kept."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    rng = np.random.default_rng(seed)
    u = rng.random((n, n))
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    lin = i * n + j
    partner = ((-i) % n) * n + ((-j) % n)
    rep = np.minimum(lin, partner)  # one draw per pair, taken at the pair's smaller linear index
    u_pair = u.ravel()[rep]
    m = (u_pair >= q).astype(np.float64)
    m[0, 0] = 1.0
    return m
