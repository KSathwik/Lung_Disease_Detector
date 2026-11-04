"""Linear source erasure on frozen features: INLP (protocol section 7) and closed-form LEACE (Belrose et al. 2023).

    from lungai.novelty.inlp import inlp, leace_fit
    res = inlp(x_train_std, src_train, x_val_std, src_val)      # res.projections[k] is P_k, k = 0..k*
    eraser = leace_fit(x_train_std, src_train); x_erased = eraser.transform(x)

INLP: at iteration k fit LogisticRegression(C=1, max_iter=500, class_weight=balanced, random_state=0) to the six
sources on the current train features; the row space of its coefficient matrix (SVD, singular values above 1e-3 x the
largest) is added to the removed subspace and all features are projected onto the orthogonal complement. Stop at the
first k* whose freshly fitted source classifier on projected train features has validation source balanced accuracy at
most 1/6 + 0.05, or at k* = 20. No TensorFlow; no randomness beyond the solver's fixed random_state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from lungai.novelty.common import logistic

ITERATION_CAP = 20
STOP_THRESHOLD = 1.0 / 6.0 + 0.05  # 0.2167
SV_REL_TOL = 1e-3
INLP_MAX_ITER = 500


def rowspace(W: np.ndarray, rel_tol: float = SV_REL_TOL) -> np.ndarray:
    """Orthonormal basis (rows) of the row space of W, keeping singular values above rel_tol x the largest."""
    _, s, vt = np.linalg.svd(np.atleast_2d(W), full_matrices=False)
    if s.size == 0 or s[0] == 0:
        return np.zeros((0, W.shape[-1]))
    return vt[s > rel_tol * s[0]]


def nullspace_projection(basis: np.ndarray, d: int) -> np.ndarray:
    """P = I - Q Q^T for an orthonormal basis Q (rows of ``basis``): symmetric and idempotent."""
    if basis.shape[0] == 0:
        return np.eye(d)
    q, _ = np.linalg.qr(basis.T)  # re-orthonormalise
    return np.eye(d) - q @ q.T


def _balanced_accuracy(y_true, y_pred) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


@dataclass
class InlpResult:
    k_star: int
    stopped_by: str  # "threshold" or "iteration cap"
    projections: List[np.ndarray]  # P_0 = I, ..., P_k*
    removed_dims: List[int]  # cumulative rank of the removed subspace per k
    val_source_ba: List[float]  # fresh source classifier on projected train features, scored on validation, per k
    source_classifiers: List[Any] = field(default_factory=list)  # the fresh classifier at each k (for test scoring)
    basis: Optional[np.ndarray] = None  # orthonormal rows; the first removed_dims[k] rows span the subspace removed at k


def projection_at(basis: np.ndarray, removed_dims: List[int], k: int) -> np.ndarray:
    """P_k rebuilt from the final basis (QR keeps the span of leading rows nested, so the prefix is the k-th subspace)."""
    return nullspace_projection(basis[:removed_dims[k]], basis.shape[1])


def inlp(x_train: np.ndarray, s_train: np.ndarray, x_val: np.ndarray, s_val: np.ndarray,
         cap: int = ITERATION_CAP, threshold: float = STOP_THRESHOLD, max_iter: int = INLP_MAX_ITER,
         clf_factory=None) -> InlpResult:
    """Iterative nullspace projection with the protocol's stopping rule.

    The classifier fitted on the iteration-k features is both the "freshly fitted source classifier" whose validation
    balanced accuracy decides stopping at k and the classifier whose row space is removed to form iteration k + 1
    (they are the same fit: same data, same settings, deterministic solver).
    """
    factory = clf_factory or (lambda: logistic(1.0, max_iter))
    d = x_train.shape[1]
    basis = np.zeros((0, d))
    P = np.eye(d)
    projections, dims, val_ba, clfs = [P], [0], [], []
    k = 0
    while True:
        clf = factory().fit(x_train @ P, s_train)
        clfs.append(clf)
        val_ba.append(_balanced_accuracy(s_val, clf.predict(x_val @ P)))
        if val_ba[-1] <= threshold:
            return InlpResult(k, "threshold", projections, dims, val_ba, clfs, basis)
        if k == cap:
            return InlpResult(k, "iteration cap", projections, dims, val_ba, clfs, basis)
        W = np.atleast_2d(clf.coef_) @ P  # the classifier acts on x P, so its effective directions are W P
        new = rowspace(W)
        if basis.shape[0]:
            new = new - (new @ basis.T) @ basis  # keep only directions not yet removed
            new = rowspace(new) if np.linalg.norm(new) > 0 else np.zeros((0, d))
        basis = np.vstack([basis, new]) if new.shape[0] else basis
        if basis.shape[0]:
            q, _ = np.linalg.qr(basis.T)
            basis = q.T
        P = nullspace_projection(basis, d)
        k += 1
        projections.append(P)
        dims.append(int(basis.shape[0]))


@dataclass
class LeaceEraser:
    """x -> x - (x - mean) E^T with E = W^+ P W (W = Sigma_XX^{-1/2}, P = projection onto colspace(W Sigma_XZ))."""
    mean: np.ndarray
    E: np.ndarray
    rank: int

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x - (x - self.mean) @ self.E.T


def leace_fit(x: np.ndarray, labels: np.ndarray, rel_tol: float = 1e-8) -> LeaceEraser:
    """Closed-form LEACE (Belrose et al. 2023, Theorem 1) with one-hot concept labels, fitted on x.

    After erasure the cross-covariance between the features and the one-hot labels is zero on the fitting data, so no
    linear classifier does better than a constant predictor in the least-squares sense.
    """
    classes = np.unique(labels)
    Z = (np.asarray(labels)[:, None] == classes[None, :]).astype(np.float64)
    mu = x.mean(0)
    xc = x - mu
    zc = Z - Z.mean(0)
    n = len(x)
    sxx = xc.T @ xc / n
    sxz = xc.T @ zc / n
    lam, V = np.linalg.eigh(sxx)
    keep = lam > rel_tol * max(lam.max(), 1e-300)
    inv_sqrt = np.zeros_like(lam)
    sqrt = np.zeros_like(lam)
    inv_sqrt[keep] = 1.0 / np.sqrt(lam[keep])
    sqrt[keep] = np.sqrt(lam[keep])
    W = (V * inv_sqrt) @ V.T
    W_pinv = (V * sqrt) @ V.T
    M = W @ sxz
    u, s, _ = np.linalg.svd(M, full_matrices=False)
    r = int((s > rel_tol * max(s.max(), 1e-300)).sum()) if s.size else 0
    U = u[:, :r]
    E = W_pinv @ (U @ U.T) @ W
    return LeaceEraser(mean=mu, E=E, rank=r)


def summary(res: InlpResult) -> Dict[str, Any]:
    return {"k_star": res.k_star, "stopped_by": res.stopped_by, "iteration_cap": ITERATION_CAP,
            "stop_threshold": STOP_THRESHOLD, "removed_dims": res.removed_dims, "val_source_ba": res.val_source_ba}


def project(x: np.ndarray, P: np.ndarray, eraser: Optional[LeaceEraser] = None) -> np.ndarray:
    return eraser.transform(x) if eraser is not None else x @ P
