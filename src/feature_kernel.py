
"""
Deterministic finite-time kernel for the featureful Linear GAE

    Z = S F W

under the Taylor-BCE surrogate

    L(W) = || M - S F W W^T F^T S ||_F^2 / n^2,
    M = 4Y - 2J,   Y = A_train + I.

Linearized SGD:
    W_{k+1} = (I + alpha A_F) W_k,
    A_F = F^T S M S F.

For isotropic initialization:
    K_k^(F)
      = S F (I + alpha A_F)^(2k) F^T S.

For a truncated eigensystem A_F ~= V Lambda V^T, evaluate

    K_k ~= H H^T
           + H V [ (I + alpha Lambda)^(2k) - I ] V^T H^T,

where H = S F.

The implementation never forms an n x n kernel or dense M.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from linear_gae_spectral import EdgeSplit, normalize_adjacency


def prepare_features(
    features: sp.spmatrix,
    mode: str = "raw",
) -> sp.csr_matrix:
    """
    Feature preprocessing.

    mode="raw"
        Use the loaded Planetoid features unchanged. This matches the
        feature-enabled Salha Linear GAE convention.

    mode="row_normalized"
        Row-normalize F so each nonzero row sums to one. Useful as an
        optional scale-controlled ablation.
    """
    f = features.astype(np.float64).tocsr().copy()

    if mode == "raw":
        return f

    if mode == "row_normalized":
        row_sum = np.asarray(f.sum(axis=1)).ravel()
        inv = np.zeros_like(row_sum, dtype=np.float64)
        nz = row_sum != 0
        inv[nz] = 1.0 / row_sum[nz]
        return (sp.diags(inv) @ f).tocsr()

    raise ValueError("feature mode must be 'raw' or 'row_normalized'")


def build_feature_encoder_matrix(
    features: sp.spmatrix,
    split: EdgeSplit,
    feature_mode: str = "raw",
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """
    Returns
        S : normalized training adjacency
        F : processed features
        H = S F
    """
    s = normalize_adjacency(split.adj_train).astype(np.float64).tocsr()
    f = prepare_features(features, mode=feature_mode)
    h = (s @ f).tocsr()
    h.eliminate_zeros()
    return s, f, h


def _apply_M(
    split: EdgeSplit,
    x: np.ndarray,
) -> np.ndarray:
    """
    Apply M = 4Y - 2J without forming either M or J.

    x can be shape (n,) or (n, r).
    """
    y = split.adj_train.astype(np.float64).tocsr() + sp.eye(
        split.adj_train.shape[0],
        dtype=np.float64,
        format="csr",
    )

    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        return 4.0 * (y @ x) - 2.0 * float(x.sum()) * np.ones_like(x)

    return 4.0 * (y @ x) - 2.0 * np.ones((x.shape[0], 1)) @ x.sum(
        axis=0,
        keepdims=True,
    )


def feature_operator(
    h: sp.csr_matrix,
    split: EdgeSplit,
) -> spla.LinearOperator:
    """
    LinearOperator for
        A_F = H^T M H = F^T S M S F.
    """
    p = h.shape[1]

    def matvec(v):
        hv = h @ np.asarray(v, dtype=np.float64)
        mhv = _apply_M(split, hv)
        return np.asarray(h.T @ mhv, dtype=np.float64).ravel()

    def matmat(v):
        hv = h @ np.asarray(v, dtype=np.float64)
        mhv = _apply_M(split, hv)
        return np.asarray(h.T @ mhv, dtype=np.float64)

    return spla.LinearOperator(
        shape=(p, p),
        matvec=matvec,
        matmat=matmat,
        rmatvec=matvec,
        dtype=np.float64,
    )


def compute_feature_spectrum(
    features: sp.spmatrix,
    split: EdgeSplit,
    positive_rank: int = 128,
    negative_rank: int = 0,
    feature_mode: str = "raw",
    seed: int = 0,
    eig_tol: float = 1e-6,
) -> Dict[str, object]:
    """
    Compute selected extreme eigenpairs of A_F = F^T S M S F.

    The returned `hv` is H V, allowing queried edge scores to be evaluated
    without forming an n x n kernel.
    """
    _, f, h = build_feature_encoder_matrix(
        features=features,
        split=split,
        feature_mode=feature_mode,
    )

    p = f.shape[1]
    if p < 2:
        raise ValueError("Need at least two feature dimensions.")

    op = feature_operator(h, split)
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=p)
    v0 /= np.linalg.norm(v0)

    eval_parts = []
    evec_parts = []
    tail_labels = []

    max_k = max(1, p - 2)

    rp = min(max(0, int(positive_rank)), max_k)
    rn = min(max(0, int(negative_rank)), max_k)

    if rp:
        vals, vecs = spla.eigsh(
            op,
            k=rp,
            which="LA",
            tol=float(eig_tol),
            v0=v0,
        )
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        eval_parts.append(vals)
        evec_parts.append(vecs)
        tail_labels.extend(["positive"] * len(vals))

    if rn:
        vals, vecs = spla.eigsh(
            op,
            k=rn,
            which="SA",
            tol=float(eig_tol),
            v0=v0,
        )
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        eval_parts.append(vals)
        evec_parts.append(vecs)
        tail_labels.extend(["negative"] * len(vals))

    if not eval_parts:
        eigenvalues = np.empty(0, dtype=np.float64)
        eigenvectors = np.empty((p, 0), dtype=np.float64)
        hv = np.empty((h.shape[0], 0), dtype=np.float64)
    else:
        eigenvalues = np.concatenate(eval_parts).astype(np.float64)
        eigenvectors = np.column_stack(evec_parts).astype(np.float64)
        hv = np.asarray(h @ eigenvectors, dtype=np.float64)

    positive_values = eigenvalues[eigenvalues > 0]
    lambda_max_positive = (
        float(np.max(positive_values))
        if len(positive_values)
        else np.nan
    )
    rho_retained = (
        float(np.max(np.abs(eigenvalues)))
        if len(eigenvalues)
        else np.nan
    )

    return {
        "H": h,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "hv": hv,
        "tail_labels": np.asarray(tail_labels, dtype=object),
        "feature_dim": int(p),
        "feature_nnz": int(f.nnz),
        "H_nnz": int(h.nnz),
        "lambda_max_positive": lambda_max_positive,
        "rho_retained": rho_retained,
        "feature_mode": feature_mode,
    }


def resolve_alpha(
    spectrum: Dict[str, object],
    n: int,
    step_mode: str = "normalized_positive",
    step_scale: float = 0.05,
    lr: float = 600.0,
) -> Tuple[float, float]:
    """
    Resolve alpha in
        W_{k+1} = (I + alpha A_F) W_k.

    normalized_positive:
        alpha = step_scale / lambda_max_positive.
        This is robust to raw feature scaling across datasets.

    lr:
        alpha = 4 * lr / n^2,
        the literal Taylor-SGD learning-rate parameterization.

    Returns (alpha, implied_lr).
    """
    if step_mode == "lr":
        alpha = 4.0 * float(lr) / float(n * n)
        return alpha, float(lr)

    if step_mode == "normalized_positive":
        lmax = float(spectrum["lambda_max_positive"])
        if not np.isfinite(lmax) or lmax <= 0:
            raise ValueError(
                "No positive retained feature-space eigenvalue; cannot use "
                "normalized_positive step mode."
            )
        alpha = float(step_scale) / lmax
        implied_lr = alpha * float(n * n) / 4.0
        return alpha, implied_lr

    raise ValueError("step_mode must be 'normalized_positive' or 'lr'")


def _baseline_hht_scores(
    edges: np.ndarray,
    h: sp.csr_matrix,
) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64)
    hu = h[edges[:, 0]]
    hv = h[edges[:, 1]]
    return np.asarray(hu.multiply(hv).sum(axis=1)).ravel().astype(np.float64)


def feature_kernel_edge_scores(
    edges: np.ndarray,
    spectrum: Dict[str, object],
    alpha: float,
    k: int,
) -> np.ndarray:
    """
    Query scores of the truncated feature kernel

        H H^T + H V diag((1 + alpha lambda)^(2k) - 1) V^T H^T.

    Uses global positive rescaling in log-space to avoid overflow; AUC/AP are
    invariant to this rescaling.
    """
    h = spectrum["H"]
    hv = np.asarray(spectrum["hv"], dtype=np.float64)
    lam = np.asarray(spectrum["eigenvalues"], dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)

    baseline = _baseline_hht_scores(edges, h)

    if int(k) == 0 or len(lam) == 0:
        return baseline

    base = np.abs(1.0 + float(alpha) * lam)

    with np.errstate(divide="ignore", invalid="ignore"):
        logw = 2.0 * int(k) * np.log(base)

    # Global rescaling keeps the sum numerically safe:
    #   score / exp(scale_log)
    finite = np.isfinite(logw)
    max_logw = float(np.max(logw[finite])) if np.any(finite) else 0.0
    scale_log = max(0.0, max_logw)

    inv_scale = np.exp(-scale_log) if scale_log < 745 else 0.0

    w_scaled = np.zeros_like(logw, dtype=np.float64)
    if np.any(finite):
        w_scaled[finite] = np.exp(logw[finite] - scale_log)

    # ((weight - 1) / scale)
    correction_weights = w_scaled - inv_scale

    u = edges[:, 0]
    v = edges[:, 1]
    correction = np.einsum(
        "ij,ij,j->i",
        hv[u],
        hv[v],
        correction_weights,
        optimize=True,
    )

    return inv_scale * baseline + correction


def feature_kernel_metrics_for_k(
    split: EdgeSplit,
    spectrum: Dict[str, object],
    alpha: float,
    k: int,
    binary_metrics,
) -> Dict[str, float]:
    val_pos = feature_kernel_edge_scores(
        split.val_edges, spectrum, alpha, k
    )
    val_neg = feature_kernel_edge_scores(
        split.val_edges_false, spectrum, alpha, k
    )
    val_auc, val_ap = binary_metrics(val_pos, val_neg)

    return {
        "val_auc": float(val_auc),
        "val_ap": float(val_ap),
    }
