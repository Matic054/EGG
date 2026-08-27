
"""
Deterministic expected kernel for the linearized Taylor-GAE dynamics.

Linearized encoder-SGD:
    W_{k+1} = (I + alpha A) W_k,
    A = S M S,
    alpha = 4 * lr / n^2.

Hence
    W_k = P_k W_0,
    P_k = (I + alpha A)^k,
and
    X_k = Z_k Z_k^T
        = S P_k W_0 W_0^T P_k S.

For isotropic Xavier initialization,
    E[W_0 W_0^T] = const * I,
so, up to a positive scalar irrelevant to AUC/AP,

    K_k = E[X_k] ∝ S (I + alpha A)^(2k) S.

If A = V diag(lambda) V^T,
    K_k = S^2
          + (S V) diag((1+alpha*lambda)^(2k)-1) (S V)^T.

We approximate only the correction using extreme eigenpairs while retaining
the exact sparse baseline S^2. This is better than simply truncating K_k to
a rank-r feature representation because eigenmodes near lambda=0 should
contribute their baseline weight 1, not disappear.

We retrieve:
  - largest algebraic eigenvalues (amplified modes),
  - smallest algebraic eigenvalues (suppressed modes).

No random initialization or embedding dimension appears in the deterministic
kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.metrics import roc_auc_score, average_precision_score

from linear_gae_spectral import (
    PlanetoidData,
    EdgeSplit,
    load_planetoid,
    split_edges_kipf,
    normalize_adjacency,
)


def _training_target_y(adj_train: sp.csr_matrix) -> sp.csr_matrix:
    n = adj_train.shape[0]
    y = (
        adj_train.astype(np.float64).tocsr()
        + sp.eye(n, dtype=np.float64, format="csr")
    )
    y.data[:] = 1.0
    y.eliminate_zeros()
    return y


def _apply_M(y: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    """
    Apply M = 4Y - 2J without forming dense J.
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        return (
            4.0 * (y @ x)
            - 2.0 * np.sum(x) * np.ones_like(x)
        )

    if x.ndim == 2:
        return (
            4.0 * (y @ x)
            - 2.0
            * np.ones((x.shape[0], 1), dtype=np.float64)
            @ np.sum(x, axis=0, keepdims=True)
        )

    raise ValueError("x must be 1D or 2D")


class SMSOperator(spla.LinearOperator):
    """Symmetric LinearOperator A = S M S."""

    def __init__(self, s: sp.csr_matrix, y: sp.csr_matrix):
        self.s = s.astype(np.float64).tocsr()
        self.y = y.astype(np.float64).tocsr()
        n = self.s.shape[0]
        super().__init__(
            dtype=np.dtype(np.float64),
            shape=(n, n),
        )

    def _matvec(self, x):
        sx = self.s @ np.asarray(x, dtype=np.float64)
        return np.asarray(
            self.s @ _apply_M(self.y, sx)
        ).reshape(-1)

    def _matmat(self, x):
        sx = self.s @ np.asarray(x, dtype=np.float64)
        return np.asarray(
            self.s @ _apply_M(self.y, sx)
        )

    def _rmatvec(self, x):
        return self._matvec(x)

    def _rmatmat(self, x):
        return self._matmat(x)


def _eigsh_robust(
    operator: spla.LinearOperator,
    k: int,
    which: Literal["LA", "SA"],
    seed: int,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
    ncv: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust symmetric extreme-eigenpair helper.
    """
    n = operator.shape[0]
    k = int(min(max(k, 1), n - 2))

    if ncv is None:
        ncv = min(
            n,
            max(2 * k + 1, 40),
        )

    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=n)
    v0 /= np.linalg.norm(v0)

    try:
        vals, vecs = spla.eigsh(
            operator,
            k=k,
            which=which,
            v0=v0,
            tol=tol,
            maxiter=maxiter,
            ncv=ncv,
        )
    except spla.ArpackNoConvergence as exc:
        vals = exc.eigenvalues
        vecs = exc.eigenvectors

        if vals is None or vecs is None or len(vals) < max(4, k // 2):
            # One more attempt with a more generous Krylov subspace.
            vals, vecs = spla.eigsh(
                operator,
                k=k,
                which=which,
                v0=v0,
                tol=max(tol, 1e-5),
                maxiter=(
                    10 * n
                    if maxiter is None
                    else max(2 * maxiter, 10 * n)
                ),
                ncv=min(n, max(4 * k + 1, 80)),
            )

    order = np.argsort(vals)
    if which == "LA":
        order = order[::-1]

    return (
        np.asarray(vals[order], dtype=np.float64),
        np.asarray(vecs[:, order], dtype=np.float64),
    )


def compute_extreme_sms_eigenpairs(
    split: EdgeSplit,
    positive_rank: int = 128,
    negative_rank: int = 32,
    seed: int = 0,
    eig_tol: float = 1e-6,
    maxiter: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compute S, S^2, and extreme eigenpairs of A = S M S.
    """
    s = normalize_adjacency(
        split.adj_train
    ).astype(np.float64).tocsr()

    s2 = (s @ s).tocsr()
    y = _training_target_y(split.adj_train)
    op = SMSOperator(s, y)

    pos_vals, pos_vecs = _eigsh_robust(
        op,
        k=positive_rank,
        which="LA",
        seed=seed,
        tol=eig_tol,
        maxiter=maxiter,
    )

    if negative_rank > 0:
        neg_vals, neg_vecs = _eigsh_robust(
            op,
            k=negative_rank,
            which="SA",
            seed=seed + 99173,
            tol=eig_tol,
            maxiter=maxiter,
        )
    else:
        neg_vals = np.empty(0, dtype=np.float64)
        neg_vecs = np.empty(
            (s.shape[0], 0),
            dtype=np.float64,
        )

    # Remove any accidental duplicate eigenpairs if the requested extreme
    # sets overlap. In normal Cora-sized runs with modest ranks they won't.
    vals = np.concatenate([pos_vals, neg_vals])
    vecs = np.concatenate([pos_vecs, neg_vecs], axis=1)

    # S V is what enters the correction kernel.
    sv = np.asarray(s @ vecs, dtype=np.float64)

    return {
        "s": s,
        "s2": s2,
        "operator": op,
        "eigenvalues": vals,
        "eigenvectors": vecs,
        "sv": sv,
        "positive_eigenvalues": pos_vals,
        "negative_eigenvalues": neg_vals,
        "positive_rank": len(pos_vals),
        "negative_rank": len(neg_vals),
    }


def _sparse_pair_values(
    matrix: sp.csr_matrix,
    edges: np.ndarray,
) -> np.ndarray:
    """
    Extract matrix[u,v] for an edge list without densifying the matrix.
    """
    edges = np.asarray(edges, dtype=np.int64)
    vals = np.asarray(
        matrix[edges[:, 0], edges[:, 1]]
    ).reshape(-1)
    return vals.astype(np.float64, copy=False)


def correction_weights(
    eigenvalues: np.ndarray,
    alpha: float,
    k: int,
) -> Tuple[np.ndarray, float]:
    """
    Return numerically scaled

        g_i(k) = (1 + alpha lambda_i)^(2k) - 1.

    We divide the entire kernel by one positive scale. Since ranking metrics
    are invariant under positive global scaling, this cannot change AUC/AP.

    Returns:
      g_scaled
      baseline_scale = 1 / global_scale

    so that the scored kernel is
      baseline_scale*S^2 + SV diag(g_scaled) SV^T.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64)
    k = int(k)

    if k < 0:
        raise ValueError("k must be nonnegative")

    if k == 0 or len(lam) == 0:
        return np.zeros_like(lam), 1.0

    base = 1.0 + float(alpha) * lam

    # Even exponent guarantees nonnegative spectral weights. Work in logs
    # to remain well behaved if we later scan more aggressive times.
    abs_base = np.abs(base)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_w = 2.0 * k * np.log(abs_base)

    # w=0 when base=0.
    log_w = np.where(
        abs_base == 0.0,
        -np.inf,
        log_w,
    )

    # Pick a positive global scale >= 1. This keeps both the baseline and
    # correction finite while preserving every pairwise ranking.
    finite = log_w[np.isfinite(log_w)]
    max_log_w = (
        float(np.max(finite))
        if len(finite)
        else 0.0
    )
    log_scale = max(0.0, max_log_w)
    baseline_scale = float(np.exp(-log_scale))

    # g/scale = exp(log_w-log_scale) - exp(-log_scale)
    scaled_w = np.exp(log_w - log_scale)
    scaled_w = np.where(
        np.isfinite(scaled_w),
        scaled_w,
        0.0,
    )
    g_scaled = scaled_w - baseline_scale

    return g_scaled, baseline_scale


def deterministic_kernel_edge_scores(
    edges: np.ndarray,
    s2: sp.csr_matrix,
    sv: np.ndarray,
    eigenvalues: np.ndarray,
    alpha: float,
    k: int,
) -> np.ndarray:
    """
    Score requested node pairs under

        K_k ≈ S^2 + SV diag(g_k(lambda)) (SV)^T,

    with global numerical rescaling handled internally.
    """
    edges = np.asarray(edges, dtype=np.int64)

    g, baseline_scale = correction_weights(
        eigenvalues=eigenvalues,
        alpha=alpha,
        k=k,
    )

    score = (
        baseline_scale
        * _sparse_pair_values(s2, edges)
    )

    if sv.shape[1] > 0:
        left = sv[edges[:, 0], :]
        right = sv[edges[:, 1], :]
        score = score + np.einsum(
            "ij,j,ij->i",
            left,
            g,
            right,
            optimize=True,
        )

    return np.asarray(score, dtype=np.float64)


def evaluate_deterministic_kernel(
    split: EdgeSplit,
    spectral_data: Dict[str, object],
    alpha: float,
    k: int,
) -> Dict[str, float]:
    """
    Evaluate validation and test AUC/AP without constructing dense K_k.
    """
    s2 = spectral_data["s2"]
    sv = spectral_data["sv"]
    vals = spectral_data["eigenvalues"]

    val_pos = deterministic_kernel_edge_scores(
        split.val_edges,
        s2,
        sv,
        vals,
        alpha,
        k,
    )
    val_neg = deterministic_kernel_edge_scores(
        split.val_edges_false,
        s2,
        sv,
        vals,
        alpha,
        k,
    )
    test_pos = deterministic_kernel_edge_scores(
        split.test_edges,
        s2,
        sv,
        vals,
        alpha,
        k,
    )
    test_neg = deterministic_kernel_edge_scores(
        split.test_edges_false,
        s2,
        sv,
        vals,
        alpha,
        k,
    )

    def metrics(pos, neg):
        y = np.concatenate(
            [
                np.ones(len(pos), dtype=np.int8),
                np.zeros(len(neg), dtype=np.int8),
            ]
        )
        score = np.concatenate([pos, neg])

        return (
            float(roc_auc_score(y, score)),
            float(average_precision_score(y, score)),
        )

    val_auc, val_ap = metrics(val_pos, val_neg)
    test_auc, test_ap = metrics(test_pos, test_neg)

    return {
        "k": int(k),
        "val_auc": val_auc,
        "val_ap": val_ap,
        "test_auc": test_auc,
        "test_ap": test_ap,
    }


def sweep_deterministic_kernel_single(
    data: PlanetoidData,
    split: EdgeSplit,
    lr: float = 600.0,
    k_values: Sequence[int] = tuple(range(0, 801, 10)),
    positive_rank: int = 128,
    negative_rank: int = 32,
    eig_seed: int = 0,
    eig_tol: float = 1e-6,
    maxiter: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Compute eigensystem once, then sweep the deterministic kernel time k.
    """
    n = data.n
    alpha = 4.0 * float(lr) / float(n * n)

    spec = compute_extreme_sms_eigenpairs(
        split=split,
        positive_rank=positive_rank,
        negative_rank=negative_rank,
        seed=eig_seed,
        eig_tol=eig_tol,
        maxiter=maxiter,
    )

    if verbose:
        pvals = spec["positive_eigenvalues"]
        nvals = spec["negative_eigenvalues"]

        print(
            f"{data.name}: alpha={alpha:.8g}, "
            f"positive_rank={len(pvals)}, negative_rank={len(nvals)}"
        )
        print(
            f"  lambda_max={np.max(pvals):.6g}, "
            f"smallest retained positive-side={np.min(pvals):.6g}"
        )
        if len(nvals):
            print(
                f"  lambda_min={np.min(nvals):.6g}, "
                f"largest retained negative-side={np.max(nvals):.6g}"
            )

    rows = []

    for k in k_values:
        result = evaluate_deterministic_kernel(
            split=split,
            spectral_data=spec,
            alpha=alpha,
            k=int(k),
        )

        result.update(
            {
                "alpha": alpha,
                "lr": float(lr),
                "positive_rank": int(positive_rank),
                "negative_rank": int(negative_rank),
            }
        )
        rows.append(result)

        if verbose:
            print(
                f"{data.name:8s} kernel "
                f"k={int(k):04d} "
                f"val={result['val_auc']:.4f} "
                f"test={result['test_auc']:.4f} "
                f"AP={result['test_ap']:.4f}"
            )

    return pd.DataFrame(rows), spec


def run_deterministic_kernel_experiment(
    dataset: str = "cora",
    n_runs: int = 3,
    base_seed: int = 0,
    lr: float = 600.0,
    k_values: Sequence[int] = tuple(range(0, 801, 10)),
    positive_rank: int = 128,
    negative_rank: int = 32,
    data_root: str | Path = "data/planetoid",
    eig_tol: float = 1e-6,
    maxiter: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Salha-style random-split deterministic kernel experiment.
    """
    data = load_planetoid(
        dataset,
        root=data_root,
        download=True,
    )

    frames = []

    for run in range(n_runs):
        split_seed = base_seed + run

        split = split_edges_kipf(
            data.adjacency,
            seed=split_seed,
        )

        if verbose:
            print(
                f"\n--- {dataset} run={run:02d} "
                f"rank=+{positive_rank}/-{negative_rank} ---"
            )

        frame, _ = sweep_deterministic_kernel_single(
            data=data,
            split=split,
            lr=lr,
            k_values=k_values,
            positive_rank=positive_rank,
            negative_rank=negative_rank,
            eig_seed=100000 + split_seed,
            eig_tol=eig_tol,
            maxiter=maxiter,
            verbose=verbose,
        )

        frame["dataset"] = dataset
        frame["run"] = run
        frame["split_seed"] = split_seed

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def validation_selected_summary(
    results: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validation-select k separately in each split.
    """
    rows = []

    grouping = [
        "dataset",
        "run",
        "positive_rank",
        "negative_rank",
        "lr",
    ]

    for _, group in results.groupby(grouping, sort=False):
        idx = group["val_auc"].idxmax()
        rows.append(results.loc[idx].copy())

    selected = pd.DataFrame(rows)

    summary = (
        selected.groupby(
            [
                "dataset",
                "positive_rank",
                "negative_rank",
                "lr",
            ],
            sort=False,
        )
        .agg(
            runs=("run", "size"),
            selected_k_mean=("k", "mean"),
            selected_k_sd=("k", "std"),
            val_auc_mean=("val_auc", "mean"),
            val_auc_sd=("val_auc", "std"),
            test_auc_mean=("test_auc", "mean"),
            test_auc_sd=("test_auc", "std"),
            test_ap_mean=("test_ap", "mean"),
            test_ap_sd=("test_ap", "std"),
        )
        .reset_index()
    )

    return selected, summary


def rank_convergence_experiment(
    dataset: str = "cora",
    ranks: Sequence[Tuple[int, int]] = (
        (32, 8),
        (64, 16),
        (128, 32),
        (256, 64),
    ),
    n_runs: int = 3,
    base_seed: int = 0,
    lr: float = 600.0,
    k_values: Sequence[int] = tuple(range(250, 551, 10)),
    data_root: str | Path = "data/planetoid",
    eig_tol: float = 1e-6,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience driver to check whether the truncated deterministic kernel
    stabilizes as more extreme SMS eigenmodes are retained.

    Note: this recomputes eigensystems for each rank for simplicity. For a
    final benchmark we can optimize this by computing the largest requested
    ranks once per split and slicing them.
    """
    all_results = []

    for positive_rank, negative_rank in ranks:
        frame = run_deterministic_kernel_experiment(
            dataset=dataset,
            n_runs=n_runs,
            base_seed=base_seed,
            lr=lr,
            k_values=k_values,
            positive_rank=positive_rank,
            negative_rank=negative_rank,
            data_root=data_root,
            eig_tol=eig_tol,
            verbose=verbose,
        )

        all_results.append(frame)

    results = pd.concat(all_results, ignore_index=True)
    selected, summary = validation_selected_summary(results)

    return results, selected, summary
