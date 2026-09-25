#!/usr/bin/env python3
"""
Reproducible diagnostic for the nonlinear/cubic term in the Taylor-GAE dynamics.

Purpose
-------
This script provides repository-visible evidence for the claim in the paper that,
during useful finite-time training, the linear and cubic terms in the Taylor-GAE
gradient become strongly aligned.

For

    L(W) = ||M - S W W^T S||_F^2 / n^2,
    Z = S W,
    M = 4Y - 2J,
    Y = A_train + I,

the gradient is

    grad_W L = (4/n^2) [
        S Z (Z^T Z) - S M Z
    ].

Define

    linear  = S M Z,
    cubic   = S Z (Z^T Z).

Gradient descent is therefore

    W <- W + alpha * (linear - cubic),
    alpha = 4 * eta / n^2,

where eta is the ordinary SGD learning rate.

At regular checkpoints this script records

    ratio   = ||cubic||_F / ||linear||_F
    cosine  = <linear,cubic>_F /
              (||linear||_F ||cubic||_F)

as well as validation AUC/AP and the exact Taylor objective value.  A cosine
near one means that the cubic term is largely collinear with the linear growth
term, so it primarily acts as a magnitude/saturation correction rather than
introducing an unrelated update direction.

The implementation never constructs dense M, J, S^2, ZZ^T, or an n x n
decoder matrix.  The Taylor objective and its exact full-batch gradient are
computed from sparse graph products and d x d Gram matrices.

Outputs
-------
The output directory contains:

    trajectory.csv
        Every recorded checkpoint from every split.

    selected_checkpoints.csv
        The validation-AUC-selected checkpoint from every split.

    selected_summary.csv
        Mean/std statistics at those validation-selected checkpoints.

    useful_regime.csv
        Checkpoints whose validation AUC is at least
        --useful-fraction times the best validation AUC in that split.

    useful_regime_summary.csv
        Aggregate diagnostics over the useful regime.

    trajectory_summary.csv
        Mean/std trajectory across splits.

    cosine_trajectory.png / .pdf
        Mean linear-vs-cubic cosine through training.

    norm_ratio_trajectory.png / .pdf
        Mean cubic/linear norm ratio through training.

    README.txt
        Plain-text description of the diagnostic and headline numbers.

Recommended paper-aligned run
-----------------------------
The default run uses Cora, 10 random splits, latent dimension 16, and the same
split seed convention used in the final benchmark:

    python run_nonlinear_term_diagnostics.py

A faster smoke test is

    python run_nonlinear_term_diagnostics.py --folds 1 --epochs 50

To reproduce the earlier exploratory d=32 diagnostic:

    python run_nonlinear_term_diagnostics.py \
        --embedding-dim 32 --epochs 1000 --lr 600

The dataset list is configurable, e.g.

    python run_nonlinear_term_diagnostics.py \
        --datasets cora,citeseer,pubmed

Dependencies
------------
numpy, pandas, scipy, scikit-learn, matplotlib, and the project's
linear_gae_spectral.py module.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linear_gae_spectral import (
    load_planetoid,
    split_edges_kipf,
    normalize_adjacency,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    datasets: Tuple[str, ...] = ("cora",)
    folds: int = 10
    fold_start: int = 0
    base_seed: int = 10_000
    data_root: str = "data/planetoid"

    embedding_dim: int = 16
    epochs: int = 600
    lr: float = 600.0
    eval_every: int = 10

    # A checkpoint belongs to the "useful regime" if its validation AUC is at
    # least this fraction of the best recorded validation AUC in that split.
    useful_fraction: float = 0.99

    verbose: bool = True


# ---------------------------------------------------------------------------
# Core mathematics
# ---------------------------------------------------------------------------

def _training_target_y(adj_train: sp.csr_matrix) -> sp.csr_matrix:
    """Binary Y = A_train + I."""
    n = adj_train.shape[0]
    y = (
        adj_train.astype(np.float64).tocsr()
        + sp.eye(n, dtype=np.float64, format="csr")
    )
    y.data[:] = 1.0
    y.eliminate_zeros()
    return y


def _apply_m(y: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    """
    Apply M = 4Y - 2J to an n x d dense matrix without materializing M or J.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must have shape (n,d)")

    return (
        4.0 * (y @ x)
        - 2.0 * np.ones((x.shape[0], 1), dtype=np.float64)
        @ np.sum(x, axis=0, keepdims=True)
    )


def _frobenius_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity after vectorizing two matrices."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    denom = na * nb
    if denom <= 1e-30:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _xavier_uniform(
    n: int,
    d: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Xavier-uniform initialization for an n x d weight matrix."""
    bound = math.sqrt(6.0 / float(n + d))
    return rng.uniform(-bound, bound, size=(n, d)).astype(np.float64)


def _binary_metrics(
    z: np.ndarray,
    pos_edges: np.ndarray,
    neg_edges: np.ndarray,
) -> Tuple[float, float]:
    """
    Inner-product decoder metrics.  Sigmoid is unnecessary because it is
    monotone and therefore does not change ROC-AUC or AP rankings.
    """
    pos_edges = np.asarray(pos_edges, dtype=np.int64)
    neg_edges = np.asarray(neg_edges, dtype=np.int64)

    pos = np.einsum(
        "ij,ij->i",
        z[pos_edges[:, 0]],
        z[pos_edges[:, 1]],
        optimize=True,
    )
    neg = np.einsum(
        "ij,ij->i",
        z[neg_edges[:, 0]],
        z[neg_edges[:, 1]],
        optimize=True,
    )

    labels = np.concatenate([
        np.ones(len(pos), dtype=np.int8),
        np.zeros(len(neg), dtype=np.int8),
    ])
    scores = np.concatenate([pos, neg])

    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def _terms_and_objective(
    s: sp.csr_matrix,
    y: sp.csr_matrix,
    w: np.ndarray,
) -> Dict[str, object]:
    """
    Compute the exact Taylor-GAE gradient decomposition and objective.

    With z = S w:

        linear = S M z
        cubic  = S z (z^T z)

    and

        grad_W L = (4/n^2) (cubic - linear).

    The objective is evaluated without forming ZZ^T:

        ||M - ZZ^T||_F^2 / n^2
          = 4
            - 2 tr(Z^T M Z)/n^2
            + ||Z^T Z||_F^2/n^2,

    because every entry of M = 4Y - 2J is either +2 or -2 and therefore
    ||M||_F^2 = 4 n^2.
    """
    n = int(s.shape[0])

    z = np.asarray(s @ w, dtype=np.float64)
    mz = _apply_m(y, z)

    gram = z.T @ z
    z_gram = z @ gram

    linear = np.asarray(s @ mz, dtype=np.float64)
    cubic = np.asarray(s @ z_gram, dtype=np.float64)

    linear_norm = float(np.linalg.norm(linear))
    cubic_norm = float(np.linalg.norm(cubic))

    ratio = cubic_norm / max(linear_norm, 1e-30)
    cosine = _frobenius_cosine(linear, cubic)

    net = linear - cubic
    net_norm = float(np.linalg.norm(net))
    net_to_linear = net_norm / max(linear_norm, 1e-30)

    objective = (
        4.0
        - 2.0 * float(np.sum(z * mz)) / float(n * n)
        + float(np.sum(gram * gram)) / float(n * n)
    )

    return {
        "z": z,
        "linear": linear,
        "cubic": cubic,
        "objective": float(objective),
        "linear_norm": linear_norm,
        "cubic_norm": cubic_norm,
        "cubic_to_linear_ratio": float(ratio),
        "linear_cubic_cosine": float(cosine),
        "net_norm": net_norm,
        "net_to_linear_ratio": float(net_to_linear),
        "z_norm": float(np.linalg.norm(z)),
        "w_norm": float(np.linalg.norm(w)),
    }


# ---------------------------------------------------------------------------
# One exact full-batch Taylor-SGD run
# ---------------------------------------------------------------------------

def run_single(
    data,
    split,
    *,
    embedding_dim: int,
    epochs: int,
    lr: float,
    eval_every: int,
    init_seed: int,
    verbose: bool,
) -> pd.DataFrame:
    n = int(data.n)

    s = normalize_adjacency(split.adj_train).astype(np.float64).tocsr()
    y = _training_target_y(split.adj_train)

    rng = np.random.RandomState(int(init_seed))
    w = _xavier_uniform(n, int(embedding_dim), rng)

    # Ordinary GD learning rate eta corresponds to alpha = 4 eta / n^2 in the
    # unscaled decomposition W <- W + alpha(linear - cubic).
    alpha = 4.0 * float(lr) / float(n * n)

    rows = []

    def record(epoch: int, terms: Dict[str, object]) -> None:
        z = terms["z"]
        val_auc, val_ap = _binary_metrics(
            z,
            split.val_edges,
            split.val_edges_false,
        )

        row = {
            "epoch": int(epoch),
            "alpha": float(alpha),
            "lr": float(lr),
            "objective": float(terms["objective"]),
            "val_auc": float(val_auc),
            "val_ap": float(val_ap),
            "linear_norm": float(terms["linear_norm"]),
            "cubic_norm": float(terms["cubic_norm"]),
            "cubic_to_linear_ratio": float(terms["cubic_to_linear_ratio"]),
            "linear_cubic_cosine": float(terms["linear_cubic_cosine"]),
            "net_norm": float(terms["net_norm"]),
            "net_to_linear_ratio": float(terms["net_to_linear_ratio"]),
            "z_norm": float(terms["z_norm"]),
            "w_norm": float(terms["w_norm"]),
        }
        rows.append(row)

        if verbose:
            print(
                f"epoch={epoch:04d} "
                f"val_auc={val_auc:.4f} "
                f"ratio={row['cubic_to_linear_ratio']:.4f} "
                f"cos={row['linear_cubic_cosine']:.4f} "
                f"loss={row['objective']:.6f}"
            )

    terms = _terms_and_objective(s, y, w)
    record(0, terms)

    for epoch in range(1, int(epochs) + 1):
        # The terms are exact at the current W.
        if epoch > 1:
            terms = _terms_and_objective(s, y, w)

        # Exact full-batch gradient-descent step:
        #
        # W <- W - eta * grad L
        #   = W + (4 eta/n^2) (linear - cubic).
        w += alpha * (terms["linear"] - terms["cubic"])

        if not np.all(np.isfinite(w)):
            raise FloatingPointError(
                f"Non-finite W encountered at epoch {epoch}. "
                "Reduce --lr."
            )

        if (
            epoch == 1
            or epoch % int(eval_every) == 0
            or epoch == int(epochs)
        ):
            post = _terms_and_objective(s, y, w)
            record(epoch, post)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _select_validation_best(trajectory: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = [
        "dataset",
        "fold",
        "split_seed",
        "init_seed",
        "embedding_dim",
        "lr",
    ]

    for _, group in trajectory.groupby(group_cols, sort=False):
        # Maximum validation AUC, then AP, then earlier epoch.
        ordered = group.sort_values(
            ["val_auc", "val_ap", "epoch"],
            ascending=[False, False, True],
        )
        rows.append(ordered.iloc[0].copy())

    return pd.DataFrame(rows).reset_index(drop=True)


def _selected_summary(selected: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "epoch",
        "val_auc",
        "val_ap",
        "objective",
        "cubic_to_linear_ratio",
        "linear_cubic_cosine",
        "net_to_linear_ratio",
    ]

    rows = []

    for dataset, group in selected.groupby("dataset", sort=False):
        for metric in metrics:
            x = group[metric].to_numpy(dtype=float)
            finite = x[np.isfinite(x)]

            rows.append({
                "dataset": dataset,
                "metric": metric,
                "n": int(len(finite)),
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "sd": (
                    float(np.std(finite, ddof=1))
                    if len(finite) > 1
                    else np.nan
                ),
                "min": float(np.min(finite)) if len(finite) else np.nan,
                "max": float(np.max(finite)) if len(finite) else np.nan,
            })

    return pd.DataFrame(rows)


def _useful_regime(
    trajectory: pd.DataFrame,
    fraction: float,
) -> pd.DataFrame:
    """
    Retain checkpoints with validation AUC >= fraction * best validation AUC
    within each split.

    This avoids supporting the alignment claim with one cherry-picked epoch.
    """
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError("--useful-fraction must be in (0,1].")

    pieces = []

    key = ["dataset", "fold", "split_seed", "init_seed"]

    for _, group in trajectory.groupby(key, sort=False):
        best = float(group["val_auc"].max())
        threshold = float(fraction) * best
        g = group[group["val_auc"] >= threshold].copy()
        g["best_val_auc_in_split"] = best
        g["useful_auc_threshold"] = threshold
        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


def _useful_summary(useful: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, group in useful.groupby("dataset", sort=False):
        for metric in (
            "cubic_to_linear_ratio",
            "linear_cubic_cosine",
            "net_to_linear_ratio",
        ):
            x = group[metric].to_numpy(dtype=float)
            finite = x[np.isfinite(x)]

            rows.append({
                "dataset": dataset,
                "metric": metric,
                "checkpoints": int(len(finite)),
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "sd": (
                    float(np.std(finite, ddof=1))
                    if len(finite) > 1
                    else np.nan
                ),
                "min": float(np.min(finite)) if len(finite) else np.nan,
                "q10": (
                    float(np.quantile(finite, 0.10))
                    if len(finite)
                    else np.nan
                ),
                "median": (
                    float(np.median(finite))
                    if len(finite)
                    else np.nan
                ),
                "q90": (
                    float(np.quantile(finite, 0.90))
                    if len(finite)
                    else np.nan
                ),
                "max": float(np.max(finite)) if len(finite) else np.nan,
            })

    return pd.DataFrame(rows)


def _trajectory_summary(trajectory: pd.DataFrame) -> pd.DataFrame:
    return (
        trajectory
        .groupby(["dataset", "epoch"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            val_auc_mean=("val_auc", "mean"),
            val_auc_sd=("val_auc", "std"),
            ratio_mean=("cubic_to_linear_ratio", "mean"),
            ratio_sd=("cubic_to_linear_ratio", "std"),
            cosine_mean=("linear_cubic_cosine", "mean"),
            cosine_sd=("linear_cubic_cosine", "std"),
            net_to_linear_mean=("net_to_linear_ratio", "mean"),
            objective_mean=("objective", "mean"),
        )
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_band(
    summary: pd.DataFrame,
    *,
    mean_col: str,
    sd_col: str,
    ylabel: str,
    out_base: Path,
    hline: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for dataset, group in summary.groupby("dataset", sort=False):
        x = group["epoch"].to_numpy(dtype=float)
        y = group[mean_col].to_numpy(dtype=float)
        sd = group[sd_col].fillna(0.0).to_numpy(dtype=float)

        ax.plot(x, y, label=dataset)
        ax.fill_between(x, y - sd, y + sd, alpha=0.18)

    if hline is not None:
        ax.axhline(hline, linestyle="--", linewidth=0.9)

    ax.set_xlabel("Taylor-SGD epoch")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(out_base.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def _metric_row(summary: pd.DataFrame, dataset: str, metric: str):
    g = summary[
        (summary["dataset"] == dataset)
        & (summary["metric"] == metric)
    ]
    return None if g.empty else g.iloc[0]


def _write_readme(
    out_dir: Path,
    cfg: Config,
    selected_summary: pd.DataFrame,
    useful_summary: pd.DataFrame,
) -> None:
    lines = [
        "NONLINEAR TERM DIAGNOSTIC",
        "=========================",
        "",
        "Gradient decomposition:",
        "  linear = S M Z",
        "  cubic  = S Z (Z^T Z)",
        "",
        "Recorded quantities:",
        "  ratio  = ||cubic||_F / ||linear||_F",
        "  cosine = <linear,cubic>_F / (||linear||_F ||cubic||_F)",
        "",
        (
            "Useful regime: checkpoints with validation AUC >= "
            f"{cfg.useful_fraction:.3f} * best validation AUC in each split."
        ),
        "",
    ]

    for dataset in cfg.datasets:
        lines.append(dataset.upper())
        lines.append("-" * len(dataset))

        sel_cos = _metric_row(
            selected_summary,
            dataset,
            "linear_cubic_cosine",
        )
        sel_ratio = _metric_row(
            selected_summary,
            dataset,
            "cubic_to_linear_ratio",
        )
        useful_cos = _metric_row(
            useful_summary,
            dataset,
            "linear_cubic_cosine",
        )
        useful_ratio = _metric_row(
            useful_summary,
            dataset,
            "cubic_to_linear_ratio",
        )

        if sel_cos is not None:
            lines.append(
                "Validation-selected checkpoint cosine: "
                f"{sel_cos['mean']:.6f} ± {sel_cos['sd']:.6f}"
            )
        if sel_ratio is not None:
            lines.append(
                "Validation-selected cubic/linear norm ratio: "
                f"{sel_ratio['mean']:.6f} ± {sel_ratio['sd']:.6f}"
            )
        if useful_cos is not None:
            lines.append(
                "Useful-regime cosine: "
                f"mean={useful_cos['mean']:.6f}, "
                f"min={useful_cos['min']:.6f}, "
                f"q10={useful_cos['q10']:.6f}, "
                f"median={useful_cos['median']:.6f}"
            )
        if useful_ratio is not None:
            lines.append(
                "Useful-regime cubic/linear norm ratio: "
                f"mean={useful_ratio['mean']:.6f}, "
                f"median={useful_ratio['median']:.6f}"
            )

        lines.append("")

    lines.extend([
        "Interpretation",
        "--------------",
        "A large norm ratio means the cubic term is not negligible in magnitude.",
        "A cosine close to one means it is nevertheless strongly aligned with",
        "the linear term, supporting the interpretation that it primarily acts",
        "as a saturation/magnitude correction rather than an unrelated direction.",
        "",
        "Do not describe the cubic term as 'small' unless the ratio actually",
        "supports that statement.  The paper's approximation is directional,",
        "not a claim that the nonlinear term vanishes.",
        "",
    ])

    (out_dir / "README.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg: Config, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2),
        encoding="utf-8",
    )

    frames = []

    for dataset in cfg.datasets:
        data = load_planetoid(
            dataset,
            root=cfg.data_root,
            download=True,
        )

        for fold in range(cfg.fold_start, cfg.fold_start + cfg.folds):
            split_seed = cfg.base_seed + fold
            init_seed = cfg.base_seed + 100_000 + fold

            split = split_edges_kipf(
                data.adjacency,
                seed=split_seed,
            )

            print(
                f"\n{'=' * 78}\n"
                f"{dataset.upper()} | fold={fold} | "
                f"split_seed={split_seed} | init_seed={init_seed}\n"
                f"{'=' * 78}"
            )

            t0 = time.perf_counter()

            frame = run_single(
                data,
                split,
                embedding_dim=cfg.embedding_dim,
                epochs=cfg.epochs,
                lr=cfg.lr,
                eval_every=cfg.eval_every,
                init_seed=init_seed,
                verbose=cfg.verbose,
            )

            frame["dataset"] = dataset
            frame["fold"] = int(fold)
            frame["split_seed"] = int(split_seed)
            frame["init_seed"] = int(init_seed)
            frame["embedding_dim"] = int(cfg.embedding_dim)
            frame["run_seconds"] = float(time.perf_counter() - t0)

            frames.append(frame)

            # Save incrementally so an interrupted run still leaves evidence.
            partial = pd.concat(frames, ignore_index=True)
            partial.to_csv(out_dir / "trajectory.csv", index=False)

    trajectory = pd.concat(frames, ignore_index=True)

    selected = _select_validation_best(trajectory)
    selected_summary = _selected_summary(selected)

    useful = _useful_regime(
        trajectory,
        fraction=cfg.useful_fraction,
    )
    useful_summary = _useful_summary(useful)

    traj_summary = _trajectory_summary(trajectory)

    trajectory.to_csv(out_dir / "trajectory.csv", index=False)
    selected.to_csv(out_dir / "selected_checkpoints.csv", index=False)
    selected_summary.to_csv(out_dir / "selected_summary.csv", index=False)
    useful.to_csv(out_dir / "useful_regime.csv", index=False)
    useful_summary.to_csv(out_dir / "useful_regime_summary.csv", index=False)
    traj_summary.to_csv(out_dir / "trajectory_summary.csv", index=False)

    _plot_band(
        traj_summary,
        mean_col="cosine_mean",
        sd_col="cosine_sd",
        ylabel="cosine(linear term, cubic term)",
        out_base=out_dir / "cosine_trajectory",
        hline=1.0,
    )

    _plot_band(
        traj_summary,
        mean_col="ratio_mean",
        sd_col="ratio_sd",
        ylabel=r"$\|\mathrm{cubic}\|_F / \|\mathrm{linear}\|_F$",
        out_base=out_dir / "norm_ratio_trajectory",
        hline=1.0,
    )

    _write_readme(
        out_dir,
        cfg,
        selected_summary,
        useful_summary,
    )

    print("\nValidation-selected summary:")
    print(selected_summary.to_string(index=False))

    print("\nUseful-regime summary:")
    print(useful_summary.to_string(index=False))

    print(f"\nWrote diagnostics to: {out_dir.resolve()}")


def _csv_tuple(value: str) -> Tuple[str, ...]:
    return tuple(
        x.strip().lower()
        for x in value.split(",")
        if x.strip()
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Measure the magnitude and directional alignment of the cubic "
            "Taylor-GAE term along exact full-batch encoder gradient descent."
        )
    )

    p.add_argument(
        "--datasets",
        default="cora",
        help="Comma-separated datasets: cora,citeseer,pubmed",
    )
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--fold-start", type=int, default=0)
    p.add_argument("--base-seed", type=int, default=10_000)
    p.add_argument("--data-root", default="data/planetoid")

    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=600.0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--useful-fraction", type=float, default=0.99)

    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/nonlinear_term_diagnostics"),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-checkpoint logging.",
    )

    args = p.parse_args()

    datasets = _csv_tuple(args.datasets)
    allowed = {"cora", "citeseer", "pubmed"}
    unknown = set(datasets) - allowed
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")

    if args.folds <= 0:
        raise ValueError("--folds must be positive")
    if args.embedding_dim <= 0:
        raise ValueError("--embedding-dim must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if not (0.0 < args.useful_fraction <= 1.0):
        raise ValueError("--useful-fraction must be in (0,1]")

    cfg = Config(
        datasets=datasets,
        folds=int(args.folds),
        fold_start=int(args.fold_start),
        base_seed=int(args.base_seed),
        data_root=str(args.data_root),
        embedding_dim=int(args.embedding_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        eval_every=int(args.eval_every),
        useful_fraction=float(args.useful_fraction),
        verbose=not bool(args.quiet),
    )

    run(cfg, args.out)


if __name__ == "__main__":
    main()
