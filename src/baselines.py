
"""
Classical topology-only link prediction baselines.

These are intentionally feature-blind, so the same scores are valid reference
rows in both the featureless and feature-available tables.
"""

from __future__ import annotations

import time
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.metrics import roc_auc_score, average_precision_score

from linear_gae_spectral import EdgeSplit


def binary_metrics(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    y = np.r_[np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)]
    score = np.r_[pos, neg]
    return (
        float(roc_auc_score(y, score)),
        float(average_precision_score(y, score)),
    )


def prepare_adjacency(adj_train):
    a = adj_train.astype(np.float64).tocsr().copy()
    a.setdiag(0)
    a.eliminate_zeros()
    a.data[:] = 1.0
    a.sort_indices()
    degree = np.diff(a.indptr).astype(np.float64)
    return a, degree


def _intersection_score_edges(edges, a, weights):
    edges = np.asarray(edges, dtype=np.int64)
    out = np.zeros(len(edges), dtype=np.float64)
    indptr, indices = a.indptr, a.indices

    for i, (u, v) in enumerate(edges):
        nu = indices[indptr[u]:indptr[u+1]]
        nv = indices[indptr[v]:indptr[v+1]]
        pu = pv = 0
        total = 0.0

        while pu < len(nu) and pv < len(nv):
            if nu[pu] == nv[pv]:
                total += weights[nu[pu]]
                pu += 1
                pv += 1
            elif nu[pu] < nv[pv]:
                pu += 1
            else:
                pv += 1

        out[i] = total
    return out


def run_local_baselines(split: EdgeSplit):
    a, degree = prepare_adjacency(split.adj_train)
    rows = []

    weights = {
        "common_neighbors": np.ones_like(degree),
        "adamic_adar": np.where(
            degree > 1,
            1.0 / np.log(np.maximum(degree, 2.0)),
            0.0,
        ),
        "resource_allocation": np.where(
            degree > 0,
            1.0 / np.maximum(degree, 1.0),
            0.0,
        ),
    }

    for method, w in weights.items():
        t0 = time.perf_counter()

        vp = _intersection_score_edges(split.val_edges, a, w)
        vn = _intersection_score_edges(split.val_edges_false, a, w)
        tp = _intersection_score_edges(split.test_edges, a, w)
        tn = _intersection_score_edges(split.test_edges_false, a, w)

        va, vap = binary_metrics(vp, vn)
        ta, tap = binary_metrics(tp, tn)

        runtime = time.perf_counter() - t0

        rows.append({
            "method": method,
            "val_auc": va,
            "val_ap": vap,
            "test_auc": ta,
            "test_ap": tap,
            "selected_hyperparameter": "",
            "execution_device": "cpu",
            "train_seconds": 0.0,
            "validation_seconds": np.nan,
            "test_seconds": np.nan,
            "total_runtime_seconds": runtime,
        })

    return rows


def _edge_targets(edges, reverse=False):
    mapping = {}
    for u, v in np.asarray(edges, dtype=np.int64):
        mapping.setdefault(int(u), set()).add(int(v))
        if reverse:
            mapping.setdefault(int(v), set()).add(int(u))
    return {
        s: np.asarray(sorted(ts), dtype=np.int64)
        for s, ts in mapping.items()
    }


def _solve_requested(lu, n, targets, rhs_scale, chunk_size):
    sources = np.asarray(sorted(targets), dtype=np.int64)
    result = {}

    for start in range(0, len(sources), int(chunk_size)):
        ss = sources[start:start+int(chunk_size)]
        b = np.zeros((n, len(ss)), dtype=np.float64)
        b[ss, np.arange(len(ss))] = float(rhs_scale)
        x = lu.solve(b)

        for j, s in enumerate(ss):
            ts = targets[int(s)]
            vals = x[ts, j]
            for t, value in zip(ts, vals):
                result[(int(s), int(t))] = float(value)

    return result


def _scores(edges, table, symmetric=False):
    edges = np.asarray(edges, dtype=np.int64)
    if not symmetric:
        return np.asarray(
            [table[(int(u), int(v))] for u, v in edges],
            dtype=np.float64,
        )

    return 0.5 * np.asarray(
        [
            table[(int(u), int(v))] + table[(int(v), int(u))]
            for u, v in edges
        ],
        dtype=np.float64,
    )


def run_katz_strict(
    split: EdgeSplit,
    gamma_values: Sequence[float],
    chunk_size: int = 128,
    verbose: bool = False,
):
    a, _ = prepare_adjacency(split.adj_train)
    n = a.shape[0]
    eye = sp.eye(n, dtype=np.float64, format="csc")

    total_start = time.perf_counter()

    rho = float(
        spla.eigsh(
            a,
            k=1,
            which="LA",
            return_eigenvectors=False,
            tol=1e-7,
        )[0]
    )

    val_edges = np.vstack([split.val_edges, split.val_edges_false])
    val_targets = _edge_targets(val_edges, reverse=False)

    tuning = []
    best = None

    for gamma in gamma_values:
        gamma = float(gamma)
        beta = gamma / rho

        t0 = time.perf_counter()
        lu = spla.splu(eye - beta * a.tocsc())
        table = _solve_requested(lu, n, val_targets, 1.0, chunk_size)

        vp = _scores(split.val_edges, table)
        vn = _scores(split.val_edges_false, table)
        va, vap = binary_metrics(vp, vn)

        row = {
            "candidate": f"gamma={gamma}",
            "gamma": gamma,
            "beta": beta,
            "val_auc": va,
            "val_ap": vap,
            "test_auc": np.nan,
            "test_ap": np.nan,
            "candidate_runtime_seconds": time.perf_counter() - t0,
        }
        tuning.append(row)

        key = (va, vap, -gamma)
        if best is None or key > best[0]:
            best = (key, row)

        if verbose:
            print(f"Katz gamma={gamma:.4g} val_auc={va:.4f}")

    chosen = best[1]
    beta = float(chosen["beta"])

    test_start = time.perf_counter()
    lu = spla.splu(eye - beta * a.tocsc())
    test_edges = np.vstack([split.test_edges, split.test_edges_false])
    test_targets = _edge_targets(test_edges, reverse=False)
    table = _solve_requested(lu, n, test_targets, 1.0, chunk_size)

    tp = _scores(split.test_edges, table)
    tn = _scores(split.test_edges_false, table)
    ta, tap = binary_metrics(tp, tn)
    test_seconds = time.perf_counter() - test_start

    selected = {
        "method": "katz",
        "val_auc": float(chosen["val_auc"]),
        "val_ap": float(chosen["val_ap"]),
        "test_auc": ta,
        "test_ap": tap,
        "selected_gamma": float(chosen["gamma"]),
        "selected_beta": float(chosen["beta"]),
        "rho_A": rho,
        "selected_hyperparameter": f"gamma={chosen['gamma']}",
        "execution_device": "cpu",
        "train_seconds": 0.0,
        "validation_seconds": np.nan,
        "test_seconds": test_seconds,
        "total_runtime_seconds": time.perf_counter() - total_start,
    }

    return selected, tuning


def run_ppr_strict(
    split: EdgeSplit,
    alpha_values: Sequence[float],
    chunk_size: int = 128,
    verbose: bool = False,
):
    a, degree = prepare_adjacency(split.adj_train)
    n = a.shape[0]

    inv_degree = np.zeros_like(degree)
    nz = degree > 0
    inv_degree[nz] = 1.0 / degree[nz]

    p = sp.diags(inv_degree) @ a
    pt = p.T.tocsc()
    eye = sp.eye(n, dtype=np.float64, format="csc")

    total_start = time.perf_counter()

    val_edges = np.vstack([split.val_edges, split.val_edges_false])
    val_targets = _edge_targets(val_edges, reverse=True)

    tuning = []
    best = None

    for alpha in alpha_values:
        alpha = float(alpha)

        t0 = time.perf_counter()
        lu = spla.splu(eye - alpha * pt)
        table = _solve_requested(
            lu, n, val_targets, 1.0-alpha, chunk_size
        )

        vp = _scores(split.val_edges, table, symmetric=True)
        vn = _scores(split.val_edges_false, table, symmetric=True)
        va, vap = binary_metrics(vp, vn)

        row = {
            "candidate": f"alpha={alpha}",
            "alpha": alpha,
            "val_auc": va,
            "val_ap": vap,
            "test_auc": np.nan,
            "test_ap": np.nan,
            "candidate_runtime_seconds": time.perf_counter() - t0,
        }
        tuning.append(row)

        key = (va, vap, -alpha)
        if best is None or key > best[0]:
            best = (key, row)

        if verbose:
            print(f"PPR alpha={alpha:.3f} val_auc={va:.4f}")

    chosen = best[1]
    alpha = float(chosen["alpha"])

    test_start = time.perf_counter()
    lu = spla.splu(eye - alpha * pt)
    test_edges = np.vstack([split.test_edges, split.test_edges_false])
    test_targets = _edge_targets(test_edges, reverse=True)
    table = _solve_requested(
        lu, n, test_targets, 1.0-alpha, chunk_size
    )

    tp = _scores(split.test_edges, table, symmetric=True)
    tn = _scores(split.test_edges_false, table, symmetric=True)
    ta, tap = binary_metrics(tp, tn)
    test_seconds = time.perf_counter() - test_start

    selected = {
        "method": "ppr",
        "val_auc": float(chosen["val_auc"]),
        "val_ap": float(chosen["val_ap"]),
        "test_auc": ta,
        "test_ap": tap,
        "selected_alpha": alpha,
        "selected_hyperparameter": f"alpha={alpha}",
        "execution_device": "cpu",
        "train_seconds": 0.0,
        "validation_seconds": np.nan,
        "test_seconds": test_seconds,
        "total_runtime_seconds": time.perf_counter() - total_start,
    }

    return selected, tuning
