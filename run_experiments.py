#!/usr/bin/env python3
"""Run the final graph-autoencoder benchmark used in the paper experiments.

Default experiment
------------------
Datasets: Cora, Citeseer, Pubmed
Splits:   10 random edge splits, seeds 10000..10009
Settings: featureless and raw Planetoid node features
Methods:
    linear_ae
    gcn_ae_2
    gcn_ae_3
    linearized_gae_kernel

Learned models are trained for 250 epochs with validation evaluated after every
training epoch.  The test set is scored only once, after the validation-best
epoch has been selected.

The proposed kernel uses only the positive retained spectrum and the normalized
step

    alpha = gamma / lambda_max_positive,

with gamma in {0.01, 0.02, 0.05, 0.10} and k=0,5,...,300.  Validation selects
(gamma, k), with AP and then smaller spectral time used only as tie-breakers.
The test set is scored once at that selected configuration.

The harness is resumable at (dataset, fold, setting, method) granularity.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence, Tuple

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

from baselines import run_katz_strict, run_local_baselines, run_ppr_strict
from deterministic_kernel import (
    compute_extreme_sms_eigenpairs,
    deterministic_kernel_edge_scores,
)
from feature_kernel import compute_feature_spectrum, feature_kernel_edge_scores
from learned_models import resolve_torch_device, train_validation_selected_gae
from linear_gae_spectral import load_planetoid, split_edges_kipf

DATASETS = ("cora", "citeseer", "pubmed")
SETTINGS = ("featureless", "features")
METHODS = ("linear_ae", "gcn_ae_2", "gcn_ae_3", "linearized_gae_kernel")
CLASSICAL_METHODS = ("ppr", "katz", "common_neighbors", "adamic_adar", "resource_allocation")
ALL_METHODS = METHODS + CLASSICAL_METHODS
LEARNED_METHODS = ("linear_ae", "gcn_ae_2", "gcn_ae_3")
LEARNED_INDEX = {m: i for i, m in enumerate(LEARNED_METHODS)}


@dataclass(frozen=True)
class Config:
    datasets: Tuple[str, ...] = DATASETS
    settings: Tuple[str, ...] = SETTINGS
    methods: Tuple[str, ...] = METHODS

    folds: int = 10
    fold_start: int = 0
    base_seed: int = 10_000
    data_root: str = "data/planetoid"

    # Learned autoencoders: final protocol.
    epochs: int = 250
    learning_rate: float = 0.01
    latent_dim: int = 16
    hidden_dim: int = 32
    feature_mode: str = "raw"

    # Runtime.
    device: str = "cpu"
    threads: int = 0
    pubmed_block_size: int = 256

    # Normalized positive-spectrum kernel: final protocol.
    kernel_gammas: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
    kernel_k_values: Tuple[int, ...] = tuple(range(0, 301, 5))
    kernel_positive_rank_featureless: int = 256
    kernel_positive_rank_features: int = 256
    kernel_eig_tol: float = 1e-6

    # Optional classical baseline grids (not shown in the main paper table).
    ppr_alphas: Tuple[float, ...] = (0.50, 0.70, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
    katz_gammas: Tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
    solve_chunk_size: int = 128

    verbose: bool = True


def binary_metrics(pos_scores, neg_scores):
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    y = np.r_[np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)]
    score = np.r_[pos, neg]
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def replace_group_csv(path: Path, frame: pd.DataFrame, group_cols: Sequence[str]):
    """Atomically replace one checkpoint group in a CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        if len(old) and all(c in old.columns for c in group_cols):
            mask = np.ones(len(old), dtype=bool)
            for c in group_cols:
                mask &= old[c].astype(str).to_numpy() == str(frame.iloc[0][c])
            old = old.loc[~mask]
        combined = pd.concat([old, frame], ignore_index=True, sort=False)
    else:
        combined = frame.copy()

    tmp = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(path)


def completed_keys(path: Path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(zip(df.dataset.astype(str), df.fold.astype(int),
                   df.setting.astype(str), df.method.astype(str)))


def environment_info(cfg: Config):
    resolved = resolve_torch_device(cfg.device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requested_device": cfg.device,
        "resolved_learned_model_device": str(resolved),
        "kernel_device": "cpu (SciPy)",
    }


def annotate_tuning(tuning, common):
    frame = pd.DataFrame(tuning)
    if frame.empty:
        return frame
    for c, v in common.items():
        frame[c] = v
    return frame


def run_learned(data, split, cfg: Config, setting: str, method: str, init_seed: int):
    kind = {"linear_ae": "linear", "gcn_ae_2": "gcn2", "gcn_ae_3": "gcn3"}[method]
    block_size = cfg.pubmed_block_size if data.name == "pubmed" else None
    return train_validation_selected_gae(
        data=data,
        split=split,
        setting=setting,
        model_kind=kind,
        epochs=cfg.epochs,
        lr=cfg.learning_rate,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        init_seed=init_seed,
        requested_device=cfg.device,
        feature_mode=cfg.feature_mode,
        block_size=block_size,
        verbose=cfg.verbose,
    )


def run_kernel(data, split, cfg: Config, setting: str, eig_seed: int):
    total_start = time.perf_counter()
    spectrum_start = time.perf_counter()

    if setting == "featureless":
        spec = compute_extreme_sms_eigenpairs(
            split=split,
            positive_rank=cfg.kernel_positive_rank_featureless,
            negative_rank=0,
            seed=eig_seed,
            eig_tol=cfg.kernel_eig_tol,
        )
        actual_pos = len(spec["positive_eigenvalues"])
        requested_pos = cfg.kernel_positive_rank_featureless
        if actual_pos != requested_pos:
            raise RuntimeError(
                f"Featureless eigensolver returned {actual_pos} positive eigenpairs; "
                f"requested {requested_pos}. Refusing a partial spectrum."
            )
        pos = np.asarray(spec["positive_eigenvalues"], dtype=np.float64)
        lambda_max = float(np.max(pos))
        feature_dim = np.nan
        feature_nnz = np.nan
    elif setting == "features":
        spec = compute_feature_spectrum(
            features=data.features,
            split=split,
            positive_rank=cfg.kernel_positive_rank_features,
            negative_rank=0,
            feature_mode=cfg.feature_mode,
            seed=eig_seed,
            eig_tol=cfg.kernel_eig_tol,
        )
        labels = np.asarray(spec["tail_labels"])
        actual_pos = int(np.sum(labels == "positive"))
        requested_pos = cfg.kernel_positive_rank_features
        if actual_pos != requested_pos:
            raise RuntimeError(
                f"Feature eigensolver returned {actual_pos} positive eigenpairs; "
                f"requested {requested_pos}."
            )
        lambda_max = float(spec["lambda_max_positive"])
        feature_dim = int(spec["feature_dim"])
        feature_nnz = int(spec["feature_nnz"])
    else:
        raise ValueError(setting)

    if not np.isfinite(lambda_max) or lambda_max <= 0:
        raise RuntimeError(f"Invalid lambda_max_positive={lambda_max}")

    spectrum_seconds = time.perf_counter() - spectrum_start
    tuning = []
    best = None
    validation_start = time.perf_counter()

    for gamma in cfg.kernel_gammas:
        gamma = float(gamma)
        alpha = gamma / lambda_max
        for k in cfg.kernel_k_values:
            k = int(k)

            if setting == "featureless":
                vp = deterministic_kernel_edge_scores(
                    split.val_edges, spec["s2"], spec["sv"],
                    spec["eigenvalues"], alpha, k,
                )
                vn = deterministic_kernel_edge_scores(
                    split.val_edges_false, spec["s2"], spec["sv"],
                    spec["eigenvalues"], alpha, k,
                )
            else:
                vp = feature_kernel_edge_scores(split.val_edges, spec, alpha, k)
                vn = feature_kernel_edge_scores(split.val_edges_false, spec, alpha, k)

            val_auc, val_ap = binary_metrics(vp, vn)
            tau = 2.0 * k * gamma
            row = {
                "candidate": f"gamma={gamma};k={k};tau={tau:.6g}",
                "gamma": gamma,
                "k": k,
                "spectral_time": tau,
                "alpha": alpha,
                "lambda_max_positive": lambda_max,
                "positive_rank": requested_pos,
                "negative_rank": 0,
                "actual_positive_rank": actual_pos,
                "actual_negative_rank": 0,
                "feature_dim": feature_dim,
                "val_auc": val_auc,
                "val_ap": val_ap,
                "test_auc": np.nan,
                "test_ap": np.nan,
            }
            tuning.append(row)

            # Same final selection rule used in the normalized benchmark:
            # validation AUC, validation AP, then less spectral evolution.
            key = (val_auc, val_ap, -tau, -gamma, -k)
            if best is None or key > best[0]:
                best = (key, row)

    validation_seconds = time.perf_counter() - validation_start
    chosen = best[1]
    gamma = float(chosen["gamma"])
    k = int(chosen["k"])
    alpha = float(chosen["alpha"])

    test_start = time.perf_counter()
    if setting == "featureless":
        tp = deterministic_kernel_edge_scores(
            split.test_edges, spec["s2"], spec["sv"],
            spec["eigenvalues"], alpha, k,
        )
        tn = deterministic_kernel_edge_scores(
            split.test_edges_false, spec["s2"], spec["sv"],
            spec["eigenvalues"], alpha, k,
        )
    else:
        tp = feature_kernel_edge_scores(split.test_edges, spec, alpha, k)
        tn = feature_kernel_edge_scores(split.test_edges_false, spec, alpha, k)
    test_auc, test_ap = binary_metrics(tp, tn)
    test_seconds = time.perf_counter() - test_start

    selected = {
        "val_auc": float(chosen["val_auc"]),
        "val_ap": float(chosen["val_ap"]),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "selected_gamma": gamma,
        "selected_k": k,
        "selected_spectral_time": float(chosen["spectral_time"]),
        "selected_hyperparameter": f"gamma={gamma};k={k};tau={chosen['spectral_time']:.6g}",
        "alpha": alpha,
        "lambda_max_positive": lambda_max,
        "positive_rank": requested_pos,
        "negative_rank": 0,
        "actual_positive_rank": actual_pos,
        "actual_negative_rank": 0,
        "feature_dim": feature_dim,
        "feature_nnz": feature_nnz,
        "feature_mode": cfg.feature_mode if setting == "features" else "identity",
        "execution_device": "cpu",
        "train_seconds": 0.0,
        "spectrum_seconds": spectrum_seconds,
        "validation_seconds": validation_seconds,
        "test_seconds": test_seconds,
        "total_runtime_seconds": time.perf_counter() - total_start,
    }
    return selected, tuning


def run(cfg: Config, out_dir: Path, force=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / "selected_results.csv"
    tuning_path = out_dir / "tuning_results.csv"

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (out_dir / "environment.json").write_text(
        json.dumps(environment_info(cfg), indent=2), encoding="utf-8"
    )

    done = set() if force else completed_keys(selected_path)

    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    limiter = (
        threadpool_limits(limits=cfg.threads)
        if cfg.threads > 0 and threadpool_limits is not None
        else None
    )
    if limiter is not None:
        limiter.__enter__()

    try:
        for dataset in cfg.datasets:
            data = load_planetoid(dataset, root=cfg.data_root, download=True)
            print(f"\n{'='*84}\n{dataset.upper()} | n={data.n} | edges={data.m}\n{'='*84}")

            for fold in range(cfg.fold_start, cfg.fold_start + cfg.folds):
                split_seed = cfg.base_seed + fold
                split = split_edges_kipf(data.adjacency, seed=split_seed)
                print(f"\nFOLD {fold} | split seed {split_seed}")

                # Topology-only classical baselines are identical in both feature
                # settings, so compute each once per split when requested.
                classical_cache = {}
                requested_classical = set(cfg.methods) & set(CLASSICAL_METHODS)
                if requested_classical:
                    if requested_classical & {"common_neighbors", "adamic_adar", "resource_allocation"}:
                        for row in run_local_baselines(split):
                            classical_cache[row["method"]] = (dict(row), [])
                    if "katz" in requested_classical:
                        classical_cache["katz"] = run_katz_strict(
                            split, cfg.katz_gammas, cfg.solve_chunk_size, cfg.verbose
                        )
                    if "ppr" in requested_classical:
                        classical_cache["ppr"] = run_ppr_strict(
                            split, cfg.ppr_alphas, cfg.solve_chunk_size, cfg.verbose
                        )

                for setting in cfg.settings:
                    for method in cfg.methods:
                        key = (dataset, fold, setting, method)
                        if key in done and not force:
                            print(f"[skip] {setting:11s} / {method}")
                            continue

                        print(f"[run ] {setting:11s} / {method}")
                        common = {
                            "dataset": dataset,
                            "fold": int(fold),
                            "split_seed": int(split_seed),
                            "setting": setting,
                            "method": method,
                        }

                        if method in CLASSICAL_METHODS:
                            selected, tuning = classical_cache[method]
                            selected = dict(selected)
                            tuning = [dict(row) for row in tuning]
                            selected["shared_topology_only_baseline"] = True
                        elif method in LEARNED_METHODS:
                            init_seed = (
                                cfg.base_seed + 100_000 + 10_000 * fold
                                + 101 * LEARNED_INDEX[method]
                                + (1 if setting == "features" else 0)
                            )
                            selected, tuning = run_learned(
                                data, split, cfg, setting, method, init_seed
                            )
                            selected["init_seed"] = init_seed
                        elif method == "linearized_gae_kernel":
                            eig_seed = (
                                cfg.base_seed + 500_000 + 1000 * fold
                                + (1 if setting == "features" else 0)
                            )
                            selected, tuning = run_kernel(data, split, cfg, setting, eig_seed)
                            selected["eig_seed"] = eig_seed
                        else:  # pragma: no cover
                            raise ValueError(method)

                        sf = pd.DataFrame([{**common, **selected}])
                        tf = annotate_tuning(tuning, common)
                        if len(tf):
                            replace_group_csv(
                                tuning_path, tf,
                                ("dataset", "fold", "setting", "method"),
                            )
                        replace_group_csv(
                            selected_path, sf,
                            ("dataset", "fold", "setting", "method"),
                        )
                        done.add(key)

                        print(
                            f"      test AUC={selected['test_auc']:.4f} "
                            f"AP={selected['test_ap']:.4f} "
                            f"time={selected['total_runtime_seconds']:.2f}s"
                        )
    finally:
        if limiter is not None:
            limiter.__exit__(None, None, None)

    print(f"\nFinished. Results: {out_dir.resolve()}")


def csv_tuple(value, cast=str):
    return tuple(cast(x.strip()) for x in value.split(",") if x.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/paper"))
    p.add_argument("--data-root", default="data/planetoid")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--settings", default=",".join(SETTINGS))
    p.add_argument("--methods", default=",".join(METHODS),
                   help="Comma-separated methods. Default is the four main paper methods.")
    p.add_argument("--include-classical", action="store_true",
                   help="Also run PPR, Katz, Common Neighbors, Adamic-Adar, and Resource Allocation.")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--fold-start", type=int, default=0)
    p.add_argument("--base-seed", type=int, default=10_000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--pubmed-block-size", type=int, default=256)
    p.add_argument("--gammas", default="0.01,0.02,0.05,0.10")
    p.add_argument("--k-max", type=int, default=300)
    p.add_argument("--k-step", type=int, default=5)
    p.add_argument("--positive-rank", type=int, default=256)
    p.add_argument("--force", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    datasets = csv_tuple(args.datasets)
    settings = csv_tuple(args.settings)
    methods = csv_tuple(args.methods)
    if args.include_classical:
        methods = tuple(dict.fromkeys(methods + CLASSICAL_METHODS))
    for values, allowed, label in [
        (datasets, DATASETS, "datasets"),
        (settings, SETTINGS, "settings"),
        (methods, ALL_METHODS, "methods"),
    ]:
        bad = sorted(set(values) - set(allowed))
        if bad:
            raise ValueError(f"Unknown {label}: {bad}")

    cfg = Config(
        datasets=datasets,
        settings=settings,
        methods=methods,
        folds=args.folds,
        fold_start=args.fold_start,
        base_seed=args.base_seed,
        data_root=args.data_root,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
        threads=args.threads,
        pubmed_block_size=args.pubmed_block_size,
        kernel_gammas=csv_tuple(args.gammas, float),
        kernel_k_values=tuple(range(0, args.k_max + 1, args.k_step)),
        kernel_positive_rank_featureless=args.positive_rank,
        kernel_positive_rank_features=args.positive_rank,
        verbose=not args.quiet,
    )
    run(cfg, args.out, force=args.force)


if __name__ == "__main__":
    main()
