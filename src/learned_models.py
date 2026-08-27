
"""
Learned graph-autoencoder baselines for the final paper benchmark.

Architectures follow the Salha / Kipf conventions:

Featureless Linear AE:
    Z = S W

Featureful Linear AE:
    Z = S F W

2-layer GCN AE:
    H1 = ReLU(S X W0)
    Z  = S H1 W1

3-layer GCN AE:
    H1 = ReLU(S X W0)
    H2 = ReLU(S H1 W1)
    Z  = S H2 W2

with X = I in the featureless setting and X = F in the featureful setting.

All use the weighted full-adjacency BCE / inner-product decoder from the
original GAE implementation. The loss is evaluated exactly in row blocks
for Pubmed to avoid materializing n x n logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from linear_gae_spectral import (
    PlanetoidData,
    EdgeSplit,
    normalize_adjacency,
    scipy_to_torch_sparse,
    paper_bce_constants,
    _dense_label_block,
    evaluate_embeddings,
)
from feature_kernel import prepare_features


def resolve_torch_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError("device must be one of: auto, cpu, cuda")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _xavier(shape, device):
    p = nn.Parameter(torch.empty(*shape, dtype=torch.float32, device=device))
    nn.init.xavier_uniform_(p)
    return p


class FinalGAE(nn.Module):
    """
    Unified Linear / 2-layer GCN / 3-layer GCN encoder.

    `features` are passed separately to encode(). If features=None, the
    featureless X=I specialization is used without constructing I.
    """

    def __init__(
        self,
        n_nodes: int,
        feature_dim: Optional[int],
        model_kind: str,
        hidden_dim: int,
        latent_dim: int,
        device: torch.device,
    ):
        super().__init__()
        self.model_kind = model_kind
        self.feature_dim = feature_dim
        self.n_nodes = int(n_nodes)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)

        input_dim = self.n_nodes if feature_dim is None else int(feature_dim)

        if model_kind == "linear":
            self.W0 = _xavier((input_dim, latent_dim), device)

        elif model_kind == "gcn2":
            self.W0 = _xavier((input_dim, hidden_dim), device)
            self.W1 = _xavier((hidden_dim, latent_dim), device)

        elif model_kind == "gcn3":
            self.W0 = _xavier((input_dim, hidden_dim), device)
            self.W1 = _xavier((hidden_dim, hidden_dim), device)
            self.W2 = _xavier((hidden_dim, latent_dim), device)

        else:
            raise ValueError("model_kind must be linear, gcn2, or gcn3")

    @staticmethod
    def _input_times_weight(
        features: Optional[torch.Tensor],
        weight: torch.Tensor,
    ) -> torch.Tensor:
        # X=I => XW = W.
        if features is None:
            return weight
        return torch.sparse.mm(features, weight)

    def encode(
        self,
        s: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        xw = self._input_times_weight(features, self.W0)

        if self.model_kind == "linear":
            return torch.sparse.mm(s, xw)

        h1 = torch.relu(torch.sparse.mm(s, xw))
        z_or_h2 = torch.sparse.mm(s, h1 @ self.W1)

        if self.model_kind == "gcn2":
            return z_or_h2

        h2 = torch.relu(z_or_h2)
        return torch.sparse.mm(s, h2 @ self.W2)


def _choose_block_size(dataset: str, n: int, requested: Optional[int]) -> int:
    if requested is not None:
        return min(int(requested), n)
    # Full matrix is convenient on Cora/Citeseer. Pubmed remains exact but
    # is evaluated in blocks because an n x n dense tensor is ridiculous.
    if dataset == "pubmed":
        return 256
    return n


def train_validation_selected_gae(
    data: PlanetoidData,
    split: EdgeSplit,
    setting: str,
    model_kind: str,
    epochs: int = 250,
    lr: float = 0.01,
    hidden_dim: int = 32,
    latent_dim: int = 16,
    init_seed: int = 0,
    requested_device: str = "auto",
    feature_mode: str = "raw",
    block_size: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, object], list[dict]]:
    """
    Train all `epochs`; evaluate validation AUC/AP after EVERY epoch; select
    best epoch by validation AUC (then AP, then earlier epoch); evaluate test
    edges exactly once at the selected checkpoint.

    Timings:
      train_seconds       forward/backward/optimizer only
      validation_seconds  validation scoring / checkpoint-selection overhead
      test_seconds        final selected test scoring
      total_runtime_seconds everything inside this function after setup starts
    """
    if setting not in {"featureless", "features"}:
        raise ValueError("setting must be featureless or features")

    dev = resolve_torch_device(requested_device)
    _seed_all(init_seed)

    total_start = time.perf_counter()

    s_sp = normalize_adjacency(split.adj_train)
    s = scipy_to_torch_sparse(s_sp, dev)

    features_t = None
    feature_dim = None
    feature_nnz = 0

    if setting == "features":
        f_sp = prepare_features(data.features, mode=feature_mode).astype(np.float32)
        feature_dim = int(f_sp.shape[1])
        feature_nnz = int(f_sp.nnz)
        features_t = scipy_to_torch_sparse(f_sp, dev)

    net = FinalGAE(
        n_nodes=data.n,
        feature_dim=feature_dim,
        model_kind=model_kind,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        device=dev,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))
    pos_weight, norm = paper_bce_constants(split.adj_train)
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=dev)

    n = data.n
    bs = _choose_block_size(data.name, n, block_size)
    starts = [0] if bs >= n else list(range(0, n, bs))

    best_key = None
    best_state = None
    best_row = None
    history = []

    train_seconds = 0.0
    validation_seconds = 0.0

    for epoch in range(1, int(epochs) + 1):
        net.train()

        _sync(dev)
        train_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        if bs >= n:
            # Small graphs: ordinary full-matrix loss/backpropagation.
            z = net.encode(s, features_t)
            logits = z @ z.T
            target = _dense_label_block(split.adj_train, 0, n, dev)

            loss = norm * F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=pos_weight_t,
                reduction="mean",
            )
            loss.backward()
            total_loss = float(loss.detach().cpu())

        else:
            # Pubmed-sized graphs: exact blockwise decoder gradient without
            # retaining every block's autograd graph.
            #
            # 1) Compute the current embedding values without an encoder graph.
            # 2) Treat Z as a leaf and accumulate dL/dZ over decoder blocks.
            # 3) Recompute Z with the encoder graph and backpropagate that one
            #    accumulated dL/dZ through the encoder.
            #
            # This is mathematically the same gradient as the full n x n BCE
            # but avoids an impressive and unnecessary GPU-memory bonfire.
            with torch.no_grad():
                z_value = net.encode(s, features_t)

            z_leaf = z_value.detach().requires_grad_(True)

            for start in starts:
                end = min(start + bs, n)
                logits = z_leaf[start:end] @ z_leaf.T
                target = _dense_label_block(
                    split.adj_train,
                    start,
                    end,
                    dev,
                )

                block_loss = (
                    norm / float(n * n)
                ) * F.binary_cross_entropy_with_logits(
                    logits,
                    target,
                    pos_weight=pos_weight_t,
                    reduction="sum",
                )

                block_loss.backward()
                total_loss += float(block_loss.detach().cpu())

            grad_z = z_leaf.grad.detach()

            z_graph = net.encode(s, features_t)
            z_graph.backward(grad_z)

        optimizer.step()

        _sync(dev)
        train_seconds += time.perf_counter() - train_start

        _sync(dev)
        val_start = time.perf_counter()

        net.eval()
        with torch.no_grad():
            z_eval = net.encode(s, features_t)
            val_auc, val_ap = evaluate_embeddings(
                z_eval,
                split.val_edges,
                split.val_edges_false,
            )

        _sync(dev)
        validation_seconds += time.perf_counter() - val_start

        row = {
            "epoch": int(epoch),
            "training_loss": float(total_loss),
            "val_auc": float(val_auc),
            "val_ap": float(val_ap),
            "test_auc": np.nan,
            "test_ap": np.nan,
        }
        history.append(row)

        # Max val AUC, then max val AP, then earliest epoch.
        key = (float(val_auc), float(val_ap), -int(epoch))
        if best_key is None or key > best_key:
            best_key = key
            best_row = row.copy()
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in net.state_dict().items()
            }

        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
            print(
                f"{data.name:8s} {setting:11s} {model_kind:6s} "
                f"epoch={epoch:03d} loss={total_loss:.6f} "
                f"val_auc={val_auc:.4f}"
            )

    # Restore validation-selected checkpoint.
    net.load_state_dict(best_state)
    net.to(dev)
    net.eval()

    _sync(dev)
    test_start = time.perf_counter()
    with torch.no_grad():
        z_best = net.encode(s, features_t)
        test_auc, test_ap = evaluate_embeddings(
            z_best,
            split.test_edges,
            split.test_edges_false,
        )
    _sync(dev)
    test_seconds = time.perf_counter() - test_start

    total_runtime_seconds = time.perf_counter() - total_start

    selected = {
        "val_auc": float(best_row["val_auc"]),
        "val_ap": float(best_row["val_ap"]),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "selected_epoch": int(best_row["epoch"]),
        "selected_hyperparameter": f"epoch={best_row['epoch']}",
        "epochs_run": int(epochs),
        "learning_rate": float(lr),
        "hidden_dim": int(hidden_dim) if model_kind != "linear" else np.nan,
        "latent_dim": int(latent_dim),
        "feature_dim": int(feature_dim) if feature_dim is not None else np.nan,
        "feature_nnz": int(feature_nnz) if feature_dim is not None else np.nan,
        "feature_mode": feature_mode if setting == "features" else "identity",
        "execution_device": str(dev),
        "train_seconds": float(train_seconds),
        "validation_seconds": float(validation_seconds),
        "test_seconds": float(test_seconds),
        "total_runtime_seconds": float(total_runtime_seconds),
    }

    return selected, history
