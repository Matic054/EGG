
"""
Minimal, notebook-friendly reproduction harness for:

1) Salha, Hennequin & Vazirgiannis (2020),
   "Simple and Effective Graph Autoencoders with One-Hop Linear Models"
2) Kipf & Welling (2016),
   "Variational Graph Auto-Encoders" -- deterministic featureless GAE baseline.

The default settings intentionally follow the old TensorFlow GAE implementation:
- undirected graph, no original self-loops
- 5% validation positives, 10% test positives
- equal number of sampled negative pairs
- A_tilde = D^{-1/2} (A_train + I) D^{-1/2}
- 200 epochs, Adam(lr=0.01), latent dimension 16
- weighted full-adjacency BCE on logits ZZ^T
- training labels A_train + I
- the original Kipf pos_weight/norm convention

The blockwise loss computes the SAME dense weighted BCE objective without
materializing the full n x n logits matrix at once. This matters for Pubmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import pickle
import random
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.metrics import average_precision_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F


DATASETS = ("cora", "citeseer", "pubmed")

# Salha et al. repository first because that is the implementation we are
# reproducing. Their README says these preprocessed files come from tkipf/gae.
DATA_BASE_URLS = (
    "https://raw.githubusercontent.com/deezer/linear_graph_autoencoders/master/data/",
    "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/",
)

REQUIRED_PLANETOID_FILES = ("x", "tx", "allx", "graph")

PAPER_TARGETS = pd.DataFrame(
    [
        # Salha et al. 2020, featureless linear AE, Table 1.
        ("salha2020", "linear_ae", "cora",     0.8319, 0.8757, 0.0113, 0.0095),
        ("salha2020", "linear_ae", "citeseer", 0.7706, 0.8305, 0.0181, 0.0125),
        ("salha2020", "linear_ae", "pubmed",   0.8185, 0.8754, 0.0032, 0.0028),

        # Salha et al. 2020, featureless 2-layer GCN AE, Table 1.
        ("salha2020", "gcn_ae", "cora",        0.8479, 0.8845, 0.0110, 0.0082),
        ("salha2020", "gcn_ae", "citeseer",    0.7825, 0.8379, 0.0169, 0.0124),
        ("salha2020", "gcn_ae", "pubmed",      0.8251, 0.8742, 0.0064, 0.0038),

        # Kipf & Welling 2016, featureless deterministic GAE (GAE*), Table 1.
        # Their table describes the uncertainty as standard error over 10
        # random initializations on fixed dataset splits. We keep only means
        # here because the displayed uncertainty scaling is not directly
        # comparable to Salha's 100-split standard deviations.
        ("kipf2016", "gcn_ae", "cora",         0.8430, 0.8810, np.nan, np.nan),
        ("kipf2016", "gcn_ae", "citeseer",     0.7870, 0.8410, np.nan, np.nan),
        ("kipf2016", "gcn_ae", "pubmed",       0.8220, 0.8740, np.nan, np.nan),
    ],
    columns=["paper", "model", "dataset", "auc", "ap", "auc_sd", "ap_sd"],
)


@dataclass
class PlanetoidData:
    name: str
    adjacency: sp.csr_matrix
    features: sp.csr_matrix

    @property
    def n(self) -> int:
        return self.adjacency.shape[0]

    @property
    def m(self) -> int:
        # adjacency is symmetric and loop-free
        return int(self.adjacency.nnz // 2)


@dataclass
class EdgeSplit:
    adj_train: sp.csr_matrix
    train_edges: np.ndarray
    val_edges: np.ndarray
    val_edges_false: np.ndarray
    test_edges: np.ndarray
    test_edges_false: np.ndarray
    seed: int


@dataclass
class TrainResult:
    model: str
    objective: str
    dataset: str
    split_seed: int
    init_seed: int
    auc: float
    ap: float
    val_auc: float
    val_ap: float
    final_loss: float
    epochs: int
    embedding_dim: int
    device: str
    embeddings: Optional[np.ndarray] = None
    history: Optional[pd.DataFrame] = None



@dataclass
class SpectralResult:
    dataset: str
    split_seed: int
    auc: float
    ap: float
    val_auc: float
    val_ap: float
    final_loss: float
    embedding_dim_requested: int
    embedding_dim_used: int
    eigenvalues: np.ndarray
    solver: str
    encoder_constraint: str
    encoder_residual: float
    embeddings: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def download_planetoid_dataset(
    dataset: str,
    root: str | Path = "data/planetoid",
    force: bool = False,
) -> Path:
    """
    Download only the Planetoid files needed for adjacency + feature loading.

    Tries the Salha et al. repository first, then the original Planetoid repo.
    """
    dataset = dataset.lower()
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    suffixes = list(REQUIRED_PLANETOID_FILES) + ["test.index"]
    for suffix in suffixes:
        filename = f"ind.{dataset}.{suffix}"
        dest = root / filename
        if dest.exists() and not force:
            continue

        errors = []
        for base in DATA_BASE_URLS:
            url = base + filename
            try:
                _download(url, dest)
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError(
                f"Could not download {filename}. Tried:\n" + "\n".join(errors)
            )
    return root


def _parse_index_file(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        return np.asarray([int(line.strip()) for line in f if line.strip()], dtype=np.int64)


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def _adjacency_from_graph_dict(graph: dict, n: int) -> sp.csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    max_node = n - 1

    for u, nbrs in graph.items():
        u = int(u)
        max_node = max(max_node, u)
        for v in nbrs:
            v = int(v)
            max_node = max(max_node, v)
            if u != v:
                rows.append(u)
                cols.append(v)

    n = max(n, max_node + 1)
    if rows:
        data = np.ones(len(rows), dtype=np.float32)
        adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
        # Match the undirected treatment used in the GAE experiments.
        adj = adj.maximum(adj.T)
        adj.data[:] = 1.0
    else:
        adj = sp.csr_matrix((n, n), dtype=np.float32)

    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def load_planetoid(
    dataset: str,
    root: str | Path = "data/planetoid",
    download: bool = True,
) -> PlanetoidData:
    """
    Load the old Planetoid pickle format used by tkipf/gae and Salha et al.

    Citeseer needs the standard isolated-test-node repair before stacking
    feature matrices.
    """
    dataset = dataset.lower()
    root = Path(root)
    if download:
        download_planetoid_dataset(dataset, root=root)

    x, tx, allx, graph = [
        _load_pickle(root / f"ind.{dataset}.{name}")
        for name in REQUIRED_PLANETOID_FILES
    ]
    test_idx_reorder = _parse_index_file(root / f"ind.{dataset}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        full_range = np.arange(test_idx_reorder.min(), test_idx_reorder.max() + 1)
        tx_extended = sp.lil_matrix((len(full_range), x.shape[1]), dtype=tx.dtype)
        tx_extended[test_idx_range - full_range.min(), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx), format="lil")
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.tocsr().astype(np.float32)

    adjacency = _adjacency_from_graph_dict(graph, n=features.shape[0])
    if adjacency.shape[0] != features.shape[0]:
        # Preserve all graph nodes if a graph dictionary contains a node not
        # represented in the stacked feature matrix.
        if adjacency.shape[0] > features.shape[0]:
            extra = adjacency.shape[0] - features.shape[0]
            features = sp.vstack(
                [features, sp.csr_matrix((extra, features.shape[1]), dtype=np.float32)],
                format="csr",
            )
        else:
            adjacency = sp.csr_matrix(
                (adjacency.data, adjacency.indices, adjacency.indptr),
                shape=(features.shape[0], features.shape[0]),
            )

    return PlanetoidData(dataset, adjacency.tocsr(), features)


def dataset_stats(data: PlanetoidData) -> Dict[str, int]:
    return {
        "dataset": data.name,
        "nodes": data.n,
        "undirected_edges_loaded": data.m,
        "features": data.features.shape[1],
    }


def _upper_edges(adj: sp.csr_matrix) -> np.ndarray:
    upper = sp.triu(adj, k=1).tocoo()
    return np.column_stack((upper.row, upper.col)).astype(np.int64, copy=False)


def _canonical_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _sample_negative_edges(
    n: int,
    count: int,
    forbidden: set[Tuple[int, int]],
    rng: np.random.RandomState,
    additional_forbidden: Optional[set[Tuple[int, int]]] = None,
) -> np.ndarray:
    if additional_forbidden is None:
        additional_forbidden = set()

    sampled: set[Tuple[int, int]] = set()
    while len(sampled) < count:
        u = int(rng.randint(0, n))
        v = int(rng.randint(0, n))
        if u == v:
            continue
        e = _canonical_edge(u, v)
        if e in forbidden or e in additional_forbidden or e in sampled:
            continue
        sampled.add(e)

    return np.asarray(sorted(sampled), dtype=np.int64)


def split_edges_kipf(
    adj: sp.csr_matrix,
    val_frac: float = 0.05,
    test_frac: float = 0.10,
    seed: int = 0,
) -> EdgeSplit:
    """
    Reproduce the classic GAE edge split:
      - unique undirected positives from the upper triangle
      - floor(5%) validation, floor(10%) test
      - same number of negative pairs
      - training adjacency rebuilt symmetrically

    The old TensorFlow code used randomized splits and warned that numbers may
    slightly deviate from the paper. This implementation makes the randomness
    explicit via `seed`.
    """
    adj = adj.tocsr().astype(np.float32).copy()
    adj.setdiag(0)
    adj.eliminate_zeros()

    edges = _upper_edges(adj)
    m = len(edges)
    num_val = int(np.floor(m * val_frac))
    num_test = int(np.floor(m * test_frac))

    rng = np.random.RandomState(seed)
    perm = rng.permutation(m)
    val_idx = perm[:num_val]
    test_idx = perm[num_val : num_val + num_test]

    holdout = np.concatenate([val_idx, test_idx])
    keep_mask = np.ones(m, dtype=bool)
    keep_mask[holdout] = False

    val_edges = edges[val_idx]
    test_edges = edges[test_idx]
    train_edges = edges[keep_mask]

    full_positive = {_canonical_edge(int(u), int(v)) for u, v in edges}

    test_false = _sample_negative_edges(
        adj.shape[0], len(test_edges), full_positive, rng
    )
    test_false_set = {tuple(e) for e in test_false.tolist()}
    val_false = _sample_negative_edges(
        adj.shape[0],
        len(val_edges),
        full_positive,
        rng,
        additional_forbidden=test_false_set,
    )

    r = np.concatenate([train_edges[:, 0], train_edges[:, 1]])
    c = np.concatenate([train_edges[:, 1], train_edges[:, 0]])
    values = np.ones(len(r), dtype=np.float32)
    adj_train = sp.csr_matrix((values, (r, c)), shape=adj.shape, dtype=np.float32)
    adj_train.eliminate_zeros()

    return EdgeSplit(
        adj_train=adj_train,
        train_edges=train_edges,
        val_edges=val_edges,
        val_edges_false=val_false,
        test_edges=test_edges,
        test_edges_false=test_false,
        seed=seed,
    )


def normalize_adjacency(adj: sp.csr_matrix) -> sp.csr_matrix:
    """D^{-1/2} (A + I) D^{-1/2}, matching Kipf/Salha preprocessing."""
    n = adj.shape[0]
    a = adj.astype(np.float32).tocsr() + sp.eye(n, dtype=np.float32, format="csr")
    degree = np.asarray(a.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nz = degree > 0
    inv_sqrt[nz] = degree[nz] ** -0.5
    d = sp.diags(inv_sqrt, format="csr")
    return (d @ a @ d).tocsr().astype(np.float32)


def scipy_to_torch_sparse(
    matrix: sp.spmatrix,
    device: torch.device | str,
) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(
        np.vstack([coo.row, coo.col]).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    tensor = torch.sparse_coo_tensor(
        indices,
        values,
        size=coo.shape,
        dtype=torch.float32,
        device=device,
    )
    return tensor.coalesce()


def _xavier_parameter(shape: Tuple[int, ...], device: torch.device) -> nn.Parameter:
    p = nn.Parameter(torch.empty(*shape, dtype=torch.float32, device=device))
    nn.init.xavier_uniform_(p)
    return p


class LinearGAE(nn.Module):
    """
    Featureless one-hop linear graph autoencoder:

        Z = A_tilde W
        logits = Z Z^T
    """

    def __init__(self, n_nodes: int, embedding_dim: int, device: torch.device):
        super().__init__()
        self.W = _xavier_parameter((n_nodes, embedding_dim), device)

    def encode(self, a_norm: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(a_norm, self.W)


class GCNGAE(nn.Module):
    """
    Featureless 2-layer deterministic GAE baseline from Kipf & Welling:

        H = ReLU(A_tilde W0)      because X = I
        Z = A_tilde H W1
        logits = Z Z^T
    """

    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int,
        embedding_dim: int,
        device: torch.device,
    ):
        super().__init__()
        self.W0 = _xavier_parameter((n_nodes, hidden_dim), device)
        self.W1 = _xavier_parameter((hidden_dim, embedding_dim), device)

    def encode(self, a_norm: torch.Tensor) -> torch.Tensor:
        h = torch.relu(torch.sparse.mm(a_norm, self.W0))
        h = h @ self.W1
        return torch.sparse.mm(a_norm, h)


def paper_bce_constants(adj_train: sp.csr_matrix) -> Tuple[float, float]:
    """
    Constants used by the original TensorFlow GAE code.

    Note the historical quirk: `pos_weight` is computed from A_train without
    self-loops, while the actual BCE labels are A_train + I.
    """
    n = adj_train.shape[0]
    n2 = float(n * n)
    positive_entries = float(adj_train.sum())
    if positive_entries <= 0:
        raise ValueError("Training graph has no positive adjacency entries.")
    pos_weight = (n2 - positive_entries) / positive_entries
    norm = n2 / ((n2 - positive_entries) * 2.0)
    return float(pos_weight), float(norm)


def _dense_label_block(
    adj_train: sp.csr_matrix,
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    target = adj_train[start:end].toarray().astype(np.float32, copy=False)
    # Original training labels are A_train + I.
    local_rows = np.arange(end - start)
    global_rows = np.arange(start, end)
    target[local_rows, global_rows] = 1.0
    return torch.from_numpy(target).to(device=device)


def edge_logits(z: torch.Tensor, edges: np.ndarray) -> torch.Tensor:
    idx = torch.as_tensor(edges, dtype=torch.long, device=z.device)
    return (z[idx[:, 0]] * z[idx[:, 1]]).sum(dim=1)


@torch.no_grad()
def evaluate_embeddings(
    z: torch.Tensor,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
) -> Tuple[float, float]:
    pos = torch.sigmoid(edge_logits(z, positive_edges)).cpu().numpy()
    neg = torch.sigmoid(edge_logits(z, negative_edges)).cpu().numpy()

    y_true = np.concatenate(
        [np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)]
    )
    y_score = np.concatenate([pos, neg])

    return (
        float(roc_auc_score(y_true, y_score)),
        float(average_precision_score(y_true, y_score)),
    )


def _choose_device(device: Optional[str | torch.device]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _choose_block_size(n: int, block_size: Optional[int]) -> int:
    if block_size is not None:
        return min(int(block_size), n)
    # Full matrix is convenient for Cora/Citeseer. Pubmed's n^2 decoder is
    # large enough to be antisocial, so default to exact row-block evaluation.
    return n if n <= 5000 else 512



def train_gae(
    data: PlanetoidData,
    split: EdgeSplit,
    model: Literal["linear_ae", "gcn_ae"] = "linear_ae",
    objective: Literal["bce", "mse", "balanced_mse", "taylor_bce"] = "bce",
    embedding_dim: int = 16,
    hidden_dim: int = 32,
    epochs: int = 200,
    lr: float = 0.01,
    init_seed: int = 0,
    device: Optional[str | torch.device] = None,
    block_size: Optional[int] = None,
    eval_every: Optional[int] = None,
    verbose: bool = False,
    keep_embeddings: bool = False,
    keep_history: bool = True,
) -> TrainResult:
    """
    Train a featureless Linear GAE or the classic featureless 2-layer GCN GAE.

    objective="bce"
        Original Kipf/Salha weighted BCE on logits ZZ^T.

    objective="mse"
        ||(A_train + I) - ZZ^T||_F^2 / n^2.

    objective="balanced_mse"
        Quadratic heuristic with positive target +1 and negative target
        -q, where q = 1 / pos_weight.

    objective="taylor_bce"
        Second-order Taylor surrogate to unweighted logistic BCE around
        logit zero. If Y=A_train+I, the quadratic target is

            M = 4Y - 2J,

        i.e. +2 for positives and -2 for negatives, and the optimized
        objective is

            ||M - ZZ^T||_F^2 / n^2.

        This is the objective solved exactly by spectral_taylor_gae().
    """
    dev = _choose_device(device)

    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)

    n = data.n
    a_norm_sp = normalize_adjacency(split.adj_train)
    a_norm = scipy_to_torch_sparse(a_norm_sp, dev)

    if model == "linear_ae":
        net: nn.Module = LinearGAE(n, embedding_dim, dev)
    elif model == "gcn_ae":
        net = GCNGAE(n, hidden_dim, embedding_dim, dev)
    else:
        raise ValueError("model must be 'linear_ae' or 'gcn_ae'")

    valid_objectives = {"bce", "mse", "balanced_mse", "taylor_bce"}
    if objective not in valid_objectives:
        raise ValueError(f"objective must be one of {sorted(valid_objectives)}")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    if objective in {"bce", "balanced_mse"}:
        pos_weight, norm = paper_bce_constants(split.adj_train)
        pos_weight_t = torch.tensor(
            pos_weight, dtype=torch.float32, device=dev
        )
        q = 1.0 / pos_weight
    else:
        pos_weight = norm = q = None
        pos_weight_t = None

    bs = _choose_block_size(n, block_size)
    history_rows = []

    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad(set_to_none=True)

        z = net.encode(a_norm)
        total_loss = 0.0

        starts = [0] if bs >= n else list(range(0, n, bs))

        for block_i, start in enumerate(starts):
            end = n if bs >= n else min(start + bs, n)
            scores = z[start:end] @ z.T
            target = _dense_label_block(split.adj_train, start, end, dev)

            if objective == "bce":
                if bs >= n:
                    block_loss = norm * F.binary_cross_entropy_with_logits(
                        scores,
                        target,
                        pos_weight=pos_weight_t,
                        reduction="mean",
                    )
                else:
                    block_loss = (
                        norm / float(n * n)
                    ) * F.binary_cross_entropy_with_logits(
                        scores,
                        target,
                        pos_weight=pos_weight_t,
                        reduction="sum",
                    )

            elif objective == "mse":
                if bs >= n:
                    block_loss = F.mse_loss(scores, target, reduction="mean")
                else:
                    block_loss = (
                        1.0 / float(n * n)
                    ) * F.mse_loss(scores, target, reduction="sum")

            elif objective == "balanced_mse":
                balanced_target = (1.0 + q) * target - q
                if bs >= n:
                    block_loss = F.mse_loss(
                        scores, balanced_target, reduction="mean"
                    )
                else:
                    block_loss = (
                        1.0 / float(n * n)
                    ) * F.mse_loss(
                        scores, balanced_target, reduction="sum"
                    )

            else:  # taylor_bce
                taylor_target = 4.0 * target - 2.0
                if bs >= n:
                    block_loss = F.mse_loss(
                        scores, taylor_target, reduction="mean"
                    )
                else:
                    block_loss = (
                        1.0 / float(n * n)
                    ) * F.mse_loss(
                        scores, taylor_target, reduction="sum"
                    )

            retain = block_i < len(starts) - 1
            block_loss.backward(retain_graph=retain)
            total_loss += float(block_loss.detach().cpu())

        optimizer.step()

        do_eval = (
            epoch == epochs
            or (eval_every is not None and epoch % eval_every == 0)
            or (verbose and epoch in {1, epochs})
        )

        if do_eval:
            net.eval()
            with torch.no_grad():
                z_eval = net.encode(a_norm)
                val_auc, val_ap = evaluate_embeddings(
                    z_eval, split.val_edges, split.val_edges_false
                )
            history_rows.append(
                {
                    "epoch": epoch,
                    "loss": total_loss,
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                }
            )
            if verbose:
                print(
                    f"{data.name:8s} {model:9s} {objective:12s} "
                    f"epoch={epoch:03d} loss={total_loss:.6f} "
                    f"val_auc={val_auc:.4f} val_ap={val_ap:.4f}"
                )
        elif keep_history:
            history_rows.append(
                {
                    "epoch": epoch,
                    "loss": total_loss,
                    "val_auc": np.nan,
                    "val_ap": np.nan,
                }
            )

    net.eval()
    with torch.no_grad():
        z_final = net.encode(a_norm)
        val_auc, val_ap = evaluate_embeddings(
            z_final, split.val_edges, split.val_edges_false
        )
        test_auc, test_ap = evaluate_embeddings(
            z_final, split.test_edges, split.test_edges_false
        )
        emb = z_final.cpu().numpy() if keep_embeddings else None

    history = pd.DataFrame(history_rows) if keep_history else None

    return TrainResult(
        model=model,
        objective=objective,
        dataset=data.name,
        split_seed=split.seed,
        init_seed=init_seed,
        auc=test_auc,
        ap=test_ap,
        val_auc=val_auc,
        val_ap=val_ap,
        final_loss=total_loss,
        epochs=epochs,
        embedding_dim=embedding_dim,
        device=str(dev),
        embeddings=emb,
        history=history,
    )

def result_to_dict(result: TrainResult) -> Dict[str, object]:
    return {
        "model": result.model,
        "objective": result.objective,
        "dataset": result.dataset,
        "split_seed": result.split_seed,
        "init_seed": result.init_seed,
        "auc": result.auc,
        "ap": result.ap,
        "val_auc": result.val_auc,
        "val_ap": result.val_ap,
        "final_loss": result.final_loss,
        "epochs": result.epochs,
        "embedding_dim": result.embedding_dim,
        "device": result.device,
    }


def evaluate_numpy_embeddings(
    z: np.ndarray,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
) -> Tuple[float, float]:
    """
    Evaluate dot-product link scores directly.

    AUC/AP are invariant under the sigmoid because it is strictly monotone,
    so there is no need to apply it here.
    """
    z = np.asarray(z, dtype=np.float64)

    pos = np.einsum(
        "ij,ij->i",
        z[positive_edges[:, 0]],
        z[positive_edges[:, 1]],
    )
    neg = np.einsum(
        "ij,ij->i",
        z[negative_edges[:, 0]],
        z[negative_edges[:, 1]],
    )

    y_true = np.concatenate(
        [
            np.ones(len(pos), dtype=np.int8),
            np.zeros(len(neg), dtype=np.int8),
        ]
    )
    y_score = np.concatenate([pos, neg])

    return (
        float(roc_auc_score(y_true, y_score)),
        float(average_precision_score(y_true, y_score)),
    )


def _taylor_y_matrix(adj_train: sp.csr_matrix) -> sp.csr_matrix:
    """Y = A_train + I as a binary symmetric sparse matrix."""
    n = adj_train.shape[0]
    y = (
        adj_train.astype(np.float64).tocsr()
        + sp.eye(n, dtype=np.float64, format="csr")
    )
    y.data[:] = 1.0
    y.eliminate_zeros()
    return y


def _apply_taylor_target(
    y: sp.csr_matrix,
    x: np.ndarray,
) -> np.ndarray:
    """
    Apply M = 4Y - 2J without ever materializing dense J or M.

    Supports a vector (n,) or matrix (n,k).
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        return 4.0 * (y @ x) - 2.0 * np.sum(x) * np.ones(y.shape[0])

    if x.ndim == 2:
        return (
            4.0 * (y @ x)
            - 2.0 * np.ones((y.shape[0], 1)) * np.sum(x, axis=0, keepdims=True)
        )

    raise ValueError("x must be a vector or a 2D matrix")


def _taylor_linear_operator(y: sp.csr_matrix) -> spla.LinearOperator:
    n = y.shape[0]

    def matvec(v):
        return _apply_taylor_target(y, v)

    def matmat(v):
        return _apply_taylor_target(y, v)

    return spla.LinearOperator(
        shape=(n, n),
        matvec=matvec,
        matmat=matmat,
        dtype=np.float64,
    )


def taylor_reconstruction_loss(
    z: np.ndarray,
    split: EdgeSplit,
) -> float:
    """
    Exact value of

        ||M - ZZ^T||_F^2 / n^2

    for M = 4(A_train+I) - 2J, without constructing either dense matrix.

    Since every entry of M is +/-2, ||M||_F^2 = 4 n^2.
    """
    z = np.asarray(z, dtype=np.float64)
    n = split.adj_train.shape[0]
    y = _taylor_y_matrix(split.adj_train)

    mz = _apply_taylor_target(y, z)

    # ||ZZ^T||_F^2 = ||Z^T Z||_F^2
    gram = z.T @ z
    x_norm_sq = float(np.sum(gram * gram))

    # <M, ZZ^T> = tr(Z^T M Z)
    cross = float(np.sum(z * mz))

    loss = (4.0 * n * n + x_norm_sq - 2.0 * cross) / float(n * n)
    return float(max(loss, 0.0))


def _top_positive_eigenpairs(
    operator: spla.LinearOperator,
    n: int,
    embedding_dim: int,
    seed: int,
    eig_tol: float,
    maxiter: Optional[int],
    positive_tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Largest algebraic eigenpairs, retaining only positive eigenvalues.

    The best rank-d PSD approximation keeps precisely the d largest positive
    eigenvalues.
    """
    if embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1")
    if n < 2:
        raise ValueError("spectral solver requires at least two nodes")

    k = min(int(embedding_dim), n - 1)
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=n)

    vals, vecs = spla.eigsh(
        operator,
        k=k,
        which="LA",
        v0=v0,
        tol=eig_tol,
        maxiter=maxiter,
    )

    order = np.argsort(vals)[::-1]
    vals = np.asarray(vals[order], dtype=np.float64)
    vecs = np.asarray(vecs[:, order], dtype=np.float64)

    scale = max(1.0, float(np.max(np.abs(vals))))
    keep = vals > positive_tol * scale

    return vals[keep], vecs[:, keep]


def _unconstrained_taylor_spectral(
    split: EdgeSplit,
    embedding_dim: int,
    seed: int,
    eig_tol: float,
    maxiter: Optional[int],
    positive_tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    y = _taylor_y_matrix(split.adj_train)
    op = _taylor_linear_operator(y)

    vals, vecs = _top_positive_eigenpairs(
        op,
        n=y.shape[0],
        embedding_dim=embedding_dim,
        seed=seed,
        eig_tol=eig_tol,
        maxiter=maxiter,
        positive_tol=positive_tol,
    )

    if len(vals) == 0:
        return np.zeros((y.shape[0], 0), dtype=np.float64), vals

    z = vecs * np.sqrt(vals)[None, :]
    return z, vals


def _exact_constrained_taylor_dense(
    split: EdgeSplit,
    embedding_dim: int,
    rank_tol: float,
    eig_tol: float,
    maxiter: Optional[int],
    positive_tol: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Exact constrained solution for singular S on moderate-sized graphs.

    S = A_tilde is symmetric. If Q is an orthonormal basis for col(S), the
    exact problem is the best rank-d PSD approximation of

        Q^T M Q.

    This routine obtains Q from a dense eigendecomposition of S. It is only a
    fallback for graphs small enough that doing this is reasonable.
    """
    s = normalize_adjacency(split.adj_train).astype(np.float64).toarray()
    s = 0.5 * (s + s.T)

    s_vals, q_full = np.linalg.eigh(s)
    scale = max(1.0, float(np.max(np.abs(s_vals))))
    keep = np.abs(s_vals) > rank_tol * scale

    q = q_full[:, keep]
    s_nonzero = s_vals[keep]

    if q.shape[1] == 0:
        raise RuntimeError("Normalized adjacency has zero numerical rank.")

    y = _taylor_y_matrix(split.adj_train)
    mq = _apply_taylor_target(y, q)
    b = q.T @ mq
    b = 0.5 * (b + b.T)

    r = b.shape[0]
    if r == 1:
        vals = np.asarray([b[0, 0]], dtype=np.float64)
        u = np.ones((1, 1), dtype=np.float64)
    else:
        k = min(embedding_dim, r - 1)
        if k == r:
            vals, u = np.linalg.eigh(b)
            order = np.argsort(vals)[::-1]
            vals, u = vals[order], u[:, order]
        else:
            rng = np.random.default_rng(seed)
            vals, u = spla.eigsh(
                b,
                k=k,
                which="LA",
                v0=rng.normal(size=r),
                tol=eig_tol,
                maxiter=maxiter,
            )
            order = np.argsort(vals)[::-1]
            vals, u = vals[order], u[:, order]

    scale_b = max(1.0, float(np.max(np.abs(vals))))
    positive = vals > positive_tol * scale_b
    vals = np.asarray(vals[positive], dtype=np.float64)
    u = np.asarray(u[:, positive], dtype=np.float64)

    if len(vals) == 0:
        z = np.zeros((s.shape[0], 0), dtype=np.float64)
        w = np.zeros((s.shape[0], 0), dtype=np.float64)
        return z, vals, w, 0.0

    coeff = u * np.sqrt(vals)[None, :]
    z = q @ coeff

    # Since S = Q diag(s_nonzero) Q^T on its range,
    # one exact preimage is W = Q diag(1/s_nonzero) coeff.
    w = q @ (coeff / s_nonzero[:, None])
    z_check = s @ w

    residual = np.linalg.norm(z_check - z) / max(np.linalg.norm(z), 1e-15)
    return z_check, vals, w, float(residual)


def spectral_taylor_gae(
    data: PlanetoidData,
    split: EdgeSplit,
    embedding_dim: int = 16,
    enforce_encoder: bool = True,
    encoder_tol: float = 1e-7,
    dense_fallback_max_n: int = 5000,
    rank_tol: float = 1e-10,
    eig_tol: float = 1e-8,
    positive_tol: float = 1e-10,
    maxiter: Optional[int] = None,
    keep_embeddings: bool = False,
    keep_weights: bool = False,
) -> SpectralResult:
    """
    Solve the Taylor-BCE quadratic objective spectrally.

    First solve the embedding-space problem

        min_{rank(Z)<=d} ||M - ZZ^T||_F^2,
        M = 4(A_train+I) - 2J.

    If enforce_encoder=True, verify that the solution can be written as

        Z = A_tilde W.

    For nonsingular A_tilde this is automatic and a sparse LU solve obtains W.
    If A_tilde is singular and n <= dense_fallback_max_n, solve the exact
    constrained problem in col(A_tilde). For a larger singular graph, raise
    rather than silently returning a solution outside the encoder's feasible
    set.
    """
    n = data.n

    z, vals = _unconstrained_taylor_spectral(
        split,
        embedding_dim=embedding_dim,
        seed=split.seed,
        eig_tol=eig_tol,
        maxiter=maxiter,
        positive_tol=positive_tol,
    )

    solver = "sparse_eigsh"
    constraint = "embedding_space_only"
    encoder_residual = np.nan
    w = None

    if enforce_encoder:
        s_sparse = normalize_adjacency(split.adj_train).astype(np.float64).tocsc()

        try:
            lu = spla.splu(s_sparse)
            if z.shape[1] == 0:
                w = np.zeros((n, 0), dtype=np.float64)
                z_encoded = z
                encoder_residual = 0.0
            else:
                w = lu.solve(z)
                z_encoded = s_sparse @ w
                encoder_residual = float(
                    np.linalg.norm(z_encoded - z)
                    / max(np.linalg.norm(z), 1e-15)
                )

            if encoder_residual <= encoder_tol:
                z = np.asarray(z_encoded)
                constraint = "exact_full_rank_encoder"
            else:
                raise RuntimeError(
                    f"sparse solve residual {encoder_residual:.3e} "
                    f"exceeds tolerance {encoder_tol:.3e}"
                )

        except Exception as exc:
            if n > dense_fallback_max_n:
                raise RuntimeError(
                    "The unconstrained spectral optimum could not be verified "
                    "as representable by Z=A_tilde W, and the graph is too "
                    "large for the exact dense constrained fallback. "
                    "Run with enforce_encoder=False only if you explicitly "
                    "want the embedding-space lower bound."
                ) from exc

            z, vals, w, encoder_residual = _exact_constrained_taylor_dense(
                split,
                embedding_dim=embedding_dim,
                rank_tol=rank_tol,
                eig_tol=eig_tol,
                maxiter=maxiter,
                positive_tol=positive_tol,
                seed=split.seed,
            )
            solver = "dense_range_projection+eigsh"
            constraint = "exact_singular_encoder"

    val_auc, val_ap = evaluate_numpy_embeddings(
        z, split.val_edges, split.val_edges_false
    )
    test_auc, test_ap = evaluate_numpy_embeddings(
        z, split.test_edges, split.test_edges_false
    )
    loss = taylor_reconstruction_loss(z, split)

    return SpectralResult(
        dataset=data.name,
        split_seed=split.seed,
        auc=test_auc,
        ap=test_ap,
        val_auc=val_auc,
        val_ap=val_ap,
        final_loss=loss,
        embedding_dim_requested=embedding_dim,
        embedding_dim_used=z.shape[1],
        eigenvalues=np.asarray(vals, dtype=np.float64),
        solver=solver,
        encoder_constraint=constraint,
        encoder_residual=float(encoder_residual),
        embeddings=z if keep_embeddings else None,
        weights=w if keep_weights else None,
    )


def spectral_result_to_dict(result: SpectralResult) -> Dict[str, object]:
    eigs = result.eigenvalues
    return {
        "model": "linear_ae",
        "objective": "taylor_bce_spectral",
        "dataset": result.dataset,
        "split_seed": result.split_seed,
        "auc": result.auc,
        "ap": result.ap,
        "val_auc": result.val_auc,
        "val_ap": result.val_ap,
        "final_loss": result.final_loss,
        "embedding_dim": result.embedding_dim_requested,
        "embedding_dim_used": result.embedding_dim_used,
        "largest_eigenvalue": float(eigs[0]) if len(eigs) else np.nan,
        "smallest_kept_eigenvalue": float(eigs[-1]) if len(eigs) else np.nan,
        "solver": result.solver,
        "encoder_constraint": result.encoder_constraint,
        "encoder_residual": result.encoder_residual,
    }


def run_spectral_reproduction(
    dataset: str,
    n_runs: int = 10,
    protocol: Literal["salha", "kipf"] = "salha",
    data_root: str | Path = "data/planetoid",
    base_seed: int = 0,
    embedding_dim: int = 16,
    enforce_encoder: bool = True,
    dense_fallback_max_n: int = 5000,
    verbose: bool = False,
    **spectral_kwargs,
) -> pd.DataFrame:
    """
    Repeated spectral Taylor-BCE evaluation on exactly the same split protocol
    as run_reproduction().
    """
    data = load_planetoid(dataset, root=data_root, download=True)
    rows = []

    fixed_split = None
    if protocol == "kipf":
        fixed_split = split_edges_kipf(data.adjacency, seed=base_seed)

    for run in range(n_runs):
        if protocol == "salha":
            split = split_edges_kipf(data.adjacency, seed=base_seed + run)
        elif protocol == "kipf":
            split = fixed_split
        else:
            raise ValueError("protocol must be 'salha' or 'kipf'")

        result = spectral_taylor_gae(
            data=data,
            split=split,
            embedding_dim=embedding_dim,
            enforce_encoder=enforce_encoder,
            dense_fallback_max_n=dense_fallback_max_n,
            keep_embeddings=False,
            keep_weights=False,
            **spectral_kwargs,
        )

        row = spectral_result_to_dict(result)
        row["run"] = run
        row["protocol"] = protocol
        rows.append(row)

        if verbose:
            print(
                f"{dataset:8s} spectral run={run:02d} "
                f"AUC={result.auc:.4f} AP={result.ap:.4f} "
                f"loss={result.final_loss:.6f} "
                f"{result.encoder_constraint} "
                f"resid={result.encoder_residual:.2e}"
            )

    return pd.DataFrame(rows)


def compare_taylor_gd_to_spectral(
    gd_results: pd.DataFrame,
    spectral_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Paired comparison by dataset/run/split seed.

    Positive delta_auc/ap means the exact spectral solution ranks held-out links
    better than the gradient-trained Taylor surrogate. A non-positive
    loss_delta_spectral_minus_gd is expected if GD has not reached the global
    quadratic optimum.
    """
    left = gd_results.copy()
    right = spectral_results.copy()

    keys = [k for k in ["dataset", "run", "split_seed"] if k in left.columns and k in right.columns]
    if not keys:
        raise ValueError("Could not find common pairing keys.")

    keep_left = keys + ["auc", "ap", "final_loss"]
    keep_right = keys + [
        "auc", "ap", "final_loss",
        "encoder_constraint", "encoder_residual",
        "embedding_dim_used",
    ]

    merged = left[keep_left].merge(
        right[keep_right],
        on=keys,
        suffixes=("_gd", "_spectral"),
    )

    merged["delta_auc"] = merged["auc_spectral"] - merged["auc_gd"]
    merged["delta_ap"] = merged["ap_spectral"] - merged["ap_gd"]
    merged["loss_delta_spectral_minus_gd"] = (
        merged["final_loss_spectral"] - merged["final_loss_gd"]
    )

    return merged



def run_reproduction(
    dataset: str,
    model: Literal["linear_ae", "gcn_ae"] = "linear_ae",
    objective: Literal["bce", "mse", "balanced_mse", "taylor_bce"] = "bce",
    n_runs: int = 10,
    protocol: Literal["salha", "kipf"] = "salha",
    data_root: str | Path = "data/planetoid",
    base_seed: int = 0,
    embedding_dim: int = 16,
    hidden_dim: int = 32,
    epochs: int = 200,
    lr: float = 0.01,
    device: Optional[str | torch.device] = None,
    block_size: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Notebook-friendly repeated evaluation.

    protocol="salha":
        new train/val/test split on every run (the 2020 paper used 100 splits)

    protocol="kipf":
        one fixed split, different initialization each run
        (the 2016 paper reports 10 initializations on fixed splits)
    """
    data = load_planetoid(dataset, root=data_root, download=True)
    rows = []

    fixed_split = None
    if protocol == "kipf":
        fixed_split = split_edges_kipf(data.adjacency, seed=base_seed)

    for run in range(n_runs):
        if protocol == "salha":
            split_seed = base_seed + run
            split = split_edges_kipf(data.adjacency, seed=split_seed)
        elif protocol == "kipf":
            split_seed = base_seed
            split = fixed_split
        else:
            raise ValueError("protocol must be 'salha' or 'kipf'")

        init_seed = base_seed + 100_000 + run
        result = train_gae(
            data=data,
            split=split,
            model=model,
            objective=objective,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            init_seed=init_seed,
            device=device,
            block_size=block_size,
            verbose=verbose,
            keep_embeddings=False,
            keep_history=False,
        )
        row = result_to_dict(result)
        row["run"] = run
        row["protocol"] = protocol
        rows.append(row)

    return pd.DataFrame(rows)


def run_benchmark(
    datasets: Sequence[str] = DATASETS,
    model: Literal["linear_ae", "gcn_ae"] = "linear_ae",
    objective: Literal["bce", "mse", "balanced_mse", "taylor_bce"] = "bce",
    n_runs: int = 10,
    protocol: Literal["salha", "kipf"] = "salha",
    **kwargs,
) -> pd.DataFrame:
    frames = [
        run_reproduction(
            dataset=d,
            model=model,
            objective=objective,
            n_runs=n_runs,
            protocol=protocol,
            **kwargs,
        )
        for d in datasets
    ]
    return pd.concat(frames, ignore_index=True)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["protocol", "model"]
    if "objective" in results.columns:
        group_cols.append("objective")
    group_cols.append("dataset")
    grouped = results.groupby(group_cols, sort=False)
    summary = grouped.agg(
        runs=("auc", "size"),
        auc_mean=("auc", "mean"),
        auc_sd=("auc", "std"),
        auc_sem=("auc", "sem"),
        ap_mean=("ap", "mean"),
        ap_sd=("ap", "std"),
        ap_sem=("ap", "sem"),
        loss_mean=("final_loss", "mean"),
    ).reset_index()
    return summary


def compare_to_paper(
    results_or_summary: pd.DataFrame,
    paper: Literal["salha2020", "kipf2016"] = "salha2020",
) -> pd.DataFrame:
    if "auc_mean" not in results_or_summary.columns:
        summary = summarize_results(results_or_summary)
    else:
        summary = results_or_summary.copy()

    targets = PAPER_TARGETS[PAPER_TARGETS["paper"] == paper].copy()
    merged = summary.merge(
        targets,
        on=["model", "dataset"],
        how="left",
        suffixes=("_ours", "_paper"),
    )

    merged["auc_delta"] = merged["auc_mean"] - merged["auc"]
    merged["ap_delta"] = merged["ap_mean"] - merged["ap"]
    cols = ["protocol", "model"]
    if "objective" in merged.columns:
        cols.append("objective")
    cols += [
        "dataset",
        "runs",
        "auc_mean",
        "auc_sd_ours",
        "auc",
        "auc_delta",
        "ap_mean",
        "ap_sd_ours",
        "ap",
        "ap_delta",
    ]
    return merged[cols].rename(
        columns={
            "auc_sd_ours": "auc_sd",
            "auc": "paper_auc",
            "ap_sd_ours": "ap_sd",
            "ap": "paper_ap",
        }
    )


def print_paper_targets() -> pd.DataFrame:
    return PAPER_TARGETS.copy()
