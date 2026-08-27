#!/usr/bin/env python3
"""Fast offline smoke test for the core model implementations."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp

from linear_gae_spectral import PlanetoidData, split_edges_kipf
from learned_models import train_validation_selected_gae
from run_experiments import Config, run_kernel

rng = np.random.default_rng(123)
n, p = 90, 20
upper = np.triu(rng.random((n, n)) < 0.08, 1)
adj = sp.csr_matrix((upper | upper.T).astype(np.float32))
features = (rng.random((n, p)) < 0.2).astype(np.float32)
for i in range(n):
    if not features[i].any():
        features[i, rng.integers(0, p)] = 1.0
features = sp.csr_matrix(features)

data = PlanetoidData("synthetic", adj, features)
split = split_edges_kipf(adj, seed=10000)

# Learned paths: one epoch, CPU.
for setting in ("featureless", "features"):
    for kind in ("linear", "gcn2", "gcn3"):
        selected, history = train_validation_selected_gae(
            data=data,
            split=split,
            setting=setting,
            model_kind=kind,
            epochs=1,
            lr=0.01,
            hidden_dim=8,
            latent_dim=4,
            init_seed=1,
            requested_device="cpu",
            block_size=None,
            verbose=False,
        )
        assert len(history) == 1
        assert np.isfinite(selected["test_auc"])

# Kernel paths: tiny rank/grid.
cfg = Config(
    kernel_gammas=(0.02,),
    kernel_k_values=(0, 2, 4),
    kernel_positive_rank_featureless=6,
    kernel_positive_rank_features=6,
    verbose=False,
)
for setting in ("featureless", "features"):
    selected, tuning = run_kernel(data, split, cfg, setting, eig_seed=7)
    assert len(tuning) == 3
    assert all(np.isnan(row["test_auc"]) for row in tuning)
    assert selected["actual_positive_rank"] == 6
    assert selected["actual_negative_rank"] == 0
    assert np.isclose(
        selected["alpha"],
        selected["selected_gamma"] / selected["lambda_max_positive"],
    )

print("Smoke test passed.")
