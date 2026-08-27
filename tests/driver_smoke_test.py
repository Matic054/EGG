#!/usr/bin/env python3
"""Offline end-to-end smoke test for the benchmark driver/checkpoint files."""
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import scipy.sparse as sp

import run_experiments as runner
from linear_gae_spectral import PlanetoidData

rng = np.random.default_rng(77)
n, p = 80, 18
upper = np.triu(rng.random((n, n)) < 0.09, 1)
adj = sp.csr_matrix((upper | upper.T).astype(np.float32))
f = (rng.random((n, p)) < 0.2).astype(np.float32)
for i in range(n):
    if not f[i].any():
        f[i, rng.integers(0, p)] = 1.0
synthetic = PlanetoidData("cora", adj, sp.csr_matrix(f))

runner.load_planetoid = lambda dataset, root, download: synthetic
cfg = runner.Config(
    datasets=("cora",),
    settings=("featureless", "features"),
    methods=runner.METHODS,
    folds=1,
    epochs=1,
    device="cpu",
    kernel_gammas=(0.02,),
    kernel_k_values=(0, 2),
    kernel_positive_rank_featureless=6,
    kernel_positive_rank_features=6,
    verbose=False,
)

out = Path(tempfile.mkdtemp(prefix="gae_driver_smoke_"))
try:
    runner.run(cfg, out)
    selected = pd.read_csv(out / "selected_results.csv")
    tuning = pd.read_csv(out / "tuning_results.csv")
    assert len(selected) == 1 * 1 * 2 * 4
    assert not selected.duplicated(["dataset", "fold", "setting", "method"]).any()
    assert tuning["test_auc"].isna().all()
    assert tuning["test_ap"].isna().all()
    k = selected[selected.method == "linearized_gae_kernel"]
    assert (k.actual_negative_rank == 0).all()
    print("Driver smoke test passed.")
finally:
    shutil.rmtree(out)
