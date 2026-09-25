# Finite-Time Spectral Kernel for Graph Autoencoders

Reproducibility code for the experiments comparing a deterministic finite-time
spectral kernel against Linear AE and 2-/3-layer GCN graph autoencoders on
Cora, Citeseer, and Pubmed.

The repository reproduces the final experimental protocol used in the paper:

- 10 random edge splits per dataset;
- 5% validation positives and 10% test positives;
- equal numbers of sampled negative validation/test pairs;
- featureless and node-feature settings;
- Linear AE, 2-layer GCN AE, 3-layer GCN AE, and the proposed kernel;
- validation-only model/hyperparameter selection;
- ROC-AUC, AP, and wall-clock runtime;
- a Table-1-style analysis similar to Salha et al. (2020), augmented with runtime.

The code automatically downloads the Planetoid-format Cora, Citeseer, and
Pubmed files on first use. Dataset files and generated results are intentionally
excluded from Git.

## Repository layout

```text
.
├── run_experiments.py          # canonical benchmark driver
├── analyze_results.py          # paper-style tables and paired analysis
├── check_device.py             # PyTorch/CUDA diagnostic
├── requirements.txt
├── src/
│   ├── learned_models.py       # Linear / 2-layer / 3-layer GAE implementations
│   ├── baselines.py            # optional PPR/Katz/CN/AA/RA baselines
│   ├── linear_gae_spectral.py  # data, splits, normalization, BCE/evaluation utilities
│   ├── deterministic_kernel.py # featureless spectral kernel implementation
│   └── feature_kernel.py       # feature-aware spectral kernel implementation
├── docs/
│   └── METHODS.md              # mathematical and experimental details
├── tests/
│   └── smoke_test.py           # offline implementation sanity check
└── results/
    └── .gitkeep
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check whether PyTorch can use CUDA:

```bash
python check_device.py
```

The learned autoencoders can run on CPU or CUDA. The spectral kernel uses
SciPy and therefore runs on CPU.

## Reproduce the paper benchmark

From the repository root:

```bash
python run_experiments.py --device cpu --out results/paper
```

Default experiment:

- datasets: `cora,citeseer,pubmed`
- folds: 10
- split seeds: `10000,...,10009`
- settings: `featureless,features`
- methods: `linear_ae,gcn_ae_2,gcn_ae_3,linearized_gae_kernel`
- learned models: 250 epochs, Adam, learning rate 0.01
- learned-model device: CPU by default (matching the reported runtime setup)
- latent dimension: 16
- GCN hidden width: 32
- raw Planetoid node features
- kernel retained rank: 256 positive modes, 0 negative modes
- kernel gamma grid: `0.01,0.02,0.05,0.10`
- kernel `k` grid: `0,5,...,300`

To require CUDA for learned models:

```bash
python run_experiments.py --device cuda --out results/paper
```

To force CPU:

```bash
python run_experiments.py --device cpu --out results/paper_cpu
```

The run is resumable. Re-running the same command skips completed
`(dataset, fold, setting, method)` combinations. Use `--force` only when you
intentionally want to recompute them.

### Running subsets

Pubmed only:

```bash
python run_experiments.py \
  --datasets pubmed \
  --out results/pubmed
```

Kernel only:

```bash
python run_experiments.py \
  --methods linearized_gae_kernel \
  --out results/kernel_only
```

The main paper table uses only the four learned/kernel methods. To also reproduce
the ancillary topology-only baselines (PPR, Katz, Common Neighbors, Adamic-Adar,
and Resource Allocation), run:

```bash
python run_experiments.py \
  --include-classical \
  --out results/paper_with_classical
```

The analysis script still keeps these methods out of the main table and writes
`classical_baselines_summary.csv` separately when they are present.

One-fold smoke experiment:

```bash
python run_experiments.py \
  --datasets cora \
  --folds 1 \
  --epochs 2 \
  --positive-rank 16 \
  --k-max 20 \
  --out results/quick_check
```

This last command is for installation/debugging only, not for reproducing the
reported numbers.

## Analyze the results

After the benchmark finishes:

```bash
python analyze_results.py results/paper
```

This creates `results/paper/analysis/` containing:

```text
summary_long.csv
fold_level_main_methods.csv
table1_like_featureless.csv
table1_like_featureless.md
table1_like_features.csv
table1_like_features.md
table1_like.tex
paired_vs_kernel.csv
win_counts.csv
selected_hyperparameters.csv
classical_baselines_summary.csv   # only if optional baselines were run
```

`table1_like.tex` is the paper-ready table. AUC and AP are reported as
mean ± sample standard deviation in percent; runtime is mean ± sample standard
deviation in seconds.

`paired_vs_kernel.csv` uses the common edge splits to report paired AUC/AP
differences, 95% t-intervals, fold wins, exact two-sided sign-test p-values,
and runtime speedups relative to each learned baseline.

## Test-set discipline

The experiment driver never uses test performance for model selection.

For the learned models, validation AUC is evaluated after every epoch. The
best epoch is selected by:

1. maximum validation AUC;
2. maximum validation AP as a tie-breaker;
3. earlier epoch as the final tie-breaker.

The test set is then evaluated once at the selected checkpoint.

For the kernel, validation selects `(gamma, k)` by:

1. maximum validation AUC;
2. maximum validation AP;
3. smaller spectral time;
4. smaller `gamma` and then smaller `k`.

Again, test edges are evaluated only after selection.

## Proposed kernel

Let

\[
S=D^{-1/2}(A_{\mathrm{train}}+I)D^{-1/2},
\qquad
Y=A_{\mathrm{train}}+I,
\qquad
M=4Y-2J.
\]

For the featureless model, define

\[
A_s=SMS.
\]

If `lambda_max+` is the largest positive retained eigenvalue, the final
cross-dataset normalization is

\[
\alpha=\frac{\gamma}{\lambda_{\max}^{+}}.
\]

Using the positive eigenpairs `A_s V = V Lambda`, the queried kernel is
approximated by

\[
K_{\gamma,k}
\approx
S^2 + (SV)
\operatorname{diag}\!\left[
(1+\alpha\lambda_i)^{2k}-1
\right]
(SV)^T.
\]

The final experiments retain 256 positive modes and no negative modes.
The associated dimensionless spectral time is

\[
\tau=2k\gamma.
\]

With node features `F`, let `H=SF` and

\[
A_F=H^TMH=F^TSMSF.
\]

The feature-aware kernel is

\[
K^{(F)}_{\gamma,k}
\approx
HH^T+(HV)
\operatorname{diag}\!\left[
(1+\alpha\lambda_i)^{2k}-1
\right]
(HV)^T,
\]

again with `alpha = gamma/lambda_max+`.

See [`docs/METHODS.md`](docs/METHODS.md) for the derivation and implementation
notes.

## Learned baselines

The learned models use the weighted full-adjacency binary cross-entropy loss
and inner-product decoder used in the classic GAE protocol.

Featureless encoders:

```text
Linear AE:       Z = S W
2-layer GCN AE:  H1 = ReLU(S W0);       Z = S H1 W1
3-layer GCN AE:  H1 = ReLU(S W0); H2 = ReLU(S H1 W1); Z = S H2 W2
```

With node features, `W0` is preceded by `F`:

```text
Linear AE:       Z = S F W
2-layer GCN AE:  H1 = ReLU(S F W0); ...
3-layer GCN AE:  H1 = ReLU(S F W0); ...
```

Pubmed uses an exact row-blocked decoder-gradient implementation to avoid
materializing the full `n x n` logits matrix. This changes memory use, not the
objective or gradient.

## Reproducibility metadata

Every run writes:

- `config.json` with the complete experiment configuration;
- `environment.json` with Python/library/device information;
- `selected_results.csv` with one selected result per method/fold/setting;
- `tuning_results.csv` with validation trajectories/grids and no test metrics
  during tuning.

For fair runtime comparisons, run all compared methods on the same machine and
report the recorded execution devices. Comparing a CUDA neural model against a
CPU SciPy kernel is a practical comparison, but not a controlled same-device
speedup.

## Nonlinear term diagnostics

To support the linearization used in the derivation, the repository includes
`src/run_nonlinear_term_diagnostics.py`. This script measures the relative
magnitude and directional alignment of the linear and cubic terms in the exact
Taylor-GAE gradient,

\[
\nabla_W \mathcal{L}
=
\frac{4}{n^2}
\left[
S Z (Z^T Z) - S M Z
\right].
\]

In particular, it defines

\[
L = SMZ,
\qquad
C = SZ(Z^TZ),
\]

and records at regular training checkpoints

\[
\frac{\|C\|_F}{\|L\|_F}
\qquad\text{and}\qquad
\cos(L,C)
=
\frac{\langle L,C\rangle_F}
{\|L\|_F\|C\|_F}.
\]

The norm ratio measures the magnitude of the nonlinear term relative to the
linear term, while the cosine similarity measures their directional alignment.
The purpose of this experiment is not to claim that the cubic term is small,
but to test whether it becomes approximately aligned with the linear component
during the finite-time training regime used for link prediction.

From the repository root, reproduce the default diagnostic with:

```bash
python src/run_nonlinear_term_diagnostics.py
```

## Offline smoke test

```bash
python tests/smoke_test.py
```

It creates a synthetic graph and exercises all six learned paths
(3 encoders × 2 feature settings) plus both kernel paths without downloading
any datasets.

## References

- Salha, G., Hennequin, R., & Vazirgiannis, M. (2020). *Simple and Effective
  Graph Autoencoders with One-Hop Linear Models*. ECML-PKDD.
- Kipf, T. N., & Welling, M. (2016). *Variational Graph Auto-Encoders*.
  arXiv:1611.07308.

## License

No license is included intentionally. Add the license appropriate for the
paper/repository before publishing the code on GitHub.

### Manual dataset placement

If automatic download is unavailable, place the standard Planetoid files under
`data/planetoid/` (or pass `--data-root`). For each dataset the loader expects
`ind.<dataset>.x`, `ind.<dataset>.tx`, `ind.<dataset>.allx`,
`ind.<dataset>.graph`, and `ind.<dataset>.test.index`.
