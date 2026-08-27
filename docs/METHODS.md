# Methods and implementation details

This document records the final method and benchmark protocol implemented by
`run_experiments.py`.

## 1. Graph and split protocol

For each Planetoid graph, original self-loops are removed and the graph is
handled as undirected. For fold `i`, the split seed is

```text
10000 + i
```

with 10 folds by default.

From unique upper-triangular graph edges:

- `floor(0.05 |E|)` positives are assigned to validation;
- `floor(0.10 |E|)` positives are assigned to test;
- the rest form the training graph;
- validation and test each receive the same number of uniformly sampled
  non-edge pairs as negatives.

All methods in a fold use exactly the same split.

## 2. Graph normalization

Let `A_train` be the symmetric training adjacency. Define

\[
S = D^{-1/2}(A_{\mathrm{train}}+I)D^{-1/2}.
\]

This is used by all learned encoders and by the proposed kernel.

## 3. Learned graph autoencoders

### Linear AE

Featureless:

\[
Z=SW.
\]

With features:

\[
Z=SFW.
\]

### Two-layer GCN AE

Featureless:

\[
H_1=\operatorname{ReLU}(SW_0),
\qquad
Z=SH_1W_1.
\]

Featureful:

\[
H_1=\operatorname{ReLU}(SFW_0),
\qquad
Z=SH_1W_1.
\]

### Three-layer GCN AE

\[
H_1=\operatorname{ReLU}(SXW_0),
\]

\[
H_2=\operatorname{ReLU}(SH_1W_1),
\]

\[
Z=SH_2W_2,
\]

where `X=I` in the featureless setting and `X=F` with node features.

### Decoder and objective

All learned models use the inner-product logits

\[
X=ZZ^T
\]

and the weighted full-adjacency BCE convention used by the classic GAE
implementation, with target `A_train + I`.

Defaults:

- Adam;
- learning rate `0.01`;
- 250 epochs;
- latent dimension 16;
- GCN hidden width 32;
- dropout 0;
- raw Planetoid features when features are enabled.

Validation AUC/AP is computed at every epoch. Test edges are evaluated only
at the validation-selected checkpoint.

For Pubmed, the dense decoder loss is evaluated in row blocks. The code first
accumulates the exact gradient with respect to `Z` over decoder blocks and then
backpropagates that accumulated gradient once through the encoder. This avoids
retaining an `n x n` autograd graph while preserving the exact objective and
gradient.

## 4. Taylor-BCE dynamics

For unweighted logistic BCE around zero logits,

\[
\ell(x,y)
=\log(1+e^x)-yx
\approx C + \frac18[x-(4y-2)]^2.
\]

Writing

\[
Y=A_{\mathrm{train}}+I,
\qquad
M=4Y-2J,
\]

leads to the squared surrogate

\[
L(Z)=\frac1{n^2}\|M-ZZ^T\|_F^2.
\]

For the featureless linear encoder `Z=SW`, gradient descent in `W` obeys

\[
W_{t+1}
= W_t + \alpha_0
\left[
SMSW_t-S^2W_t(W_t^TS^2W_t)
\right],
\]

where the second term is cubic in the embedding/factor.

Dropping the cubic saturation term gives the finite-time linearized dynamics

\[
W_{t+1}=(I+\alpha SMS)W_t.
\]

For an isotropic initialization,

\[
\mathbb{E}[W_0W_0^T]\propto I,
\]

so the expected decoder kernel is, up to a positive scalar,

\[
K_k=S(I+\alpha SMS)^{2k}S.
\]

## 5. Positive-spectrum approximation

Let

\[
SMS=V\Lambda V^T.
\]

Instead of forming the dense kernel, the implementation keeps the exact sparse
baseline `S^2` and approximates only the spectral correction:

\[
K_k
\approx
S^2+(SV)
\operatorname{diag}\left[(1+\alpha\lambda_i)^{2k}-1\right]
(SV)^T.
\]

Only queried validation/test edge scores are evaluated.

The final benchmark retains:

```text
positive rank = 256
negative rank = 0
```

The positive-only choice is intentional. Exploratory Pubmed diagnostics showed
that explicitly retaining the extreme negative spectrum can severely degrade
link-ranking performance, while the positive spectrum gives stable behavior
across all three datasets.

## 6. Cross-dataset normalization

The initial derivation gives `alpha = 4 eta / n^2`, but keeping a fixed `eta`
across datasets makes the effective spectral evolution strongly graph-size
dependent.

The final method therefore normalizes by the largest positive retained
eigenvalue:

\[
\boxed{
\alpha=\frac{\gamma}{\lambda_{\max}^{+}}
}
\]

with

```text
gamma ∈ {0.01, 0.02, 0.05, 0.10}
k     ∈ {0, 5, 10, ..., 300}
```

and validation selection of `(gamma,k)`.

A useful dimensionless time coordinate is

\[
\boxed{
\tau=2k\gamma.
}
\]

For small `gamma`, the relative spectral weights satisfy approximately

\[
\left(1+\gamma\frac{\lambda}{\lambda_{\max}^+}\right)^{2k}
\approx
\exp\left(\tau\frac{\lambda}{\lambda_{\max}^+}\right).
\]

Thus different `(gamma,k)` pairs at similar `tau` represent nearly the same
spectral evolution.

## 7. Feature-aware kernel

With features `F`, let

\[
H=SF.
\]

The Taylor objective becomes

\[
L(W)=\frac1{n^2}\|M-HWW^TH^T\|_F^2.
\]

The linearized feature-space operator is

\[
A_F=H^TMH=F^TSMSF.
\]

If

\[
A_F=V\Lambda V^T,
\]

the expected finite-time kernel is

\[
K_k^{(F)}
=HF_kH^T,
\qquad
F_k=(I+\alpha A_F)^{2k},
\]

implemented as

\[
K_k^{(F)}
\approx
HH^T+(HV)
\operatorname{diag}\left[(1+\alpha\lambda_i)^{2k}-1\right]
(HV)^T.
\]

The same normalization is used:

\[
\alpha=\gamma/\lambda_{\max}^{+}(A_F).
\]

## 8. Evaluation

For every selected model/configuration, the final metrics are:

- ROC-AUC;
- average precision (AP);
- wall-clock runtime.

The analysis script reports mean and sample standard deviation over the 10
random edge splits and computes paired fold-level comparisons against the
kernel.
