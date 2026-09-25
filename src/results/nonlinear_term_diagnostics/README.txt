NONLINEAR TERM DIAGNOSTIC
=========================

Gradient decomposition:
  linear = S M Z
  cubic  = S Z (Z^T Z)

Recorded quantities:
  ratio  = ||cubic||_F / ||linear||_F
  cosine = <linear,cubic>_F / (||linear||_F ||cubic||_F)

Useful regime: checkpoints with validation AUC >= 0.990 * best validation AUC in each split.

CORA
----
Validation-selected checkpoint cosine: 0.959444 ± 0.021063
Validation-selected cubic/linear norm ratio: 0.802407 ± 0.083804
Useful-regime cosine: mean=0.953440, min=0.904148, q10=0.926479, median=0.956822
Useful-regime cubic/linear norm ratio: mean=0.755382, median=0.785774

Interpretation
--------------
A large norm ratio means the cubic term is not negligible in magnitude.
A cosine close to one means it is nevertheless strongly aligned with
the linear term, supporting the interpretation that it primarily acts
as a saturation/magnitude correction rather than an unrelated direction.

Do not describe the cubic term as 'small' unless the ratio actually
supports that statement.  The paper's approximation is directional,
not a claim that the nonlinear term vanishes.
