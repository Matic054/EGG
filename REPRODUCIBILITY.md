# Reproducibility checklist

## Canonical command

```bash
python run_experiments.py --out results/paper
python analyze_results.py results/paper
```

## Expected benchmark cardinality

A complete default run contains

```text
3 datasets × 10 folds × 2 settings × 4 methods = 240 selected rows
```

in `results/paper/selected_results.csv`.

If `--include-classical` is used, the complete raw experiment has

```text
3 datasets × 10 folds × 2 settings × 9 methods = 540 selected rows
```

(the topology-only baselines are written into both feature-setting comparison
blocks, though computed only once per split).

The tuning file is larger because it contains every learned epoch and every
kernel `(gamma,k)` validation candidate.

## Before a long run

```bash
python check_device.py
python tests/smoke_test.py
```

## Files to archive with a paper release

At minimum, retain:

```text
results/paper/config.json
results/paper/environment.json
results/paper/selected_results.csv
results/paper/tuning_results.csv
results/paper/analysis/
```

These files are sufficient to identify the full configuration and recreate the
reported aggregate tables without retraining the models.

## Determinism caveat

Random seeds are fixed for graph splits, learned-model initialization, NumPy,
and PyTorch. Exact floating-point results can still vary slightly across
PyTorch/SciPy versions, CPU BLAS libraries, CUDA versions, and hardware.

## Runtime caveat

Runtime should only be interpreted as a controlled speed comparison when the
methods being compared were executed on comparable hardware/device settings.
The kernel is a SciPy CPU implementation. Learned models may use CPU or CUDA.
