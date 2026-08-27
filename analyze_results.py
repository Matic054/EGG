#!/usr/bin/env python3
"""Analyze a completed benchmark and generate paper-ready tables.

The main table mirrors the layout of Table 1 in Salha et al. (2020), with an
additional runtime column for each dataset.  It contains only the four methods
used in the final comparison:

    Linear AE
    2-layer GCN AE
    3-layer GCN AE
    Our normalized finite-time kernel

Outputs also include paired fold-wise differences, win counts, selected epochs
and kernel spectral times.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t

METHODS = ["linear_ae", "gcn_ae_2", "gcn_ae_3", "linearized_gae_kernel"]
METHOD_LABELS = {
    "linear_ae": "Linear AE",
    "gcn_ae_2": "2-layer GCN AE",
    "gcn_ae_3": "3-layer GCN AE",
    "linearized_gae_kernel": "Our kernel",
}
DATASETS = ["cora", "citeseer", "pubmed"]
DATASET_LABELS = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "Pubmed"}
SETTINGS = ["featureless", "features"]
SETTING_LABELS = {
    "featureless": "Without node features",
    "features": "With node features",
}
KERNEL = "linearized_gae_kernel"
CLASSICAL = ["ppr", "katz", "common_neighbors", "adamic_adar", "resource_allocation"]


def fmt_pct(mean, sd):
    return f"{100*mean:.2f} ± {100*sd:.2f}"


def fmt_sec(mean, sd):
    return f"{mean:.2f} ± {sd:.2f}"


def geomean(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    return np.nan if len(a) == 0 else float(np.exp(np.mean(np.log(a))))


def paired_ci(diff, confidence=0.95):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan, np.nan
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    q = t.ppf((1 + confidence) / 2, df=len(d) - 1)
    return float(mean - q * se), float(mean + q * se)


def summarize(df):
    x = df[df.method.isin(METHODS)].copy()
    return (
        x.groupby(["setting", "dataset", "method"], sort=False)
        .agg(
            folds=("fold", "size"),
            auc_mean=("test_auc", "mean"),
            auc_sd=("test_auc", "std"),
            auc_sem=("test_auc", "sem"),
            ap_mean=("test_ap", "mean"),
            ap_sd=("test_ap", "std"),
            ap_sem=("test_ap", "sem"),
            runtime_mean=("total_runtime_seconds", "mean"),
            runtime_sd=("total_runtime_seconds", "std"),
            runtime_median=("total_runtime_seconds", "median"),
            val_auc_mean=("val_auc", "mean"),
        )
        .reset_index()
    )


def table_panel(summary, setting):
    rows = []
    for method in METHODS:
        row = {"Model": METHOD_LABELS[method]}
        for dataset in DATASETS:
            g = summary[
                (summary.setting == setting)
                & (summary.dataset == dataset)
                & (summary.method == method)
            ]
            if g.empty:
                auc = ap = runtime = ""
            else:
                r = g.iloc[0]
                auc = fmt_pct(r.auc_mean, r.auc_sd)
                ap = fmt_pct(r.ap_mean, r.ap_sd)
                runtime = fmt_sec(r.runtime_mean, r.runtime_sd)
            row[f"{DATASET_LABELS[dataset]} AUC (%)"] = auc
            row[f"{DATASET_LABELS[dataset]} AP (%)"] = ap
            row[f"{DATASET_LABELS[dataset]} Runtime (s)"] = runtime
        rows.append(row)
    return pd.DataFrame(rows)


def best_method(summary, setting, dataset, metric, lower=False):
    g = summary[(summary.setting == setting) & (summary.dataset == dataset)]
    if g.empty:
        return None
    idx = g[metric].idxmin() if lower else g[metric].idxmax()
    return g.loc[idx, "method"]


def tex_metric(mean, sd, pct=False, bold=False):
    if pct:
        mean, sd = 100*mean, 100*sd
    s = f"{mean:.2f} $\\pm$ {sd:.2f}"
    return r"\textbf{" + s + "}" if bold else s


def make_latex(summary):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Link prediction on Cora, Citeseer, and Pubmed. Results are mean $\pm$ sample standard deviation across random edge splits. AUC and AP are in percent; runtime is wall-clock seconds.}",
        r"\label{tab:main_link_prediction}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l|ccc|ccc|ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c|}{Cora} & \multicolumn{3}{c|}{Citeseer} & \multicolumn{3}{c}{Pubmed} \\",
        r"Model & AUC (\%) $\uparrow$ & AP (\%) $\uparrow$ & Runtime (s) $\downarrow$ & AUC (\%) $\uparrow$ & AP (\%) $\uparrow$ & Runtime (s) $\downarrow$ & AUC (\%) $\uparrow$ & AP (\%) $\uparrow$ & Runtime (s) $\downarrow$ \\",
        r"\midrule",
    ]
    for si, setting in enumerate(SETTINGS):
        lines.append(r"\multicolumn{10}{l}{\textit{" + SETTING_LABELS[setting] + r"}} \\")
        for method in METHODS:
            cells = [METHOD_LABELS[method]]
            for dataset in DATASETS:
                g = summary[
                    (summary.setting == setting)
                    & (summary.dataset == dataset)
                    & (summary.method == method)
                ]
                if g.empty:
                    cells += ["--", "--", "--"]
                    continue
                r = g.iloc[0]
                cells += [
                    tex_metric(r.auc_mean, r.auc_sd, pct=True,
                               bold=best_method(summary, setting, dataset, "auc_mean") == method),
                    tex_metric(r.ap_mean, r.ap_sd, pct=True,
                               bold=best_method(summary, setting, dataset, "ap_mean") == method),
                    tex_metric(r.runtime_mean, r.runtime_sd,
                               bold=best_method(summary, setting, dataset, "runtime_mean", lower=True) == method),
                ]
            lines.append(" & ".join(cells) + r" \\")
        if si == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    return "\n".join(lines)


def paired_analysis(df):
    rows = []
    for setting in SETTINGS:
        for dataset in DATASETS:
            k = df[
                (df.setting == setting) & (df.dataset == dataset) & (df.method == KERNEL)
            ][["fold", "test_auc", "test_ap", "total_runtime_seconds"]].rename(
                columns={"test_auc":"k_auc", "test_ap":"k_ap", "total_runtime_seconds":"k_time"}
            )
            for baseline in METHODS[:-1]:
                b = df[
                    (df.setting == setting) & (df.dataset == dataset) & (df.method == baseline)
                ][["fold", "test_auc", "test_ap", "total_runtime_seconds"]].rename(
                    columns={"test_auc":"b_auc", "test_ap":"b_ap", "total_runtime_seconds":"b_time"}
                )
                m = k.merge(b, on="fold")
                if m.empty:
                    continue
                da = m.k_auc - m.b_auc
                dp = m.k_ap - m.b_ap
                auc_lo, auc_hi = paired_ci(da)
                ap_lo, ap_hi = paired_ci(dp)
                auc_wins = int((da > 0).sum())
                ap_wins = int((dp > 0).sum())
                n_nonzero_auc = int((da != 0).sum())
                n_nonzero_ap = int((dp != 0).sum())
                rows.append({
                    "setting": setting,
                    "dataset": dataset,
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS[baseline],
                    "paired_folds": len(m),
                    "auc_gain_mean": da.mean(),
                    "auc_gain_sd": da.std(ddof=1),
                    "auc_gain_ci95_low": auc_lo,
                    "auc_gain_ci95_high": auc_hi,
                    "ap_gain_mean": dp.mean(),
                    "ap_gain_sd": dp.std(ddof=1),
                    "ap_gain_ci95_low": ap_lo,
                    "ap_gain_ci95_high": ap_hi,
                    "kernel_auc_wins": auc_wins,
                    "kernel_ap_wins": ap_wins,
                    "auc_sign_test_p_two_sided": (
                        binomtest(auc_wins, n_nonzero_auc, .5).pvalue if n_nonzero_auc else np.nan
                    ),
                    "ap_sign_test_p_two_sided": (
                        binomtest(ap_wins, n_nonzero_ap, .5).pvalue if n_nonzero_ap else np.nan
                    ),
                    "speedup_geomean": geomean(m.b_time / m.k_time),
                    "speedup_median": float(np.median(m.b_time / m.k_time)),
                })
    return pd.DataFrame(rows)


def win_counts(df):
    x = df[df.method.isin(METHODS)].copy()
    rows = []
    for (setting, dataset, fold), g in x.groupby(["setting", "dataset", "fold"]):
        amax, pmax = g.test_auc.max(), g.test_ap.max()
        for _, r in g.iterrows():
            rows.append({
                "setting": setting, "dataset": dataset, "fold": fold, "method": r.method,
                "auc_best": bool(np.isclose(r.test_auc, amax)),
                "ap_best": bool(np.isclose(r.test_ap, pmax)),
            })
    raw = pd.DataFrame(rows)
    return raw.groupby(["setting", "dataset", "method"], as_index=False).agg(
        folds=("fold", "size"), auc_best_folds=("auc_best", "sum"), ap_best_folds=("ap_best", "sum")
    )


def selected_hparams(df):
    cols = [
        "dataset", "fold", "setting", "method", "selected_epoch",
        "selected_gamma", "selected_k", "selected_spectral_time",
        "selected_hyperparameter", "learning_rate", "hidden_dim", "latent_dim",
        "alpha", "lambda_max_positive", "positive_rank", "actual_positive_rank",
    ]
    return df[df.method.isin(METHODS)][[c for c in cols if c in df.columns]].copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path, nargs="?", default=Path("results/paper"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    source = args.results / "selected_results.csv" if args.results.is_dir() else args.results
    if not source.exists():
        raise FileNotFoundError(source)
    df = pd.read_csv(source)
    missing = [m for m in METHODS if m not in set(df.method.astype(str))]
    if missing:
        print(f"Warning: missing methods: {missing}", file=sys.stderr)

    out = args.out or source.parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    s = summarize(df)
    s.to_csv(out / "summary_long.csv", index=False)
    df[df.method.isin(METHODS)].to_csv(out / "fold_level_main_methods.csv", index=False)

    # Optional classical baselines are deliberately excluded from the main table,
    # but summarized separately when present so ancillary claims remain reproducible.
    classical = df[df.method.isin(CLASSICAL)].copy()
    if len(classical):
        classical_summary = (
            classical.groupby(["setting", "dataset", "method"], sort=False)
            .agg(
                folds=("fold", "size"),
                auc_mean=("test_auc", "mean"), auc_sd=("test_auc", "std"),
                ap_mean=("test_ap", "mean"), ap_sd=("test_ap", "std"),
                runtime_mean=("total_runtime_seconds", "mean"),
                runtime_sd=("total_runtime_seconds", "std"),
            )
            .reset_index()
        )
        classical_summary.to_csv(out / "classical_baselines_summary.csv", index=False)

    for setting in SETTINGS:
        table = table_panel(s, setting)
        table.to_csv(out / f"table1_like_{setting}.csv", index=False)
        try:
            md = table.to_markdown(index=False)
        except Exception:
            md = table.to_csv(index=False)
        (out / f"table1_like_{setting}.md").write_text(md, encoding="utf-8")

    (out / "table1_like.tex").write_text(make_latex(s), encoding="utf-8")
    paired_analysis(df).to_csv(out / "paired_vs_kernel.csv", index=False)
    win_counts(df).to_csv(out / "win_counts.csv", index=False)
    selected_hparams(df).to_csv(out / "selected_hyperparameters.csv", index=False)

    print("\nWITHOUT NODE FEATURES")
    print(table_panel(s, "featureless").to_string(index=False))
    print("\nWITH NODE FEATURES")
    print(table_panel(s, "features").to_string(index=False))
    print(f"\nAnalysis written to: {out.resolve()}")


if __name__ == "__main__":
    main()
