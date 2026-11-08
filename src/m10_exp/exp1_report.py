"""Tables and plots for Experiment 1, written from fold_metrics.csv only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.m8_eval.stats import corrected_resampled_ttest, holm, wilcoxon_p  # noqa: E402
from src.m10_exp.common import markdown_table, report_header, summarise  # noqa: E402

HEAD = ["balanced_accuracy", "recall", "roc_auc", "pr_auc"]


def _ratio(results: pd.DataFrame) -> float:
    return float(results["n_val"].mean() / results["n_fold_train"].mean())


def paired(results, keys_fixed: dict, vary: str, a, b, metric="roc_auc") -> dict:
    sel = results
    for k, v in keys_fixed.items():
        sel = sel[sel[k] == v]
    pa = sel[sel[vary] == a].set_index(["repeat", "fold"])[metric]
    pb = sel[sel[vary] == b].set_index(["repeat", "fold"])[metric]
    common = pa.index.intersection(pb.index)
    test = corrected_resampled_ttest(pa[common].to_numpy(), pb[common].to_numpy(), _ratio(results))
    test["p_nb_t"] = test.pop("p")
    test["wilcoxon_p"] = wilcoxon_p(pa[common].to_numpy(), pb[common].to_numpy())
    return test


def encoding_tests(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q = results[results.model.isin(["qsvm", "vqc", "hybrid"])]
    for (model, method, n), g in q.groupby(["model", "feature_method", "n_qubits"]):
        means = g.groupby("encoding")["roc_auc"].mean().sort_values(ascending=False)
        best = means.index[0]
        for other in means.index[1:]:
            t = paired(q, {"model": model, "feature_method": method, "n_qubits": n}, "encoding", best, other)
            rows.append({"model": model, "feature_method": method, "n_qubits": n, "best": best,
                         "vs": other, "auc_best": means[best], "auc_other": means[other], **t})
    out = pd.DataFrame(rows)
    if len(out):
        out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"])
        out["wilcoxon_p_holm_in_config"] = out.groupby(["model", "feature_method", "n_qubits"])[
            "wilcoxon_p"].transform(lambda p: holm(p.to_numpy()))
    return out


def swing_table(results: pd.DataFrame) -> pd.DataFrame:
    q = results[results.model.isin(["qsvm", "vqc", "hybrid"])]
    m = q.groupby(["model", "feature_method", "n_qubits", "encoding"])[["roc_auc", "balanced_accuracy"]].mean().reset_index()
    rows = []
    for (model, method, n), g in m.groupby(["model", "feature_method", "n_qubits"]):
        hi, lo = g.loc[g.roc_auc.idxmax()], g.loc[g.roc_auc.idxmin()]
        rows.append({"model": model, "feature_method": method, "n_qubits": n,
                     "best_encoding": hi.encoding, "best_auc": hi.roc_auc,
                     "worst_encoding": lo.encoding, "worst_auc": lo.roc_auc,
                     "auc_swing": hi.roc_auc - lo.roc_auc,
                     "auc_swing_pct_of_worst": 100 * (hi.roc_auc - lo.roc_auc) / lo.roc_auc,
                     "bal_acc_swing": g.balanced_accuracy.max() - g.balanced_accuracy.min()})
    return pd.DataFrame(rows)


def method_tests(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q = results[results.feature_method.isin(["mi", "pca"])]
    for (model, enc, n), g in q.groupby(["model", "encoding", "n_qubits"]):
        if g.feature_method.nunique() < 2:
            continue
        t = paired(q, {"model": model, "encoding": enc, "n_qubits": n}, "feature_method", "mi", "pca")
        rows.append({"model": model, "encoding": enc, "n_qubits": n,
                     "auc_mi": g[g.feature_method == "mi"].roc_auc.mean(),
                     "auc_pca": g[g.feature_method == "pca"].roc_auc.mean(), **t})
    out = pd.DataFrame(rows)
    if len(out):
        out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"])
    return out


def _fmt_tests(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and float(v).is_integer() and not c.startswith(("p", "wilcoxon", "auc", "mean", "ci", "best_auc", "worst_auc")):
                cells.append(str(int(v)))
            elif isinstance(v, float):
                cells.append(f"{v:.2g}" if c.startswith(("p_", "wilcoxon")) else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_encodings(summary: pd.DataFrame, run_dir: Path) -> None:
    models = [m for m in ["qsvm", "vqc", "hybrid"] if m in set(summary.model)]
    methods = ["mi", "pca", "learned"]
    panels = [(m, f) for m in models for f in methods if ((summary.model == m) & (summary.feature_method == f)).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), sharey=True, squeeze=False)
    encodings = ["angle", "amplitude", "zz", "reupload"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    ref = summary[summary.model == "logreg"]
    for ax, (model, method) in zip(axes[0], panels):
        s = summary[(summary.model == model) & (summary.feature_method == method)]
        qubits = sorted(s.n_qubits.unique())
        width = 0.2
        for i, (enc, col) in enumerate(zip(encodings, colors)):
            e = s[s.encoding == enc].set_index("n_qubits").reindex(qubits)
            ax.bar(np.arange(len(qubits)) + (i - 1.5) * width, e.roc_auc_mean, width,
                   yerr=e.roc_auc_std, color=col, label=enc, capsize=2)
        if method in ("mi", "pca") and len(ref):
            r = ref[ref.feature_method == method].set_index("n_qubits").reindex(qubits)
            ax.plot(np.arange(len(qubits)), r.roc_auc_mean, "k_", markersize=28, mew=2, label="LogReg (same rows/features)")
        ax.set_xticks(np.arange(len(qubits)), [f"{q}q" for q in qubits])
        ax.set_title(f"{model.upper()} — {method}")
        ax.set_ylim(0.5, 0.9)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("ROC-AUC (mean ± std over folds)")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(run_dir / "exp1_encoding_auc.png", dpi=160)
    plt.close(fig)


def write_report(run_dir: Path, results: pd.DataFrame, resources: pd.DataFrame, loader, config) -> None:
    keys = ["model", "encoding", "feature_method", "n_qubits"]
    summary = summarise(results, keys, HEAD + ["fit_seconds", "tuning_seconds", "n_params"])
    summary = summary.sort_values(keys).reset_index(drop=True)
    summary.insert(0, "data_source", loader.source_tag())
    summary.to_csv(run_dir / "summary.csv", index=False)
    enc_t, swing, meth_t = encoding_tests(results), swing_table(results), method_tests(results)
    enc_t.to_csv(run_dir / "encoding_tests.csv", index=False)
    swing.to_csv(run_dir / "encoding_swing.csv", index=False)
    meth_t.to_csv(run_dir / "mi_vs_pca_tests.csv", index=False)
    plot_encodings(summary, run_dir)

    cost = summary[summary.model != "logreg"][["model", "encoding", "feature_method", "n_qubits",
                                               "n_params_mean", "fit_seconds_mean", "tuning_seconds_mean"]]
    counts = results[[c for c in results.columns if c.startswith("n_") and c.endswith(("pos", "val", "train", "subsample"))]]
    q = summary[summary.model.isin(["qsvm", "vqc", "hybrid"])]
    best = q.loc[q.roc_auc_mean.idxmax()]

    lines = report_header("Experiment 1 — quantum encoding comparison (no privacy)", loader)
    lines += [
        f"Folds: {results.groupby(['repeat', 'fold']).ngroups} outer folds (identical to Phase 1). "
        f"Quantum training uses a stratified subsample of each fold's training rows: "
        f"{int(counts.n_subsample.iloc[0])} rows "
        f"(positives {counts.n_subsample_pos.min()}–{counts.n_subsample_pos.max()}), split into inner-train "
        f"{int(counts.n_inner_train.iloc[0])} (positives {counts.n_inner_train_pos.min()}–{counts.n_inner_train_pos.max()}) "
        f"and inner-val {int(counts.n_inner_val.iloc[0])} (positives {counts.n_inner_val_pos.min()}–{counts.n_inner_val_pos.max()}). "
        f"Scored on the full outer validation fold: {int(counts.n_val.iloc[0])} rows "
        f"(positives {counts.n_val_pos.min()}–{counts.n_val_pos.max()}).",
        "",
        "Hyperparameters, early-stopping epoch and decision threshold (Youden J) are selected on inner-val only. "
        "Significance: two-sided Wilcoxon signed-rank test on paired per-fold ROC-AUC (primary; Holm-adjusted "
        "within each table). The Nadeau–Bengio corrected resampled t-test "
        f"(test/train ratio {_ratio(results):.3f}) is reported as a more conservative secondary check, because "
        "CV folds share training rows. CI = 95% interval of the mean paired difference from the corrected t.",
        "",
        f"**Best quantum setup by mean ROC-AUC:** {best.model} / {best.encoding} / {best.feature_method} / "
        f"{int(best.n_qubits)} qubits — ROC-AUC {best.roc_auc_mean:.3f} ± {best.roc_auc_std:.3f}, "
        f"balanced accuracy {best.balanced_accuracy_mean:.3f} ± {best.balanced_accuracy_std:.3f}, "
        f"recall {best.recall_mean:.3f} ± {best.recall_std:.3f}.",
        "",
        "## Utility (mean ± std over folds)", "",
        markdown_table(summary, keys, HEAD), "",
        "## Encoding swing (best minus worst encoding, mean ROC-AUC)", "",
        _fmt_tests(swing, ["model", "feature_method", "n_qubits", "best_encoding", "best_auc",
                           "worst_encoding", "worst_auc", "auc_swing", "auc_swing_pct_of_worst", "bal_acc_swing"]), "",
        "## Best encoding vs each alternative (paired ROC-AUC)", "",
        _fmt_tests(enc_t, ["model", "feature_method", "n_qubits", "best", "vs", "mean_diff",
                           "ci95_low", "ci95_high", "wilcoxon_p", "wilcoxon_p_holm", "wilcoxon_p_holm_in_config", "p_nb_t"]), "",
        f"Note: with k = {results.groupby(['repeat', 'fold']).ngroups} paired folds the smallest attainable two-sided "
        f"Wilcoxon p is {2 / 2 ** results.groupby(['repeat', 'fold']).ngroups:.4f} (all folds agree). Holm over the "
        "whole table (pre-declared, `wilcoxon_p_holm`) therefore cannot fall below that floor times the family size; "
        "`wilcoxon_p_holm_in_config` (secondary, family = the 3 comparisons within one model/feature/qubit "
        "configuration) is shown for interpretation.", "",
        "## MI vs PCA (paired ROC-AUC, positive = MI better)", "",
        _fmt_tests(meth_t, ["model", "encoding", "n_qubits", "auc_mi", "auc_pca", "mean_diff", "wilcoxon_p", "wilcoxon_p_holm", "p_nb_t"]), "",
        "## Circuit cost (decomposed to RX/RY/RZ/CNOT/H)", "",
        _fmt_tests(resources.astype(object).where(resources.notna(), ""), list(resources.columns)), "",
        "## Training cost (mean per fold; tuning = all grid points)", "",
        _fmt_tests(cost, list(cost.columns)), "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
