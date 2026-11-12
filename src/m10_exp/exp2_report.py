"""Tables and plots for Experiment 2, built from fold_metrics.csv and val_scores.npz."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src.m8_eval.stats import add_holm, corrected_resampled_ttest, wilcoxon_p  # noqa: E402
from src.m10_exp.common import markdown_table, report_header, stamp_synthetic, summarise  # noqa: E402

HEAD = ["balanced_accuracy", "recall", "roc_auc"]
KEYS = ["model", "regime", "feature_set"]


def _arm(results, model, regime, fs):
    return results[(results.model == model) & (results.regime == regime) & (results.feature_set == fs)]


def compare(results, a: tuple, b: tuple, ratio: float) -> list[dict]:
    ra = _arm(results, *a).set_index(["repeat", "fold"])
    rb = _arm(results, *b).set_index(["repeat", "fold"])
    idx = ra.index.intersection(rb.index)
    out = []
    for metric in HEAD:
        x, y = ra.loc[idx, metric].to_numpy(), rb.loc[idx, metric].to_numpy()
        t = corrected_resampled_ttest(x, y, ratio)
        out.append({"quantum": "/".join(a), "classical": "/".join(b), "metric": metric,
                    "quantum_mean": x.mean(), "classical_mean": y.mean(), "mean_diff": t["mean_diff"],
                    "ci95_low": t["ci95_low"], "ci95_high": t["ci95_high"],
                    "quantum_wins": int((x > y).sum()), "k": len(idx),
                    "wilcoxon_p": wilcoxon_p(x, y), "p_nb_t": t["p"]})
    return out


def comparison_tables(results: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ratio = float(results["n_val"].mean() / results["n_fold_train"].mean())
    quantum = config["experiment"]["quantum"]
    classical = config["experiment"]["classical_models"]
    same = []
    for q, spec in quantum.items():
        for c in classical:
            same += compare(results, (q, "subsample", spec["feature_set"]), (c, "subsample", spec["feature_set"]), ratio)
    same = pd.DataFrame(same)
    full = results[(results.regime == "full") & (results.feature_set == "all95") & (results.family == "classical")]
    ceiling = full.groupby("model").roc_auc.mean().idxmax()
    ceil = []
    for q, spec in quantum.items():
        ceil += compare(results, (q, "subsample", spec["feature_set"]), (ceiling, "full", "all95"), ratio)
    ceil = pd.DataFrame(ceil)
    same, ceil = add_holm(same, ["metric"]), add_holm(ceil, ["metric"])
    return same, ceil, ceiling


def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:.2g}" if c.startswith(("wilcoxon", "p_")) else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mean_roc(run_dir: Path, results: pd.DataFrame, labels: np.ndarray, arms: list[tuple], out: Path, loader=None) -> None:
    scores = np.load(run_dir / "val_scores.npz")
    val_rows = pd.read_csv(run_dir / "cv_val_rows.csv")
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    for label, (model, regime, fs), style in arms:
        tprs, aucs = [], []
        for (rep, fold), _ in _arm(results, model, regime, fs).groupby(["repeat", "fold"]):
            key = f"{model}__{regime}__{fs}__r{rep}f{fold}"
            if key not in scores:
                continue
            y = labels[val_rows[(val_rows.repeat == rep) & (val_rows.fold == fold)].row.to_numpy()]
            fpr, tpr, _ = roc_curve(y, scores[key])
            tprs.append(np.interp(grid, fpr, tpr))
        if not tprs:
            continue
        auc = _arm(results, model, regime, fs).roc_auc
        ax.plot(grid, np.mean(tprs, axis=0), style, lw=1.6,
                label=f"{label} (AUC {auc.mean():.3f} ± {auc.std(ddof=1):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("Mean ROC over outer folds")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if loader is not None:
        stamp_synthetic(fig, loader)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_report(run_dir: Path, results: pd.DataFrame, loader, config: dict) -> None:
    # Only report models named in the config (RandomForest was dropped on supervisor instruction).
    keep = set(config["experiment"]["classical_models"]) | set(config["experiment"]["quantum"])
    results = results[results.model.isin(keep)].reset_index(drop=True)
    summary = summarise(results, KEYS, HEAD + ["pr_auc", "accuracy", "fit_seconds", "tuning_seconds", "n_params"])
    summary.insert(0, "data_source", loader.source_tag())
    summary = summary.sort_values(["regime", "feature_set", "model"]).reset_index(drop=True)
    summary.to_csv(run_dir / "summary.csv", index=False)
    same, ceil, ceiling = comparison_tables(results, config)
    same.to_csv(run_dir / "tests_same_rows_features.csv", index=False)
    ceil.to_csv(run_dir / "tests_vs_full_data_ceiling.csv", index=False)

    frame = loader.load()
    labels = frame[loader.label_column].to_numpy(dtype=int)
    q = config["experiment"]["quantum"]
    arms = [("QSVM angle PCA-8 (1.5k rows)", ("qsvm", "subsample", q["qsvm"]["feature_set"]), "-"),
            ("VQC re-upload PCA-4 (1.5k rows)", ("vqc", "subsample", q["vqc"]["feature_set"]), "-"),
            ("Hybrid QNN angle 8q (1.5k rows)", ("hybrid", "subsample", q["hybrid"]["feature_set"]), "-"),
            ("LogReg PCA-8 (1.5k rows)", ("logreg", "subsample", "pca8"), "--"),
            (f"{ceiling} all-95 (full data)", (ceiling, "full", "all95"), "k-")]
    mean_roc(run_dir, results, labels, arms, run_dir / "exp2_roc.png", loader)

    counts = results[results.regime == "subsample"]
    k = results.groupby(["repeat", "fold"]).ngroups
    lines = report_header("Experiment 2 — quantum vs classical (no privacy)", loader)
    lines += [
        f"{k} outer folds (3 × 5 repeated stratified CV, identical to Phase 1). Quantum models and the "
        f"'subsample' classical arms train on the same stratified {int(counts.n_train_rows.iloc[0])}-row subsample "
        f"per fold ({counts.n_train_pos.min()}–{counts.n_train_pos.max()} positives; inner-train "
        f"{int(counts.n_inner_train.iloc[0])}, inner-val {int(counts.n_inner_val.iloc[0])} with "
        f"{counts.n_inner_val_pos.min()}–{counts.n_inner_val_pos.max()} positives). 'full' arms use all fold-training rows. "
        f"All arms are scored on the same outer folds ({int(results.n_val.iloc[0])} rows, "
        f"{results.n_val_pos.min()}–{results.n_val_pos.max()} positives).", "",
        "Every model uses the same tuning protocol (grid, early-stopping epoch and Youden threshold chosen on inner-val "
        "only). Tests are paired over folds. Primary: Nadeau–Bengio corrected resampled t (`p_nb_t`), Holm-adjusted "
        "within each metric (`p_nb_t_holm`; `significant` = p_nb_t_holm < 0.05). Sensitivity check: two-sided Wilcoxon "
        "signed-rank, Holm-adjusted the same way (`wilcoxon_p_holm`). `quantum_wins` = folds where the quantum model "
        "scored higher.", "",
        "## Utility (mean ± std over folds)", "",
        markdown_table(summary, KEYS, HEAD + ["pr_auc", "accuracy"]), "",
        "## Quantum vs classical on identical rows and features", "",
        _fmt(same, ["quantum", "classical", "metric", "quantum_mean", "classical_mean", "mean_diff",
                    "ci95_low", "ci95_high", "quantum_wins", "p_nb_t", "p_nb_t_holm", "significant", "wilcoxon_p", "wilcoxon_p_holm"]), "",
        f"## Quantum (1.5k rows, reduced features) vs the full-data classical ceiling ({ceiling}, all 95 features)", "",
        _fmt(ceil, ["quantum", "classical", "metric", "quantum_mean", "classical_mean", "mean_diff",
                    "ci95_low", "ci95_high", "quantum_wins", "p_nb_t", "p_nb_t_holm", "significant", "wilcoxon_p", "wilcoxon_p_holm"]), "",
        "## Cost (mean per fold)", "",
        _fmt(summary[KEYS + ["n_params_mean", "fit_seconds_mean", "tuning_seconds_mean"]],
             KEYS + ["n_params_mean", "fit_seconds_mean", "tuning_seconds_mean"]), "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
