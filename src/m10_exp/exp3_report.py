"""Tables and the privacy-utility curve for Experiment 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.m8_eval.stats import holm, wilcoxon_p  # noqa: E402
from src.m10_exp.common import report_header, stamp_synthetic  # noqa: E402

HEAD = ["roc_auc", "balanced_accuracy", "recall"]
# Validated categorical slots 1-3 (dataviz reference palette, light surface).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def eps_order(values) -> list[str]:
    finite = sorted({v for v in values if v != "inf"}, key=float)
    return finite + (["inf"] if "inf" in set(values) else [])


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby(["arm", "target_epsilon"], sort=False)
    out = g[HEAD].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["epsilon_spent_mean"] = g["epsilon_spent"].mean()
    out["epsilon_spent_max"] = g["epsilon_spent"].max()
    out["noise_multiplier"] = g["noise_multiplier"].mean()
    out["fit_seconds_mean"] = g["fit_seconds"].mean()
    out["n_folds"] = g.size()
    return out.reset_index()


def _series(results, arm, eps, metric="roc_auc") -> pd.Series:
    sel = results[(results.arm == arm) & (results.target_epsilon == eps)]
    return sel.set_index(["repeat", "fold"])[metric].sort_index()


def _pairs(results, a_arm, a_eps, b_arm, b_eps, metric="roc_auc"):
    """Values of two arms aligned on (repeat, fold)."""
    a, b = _series(results, a_arm, a_eps, metric), _series(results, b_arm, b_eps, metric)
    idx = a.index.intersection(b.index).sort_values()
    return a[idx].to_numpy(), b[idx].to_numpy()


def cost_of_privacy(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in results.arm.unique():
        for eps in eps_order(results[results.arm == arm].target_epsilon):
            if eps == "inf":
                continue
            for metric in HEAD:
                x, ref = _pairs(results, arm, eps, arm, "inf", metric)
                rows.append({"arm": arm, "epsilon": eps, "metric": metric, "private_mean": x.mean(),
                             "nonprivate_mean": ref.mean(), "change": (x - ref).mean(),
                             "wilcoxon_p": wilcoxon_p(x, ref)})
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = out.groupby(["arm", "metric"])["wilcoxon_p"].transform(lambda p: holm(p.to_numpy()))
    return out


def quantum_vs_classical(results: pd.DataFrame, pairs: list[list[str]]) -> pd.DataFrame:
    rows = []
    for q_arm, c_arm in pairs:
        for eps in eps_order(results[results.arm == q_arm].target_epsilon):
            xq, xc = _pairs(results, q_arm, eps, c_arm, eps)
            row = {"quantum": q_arm, "classical": c_arm, "epsilon": eps, "quantum_auc": xq.mean(),
                   "classical_auc": xc.mean(), "auc_diff": (xq - xc).mean(), "quantum_wins": int((xq > xc).sum()),
                   "k": len(xq), "wilcoxon_p": wilcoxon_p(xq, xc)}
            if eps != "inf":  # degradation relative to each model's own non-private AUC
                dq = _series(results, q_arm, eps) - _series(results, q_arm, "inf")
                dc = _series(results, c_arm, eps) - _series(results, c_arm, "inf")
                idx = dq.dropna().index.intersection(dc.dropna().index).sort_values()
                dq, dc = dq[idx].to_numpy(), dc[idx].to_numpy()
                row.update({"quantum_auc_drop": -dq.mean(), "classical_auc_drop": -dc.mean(),
                            "drop_diff_wilcoxon_p": wilcoxon_p(dq, dc)})
            rows.append(row)
    out = pd.DataFrame(rows)
    for col in ("wilcoxon_p", "drop_diff_wilcoxon_p"):
        if col in out:
            mask = out[col].notna()
            out.loc[mask, col + "_holm"] = out[mask].groupby(["quantum", "classical"])[col].transform(
                lambda p: holm(p.to_numpy()))
    return out


def plot_curve(summary: pd.DataFrame, labels: dict, out: Path, delta: float, loader=None) -> None:
    eps = eps_order(summary.target_epsilon)
    x_pos = {e: i for i, e in enumerate(eps)}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, metric, title in [(axes[0], "roc_auc", "ROC-AUC"), (axes[1], "balanced_accuracy", "Balanced accuracy")]:
        for i, arm in enumerate(labels):
            s = summary[summary.arm == arm].set_index("target_epsilon").reindex(eps)
            xs = [x_pos[e] for e in eps]
            ys = s[f"{metric}_mean"].to_numpy()
            ax.errorbar(xs, ys, yerr=s[f"{metric}_std"], color=COLORS[i], marker=MARKERS[i],
                        markersize=8, lw=0, elinewidth=1.5, capsize=3, label=labels[arm],
                        markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=3)
            ax.plot(xs[:-1], ys[:-1], color=COLORS[i], lw=2, zorder=2)  # finite budgets
            ax.plot(xs[-2:], ys[-2:], color=COLORS[i], lw=1.2, ls=":", zorder=2)  # break to infinity
            ax.annotate(f"{s[f'{metric}_mean'].iloc[-1]:.3f}", (xs[-1], s[f"{metric}_mean"].iloc[-1]),
                        xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(eps)), [("∞\n(non-private)" if e == "inf" else e) for e in eps])
        ax.axvline(len(eps) - 1.5, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel(f"Privacy budget ε  (δ = {delta:g}; smaller ε = stronger privacy)", color=MUTED)
        ax.set_ylabel(f"{title} (mean ± std over folds)", color=MUTED)
        ax.set_title(title, color=INK, loc="left")
        ax.grid(axis="y", color=GRID)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlim(-0.4, len(eps) - 0.4)
    axes[0].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("Privacy–utility trade-off: DP-SGD (Opacus RDP accounting)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    if loader is not None:
        stamp_synthetic(fig, loader)
    fig.savefig(out, dpi=170, facecolor="#fcfcfb")
    plt.close(fig)


def _fmt(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else (f"{v:.2g}" if "wilcoxon" in c else f"{v:.3f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(run_dir: Path, results: pd.DataFrame, loader, config: dict) -> None:
    results = results.copy()
    results["target_epsilon"] = [
        "inf" if str(v) in ("inf", "nan", "None") else f"{float(v):g}" for v in results["target_epsilon"]]
    exp = config["experiment"]
    summary = summary_table(results)
    arm_rank = {a: i for i, a in enumerate(exp["arms"])}
    eps_rank = {e: i for i, e in enumerate(eps_order(results.target_epsilon))}
    summary = summary.sort_values(by=["arm", "target_epsilon"], key=lambda c: c.map(
        arm_rank if c.name == "arm" else eps_rank)).reset_index(drop=True)
    summary.insert(0, "data_source", loader.source_tag())
    summary.to_csv(run_dir / "summary.csv", index=False)
    cost = cost_of_privacy(results)
    cost.to_csv(run_dir / "tests_private_vs_nonprivate.csv", index=False)
    qvc = quantum_vs_classical(results, exp["comparisons"])
    qvc.to_csv(run_dir / "tests_quantum_vs_classical.csv", index=False)
    labels = {arm: spec.get("label", arm) for arm, spec in exp["arms"].items()}
    plot_curve(summary, labels, run_dir / "privacy_utility_curve.png", float(exp["delta"]), loader)

    spent = results[results.dp_trained.astype(str) == "True"]
    k = results.groupby(["repeat", "fold"]).ngroups
    table = summary[["arm", "target_epsilon", "roc_auc_mean", "roc_auc_std", "balanced_accuracy_mean",
                     "balanced_accuracy_std", "recall_mean", "recall_std", "epsilon_spent_mean",
                     "noise_multiplier", "fit_seconds_mean"]]
    lines = report_header("Experiment 3 — privacy–utility curve (DP-SGD)", loader)
    lines += [
        f"{k} outer folds. DP-SGD via Opacus (Poisson sampling, per-sample clipping, Gaussian noise); ε is calibrated "
        f"and read back from Opacus's RDP accountant (δ = {exp['delta']}). Achieved ε within "
        f"{(spent.epsilon_spent / spent.target_epsilon.astype(float)).min():.3f}–"
        f"{(spent.epsilon_spent / spent.target_epsilon.astype(float)).max():.3f} × target across all private runs. "
        "ε = ∞ is the same model and optimiser without clipping or noise. QSVM is excluded (not gradient-trained). "
        "Not covered by ε: fixed hyperparameters and the decision threshold chosen on inner-val rows. "
        "Privacy comes from DP-SGD only; the quantum circuit is a representation layer.", "",
        "Arms: " + "; ".join(f"`{a}` = {s.get('label', a)} (lr {s['learning_rate']}, {s['epochs']} epochs, "
                              f"expected batch {s['batch_size']}, clip {s.get('max_grad_norm', 1.0)})"
                              for a, s in exp["arms"].items()), "",
        "![privacy-utility curve](privacy_utility_curve.png)", "",
        "## Utility by ε (mean / std over folds)", "", _fmt(table), "",
        "## Cost of privacy: each ε vs the same arm at ε = ∞ (paired Wilcoxon, Holm within arm × metric)", "",
        _fmt(cost), "",
        "## Quantum vs classical at each ε, and whether quantum degrades differently", "",
        "`auc_drop` = mean fold-wise loss of ROC-AUC relative to the arm's own ε = ∞ model; "
        "`drop_diff_wilcoxon_p` tests whether the quantum drop differs from the classical drop on the same folds.", "",
        _fmt(qvc), "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
