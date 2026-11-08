"""Tables and convergence plots for Experiment 4 (federated + DP)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.m8_eval.stats import holm, wilcoxon_p  # noqa: E402
from src.m10_exp.common import report_header, stamp_synthetic  # noqa: E402

HEAD = ["roc_auc", "balanced_accuracy", "recall"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated slots 1-3
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _norm_eps(series: pd.Series) -> list[str]:
    return ["inf" if str(v) in ("inf", "nan", "None") else f"{float(v):g}" for v in series]


def load_reference(config: dict, folds: pd.DataFrame) -> pd.DataFrame:
    """Centralised rows (Experiment 3) for the same arms, epsilons and folds."""
    ref_dir = Path(config["experiment"]["reference_run"])
    ref = pd.read_csv(ref_dir / "fold_metrics.csv")
    ref["target_epsilon"] = _norm_eps(ref["target_epsilon"])
    wanted = {"inf" if e is None else f"{float(e):g}" for e in config["experiment"]["epsilons"]}
    ref = ref[ref.arm.isin(config["experiment"]["arms"]) & ref.target_epsilon.isin(wanted)]
    ref = ref.merge(folds, on=["repeat", "fold"])
    ref["setting"] = "centralised"
    ref["n_clients"] = 1
    ref["partition"] = "-"
    return ref


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


def convergence_plot(history: pd.DataFrame, combined: pd.DataFrame, config: dict, out: Path, loader) -> None:
    arms = [a for a in config["experiment"]["arms"] if config["experiment"]["arms"][a]["family"] == "quantum"]
    arms += [a for a in config["experiment"]["arms"] if a not in arms][:1]
    partitions = config["experiment"]["partitions"]
    eps_values = sorted(set(history.target_epsilon), key=lambda e: (e == "inf", float(e) if e != "inf" else 0))
    fig, axes = plt.subplots(len(arms), len(partitions) * len(eps_values), figsize=(3.4 * len(partitions) * len(eps_values),
                             3.0 * len(arms)), squeeze=False, sharey=True)
    labels = config["experiment"]["arms"]
    for i, arm in enumerate(arms):
        col = 0
        for eps in eps_values:
            for partition in partitions:
                ax = axes[i][col]
                col += 1
                h = history[(history.arm == arm) & (history.partition == partition) & (history.target_epsilon == eps)]
                for j, n in enumerate(sorted(h.n_clients.unique())):
                    g = h[h.n_clients == n].groupby("round").val_auc
                    m, s = g.mean(), g.std()
                    ax.plot(m.index, m.values, color=COLORS[j], lw=2, label=f"{n} clients")
                    ax.fill_between(m.index, m - s, m + s, color=COLORS[j], alpha=0.12, lw=0)
                    ax.plot(m.index[-1], m.values[-1], MARKERS[j], color=COLORS[j], markersize=8,
                            markeredgecolor="#fcfcfb")
                ref = combined[(combined.arm == arm) & (combined.setting == "centralised") & (combined.target_epsilon == eps)]
                if len(ref):
                    ax.axhline(ref.roc_auc.mean(), color=INK, lw=1.2, ls="--", label="centralised")
                eps_txt = "non-private" if eps == "inf" else f"ε = {eps}"
                ax.set_title(f"{labels[arm].get('label', arm)}\n{partition}, {eps_txt}", fontsize=8, loc="left", color=INK)
                ax.grid(color=GRID)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                if i == len(arms) - 1:
                    ax.set_xlabel("FedAvg round", color=MUTED)
        axes[i][0].set_ylabel("Outer-fold ROC-AUC", color=MUTED)
    handles, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=len(lab), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    stamp_synthetic(fig, loader)
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def write_report(run_dir: Path, results: pd.DataFrame, history: pd.DataFrame, loader, config: dict) -> None:
    exp = config["experiment"]
    results = results.copy()
    results["target_epsilon"] = _norm_eps(results["target_epsilon"])
    history = history.copy()
    history["target_epsilon"] = _norm_eps(history["target_epsilon"])
    folds = results[["repeat", "fold"]].drop_duplicates()
    ref = load_reference(config, folds)
    combined = pd.concat([ref, results], ignore_index=True, sort=False)
    combined.to_csv(run_dir / "combined_with_centralised.csv", index=False)

    keys = ["arm", "setting", "partition", "n_clients", "target_epsilon"]
    g = combined.groupby(keys, sort=False)
    summary = g[HEAD].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary["epsilon_spent_max"] = g["epsilon_spent"].max()
    summary["total_kB"] = g["total_bytes"].mean() / 1e3 if "total_bytes" in combined else np.nan
    summary["fit_seconds_mean"] = g["fit_seconds"].mean()
    summary["n_folds"] = g.size()
    summary = summary.reset_index()
    arm_rank = {a: i for i, a in enumerate(exp["arms"])}
    summary = summary.sort_values(["arm", "target_epsilon", "setting", "partition", "n_clients"],
                                  key=lambda c: c.map(arm_rank) if c.name == "arm" else c).reset_index(drop=True)
    summary.insert(0, "data_source", loader.source_tag())
    summary.to_csv(run_dir / "summary.csv", index=False)

    tests = []
    for arm in exp["arms"]:
        for eps in sorted(set(results.target_epsilon)):
            central = ref[(ref.arm == arm) & (ref.target_epsilon == eps)].set_index(["repeat", "fold"]).roc_auc
            central_np = ref[(ref.arm == arm) & (ref.target_epsilon == "inf")].set_index(["repeat", "fold"]).roc_auc
            for (partition, n), grp in results[(results.arm == arm) & (results.target_epsilon == eps)].groupby(
                    ["partition", "n_clients"]):
                fed = grp.set_index(["repeat", "fold"]).roc_auc
                for ref_name, base in (("centralised same ε", central), ("centralised non-private", central_np)):
                    if eps == "inf" and ref_name == "centralised non-private":
                        continue
                    idx = fed.index.intersection(base.index).sort_values()
                    if len(idx) < 2:
                        continue
                    tests.append({"arm": arm, "epsilon": eps, "partition": partition, "n_clients": n,
                                  "vs": ref_name, "federated_auc": fed[idx].mean(), "reference_auc": base[idx].mean(),
                                  "auc_change": (fed[idx] - base[idx]).mean(), "k": len(idx),
                                  "wilcoxon_p": wilcoxon_p(fed[idx].to_numpy(), base[idx].to_numpy())})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["wilcoxon_p_holm"] = tests.groupby(["arm", "vs"])["wilcoxon_p"].transform(lambda p: holm(p.to_numpy()))
    tests.to_csv(run_dir / "tests_federated_vs_centralised.csv", index=False)
    convergence_plot(history, combined, config, run_dir / "federated_convergence.png", loader)

    comm = results.groupby(["arm", "n_clients"]).agg(n_params=("n_params", "first"),
                                                     bytes_per_round=("bytes_per_round", "first"),
                                                     total_kB=("total_bytes", lambda b: b.mean() / 1e3)).reset_index()
    sizes = results.groupby(["arm", "partition", "n_clients"]).agg(
        client_rows_min=("client_size_min", "min"), client_rows_max=("client_size_max", "max"),
        client_pos_min=("client_pos_min", "min"), client_pos_max=("client_pos_max", "max"),
        prevalence_min=("client_prevalence_min", "min"), prevalence_max=("client_prevalence_max", "max")).reset_index()

    k = results.groupby(["repeat", "fold"]).ngroups
    table = summary[["arm", "setting", "partition", "n_clients", "target_epsilon", "roc_auc_mean", "roc_auc_std",
                     "balanced_accuracy_mean", "balanced_accuracy_std", "recall_mean", "recall_std",
                     "epsilon_spent_max", "total_kB"]]
    lines = report_header("Experiment 4 — federated learning + DP-SGD", loader)
    lines += [
        f"{k} outer folds. FedAvg, {exp['rounds']} rounds × {exp['local_epochs']} local epoch; each client runs "
        "Opacus DP-SGD with its own RDP accountant (noise calibrated for all its local steps). Every record belongs "
        "to one client, so the record-level guarantee is the maximum client ε reported. Centralised rows are the "
        "Experiment 3 runs on the same folds and arms. Secure aggregation is not implemented: the server sees "
        "individual DP-noised updates. Threshold chosen on central inner-val rows, as in Experiment 3.", "",
        "Partitions: `iid` = stratified random clients; `by_state` = non-IID, whole U.S. states assigned to clients "
        "(\"schools\").", "",
        "![federated convergence](federated_convergence.png)", "",
        "## Utility (mean / std over folds)", "", _fmt(table), "",
        "## Federated vs centralised (paired ROC-AUC, Wilcoxon, Holm within arm × reference)", "", _fmt(tests), "",
        "## Communication cost (float64 weights, all clients every round, download + upload)", "", _fmt(comm), "",
        "## Client sizes and prevalence", "", _fmt(sizes), "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
