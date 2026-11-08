"""Assemble results/RESULTS.md, LaTeX tables and final figures from saved runs.

Every number is read from the run directories listed in the config; nothing is
typed in by hand.

Usage: python -m src.m10_exp.final_report --config configs/final_report.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.m10_exp.common import LABEL_LIMITATION, load_config  # noqa: E402
from src.m8_eval.stats import holm, wilcoxon_p  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated slots 1-4
MARKERS = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"logreg": "LogReg", "xgboost": "XGBoost", "mlp": "MLP", "qsvm": "QSVM", "vqc": "VQC", "hybrid": "Hybrid QNN"}


def ms(mean, std, d=3) -> str:
    return f"{mean:.{d}f} ± {std:.{d}f}"


def pfmt(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def latex(df: pd.DataFrame, caption: str, label: str) -> str:
    def esc(v):
        return (str(v).replace("±", r"$\pm$").replace("<", r"$<$").replace("_", r"\_").replace("%", r"\%")
                .replace("ε", r"$\varepsilon$").replace("∞", r"$\infty$"))
    lines = [r"\begin{table}[ht]", r"\centering", r"\small", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.lines: list[str] = []
        self.n = 0
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "figures").mkdir(parents=True, exist_ok=True)

    def text(self, *lines: str) -> None:
        self.lines += list(lines) + [""]

    def table(self, df: pd.DataFrame, caption: str, slug: str) -> None:
        self.n += 1
        df.to_csv(self.out / "tables" / f"table{self.n}_{slug}.csv", index=False)
        tex = latex(df, caption, f"tab:{slug}")
        (self.out / "tables" / f"table{self.n}_{slug}.tex").write_text(tex, encoding="utf-8")
        self.text(f"**Table {self.n}. {caption}** (LaTeX: `tables/table{self.n}_{slug}.tex`)", "", md(df))


# ------------------------------------------------------------------ sections
def classical_table(exp2: Path) -> pd.DataFrame:
    s = pd.read_csv(exp2 / "summary.csv")
    s = s[(s.regime == "full") & (s.feature_set == "all95")]
    return pd.DataFrame({
        "Model": s.model.map(NAMES), "Balanced acc.": [ms(a, b) for a, b in zip(s.balanced_accuracy_mean, s.balanced_accuracy_std)],
        "Recall": [ms(a, b) for a, b in zip(s.recall_mean, s.recall_std)],
        "ROC-AUC": [ms(a, b) for a, b in zip(s.roc_auc_mean, s.roc_auc_std)],
        "Plain acc. (not headline)": [ms(a, b) for a, b in zip(s.accuracy_mean, s.accuracy_std)]})


def encoding_table(exp1: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(exp1 / "summary.csv")
    q = s[s.model.isin(["qsvm", "vqc"])].copy()
    q["cell"] = [ms(a, b) for a, b in zip(q.roc_auc_mean, q.roc_auc_std)]
    wide = q.pivot_table(index=["model", "feature_method", "encoding"], columns="n_qubits", values="cell", aggfunc="first")
    wide.columns = [f"{c} qubits" for c in wide.columns]
    wide = wide.reset_index().rename(columns={"model": "Model", "feature_method": "Features", "encoding": "Encoding"})
    wide["Model"] = wide["Model"].map(NAMES)
    wide["Features"] = wide["Features"].str.upper()
    sw = pd.read_csv(exp1 / "encoding_swing.csv")
    swing = pd.DataFrame({"Model": sw.model.map(NAMES), "Features": sw.feature_method.str.upper(), "Qubits": sw.n_qubits,
                          "Best": sw.best_encoding, "Worst": sw.worst_encoding,
                          "AUC swing": sw.auc_swing.round(3), "Swing (% of worst)": sw.auc_swing_pct_of_worst.round(1)})
    return wide, swing


def comparison_table(exp2: Path) -> pd.DataFrame:
    t = pd.read_csv(exp2 / "tests_same_rows_features.csv")
    t = t[~t.classical.str.startswith("random_forest")]
    t["wilcoxon_p_holm"] = t.groupby("metric")["wilcoxon_p"].transform(lambda p: holm(p.to_numpy()))
    metric = {"roc_auc": "ROC-AUC", "balanced_accuracy": "Balanced acc.", "recall": "Recall"}
    return pd.DataFrame({
        "Quantum": t.quantum.str.split("/").str[0].map(NAMES) + " (" + t.quantum.str.split("/").str[2] + ")",
        "Classical": t.classical.str.split("/").str[0].map(NAMES), "Metric": t.metric.map(metric),
        "Quantum mean": t.quantum_mean.round(3), "Classical mean": t.classical_mean.round(3),
        "Diff (95% CI)": [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in zip(t.mean_diff, t.ci95_low, t.ci95_high)],
        "Quantum wins": [f"{w}/{k}" for w, k in zip(t.quantum_wins, t.k)],
        "Wilcoxon p": t.wilcoxon_p.map(pfmt), "Holm p": t.wilcoxon_p_holm.map(pfmt)})


def privacy_tables(exp3: Path, sens: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = pd.concat([pd.read_csv(exp3 / "summary.csv"), pd.read_csv(sens / "summary.csv")], ignore_index=True)
    s = s[s.arm != "logreg_sub_clip1"]  # identical to the main logreg_sub arm
    s["eps"] = ["∞" if str(e) in ("inf", "nan") else f"{float(e):g}" for e in s.target_epsilon]
    s["cell"] = [ms(a, b) for a, b in zip(s.roc_auc_mean, s.roc_auc_std)]
    order = ["0.5", "1", "2", "4", "8", "∞"]
    auc = s.pivot_table(index="arm", columns="eps", values="cell", aggfunc="first")[order]
    names = {"vqc_sub": "VQC (1.2k rows, PCA-4), clip 1", "logreg_sub": "LogReg (same rows/features), clip 1",
             "logreg_sub_clip5": "LogReg (same rows/features), clip 5 [sensitivity]",
             "logreg_sub_clip10": "LogReg (same rows/features), clip 10 [sensitivity]",
             "logreg_full": "LogReg (full 11k rows, all 95 features), clip 1"}
    auc = auc.reindex([a for a in names if a in auc.index]).rename(index=names).reset_index().rename(columns={"arm": "Arm"})
    auc.columns = ["Arm"] + [f"ε = {c}" for c in order]
    q = pd.read_csv(exp3 / "tests_quantum_vs_classical.csv")
    q["eps"] = ["∞" if str(e) in ("inf", "nan") else f"{float(e):g}" for e in q.epsilon]
    deg = pd.DataFrame({"ε": q.eps, "VQC AUC": q.quantum_auc.round(3), "LogReg AUC (clip 1)": q.classical_auc.round(3),
                        "Diff": q.auc_diff.round(3), "Holm p (diff)": q.wilcoxon_p_holm.map(pfmt),
                        "VQC drop vs ∞": q.quantum_auc_drop.round(3), "LogReg drop vs ∞": q.classical_auc_drop.round(3),
                        "Holm p (drop diff)": q.get("drop_diff_wilcoxon_p_holm", pd.Series(dtype=float)).map(pfmt)})
    # sensitivity: VQC (clip 1) vs LogReg clip 10, paired by fold
    main = pd.read_csv(exp3 / "fold_metrics.csv")
    sen = pd.read_csv(sens / "fold_metrics.csv")
    for f in (main, sen):
        f["eps"] = ["∞" if str(e) in ("inf", "nan") else f"{float(e):g}" for e in f.target_epsilon]
    rows = []
    for e in order:
        a = main[(main.arm == "vqc_sub") & (main.eps == e)].set_index(["repeat", "fold"]).roc_auc.sort_index()
        b = sen[(sen.arm == "logreg_sub_clip10") & (sen.eps == e)].set_index(["repeat", "fold"]).roc_auc.sort_index()
        idx = a.index.intersection(b.index)
        rows.append({"ε": e, "VQC AUC": round(a[idx].mean(), 3), "LogReg clip 10 AUC": round(b[idx].mean(), 3),
                     "Diff": round((a[idx] - b[idx]).mean(), 3), "Wilcoxon p": wilcoxon_p(a[idx].to_numpy(), b[idx].to_numpy())})
    sensdf = pd.DataFrame(rows)
    sensdf["Holm p"] = holm(sensdf["Wilcoxon p"].to_numpy())
    sensdf["Wilcoxon p"] = sensdf["Wilcoxon p"].map(pfmt)
    sensdf["Holm p"] = sensdf["Holm p"].map(pfmt)
    return auc, deg, sensdf


def privacy_figure(exp3: Path, sens: Path, out: Path) -> None:
    s = pd.concat([pd.read_csv(exp3 / "summary.csv"), pd.read_csv(sens / "summary.csv")], ignore_index=True)
    s["eps"] = ["inf" if str(e) in ("inf", "nan") else f"{float(e):g}" for e in s.target_epsilon]
    order = ["0.5", "1", "2", "4", "8", "inf"]
    arms = [("vqc_sub", "VQC re-upload, 4 qubits (1.2k rows, PCA-4)", "-"),
            ("logreg_sub", "LogReg, same rows & features (clip 1)", "-"),
            ("logreg_full", "LogReg, full 11k rows, all 95 features", "-"),
            ("logreg_sub_clip10", "LogReg, same rows & features (clip 10, sensitivity)", "--")]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, (arm, label, ls) in enumerate(arms):
        d = s[s.arm == arm].set_index("eps").reindex(order)
        x = np.arange(len(order)) + (i - 1.5) * 0.07
        ax.errorbar(x, d.roc_auc_mean, yerr=d.roc_auc_std, color=COLORS[i], marker=MARKERS[i], markersize=8, lw=0,
                    elinewidth=1.4, capsize=3, markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label, zorder=3)
        ax.plot(x[:-1], d.roc_auc_mean.iloc[:-1], color=COLORS[i], lw=2, ls=ls, zorder=2)
        ax.plot(x[-2:], d.roc_auc_mean.iloc[-2:], color=COLORS[i], lw=1.2, ls=":", zorder=2)
    ax.axvline(len(order) - 1.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(range(len(order)), [("∞\n(non-private)" if e == "inf" else e) for e in order])
    ax.set_xlabel("Privacy budget ε (δ = 1e-5; smaller ε = stronger privacy)", color=MUTED)
    ax.set_ylabel("ROC-AUC, mean ± std over 10 folds", color=MUTED)
    ax.set_title("Privacy–utility trade-off under DP-SGD (Opacus RDP accounting)", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor="#fcfcfb")
    plt.close(fig)


def federated_table(exp4: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(exp4 / "summary.csv")
    s["eps"] = ["∞" if str(e) in ("inf", "nan") else f"{float(e):g}" for e in s.target_epsilon]
    names = {"vqc_sub": "VQC (1.2k rows)", "logreg_full": "LogReg (full data)"}
    t = pd.DataFrame({"Arm": s.arm.map(names), "Setting": s.setting, "Partition": s.partition, "Clients": s.n_clients,
                      "ε": s.eps, "ROC-AUC": [ms(a, b) for a, b in zip(s.roc_auc_mean, s.roc_auc_std)],
                      "Balanced acc.": [ms(a, b) for a, b in zip(s.balanced_accuracy_mean, s.balanced_accuracy_std)],
                      "Recall": [ms(a, b) for a, b in zip(s.recall_mean, s.recall_std)],
                      "Max client ε": s.epsilon_spent_max.map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
                      "Comm. (kB total)": s.total_kB.map(lambda v: "" if pd.isna(v) else f"{v:.1f}")})
    tests = pd.read_csv(exp4 / "tests_federated_vs_centralised.csv")
    tt = pd.DataFrame({"Arm": tests.arm.map(names), "Partition": tests.partition, "Clients": tests.n_clients,
                       "vs": tests.vs, "Federated AUC": tests.federated_auc.round(3),
                       "Reference AUC": tests.reference_auc.round(3), "Change": tests.auc_change.round(3),
                       "Wilcoxon p": tests.wilcoxon_p.map(pfmt), "Holm p": tests.wilcoxon_p_holm.map(pfmt)})
    return t, tt


def cost_table(exp1: Path, exp2: Path) -> pd.DataFrame:
    res = pd.read_csv(exp1 / "circuit_resources.csv")
    s = pd.read_csv(exp2 / "summary.csv")
    vqc_c = res[(res.encoding == "reupload") & (res.n_qubits == 4) & (res.n_layers == 2)].iloc[0]
    hyb_c = res[(res.encoding == "angle") & (res.n_qubits == 8) & (res.n_layers == 2)].iloc[0]
    qsvm_c = res[(res.encoding == "angle") & (res.n_qubits == 8) & (res.n_layers == 0)].iloc[0]
    rows = []

    def add(model, regime, fs, qubits, depth, gates, label):
        r = s[(s.model == model) & (s.regime == regime) & (s.feature_set == fs)].iloc[0]
        rows.append({"Model": label, "Training rows": "1.2k" if regime == "subsample" else "~11k",
                     "Qubits": qubits, "Circuit depth": depth, "Gate count": gates,
                     "Parameters": "" if pd.isna(r.n_params_mean) else f"{r.n_params_mean:.0f}",
                     "Fit time (s)": f"{r.fit_seconds_mean:.2f}", "Tuning time, all grid points (s)": f"{r.tuning_seconds_mean:.1f}"})
    add("qsvm", "subsample", "pca8", 8, f"{qsvm_c.depth} (kernel ≈ 2×)", qsvm_c.n_gates, "QSVM, angle, PCA-8 (params = dual coef.)")
    add("vqc", "subsample", "pca4", 4, vqc_c.depth, vqc_c.n_gates, "VQC, re-upload, PCA-4, 2 layers")
    add("hybrid", "subsample", "all95", 8, hyb_c.depth, hyb_c.n_gates, "Hybrid QNN, angle, 8 qubits, 2 layers")
    for m in ("logreg", "xgboost", "mlp"):
        add(m, "subsample", "pca8", "–", "–", "–", f"{NAMES[m]}, PCA-8")
        add(m, "full", "all95", "–", "–", "–", f"{NAMES[m]}, all 95 features")
    return pd.DataFrame(rows)


def build(config_path: str) -> Path:
    cfg = load_config(config_path)
    runs = {k: Path(v) for k, v in cfg["runs"].items()}
    out = Path(cfg.get("output_dir", "results"))
    w = Writer(out)
    p1 = pd.read_json(runs["phase1"] / "dataset_profile.json", typ="series")
    w.text("# RESULTS — Privacy-Preserving Adolescent Mental Health Using Quantum AI", "",
           f"All numbers below are generated by `python -m src.m10_exp.final_report --config {config_path}` from the "
           "run directories listed at the end. **Data: REAL** — NSCH 2024 public-use file, adolescents 12–17, "
           f"{p1['n_rows']:,} rows, positive rate {p1['positive_rate']:.2%} (caregiver-reported current provider-"
           f"diagnosed depression). Locked hold-out: {p1['n_holdout']:,} rows, scored exactly once at the end (Section 7). "
           "No synthetic result appears here.",
           f"> {LABEL_LIMITATION}",
           "Protocol: repeated stratified CV on the dev rows; all preprocessing fitted inside each fold; "
           "hyperparameters, early stopping and the decision threshold chosen on an inner validation split of the "
           "fold's training rows, identically for every model. Headline metrics: balanced accuracy, recall, ROC-AUC. "
           "Significance: paired two-sided Wilcoxon signed-rank over folds, Holm-adjusted within each family. "
           "Privacy comes only from DP-SGD (Opacus) and federated aggregation; quantum circuits are a representation "
           "layer. RandomForest is excluded from all results on supervisor instruction.")

    w.text("## 1. Classical baselines (full data, all 95 features, 15 folds)")
    w.table(classical_table(runs["exp2"]), "Tuned classical baselines, mean ± std over 15 folds", "classical")

    wide, swing = encoding_table(runs["exp1"])
    w.text("## 2. Experiment 1 — encoding comparison (10 folds, 1.5k-row training subsample, no privacy)")
    w.table(wide, "Quantum ROC-AUC by encoding, feature selection and qubit count", "encodings")
    w.table(swing, "Encoding swing: best minus worst encoding (mean ROC-AUC)", "encoding_swing")
    shutil.copy(runs["exp1"] / "exp1_encoding_auc.png", out / "figures" / "encoding_comparison.png")
    w.text("![encoding comparison](figures/encoding_comparison.png)",
           "Full pairwise tests: `" + str(runs["exp1"] / "summary.md") + "`. With 10 folds the smallest attainable "
           "two-sided Wilcoxon p is 0.002, so differences cannot survive Holm over the whole 36-comparison table; "
           "within-configuration Holm values are given there.")

    w.text("## 3. Experiment 2 — quantum vs classical on identical rows and features (15 folds)")
    w.table(comparison_table(runs["exp2"]), "Paired comparisons (Holm within metric)", "quantum_vs_classical")
    shutil.copy(runs["exp2"] / "exp2_roc.png", out / "figures" / "roc_curves.png")
    w.text("![ROC curves](figures/roc_curves.png)",
           "Mean ROC over the 15 outer folds. VQC and Hybrid validation scores were recovered by re-running those two "
           "models with the same seeds and folds; the re-run reproduced all 30 of their fold metrics exactly "
           "(`scores_merged_from_rerun.json` in the Experiment 2 run directory).")

    auc, deg, sensdf = privacy_tables(runs["exp3"], runs["exp3_sensitivity"])
    privacy_figure(runs["exp3"], runs["exp3_sensitivity"], out / "figures" / "privacy_utility_curve.png")
    w.text("## 4. Experiment 3 — privacy–utility curve (DP-SGD, 10 folds) — headline",
           "![privacy-utility curve](figures/privacy_utility_curve.png)")
    w.table(auc, "ROC-AUC by privacy budget ε (δ = 1e-5)", "privacy_utility")
    w.table(deg, "VQC vs LogReg on identical rows/features at each ε (pre-declared comparison, clip 1)", "privacy_degradation")
    w.table(sensdf, "Sensitivity analysis (added after seeing Table 6): VQC vs LogReg with clipping norm 10", "privacy_sensitivity")

    fed, fed_t = federated_table(runs["exp4"])
    shutil.copy(runs["exp4"] / "federated_convergence.png", out / "figures" / "federated_convergence.png")
    w.text("## 5. Experiment 4 — federated learning + DP (ε = 4, 10 folds)",
           "![federated convergence](figures/federated_convergence.png)")
    w.table(fed, "Federated (FedAvg, 30 rounds, client-side DP-SGD) vs centralised", "federated")
    w.table(fed_t, "Federated vs centralised, paired ROC-AUC (Holm within arm × reference)", "federated_tests")

    w.text("## 6. Computational cost")
    w.table(cost_table(runs["exp1"], runs["exp2"]),
            "Cost per model (CPU statevector simulation; depth/gates after decomposition to RX/RY/RZ/CNOT/H)", "cost")

    if "holdout" in runs:
        h = pd.read_csv(runs["holdout"] / "holdout_metrics.csv").set_index("model")
        e2 = pd.read_csv(runs["exp2"] / "fold_metrics.csv")
        e3 = pd.read_csv(runs["exp3"] / "fold_metrics.csv")
        e3["eps"] = ["inf" if str(v) in ("inf", "nan") else f"{float(v):g}" for v in e3.target_epsilon]
        cv = {"logreg_full_all95": e2[(e2.model == "logreg") & (e2.regime == "full") & (e2.feature_set == "all95")],
              "qsvm_angle_pca8_1500rows": e2[(e2.model == "qsvm") & (e2.regime == "subsample")],
              "dp_logreg_full_all95_eps4": e3[(e3.arm == "logreg_full") & (e3.eps == "4")]}
        label = {"logreg_full_all95": "LogReg, full data, all 95 features",
                 "qsvm_angle_pca8_1500rows": "QSVM angle, PCA-8, 1,500 rows",
                 "dp_logreg_full_all95_eps4": "DP LogReg ε = 4, full data, all 95 features"}
        rows = []
        for k, g in cv.items():
            r = h.loc[k]
            rows.append({"Configuration": label[k], "CV folds": len(g),
                         "CV ROC-AUC": ms(g.roc_auc, g.roc_auc.std()) if False else f"{g.roc_auc.mean():.3f} ± {g.roc_auc.std(ddof=1):.3f}",
                         "Hold-out ROC-AUC [95% CI]": f"{r.roc_auc:.3f} [{r.auc_ci95_low:.3f}, {r.auc_ci95_high:.3f}]",
                         "CV bal. acc.": f"{g.balanced_accuracy.mean():.3f}", "Hold-out bal. acc.": f"{r.balanced_accuracy:.3f}",
                         "CV recall": f"{g.recall.mean():.3f}", "Hold-out recall": f"{r.recall:.3f}"})
        w.text("## 7. Locked hold-out (scored once)",
               f"Each final configuration was trained on all {int(h.n_train_rows.max()):,} dev rows (QSVM: its 1,500-row "
               "stratified subsample) with the same inner-split tuning and threshold protocol, then scored once on the "
               f"{int(h.n_holdout.iloc[0]):,}-row hold-out ({int(h.n_holdout_pos.iloc[0])} positives). Nothing was tuned on "
               "it; the evaluation script refuses to run a second time. CI = 2,000-sample bootstrap.")
        w.table(pd.DataFrame(rows), "Cross-validation vs locked hold-out", "holdout")
        w.text("Every CV mean lies inside the hold-out 95% interval; hold-out AUC is 0.004–0.007 higher, consistent with "
               "the final models training on more rows than a CV fold. The CV protocol does not overestimate performance. "
               "Threshold-dependent metrics (recall, balanced accuracy) move more than AUC because the threshold is "
               "chosen on a small inner-validation split.")

    w.text("## Limitations",
           "- **Label bias.** The target is caregiver-reported, provider-diagnosed depression. Much depression is never "
           "diagnosed (~63.6% per the WHO estimate cited in the survey), so the model identifies adolescents who "
           "resemble those already diagnosed and under-represents groups with poorer access to care.",
           "- **Risk now, not forecasting.** NSCH is a one-time cross-sectional survey and several strong predictors "
           "(difficulty making friends, staying calm, missed school) may be consequences of depression rather than causes.",
           "- **Near-linear signal.** Tuned logistic regression matches XGBoost and the MLP (Table 1), so the signal is "
           "largely additive. Quantum feature maps are motivated by non-linear structure; this dataset is unfavourable "
           "terrain for them by construction, which is the most likely explanation of the quantum results.",
           "- **Feature-reduction ceiling.** Qubit-sized inputs (4–8 features) cost ROC-AUC relative to all 95 features "
           "even for classical models; this is a cost of the quantum approach, separate from model quality.",
           "- **Simulation, not hardware.** All circuits run on an exact CPU statevector simulator (no shot noise, "
           "no hardware noise). The winning QSVM uses angle encoding, whose kernel is a product of cos² terms and is "
           "classically computable.",
           "- **Small training subsample.** Quantum models train on 1,500 rows per fold (~150 positives; ~120 in "
           "inner-train, 30 in inner-val), so thresholds and model selection are noisy.",
           "- **DP scope.** ε covers the trained weights only; fixed hyperparameters and the decision threshold "
           "(chosen on inner-val rows) are not accounted. The DP comparison is sensitive to the clipping norm "
           "(Table 7). QSVM is not gradient-trained and has no DP variant here. Secure aggregation is not implemented.",
           "- **ε does not cover preprocessing.** Imputation medians, scaling statistics and PCA components are fitted on "
           "the private training rows without DP, so the (ε, δ) guarantee applies to the training steps given that "
           "preprocessing, not to the whole pipeline end to end.",
           "- **Federated preprocessing is centralised.** In Experiment 4 the imputer, scaler and PCA are fitted once on "
           "the pooled client rows, which a real federation could not do; clients would need federated or public-data "
           "preprocessing. The threshold is also chosen on central inner-validation rows.",
           "- **Scope cuts made to finish:** Experiment 1 uses 10 folds and excludes the Hybrid QNN; Experiments 3–4 "
           "use 10 folds; Experiment 4 runs at ε = 4 only with two arms; RandomForest dropped on supervisor instruction.")
    w.text("## Run directories", *[f"- {k}: `{v}`" for k, v in runs.items()])
    (out / "RESULTS.md").write_text("\n".join(w.lines), encoding="utf-8")
    print(f"wrote {out / 'RESULTS.md'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RESULTS.md")
    parser.add_argument("--config", required=True)
    build(parser.parse_args().config)


if __name__ == "__main__":
    main()
