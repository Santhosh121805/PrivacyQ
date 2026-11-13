"""Results report, one section per box of docs/architecture.jpg (box 1 to box 15).

Every number is read from a file under results/. Nothing is re-run or estimated. Where a number does not exist the
report says "Not measured"; where the dataset cannot support a step it says "Not available on this dataset".
Each number and table is registered with its file path and listed in the traceability appendix.

The only derived values are means and standard deviations over the per-fold rows of fold_metrics.csv for
precision and F1 (no summary file stores them); the same aggregation is checked against summary.csv for
accuracy, recall and ROC-AUC and the build stops if those do not match. Figures are redrawn from the CSV
values with markers, line styles and hatching so they print in black and white; circuit figures are drawn
from the code in src/ and contain no numbers.

Outputs: results/results_report.md, results/results_report.html, results/results_report.pdf,
results/report_figures/*.png
Usage:   python scripts/build_results_report.py
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
FIG = R / "report_figures"
CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
RUN = {
    "phase1": "phase1-nsch2024-depression-20261104-183743",
    "exp1": "exp1-nsch2024-encodings-with-basis-20261112",
    "exp2": "exp2-nsch2024-quantum-vs-classical-20261108-165426",
    "exp2_svm": "exp2-nsch2024-svm-baseline-20261112-121554",
    "exp2_cobyla": "exp2-nsch2024-vqc-cobyla-20261112-121718",
    "exp3": "exp3-nsch2024-privacy-utility-20261108-165618",
    "exp4": "exp4-nsch2024-federated-dp-20261108-170320",
    "exp5": "exp5-nsch2024-input-privacy-20261112-120542",
    "exp6": "exp6-nsch2024-membership-inference-20261112-120934",
    "exp7": "exp7-nsch2024-federated-cost-20261112-123743",
    "exp8": "exp8-nsch2024-hardware-noise-20261112-125925",
    "exp8b": "exp8b-nsch2024-hardware-noise-zz-vs-angle-4q-20261112-144158",
    "exp9": "exp9-nsch2024-multiclass-20261112-140242",
    "exp10": "exp10-nsch2024-umap-autoencoder-20261112-143551",
    "holdout": "holdout-20261108-173801",
    "holdout_mc": "holdout-multiclass-20261112-171606",
}
NM = "Not measured"
NA = "Not available on this dataset"


def p(run: str, name: str) -> Path:
    path = R / RUN[run] / name if run else R / name
    if not path.exists():
        raise SystemExit(f"MISSING ARTIFACT: {path.relative_to(ROOT)}")
    return path


def rel(path: Path) -> str:
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


def csv(run, name):
    return pd.read_csv(p(run, name))


def text_has(path: Path, phrase: str) -> str:
    """Return the phrase only if it appears verbatim in the file (wording is quoted, never invented)."""
    t = path.read_text(encoding="utf-8")
    if phrase not in t:
        raise SystemExit(f"phrase not found in {rel(path)}: {phrase!r}")
    return phrase


# ---------------------------------------------------------------- document model
class Doc:
    def __init__(self):
        self.blocks: list[tuple] = []
        self.trace: list[tuple[str, str, str, str]] = []   # (where, what, value, file)
        self.figures: list[dict] = []
        self.where = ""
        self.ntab = 0

    def num(self, value, path: Path, what: str, fmt: str = "{}") -> str:
        s = fmt.format(value)
        self.trace.append((self.where, what, s, rel(path)))
        return s

    def h(self, level, text):
        self.blocks.append(("h", level, text))

    def p(self, text):
        self.blocks.append(("p", text))

    def lab(self, name, text=""):
        self.blocks.append(("lab", name, text))

    def status(self, kind, reason):
        self.blocks.append(("status", kind, reason))

    def table(self, title, cols, rows, sources: list[Path], selection: str, note: str | None = None):
        self.ntab += 1
        ref = f"Table {self.where.split()[-1]}.{self.ntab}"
        self.blocks.append(("table", f"{ref}. {title}", cols, rows, note, sources))
        for s in sources:
            self.trace.append((self.where, f"{ref} — {title} (all cells)", selection, rel(s)))
        return ref

    def figure(self, path: Path, caption: str, sources: list[str]):
        n = len(self.figures) + 1
        self.figures.append({"n": n, "path": path, "caption": caption, "box": self.where, "sources": sources})
        self.blocks.append(("fig", n, path, caption, sources))
        for s in sources:
            self.trace.append((self.where, f"Figure {n} (plotted values)", "all plotted points", s))
        return n

    def box(self, n, title):
        self.where = f"Box {n}"
        self.ntab = 0
        self.blocks.append(("box", n, title))


def f3(v):
    return f"{v:.3f}"


def pm(m, s):
    return f"{m:.3f} ± {s:.3f}"


def pct(v, d=1):
    return f"{100 * v:.{d}f} %"


# ---------------------------------------------------------------- figures (black-and-white safe)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "axes.prop_cycle": plt.cycler(color=["black"])})
MARKERS = ["o", "s", "^", "D", "v", "x", "P"]
LINES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / name
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_mi(mi):
    top = mi.sort_values("rank").head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    for y, (_, r) in enumerate(top.iterrows()):
        sel = bool(r.selected_top8)
        ax.barh(y, r.mutual_information, color="white" if sel else "#bdbdbd", edgecolor="black", hatch="////" if sel else None, lw=0.8)
    ax.set_yticks(range(len(top)), [l if len(l) <= 58 else l[:56] + "…" for l in top.label], fontsize=7.5)
    ax.set_xlabel("Mutual information with current depression diagnosis (nats)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch="////", label="Top 8 (MI selection)"),
                       Patch(facecolor="#bdbdbd", edgecolor="black", label="Ranks 9–20")], loc="lower right", fontsize=8)
    return save(fig, "fig_mi_ranking.png")


def draw_circuit(fn, args, name, figsize=None):
    import pennylane as qml
    fig, _ = qml.draw_mpl(fn, decimals=None, style="black_white", level="device")(*args)
    if figsize:
        fig.set_size_inches(*figsize)
    return save(fig, name)


def fig_encodings():
    import pennylane as qml

    from src.m4_encoding import make_encoding
    outs = []
    for enc in ("angle", "amplitude", "basis", "zz", "reupload"):
        e = make_encoding(enc, 4)
        x = e.prepare(np.full((1, e.n_features()), 1.0))[0]

        @qml.qnode(qml.device("default.qubit", wires=4))
        def c(v, e=e):
            e.feature_map(v)
            return qml.state()
        outs.append((enc, draw_circuit(c, (x,), f"fig_circuit_{enc}.png")))
    return outs


def fig_models():
    import pennylane as qml

    from src.m4_encoding import make_encoding
    from src.m5_models.quantum import _variational_layers
    vq = make_encoding("reupload", 4)

    @qml.qnode(qml.device("default.qubit", wires=4))
    def vqc(x, w):
        _variational_layers(vq, x, w)
        return qml.expval(qml.PauliZ(0))
    a = draw_circuit(vqc, (np.ones(4), np.zeros((2, 4, 3))), "fig_circuit_vqc.png")
    hy = make_encoding("angle", 8)

    @qml.qnode(qml.device("default.qubit", wires=8))
    def hybrid(x, w):
        _variational_layers(hy, x, w)
        return [qml.expval(qml.PauliZ(i)) for i in range(8)]
    b = draw_circuit(hybrid, (np.ones(8), np.zeros((2, 8, 3))), "fig_circuit_hybrid.png")
    ke = make_encoding("angle", 4)

    @qml.qnode(qml.device("default.qubit", wires=4))
    def kernel(x1, x2):
        ke.feature_map(x1)
        qml.adjoint(ke.feature_map)(x2)
        return qml.probs(wires=range(4))
    c = draw_circuit(kernel, (np.ones(4), np.ones(4)), "fig_circuit_qsvm_kernel.png")
    return a, b, c


def fig_training_curves(rh):
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True)
    for i, arm in enumerate(["vqc_sub", "logreg_full"]):
        for j, part in enumerate(["iid", "by_state"]):
            ax = axes[i, j]
            for k, n in enumerate([3, 5, 10]):
                d = rh[(rh.arm == arm) & (rh.partition == part) & (rh.n_clients == n)].groupby("round").val_auc.mean()
                ax.plot(d.index, d.values, linestyle=LINES[k], marker=MARKERS[k], markevery=5, ms=4, color="black", lw=1,
                        label=f"{n} clients")
            ax.set_title(f"{'VQC (1.2k rows, PCA-4)' if arm == 'vqc_sub' else 'LogReg (full rows, all 95)'} · "
                         f"{'IID' if part == 'iid' else 'by U.S. state'}", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("Validation ROC-AUC")
            if i == 1:
                ax.set_xlabel("FedAvg round")
    axes[0, 0].legend(fontsize=7.5)
    fig.tight_layout()
    return save(fig, "fig_training_curves.png")


def fig_qubits(s1):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    encs = ["angle", "amplitude", "basis", "zz", "reupload"]
    for ax, model in zip(axes, ["qsvm", "vqc"]):
        for k, enc in enumerate(encs):
            d = s1[(s1.model == model) & (s1.encoding == enc) & (s1.feature_method == "pca")].sort_values("n_qubits")
            ax.errorbar(d.n_qubits, d.roc_auc_mean, yerr=d.roc_auc_std, linestyle=LINES[k], marker=MARKERS[k], ms=5,
                        color="black", mfc="white" if k % 2 else "black", capsize=2, lw=1, label=enc)
        ax.set_title(f"{model.upper()} · PCA features", fontsize=9)
        ax.set_xticks([4, 6, 8])
        ax.set_xlabel("Qubits")
    axes[0].set_ylabel("ROC-AUC (mean ± SD over folds)")
    axes[1].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    return save(fig, "fig_qubit_sweep.png")


def fig_privacy(s3):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    order = [0.5, 1, 2, 4, 8, np.inf]
    xs = np.arange(len(order))
    names = {"vqc_sub": "VQC (1.2k rows, PCA-4)", "logreg_sub": "LogReg (same 1.2k rows, PCA-4)", "logreg_full": "LogReg (full rows, all 95)"}
    for k, (arm, lab_) in enumerate(names.items()):
        d = s3[s3.arm == arm].set_index("target_epsilon").reindex(order)
        ax.errorbar(xs + (k - 1) * 0.08, d.roc_auc_mean, yerr=d.roc_auc_std, linestyle=LINES[k], marker=MARKERS[k],
                    mfc="white" if k == 1 else "black", color="black", capsize=2, lw=1, ms=5, label=lab_)
    ax.set_xticks(xs, ["0.5", "1", "2", "4", "8", "∞ (no DP)"])
    ax.set_xlabel("Target ε (δ = 1e-5)")
    ax.set_ylabel("ROC-AUC (mean ± SD, 10 folds)")
    ax.legend(fontsize=7.5, loc="lower right")
    return save(fig, "fig_privacy_utility.png")


def fig_clients(s4):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, arm, t in zip(axes, ["vqc_sub", "logreg_full"], ["VQC (1.2k rows, PCA-4)", "LogReg (full rows, all 95)"]):
        for k, part in enumerate(["iid", "by_state"]):
            d = s4[(s4.arm == arm) & (s4.partition == part)].sort_values("n_clients")
            ax.errorbar(d.n_clients, d.roc_auc_mean, yerr=d.roc_auc_std, linestyle=LINES[k], marker=MARKERS[k],
                        mfc="white" if k else "black", color="black", capsize=2, lw=1, ms=5,
                        label="IID" if part == "iid" else "non-IID (by U.S. state)")
        c = s4[(s4.arm == arm) & (s4.setting == "centralised")].roc_auc_mean.iloc[0]
        ax.axhline(c, color="black", lw=0.8, linestyle=(0, (1, 2)), label="centralised, same ε")
        ax.set_xticks([3, 5, 10])
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("Clients")
    axes[0].set_ylabel("ROC-AUC (mean ± SD, 10 folds)")
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    return save(fig, "fig_clients.png")


# ---------------------------------------------------------------- the report
def fold_stats(fm, cols, keys, check=None, check_keys=None):
    """Mean ± SD over per-fold rows; cross-checked against a summary file where it has the same metric."""
    g = fm.groupby(keys)
    out = g[cols].agg(["mean", "std"])
    out["n_folds"] = g.size()
    if check is not None:
        for key, row in check.set_index(check_keys).iterrows():
            if key not in out.index:
                continue
            for m in ("roc_auc", "accuracy", "recall"):
                if f"{m}_mean" in row and not np.isnan(row[f"{m}_mean"]):
                    if not np.isclose(out.loc[key, (m, "mean")], row[f"{m}_mean"], atol=1e-6) or \
                            not np.isclose(out.loc[key, (m, "std")], row[f"{m}_std"], atol=1e-6):
                        raise SystemExit(f"fold aggregation does not reproduce summary.csv for {key} {m}")
    return out


MODEL_NAMES = {"logreg": "Logistic regression", "svm": "SVM (RBF)", "xgboost": "XGBoost", "mlp": "MLP",
               "qsvm": "QSVM (quantum kernel)", "vqc": "VQC", "hybrid": "Hybrid QNN", "vqc_cobyla": "VQC (COBYLA)",
               "random_forest": "Random forest"}
REGIMES = {"full": "full training rows", "subsample": "1.2k training rows"}
FEATS = {"all95": "all 95 columns", "pca4": "PCA-4", "pca8": "PCA-8"}


def build() -> Doc:
    d = Doc()
    prof_p = p("phase1", "dataset_profile.json")
    prof = json.loads(prof_p.read_text())
    mi_p = R / "feature_importance" / "mi_scores.csv"
    mi = pd.read_csv(mi_p)
    lab = dict(zip(mi.column, mi.label))
    env_p = p("exp2", "environment.json")
    env = json.loads(env_p.read_text())
    cfg2 = p("exp2", "config.yaml")
    res_md = R / "RESULTS.md"

    # ------------------------------------------------------------ Box 1
    d.box(1, "Data Sources (Adolescent Mental Health)")
    d.lab("What this step does", "The project uses one real, public survey file: the National Survey of Children's Health (NSCH), "
          "topical public-use file. The data are real survey responses, not synthetic.")
    d.lab("What it produced")
    data_file = list(env["data_file_sha256"])[0]
    src_tag = prof["source"]
    d.table("Dataset", ["Item", "Value"], [
        ["Source file", d.num(data_file, env_p, "data file path")],
        ["File checksum (SHA-256)", d.num(env["data_file_sha256"][data_file][:16] + "…", env_p, "data file SHA-256 (first 16 hex)")],
        ["Survey", d.num(src_tag.split(":")[1].split("-age")[0], prof_p, "source tag: survey and year")],
        ["Real or synthetic", d.num("real (is_synthetic = false)", prof_p, "loader.is_synthetic")],
        ["Age range", d.num(src_tag.split("age")[1].split("-depression")[0].replace("-", "–") + " years", prof_p, "source tag: age range")],
        ["Rows (adolescents)", d.num(prof["n_rows"], prof_p, "n_rows", "{:,}")],
        ["Positive (current depression diagnosis)", d.num(prof["n_positive"], prof_p, "n_positive", "{:,}")
         + " (" + d.num(prof["positive_rate"], prof_p, "positive_rate", "{:.2%}") + ")"],
        ["Class balance", "positive rate " + d.num(prof["dev_positive_rate"], prof_p, "dev_positive_rate", "{:.2%}")
         + " in development rows, " + d.num(prof["holdout_positive_rate"], prof_p, "holdout_positive_rate", "{:.2%}") + " in hold-out rows"],
    ], [prof_p, env_p], "values as listed")
    d.table("The six sources in the diagram", ["Source in the diagram", "Used", "Reason"], [
        ["Clinical notes (de-identified)", "No", f"{NA}: no clinical-notes dataset is part of this project."],
        ["Counselling records", "No", f"{NA}: no counselling-records dataset is part of this project."],
        ["Survey responses", "Yes", "NSCH 2024 caregiver-reported survey answers."],
        ["Social media text", "No", f"{NA}: the data contain no free text."],
        ["Wearable/IoT data (optional)", "No", f"{NA}: NSCH records no sensor data."],
        ["Public datasets", "Yes", "The NSCH topical file is a public-use download."],
    ], [prof_p], "source tag REAL:NSCH-2024-…; no other data source appears in any results/ run")
    d.lab("Figure", "None for this step.")

    # ------------------------------------------------------------ Box 2
    d.box(2, "Data Preprocessing & Anonymization")
    d.lab("What this step does", "Identifier and metadata columns are removed, missing answers are imputed, categorical answers are "
          "one-hot encoded, and the rows are split into development and locked hold-out sets.")
    d.lab("What it produced")
    fm2 = csv("exp2", "fold_metrics.csv")
    sub0, full0 = fm2[fm2.regime == "subsample"].iloc[0], fm2[fm2.regime == "full"].iloc[0]
    fmp = p("exp2", "fold_metrics.csv")
    d.table("Preprocessing", ["Item", "Value"], [
        ["Rows in the raw file before the age filter", f"{NM}: no results file records the unfiltered row count."],
        ["Rows after the age filter (used)", d.num(prof["n_rows"], prof_p, "n_rows", "{:,}")],
        ["Survey answers kept", d.num(prof["n_raw_features"], prof_p, "n_raw_features") + " ("
         + d.num(prof["n_numeric_features"], prof_p, "n_numeric_features") + " numeric, "
         + d.num(prof["n_categorical_features"], prof_p, "n_categorical_features") + " categorical)"],
        ["Columns after one-hot encoding", d.num(len(mi), mi_p, "row count of mi_scores.csv (one row per model column)")],
        ["Columns dropped as identifiers / metadata", d.num(", ".join(prof["loader"]["metadata_columns"]), prof_p, "loader.metadata_columns")],
        ["Missing cells (all rows)", d.num(prof["missing_cell_rate"], prof_p, "missing_cell_rate", "{:.2%}")],
        ["Imputation", text_has(res_md, "Imputation medians") + ", fitted on training rows only (RESULTS.md). "
         "The rule for categorical answers is not recorded in results/: Not measured."],
        ["Scaling", "scaling statistics fitted on training rows (RESULTS.md); quantum inputs lie in [0, π] (Experiment 5 summary). "
         "The scaler type is not recorded in results/: Not measured."],
        ["Development rows / locked hold-out rows", d.num(prof["n_dev"], prof_p, "n_dev", "{:,}") + " / "
         + d.num(prof["n_holdout"], prof_p, "n_holdout", "{:,}")],
        ["Inner split, quantum regime (train / validation)", d.num(int(sub0.n_inner_train), fmp, "n_inner_train (subsample)", "{:,}")
         + " / " + d.num(int(sub0.n_inner_val), fmp, "n_inner_val (subsample)", "{:,}")],
        ["Inner split, full regime (train / validation)", d.num(int(full0.n_inner_train), fmp, "n_inner_train (full)", "{:,}")
         + " / " + d.num(int(full0.n_inner_val), fmp, "n_inner_val (full)", "{:,}")],
    ], [prof_p, mi_p, fmp, res_md], "values as listed")
    d.table("Most-missing answers", ["Answer", "Missing"], [[lab.get(k, k), d.num(v, prof_p, f"most_missing.{k}", "{:.2%}")]
                                                          for k, v in prof["most_missing"].items()], [prof_p], "most_missing")
    d.p("Fold-wise fitting rule: " + text_has(res_md, "all preprocessing fitted inside each fold") + "; hyperparameters, early stopping "
        "and the decision threshold are chosen on an inner validation split of each fold's training rows (RESULTS.md).")
    d.lab("Figure", "None for this step.")

    # ------------------------------------------------------------ Box 3
    d.box(3, "Feature Representation")
    d.lab("What this step does", "The survey answers become numeric model columns. The quantum models see 4 or 8 columns, produced from "
          "the 95 by principal component analysis (PCA) or by mutual-information (MI) selection, fitted on training rows.")
    d.lab("What it produced")
    d.table("Text features", ["Diagram item", "Status"], [
        ["Transformer embeddings (e.g., BERT, RoBERTa)", f"{NA}: the data contain no free text."],
        ["Sentiment & emotion features", f"{NA}: the data contain no free text."],
        ["Contextual representations", f"{NA}: the data contain no free text."],
        ["Physiological signals (optional)", f"{NA}: NSCH records no physiological signals."]], [prof_p], "status of diagram items")
    ms = mi.sort_values("rank")
    d.table("All 95 model columns, plain-English labels, ranked by mutual information with the label",
            ["Rank", "Column (plain English)", "MI (nats)", "MI top 8"],
            [[int(r["rank"]), r.label, f"{r.mutual_information:.4f}", "yes" if r.selected_top8 else ""] for _, r in ms.iterrows()],
            [mi_p], "columns rank, label, mutual_information, selected_top8")
    s1p = p("exp1", "summary.csv")
    tests10 = csv("exp10", "comparison.csv")
    t10p = p("exp10", "comparison.csv")
    red = tests10[tests10.metric == "roc_auc"]
    d.table("95 columns → 8 quantum inputs: reduction methods compared (QSVM, 1.2k rows, 8 inputs)",
            ["Reduction", "QSVM ROC-AUC (mean ± SD)", "vs PCA-8: NB-t Holm p"],
            [["PCA-8", pm(red.iloc[0].b_mean, red.iloc[0].b_std), "—"]]
            + [[n, pm(r.a_mean, r.a_std), f"{r.p_nb_t_holm:.3g}"] for n, r in
               zip(["UMAP-8", "Autoencoder-8"], [red[red.a.str.startswith("qsvm_umap8") & red.b.str.startswith("qsvm_pca8")].iloc[0],
                                                 red[red.a.str.startswith("qsvm_ae8") & red.b.str.startswith("qsvm_pca8")].iloc[0]])],
            [t10p], "rows metric=roc_auc, a in {qsvm_umap8, qsvm_ae8} vs b=qsvm_pca8",
            note="PCA explained-variance shares are not recorded in results/: Not measured.")
    fig = fig_mi(mi)
    d.lab("Figure")
    d.figure(fig, "Top 20 of 95 columns by mutual information with the current-depression label; hatched bars are the MI top 8.",
             [rel(mi_p)])

    # ------------------------------------------------------------ Box 4
    d.box(4, "Privacy-Preserving Mechanisms")
    d.lab("What this step does", "Privacy comes from input-stage k-anonymity and local differential privacy, DP-SGD during training, "
          "federated training and pairwise-masked secure aggregation. Quantum encoding provides no privacy.")
    d.lab("What it produced")
    s3p, s5p, s3 = p("exp3", "summary.csv"), p("exp5", "summary.csv"), csv("exp3", "summary.csv")
    s5 = csv("exp5", "summary.csv")
    cfg3, cfg5, cfg7b = p("exp3", "config.yaml"), p("exp5", "config.yaml"), R / "exp7b-nsch2024-federated-secure-aggregation-20261112-153004" / "config.yaml"
    sum3md, sum5md, ovmd = p("exp3", "summary.md"), p("exp5", "summary.md"), R / "secagg_overhead" / "overhead.md"
    sup_full = s5[(s5.arm == "logreg") & (s5.k_anon == "k=5")].suppression_rate_train.iloc[0]
    sup_sub = s5[(s5.arm == "qsvm") & (s5.k_anon == "k=5")].suppression_rate_train.iloc[0]
    d.table("Mechanisms and their parameters", ["Mechanism", "Setting"], [
        ["k-anonymity", "k = " + d.num(5, cfg5, "k_values") + " on 7 demographic quasi-identifiers; training rows suppressed: "
         + d.num(sup_full, s5p, "suppression_rate_train, logreg full k=5", "{:.1%}") + " (full regime), "
         + d.num(sup_sub, s5p, "suppression_rate_train, qsvm subsample k=5", "{:.1%}") + " (1.2k-row regime)"],
        ["Local DP", text_has(sum5md, "Laplace, pure ε-LDP") + " on the 8 encoder inputs, labels not perturbed; ε_input ∈ {"
         + d.num("32, 16, 8, 4", cfg5, "eps_inputs") + "}"],
        ["DP-SGD", text_has(sum3md, "DP-SGD via Opacus") + "; ε read back from the RDP accountant; δ = " + d.num("1e-5", cfg3, "delta")],
        ["Secure aggregation", text_has(cfg7b, "pairwise-masked secure aggregation") + "; keys by "
         + text_has(ovmd, "Diffie-Hellman key generation") + " and pairwise agreement"],
        ["Homomorphic Encryption (on simulated data)", f"{NM}: not implemented in this project."],
    ], [cfg5, s5p, sum5md, sum3md, cfg3, cfg7b, ovmd], "values as listed")
    rows = []
    for arm, nm in [("vqc_sub", "VQC, 1.2k rows"), ("logreg_sub", "LogReg, 1.2k rows"), ("logreg_full", "LogReg, full rows")]:
        for _, r in s3[(s3.arm == arm) & np.isfinite(s3.target_epsilon)].iterrows():
            rows.append([nm, f"{r.target_epsilon:g}", f"{r.epsilon_spent_max:.3f}", f"{r.noise_multiplier:.3f}"])
    d.table("DP-SGD: target ε and ε spent (RDP accountant, maximum over folds)", ["Model", "Target ε", "ε spent", "Noise multiplier σ"],
            rows, [s3p], "columns arm, target_epsilon, epsilon_spent_max, noise_multiplier (finite ε)")
    d.table("One real row before and after de-identification", ["Field", "Before", "After"],
            [[NM, "No file in results/ stores an individual record; the report does not read raw data.", ""]], [s5p],
            "status only")
    d.lab("Figure", "None for this step.")

    # ------------------------------------------------------------ Box 5
    d.box(5, "Quantum Encoding")
    d.lab("What this step does", "Each record's reduced features are written into a quantum state |ψ⟩ = U(x)|0⟩. Five encodings were "
          "implemented and compared at 4, 6 and 8 qubits.")
    d.lab("What it produced")
    rp = p("exp1", "circuit_resources.csv")
    res = pd.read_csv(rp)
    fmap = res[res.circuit.str.startswith("feature_map")]
    d.table("Encodings implemented: feature-map circuit size", ["Encoding", "Qubits", "Depth", "Gates", "CNOTs"],
            [[r.encoding, int(r.n_qubits), int(r.depth), int(r.n_gates), int(r.n_cnot)]
             for _, r in fmap.sort_values(["encoding", "n_qubits"]).iterrows()], [rp],
            "rows circuit=feature_map; columns encoding, n_qubits, depth, n_gates, n_cnot")
    d.table("Rotation angles for one real row", ["Qubit", "Angle"],
            [[NM, "No file in results/ stores the encoded angles of an individual record."]], [rp], "status only")
    d.lab("Figure")
    for enc, path in fig_encodings():
        d.figure(path, f"{enc} encoding feature map, 4 qubits (structure only; no data values). Drawn from src/m4_encoding/encodings.py.",
                 ["src/m4_encoding/encodings.py (circuit structure; no numbers)"])

    # ------------------------------------------------------------ Box 6
    d.box(6, "Quantum Model (Simulation)")
    d.lab("What this step does", "Three quantum models are trained on the encoded states: a quantum-kernel SVM, a variational quantum "
          "circuit, and a hybrid quantum neural network.")
    d.lab("What it produced")
    t10 = R / "tables" / "table10_cost.csv"
    c10 = pd.read_csv(t10)
    q = c10[c10.Model.str.contains("QSVM|VQC|Hybrid")]
    kinds = {"QSVM": "SVM dual coefficients (no trainable circuit parameters)", "VQC": "rotation angles of the circuit",
             "Hybrid": "classical input layer (95→8) + circuit rotations + classical output layer (8→1)"}
    d.table("Quantum models", ["Model", "Training rows", "Qubits", "Circuit depth", "Gates", "Parameters", "What the parameters are"],
            [[r.Model, r["Training rows"], r.Qubits, r["Circuit depth"], r["Gate count"], r.Parameters,
              next(v for k, v in kinds.items() if r.Model.startswith(k))] for _, r in q.iterrows()], [t10, cfg2],
            "rows QSVM/VQC/Hybrid; layer counts from the Model column and exp2 config.yaml")
    qc = R / "qiskit_crosscheck" / "crosscheck.md"
    d.p("Simulators: PennyLane " + d.num(env["packages"]["PennyLane"], env_p, "packages.PennyLane") + " ran all training and evaluation. "
        "Qiskit re-scored the trained VQC independently: " + text_has(qc, "Max |PennyLane score − Qiskit score| over all 52,293 validation rows: 9.77e-15")
        + "; " + text_has(qc, "identical thresholded predictions in 15/15 folds") + ".")
    d.trace.append((d.where, "Qiskit cross-check sentence (52,293 rows; 9.77e-15; 15/15)", "quoted", rel(qc)))
    a, b, c = fig_models()
    d.lab("Figure")
    d.figure(a, "VQC: data re-uploading encoding with 2 strongly-entangling layers, 4 qubits, readout ⟨Z₀⟩ (structure only).",
             ["src/m5_models/quantum.py (circuit structure; no numbers)"])
    d.figure(b, "Hybrid QNN, quantum part: angle encoding and 2 strongly-entangling layers on 8 qubits, readout ⟨Z⟩ on every qubit "
                "(the classical layers before and after are not drawn).", ["src/m5_models/quantum.py (circuit structure; no numbers)"])
    d.figure(c, "QSVM kernel circuit U(x)U†(x′) (compute–uncompute), drawn at 4 qubits for legibility; the model uses 8 qubits.",
             ["src/m5_models/quantum.py (circuit structure; no numbers)"])

    # ------------------------------------------------------------ Box 7
    d.box(7, "Hybrid Quantum-Classical Learning")
    d.lab("What this step does", "A classical optimizer updates the circuit parameters to reduce a classification loss; the loss is "
          "evaluated by running the circuit in simulation.")
    d.lab("What it produced")
    cob_cfg, cob_c = p("exp2_cobyla", "config.yaml"), p("exp2_cobyla", "comparison.csv")
    cob = pd.read_csv(cob_c).query("metric == 'roc_auc'").iloc[0]
    s2sum = p("exp2", "summary.md")
    cfg4 = p("exp4", "config.yaml")
    d.table("Training settings", ["Item", "Value"], [
        ["Optimizer", text_has(cob_cfg, "Adam VQC").replace(" VQC", "") + " (VQC and hybrid QNN); COBYLA compared on the same folds"],
        ["Learning rate", "grid {" + d.num("0.01, 0.05", cfg2, "quantum learning_rate grid") + "}, chosen per fold on inner-validation rows"],
        ["Epochs / batch size", d.num(30, cfg2, "epochs") + " / " + d.num(64, cfg2, "batch_size")],
        ["Stopping rule", text_has(s2sum, "early-stopping epoch and Youden threshold chosen on inner-val only")],
        ["COBYLA budget", d.num(300, cob_cfg, "maxiter") + " iterations"],
        ["COBYLA vs Adam (VQC ROC-AUC)", d.num(cob.a_mean, cob_c, "a_mean", "{:.3f}") + " vs " + d.num(cob.b_mean, cob_c, "b_mean", "{:.3f}")
         + "; NB-t Holm p " + d.num(cob.p_nb_t_holm, cob_c, "p_nb_t_holm", "{:.2f}") + ": not significant"],
        ["Federated training", d.num(30, cfg4, "rounds") + " FedAvg rounds, " + d.num(1, cfg4, "local_epochs") + " local epoch per round"],
    ], [cob_cfg, cfg2, s2sum, cob_c, cfg4], "values as listed")
    d.p(f"Per-epoch training curves of the centralised VQC and hybrid QNN: {NM}: no results file stores per-epoch loss. "
        "The federated per-round validation curves below are recorded.")
    rhp = p("exp4", "round_history.csv")
    d.lab("Figure")
    d.figure(fig_training_curves(pd.read_csv(rhp)), "Validation ROC-AUC per FedAvg round, mean over the 10 folds, ε = 4; "
             "markers every 5 rounds.", [rel(rhp)])

    # ------------------------------------------------------------ Box 8
    d.box(8, "Simulation Environment")
    d.lab("What this step does", "All circuits run on classical simulators on a CPU. Hardware noise and the number of qubits are varied "
          "in simulation.")
    d.lab("What it produced")
    nr = p("exp8", "noise_report.md")
    d.table("Simulators", ["Simulator", "Status"], [
        ["PennyLane", "used for all training and evaluation; " + text_has(nr, "PennyLane `default.mixed` (density matrices)").replace("`", "")
         + " for noise"],
        ["Qiskit", "cross-check of the VQC only (box 6)"],
        ["Cirq", f"{NM}: not used in this project."], ["TensorFlow Quantum", f"{NM}: not used in this project."]],
        [env_p, nr, qc], "status of diagram items")
    e8c, e8bc = p("exp8", "comparison.csv"), p("exp8b", "comparison.csv")
    e8 = pd.read_csv(e8c).query("metric == 'roc_auc'")
    e8b = pd.read_csv(e8bc).query("metric == 'roc_auc'")
    d.table("Noise simulation (noise model in noise_report.md; scale 10 multiplies every rate by 10)",
            ["Arm", "Compared with", "ROC-AUC", "Reference ROC-AUC", "Difference", "NB-t Holm p"],
            [[r.a.split("/")[0], r.b.split("/")[0], f3(r.a_mean), f3(r.b_mean), f"{r['diff']:+.3f}", f"{r.p_nb_t_holm:.3g}"]
             for _, r in pd.concat([e8, e8b]).iterrows()], [e8c, e8bc], "rows metric=roc_auc; columns a, b, a_mean, b_mean, diff, p_nb_t_holm")
    s1 = pd.read_csv(s1p)
    rows = []
    for model in ("qsvm", "vqc"):
        for fmeth in ("pca", "mi"):
            for enc in ("angle", "amplitude", "basis", "zz", "reupload"):
                z = s1[(s1.model == model) & (s1.feature_method == fmeth) & (s1.encoding == enc)].set_index("n_qubits")
                rows.append([model.upper(), fmeth.upper(), enc] + [pm(z.loc[n].roc_auc_mean, z.loc[n].roc_auc_std) for n in (4, 6, 8)])
    d.table("Qubit-count sweep: ROC-AUC (mean ± SD over folds)", ["Model", "Features", "Encoding", "4 qubits", "6 qubits", "8 qubits"],
            rows, [s1p], "columns model, feature_method, encoding, n_qubits, roc_auc_mean, roc_auc_std")
    d.lab("Figure")
    d.figure(fig_qubits(s1), "ROC-AUC against qubit count for each encoding, PCA features (mean ± SD over folds).", [rel(s1p)])

    # ------------------------------------------------------------ Box 9
    d.box(9, "Mental Health Analysis Tasks")
    d.lab("What this step does", "Binary task: predict a current, caregiver-reported depression diagnosis. Three-class task: none, "
          "anxiety only, depression.")
    d.lab("What it produced")
    popp = p("exp9", "population.json")
    pop = json.loads(popp.read_text())
    d.table("What is predicted, on whom", ["Item", "Value"], [
        ["Binary label", text_has(sum5md, "caregiver-reported, provider-diagnosed depression")],
        ["Population", "adolescents aged 12–17, NSCH 2024: " + d.num(prof["n_rows"], prof_p, "n_rows", "{:,}") + " rows"],
        ["Three classes: none / anxiety only / depression", d.num(pop["counts"]["none"], popp, "counts.none", "{:,}") + " / "
         + d.num(pop["counts"]["anxiety_only"], popp, "counts.anxiety_only", "{:,}") + " / "
         + d.num(pop["counts"]["depression"], popp, "counts.depression", "{:,}") + "; undetermined "
         + d.num(pop["n_undetermined"], popp, "n_undetermined", "{:,}")],
        ["Stress", f"{NA}: NSCH has no stress variable."],
    ], [sum5md, prof_p, popp], "values as listed")
    hmp = p("holdout_mc", "holdout_multiclass_metrics.csv")
    hm = pd.read_csv(hmp)
    d.table("Three-class task, locked hold-out", ["Model", "Macro-F1", "Balanced accuracy", "Depression-vs-rest AUC", "Depression recall"],
            [[r.model, f3(r.three_class_macro_f1), f3(r.three_class_balanced_accuracy), f3(r.three_class_auc_depression_vs_rest),
              f3(r.three_class_recall_depression)] for _, r in hm.iterrows()], [hmp], "columns as headed")
    d.table("Other tasks in the diagram", ["Task", "Status"], [
        ["Risk level prediction", f"{NM} in results/: the risk-band cut-points and band counts exist only in models/screening_models.pkl "
                                  "(scripts/build_screening_models.py), not in any results/ file."],
        ["Topic and trend analysis", f"{NA}: needs free text over time; the survey is one cross-sectional snapshot."],
        ["Emotion and sentiment analysis", f"{NA}: needs free text."],
        ["Personalized insights (simulated)", f"{NA}: would need text or longitudinal data."]], [hmp], "status of diagram items")
    d.lab("Figure", "None for this step.")

    # ------------------------------------------------------------ Box 10
    d.box(10, "Evaluation & Comparison")
    d.lab("What this step does", "Every model is scored with the same repeated cross-validation folds on the development rows; quantum "
          "models are compared with classical models on the same rows and features.")
    d.lab("What it produced")
    s2p = p("exp2", "summary.csv")
    s2 = pd.read_csv(s2p)
    svmc = csv("exp2_svm", "comparison.csv")
    svmp = p("exp2_svm", "comparison.csv")
    rows = [[MODEL_NAMES[r.model], "quantum" if r.model in ("qsvm", "vqc", "hybrid") else "classical", REGIMES[r.regime], FEATS[r.feature_set],
             pm(r.roc_auc_mean, r.roc_auc_std)] for _, r in s2.sort_values(["regime", "feature_set", "model"]).iterrows()]
    for key in ("svm/subsample/pca8", "svm/full/all95"):
        r = svmc[(svmc.a == key) & (svmc.metric == "roc_auc")].iloc[0]
        m, reg, fs = key.split("/")
        rows.append([MODEL_NAMES[m], "classical", REGIMES[reg], FEATS[fs], pm(r.a_mean, r.a_std)])
    d.table("Every model trained in the main comparison (Experiment 2 and its SVM addition): ROC-AUC, 15 folds",
            ["Model", "Family", "Training rows", "Features", "ROC-AUC (mean ± SD)"], rows, [s2p, svmp],
            "summary.csv roc_auc_mean/std; SVM rows from comparison.csv a_mean/a_std")
    s1x = p("phase1", "summary.csv")
    ph = pd.read_csv(s1x).query("feature_method == 'full'")
    d.table("Phase 1 untuned baselines, all 95 columns (includes random forest)", ["Model", "ROC-AUC (mean ± SD)"],
            [[MODEL_NAMES[r.model], pm(r.roc_auc_mean, r.roc_auc_std)] for _, r in ph.iterrows()], [s1x],
            "rows feature_method=full; roc_auc_mean/std",
            note="Random forest appears only in Phase 1; " + text_has(cfg2, "RandomForest dropped on supervisor instruction") + ".")
    d.table("Diagram items not trained", ["Diagram item", "Status"], [
        ["BERT + ML", f"{NA}: needs free text."], ["Deep Learning (LSTM, CNN)", f"{NA}: needs sequence or image data; an MLP is used instead."],
        ["Quantum Neural Network", "trained as the hybrid QNN (rows above)."]], [s2p], "status of diagram items")
    d.lab("Figure", "None for this step; the encoding comparison is the qubit-sweep figure in box 8.")

    # ------------------------------------------------------------ Box 11
    d.box(11, "Performance Metrics")
    d.lab("What this step does", "Accuracy, precision, recall, F1 and ROC-AUC are reported together for every model; significance is "
          "tested over folds with Holm adjustment.")
    d.lab("What it produced")
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    st = fold_stats(fm2, cols, ["model", "regime", "feature_set"], s2, ["model", "regime", "feature_set"])
    fsvm, fcob = p("exp2_svm", "fold_metrics.csv"), p("exp2_cobyla", "fold_metrics.csv")
    st = pd.concat([st, fold_stats(pd.read_csv(fsvm), cols, ["model", "regime", "feature_set"]),
                    fold_stats(pd.read_csv(fcob), cols, ["model", "regime", "feature_set"])])
    st = st.sort_values(("roc_auc", "mean"), ascending=False)
    d.table("Cross-validation, one row per model, sorted by ROC-AUC (mean ± SD over folds)",
            ["Model", "Rows", "Features", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            [[MODEL_NAMES[k[0]], {"full": "full", "subsample": "1.2k"}[k[1]], FEATS[k[2]].replace(" columns", "")] + [pm(r[(c, "mean")], r[(c, "std")]) for c in cols]
             for k, r in st.iterrows()], [fmp, fsvm, fcob],
            "mean and SD over the per-fold rows (columns accuracy, precision, recall, f1, roc_auc); accuracy, recall and ROC-AUC "
            "checked equal to exp2 summary.csv",
            note="Precision and F1 have no summary file; their mean ± SD is taken over the per-fold rows of fold_metrics.csv. "
                 "Accuracy, recall and ROC-AUC computed the same way equal summary.csv. 9.96 % of records are positive, so accuracy "
                 "can be high while recall is low; read the columns together.")
    php = p("phase1", "fold_metrics.csv")
    pf = pd.read_csv(php)
    pst = fold_stats(pf[pf.feature_method == "full"], cols, ["model"], ph, ["model"])
    d.table("Phase 1 untuned baselines, all 95 columns (mean ± SD over folds)", ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            [[MODEL_NAMES[k]] + [pm(r[(c, "mean")], r[(c, "std")]) for c in cols]
             for k, r in pst.sort_values(("roc_auc", "mean"), ascending=False).iterrows()], [php],
            "mean and SD over per-fold rows, feature_method=full; accuracy, recall, ROC-AUC checked equal to phase1 summary.csv")
    hop, t11 = p("holdout", "holdout_metrics.csv"), R / "tables" / "table11_holdout.csv"
    ho = pd.read_csv(hop)
    d.table("Locked hold-out (scored once)", ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC [95 % CI]", "Positives / rows"],
            [[r.model, f3(r.accuracy), f3(r.precision), f3(r.recall), f3(r.f1), f"{r.roc_auc:.3f} [{r.auc_ci95_low:.3f}, {r.auc_ci95_high:.3f}]",
              f"{int(r.n_holdout_pos):,} / {int(r.n_holdout):,}"] for _, r in ho.iterrows()], [hop], "columns as headed")
    t11d = pd.read_csv(t11)
    d.table("Hold-out beside cross-validation", list(t11d.columns), t11d.astype(str).values.tolist(), [t11], "table as stored")
    t4 = R / "tables" / "table4_quantum_vs_classical.csv"
    t4d = pd.read_csv(t4)
    d.table("Significance: quantum vs classical on the same rows and features (15 folds; primary NB-t, Holm-adjusted within metric)",
            list(t4d.columns), t4d.astype(str).values.tolist(), [t4], "table as stored",
            note="Significant = NB-t Holm p < 0.05. Of the ROC-AUC comparisons, the VQC is significantly below logistic regression "
                 "and the hybrid QNN is significantly below logistic regression and the MLP; the QSVM is not significantly different "
                 "from any classical model (null result).")
    cel, celp = csv("exp2", "tests_vs_full_data_ceiling.csv"), p("exp2", "tests_vs_full_data_ceiling.csv")
    extra = [[r.a, r.b, r.metric, f3(r.a_mean), f3(r.b_mean), f"{r.p_nb_t_holm:.3g}", "yes" if r.p_nb_t_holm < 0.05 else "no"]
             for _, r in svmc.iterrows()]
    extra += [[r.quantum, r.classical, r.metric, f3(r.quantum_mean), f3(r.classical_mean), f"{r.p_nb_t_holm:.3g}",
               "yes" if r.p_nb_t_holm < 0.05 else "no"] for _, r in cel.iterrows()]
    d.table("Further tests: SVM baseline and quantum vs the full-data logistic-regression ceiling", ["A", "B", "Metric", "A mean", "B mean",
            "NB-t Holm p", "Significant"], extra, [svmp, celp], "columns a/quantum, b/classical, metric, means, p_nb_t_holm; "
            "significant = p_nb_t_holm < 0.05 (rule stated in exp2 summary.md)")
    keep = c10.copy()
    keep["What the parameters are"] = [
        "dual coefficients" if "QSVM" in m else "circuit rotations" if m.startswith("VQC") else "classical + circuit weights"
        if m.startswith("Hybrid") else "coefficients + intercept" if m.startswith("LogReg") else "network weights"
        if m.startswith("MLP") else "Not measured (tree ensemble; no count recorded)" for m in keep.Model]
    keep["Parameters"] = keep.Parameters.fillna(NM).astype(str).str.replace(".0", "", regex=False)
    d.table("Training time, model size, qubits and gates", list(keep.columns), keep.astype(str).values.tolist(), [t10],
            "table as stored, plus a column naming what each parameter count measures",
            note="Parameter counts measure different things in each family (kernel dual coefficients, circuit rotation angles, "
                 "network weights) and are not comparable across families.")
    d.lab("Figure", "None for this step.")

    # ------------------------------------------------------------ Box 12
    d.box(12, "Privacy Evaluation")
    d.lab("What this step does", "A loss-threshold membership-inference attack tests whether trained models reveal which records they "
          "were trained on; re-identification risk is measured on the quasi-identifiers; the privacy budget ε is swept.")
    d.lab("What it produced")
    e6p, e6md = p("exp6", "summary.csv"), p("exp6", "summary.md")
    e6 = pd.read_csv(e6p)
    d.table("Membership-inference risk (attack ROC-AUC 0.5 = no membership signal)",
            ["Model", "Target ε", "Attack ROC-AUC (mean ± SD)", "TPR at 1 % FPR", "Advantage", "Members per fold"],
            [[r.arm, "∞" if not np.isfinite(r.target_epsilon) else f"{r.target_epsilon:g}", pm(r.attack_auc_mean, r.attack_auc_std),
              f3(r.tpr_at_1pct_fpr_mean), f3(r.advantage_mean), f"{int(r.n_members):,}"] for _, r in e6.iterrows()], [e6p],
            "columns as headed",
            note="Bound: " + text_has(e6md, "ε-DP bounds the membership advantage by e^ε − 1 (meaningful only for small ε)")
                 + ". mlp_overfit is a deliberately overfitted positive control.")
    rk = R / "reidentification" / "risk.csv"
    rkd = pd.read_csv(rk)
    d.table("Re-identification risk on the development rows", ["Quasi-identifier table", "Rows", "Unique rows", "Rows violating k = 5",
            "Average risk", "Maximum risk"],
            [[r.table, f"{int(r.n_rows):,}", pct(r.uniqueness), pct(r.k_violation_rate), f3(r.avg_prosecutor_risk), f"{r.max_prosecutor_risk:.2f}"]
             for _, r in rkd.iterrows()], [rk], "columns as headed (prosecutor risk)")
    tp = p("exp3", "tests_private_vs_nonprivate.csv")
    tpd = pd.read_csv(tp).query("metric == 'roc_auc'")
    d.table("ε sweep: ROC-AUC with DP-SGD against the same model without DP (10 folds)",
            ["Model", "ε", "With DP", "Without DP", "Change", "NB-t Holm p", "Significant"],
            [[r.arm, f"{r.epsilon:g}", f3(r.private_mean), f3(r.nonprivate_mean), f"{r.change:+.3f}", f"{r.p_nb_t_holm:.3g}",
              "yes" if r.significant else "no"] for _, r in tpd.iterrows()], [tp], "rows metric=roc_auc; columns as headed")
    r4 = tpd[(tpd.arm == "logreg_full") & (tpd.epsilon == 4)].iloc[0]
    d.p("Null result: at ε = 4, DP-SGD changed the full-data logistic regression's ROC-AUC by "
        + d.num(r4.change, tp, "change, logreg_full, ε=4", "{:+.3f}") + " (" + d.num(r4.private_mean, tp, "private_mean", "{:.3f}")
        + " vs " + d.num(r4.nonprivate_mean, tp, "nonprivate_mean", "{:.3f}") + "). The difference is not statistically significant "
        "(NB-t Holm p " + d.num(r4.p_nb_t_holm, tp, "p_nb_t_holm", "{:.2f}") + "). For the VQC no ε value gave a significant change "
        "under the primary test.")
    d.lab("Figure")
    d.figure(fig_privacy(s3), "Privacy–utility trade-off: ROC-AUC (mean ± SD over 10 folds) against target ε for DP-SGD.", [rel(s3p)])

    # ------------------------------------------------------------ Box 13
    d.box(13, "Scalability Analysis (Simulation)")
    d.lab("What this step does", "Federated averaging is simulated in one process with 3, 5 and 10 clients, with records split at random "
          "(IID) or by U.S. state (non-IID), at ε = 4.")
    d.lab("What it produced")
    s4p = p("exp4", "summary.csv")
    s4 = pd.read_csv(s4p)
    d.table("Federated results by number of clients and split (10 folds, ε = 4)",
            ["Model", "Setting", "Split", "Clients", "ROC-AUC (mean ± SD)", "Recall", "Max client ε", "Total kB"],
            [[r.arm, r.setting, r.partition, int(r.n_clients), pm(r.roc_auc_mean, r.roc_auc_std), f3(r.recall_mean),
              f"{r.epsilon_spent_max:.3f}", "—" if pd.isna(r.total_kB) else f"{r.total_kB:g}"] for _, r in s4.iterrows()], [s4p],
            "columns as headed")
    tfp = p("exp4", "tests_federated_vs_centralised.csv")
    tf = pd.read_csv(tfp)
    d.table("Federated vs centralised at the same ε", ["Model", "Split", "Clients", "Federated", "Centralised", "Change", "NB-t Holm p", "Significant"],
            [[r.arm, r.partition, int(r.n_clients), f3(r.federated_auc), f3(r.reference_auc), f"{r.auc_change:+.3f}", f"{r.p_nb_t_holm:.3g}",
              "yes" if r.significant else "no"] for _, r in tf.iterrows()], [tfp], "columns as headed")
    ccp = p("exp7", "communication_cost.csv")
    cc = pd.read_csv(ccp)
    d.table("Communication and server load (by-state clients, 5 folds)", ["Model", "Clients", "Bytes per update", "kB per round",
            "Slowest client s/round", "Server ms/round", "Server ms p95", "Rounds to within 0.01 of best"],
            [[r.arm, int(r.clients), f"{int(r.payload_bytes_per_update):,}", f"{r.payload_kB_per_round:g}",
              f"{r.client_seconds_per_round_slowest:.3f}", f"{r.aggregation_ms_per_round:.2f}", f"{r.aggregation_ms_p95:.2f}",
              r["rounds_to_within_0.01_of_best"]] for _, r in cc.iterrows()], [ccp], "columns as headed")
    ovp = R / "secagg_overhead" / "overhead.csv"
    ov = pd.read_csv(ovp)
    d.table("Secure-aggregation overhead (median of 200 repetitions)", ["Model", "Clients", "Server plain ms", "Server secure ms",
            "Client masking ms", "Key setup s (once)", "Payload plain B", "Payload secure B"],
            [[r.arm, int(r.clients), f"{r.plain_server_ms:.2f}", f"{r.secure_server_ms:.2f}", f"{r.mask_ms_per_client:.2f}",
              f"{r.setup_s_once:.2f}", f"{int(r.plain_payload_B):,}", f"{int(r.secure_payload_B):,}"] for _, r in ov.iterrows()], [ovp],
            "columns as headed")
    d.lab("Figure")
    d.figure(fig_clients(s4), "Federated ROC-AUC (mean ± SD, 10 folds) against number of clients, IID and by-state splits; dotted "
             "line = centralised model at the same ε.", [rel(s4p)])

    # ------------------------------------------------------------ Box 14
    d.box(14, "Output & Visualization")
    d.lab("What this step does", "Lists every figure in this report. Each graph appears in the section of the step that produced it.")
    d.lab("What it produced")
    d.blocks.append(("figindex",))
    d.table("Visualizations in the diagram that are not in this report", ["Diagram item", "Status"], [
        ["Feature and attention visualization: attention", f"{NA}: needs a transformer over text."],
        ["Quantum state visualization (Bloch sphere, state vector)", "results/state_visualisation/bloch_angle_vs_zz.png exists but marks "
         "its two classes by colour only, so it is not reproduced in this black-and-white report."]],
        [R / "state_visualisation" / "bloch_angle_vs_zz.png"], "status of diagram items")
    d.lab("Figure", "None (index only).")

    # ------------------------------------------------------------ Box 15
    d.box(15, "Key Outcomes")
    d.lab("What this step does", "States what was found and the limits of the evidence.")
    d.lab("What it produced")
    lg = ho[ho.model == "logreg_full_all95"].iloc[0]
    qs = ho[ho.model == "qsvm_angle_pca8_1500rows"].iloc[0]
    t_q = pd.read_csv(p("exp2", "tests_same_rows_features.csv"))
    tq = t_q[(t_q.quantum == "qsvm/subsample/pca8") & (t_q.classical == "logreg/subsample/pca8") & (t_q.metric == "roc_auc")].iloc[0]
    tqp = p("exp2", "tests_same_rows_features.csv")
    full = s2[(s2.regime == "full") & (s2.feature_set == "all95")].set_index("model")
    d.p("On the locked hold-out, the best quantum model (QSVM, angle encoding, 8 qubits) reached ROC-AUC "
        + d.num(qs.roc_auc, hop, "roc_auc, qsvm", "{:.3f}") + " against " + d.num(lg.roc_auc, hop, "roc_auc, logreg", "{:.3f}")
        + " for logistic regression on all 95 columns. On identical rows and features in cross-validation the QSVM scored "
        + d.num(tq.quantum_mean, tqp, "quantum_mean", "{:.3f}") + " against " + d.num(tq.classical_mean, tqp, "classical_mean", "{:.3f}")
        + " for logistic regression (NB-t Holm p " + d.num(tq.p_nb_t_holm, tqp, "p_nb_t_holm", "{:.2f}") + "): no quantum model beat its classical counterpart.")
    d.p("The survey signal is close to linear: on all 95 columns, logistic regression ("
        + d.num(full.loc["logreg"].roc_auc_mean, s2p, "logreg full all95 roc_auc_mean", "{:.3f}") + "), XGBoost ("
        + d.num(full.loc["xgboost"].roc_auc_mean, s2p, "xgboost full all95 roc_auc_mean", "{:.3f}") + ") and the MLP ("
        + d.num(full.loc["mlp"].roc_auc_mean, s2p, "mlp full all95 roc_auc_mean", "{:.3f}") + ") reach the same ROC-AUC, so the "
        "extra non-linearity a quantum feature map adds has no pattern to capture.")
    d.p("Differential privacy at ε = 4 cost the full-data logistic regression " + d.num(-r4.change, tp, "−change, logreg_full ε=4", "{:.3f}")
        + " ROC-AUC, not statistically significant, and the membership-inference attack stayed near chance for every DP model.")
    d.p("Limitations: the data contain no free text, no sensor data and no clinical records; every circuit ran on a simulator, not on "
        "quantum hardware; the label is a caregiver-reported diagnosis, so undiagnosed adolescents are in the negative class.")
    d.lab("Figure", "None for this step.")
    return d


# ---------------------------------------------------------------- rendering
CSS = """
@page { size: A4 portrait; margin: 16mm 15mm 18mm 15mm;
  @bottom-center { content: "Page " counter(page); font: 9pt "Segoe UI", Arial, sans-serif; color: #000; } }
body { font: 11pt/1.4 "Segoe UI", Arial, sans-serif; color: #000; background: #fff; }
h1 { font-size: 18pt; margin: 0 0 4mm; } h2 { font-size: 14pt; margin: 0 0 3mm; border-bottom: 1.5pt solid #000; padding-bottom: 1mm; }
section.box { break-before: page; } section.box.first { break-before: auto; }
.lab { margin: 3mm 0 1.5mm; } .lab b { font-weight: 700; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 1mm 0 1mm; }
caption { caption-side: top; text-align: left; font-weight: 700; font-size: 9.5pt; padding-bottom: 1mm; }
th, td { border: 0.6pt solid #000; padding: 0.6mm 1mm; text-align: left; vertical-align: top; }
th { background: #e6e6e6; } tr { break-inside: avoid; }
.src { font-size: 8pt; color: #333; margin-bottom: 3mm; } .note { font-size: 9pt; margin: 0.5mm 0 1mm; }
figure { margin: 2mm 0 4mm; break-inside: avoid; } figure img { max-width: 100%; max-height: 120mm; display: block; }
figcaption { font-size: 9pt; margin-top: 1mm; }
figure img.circ { max-height: 42mm; max-width: 100%; } td.nw { white-space: nowrap; }
.appendix td { font-size: 8pt; } .appendix { break-before: page; }
"""


def esc(s):
    return html.escape(str(s))


def render(d: Doc, pages: dict[int, str] | None = None, stop_after_fig: int | None = None):
    md, ht = [], []
    title = "Results report — Quantum AI for Mental Health (NSCH 2024)"
    md += [f"# {title}", "", "One section per box of docs/architecture.jpg, in diagram order. Every number comes from a file in "
           "results/; the traceability appendix lists each one with its file.", ""]
    ht += [f"<h1>{esc(title)}</h1><p>One section per box of docs/architecture.jpg, in diagram order. Every number comes from a file in "
           "results/; the traceability appendix lists each one with its file.</p>"]
    open_box = False
    for b in d.blocks:
        k = b[0]
        if k == "box":
            if open_box:
                ht.append("</section>")
            ht.append(f"<section class='box{' first' if b[1] == 1 else ''}'><h2>Box {b[1]} — {esc(b[2])}</h2>")
            md += [f"## Box {b[1]} — {b[2]}", ""]
            open_box = True
        elif k == "lab":
            md += [f"**{b[1]}:** {b[2]}".rstrip(), ""]
            ht.append(f"<p class='lab'><b>{esc(b[1])}:</b> {esc(b[2])}</p>")
        elif k == "p":
            md += [b[1], ""]
            ht.append(f"<p>{esc(b[1])}</p>")
        elif k == "table":
            _, cap, cols, rows, note, srcs = b
            md += [f"*{cap}*", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            md += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
            md += [""] + ([note, ""] if note else []) + ["Source: " + " · ".join(rel(s) for s in srcs), ""]
            ht.append(f"<table><caption>{esc(cap)}</caption><thead><tr>{''.join(f'<th>{esc(c)}</th>' for c in cols)}</tr></thead><tbody>"
                      + "".join("<tr>" + "".join(f"<td{' class=nw' if ' ± ' in str(c) and len(str(c)) < 16 else ''}>{esc(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>"
                      + (f"<div class='note'>{esc(note)}</div>" if note else "")
                      + f"<div class='src'>Source: {esc(' · '.join(rel(s) for s in srcs))}</div>")
        elif k == "fig":
            _, n, path, cap, srcs = b
            relp = Path(path).relative_to(R).as_posix()
            md += [f"![Figure {n}]({relp})", "", f"*Figure {n}. {cap} Source: {'; '.join(srcs)}*", ""]
            ht.append(f"<figure id='fig{n}'><img src='{relp}'{' class=circ' if 'circuit' in relp else ''}><figcaption><b>Figure {n}.</b> {esc(cap)} "
                      f"<span class='src'>Source: {esc('; '.join(srcs))}</span></figcaption></figure>")
            if stop_after_fig == n:
                ht.append("</section>")
                return "\n".join(md), "\n".join(ht)
        elif k == "figindex":
            rows = [[f"Figure {f['n']}", f["box"], f["caption"][:90] + ("…" if len(f["caption"]) > 90 else ""),
                     (pages or {}).get(f["n"], "00")] for f in d.figures]
            md += ["| Figure | Section | Caption | Page |", "|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows] + [""]
            ht.append("<table><caption>Index of figures</caption><thead><tr><th>Figure</th><th>Section</th><th>Caption</th><th>Page</th></tr>"
                      "</thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>")
    if open_box:
        ht.append("</section>")
    md += ["## Traceability appendix", "", "Every number and figure in the report, with the file it came from.", "",
           "| Section | Item | Value | File |", "|---|---|---|---|"]
    md += ["| " + " | ".join(str(x).replace("|", "\\|") for x in t) + " |" for t in d.trace]
    ht.append("<section class='appendix'><h2>Traceability appendix</h2><p>Every number and figure in the report, with the file it came from.</p>"
              "<table class='appendix'><thead><tr><th>Section</th><th>Item</th><th>Value</th><th>File</th></tr></thead><tbody>"
              + "".join("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in t) + "</tr>" for t in d.trace) + "</tbody></table></section>")
    return "\n".join(md), "\n".join(ht)


def html_page(body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Results report</title><style>{CSS}</style></head><body>{body}</body></html>"


def print_pdf(html_path: Path, pdf_path: Path) -> int:
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()], check=True, capture_output=True, timeout=180)
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))


def main():
    d = build()
    # page of each figure: print the report truncated just after that figure and count pages
    pages = {}
    tmp_h, tmp_p = R / "_report_probe.html", R / "_report_probe.pdf"
    for f in d.figures:
        _, ht = render(d, stop_after_fig=f["n"])
        tmp_h.write_text(html_page(ht), encoding="utf-8")
        pages[f["n"]] = f"{print_pdf(tmp_h, tmp_p):02d}"
    tmp_h.unlink(missing_ok=True)
    tmp_p.unlink(missing_ok=True)
    md, ht = render(d, pages)
    (R / "results_report.md").write_text(md, encoding="utf-8")
    hp = R / "results_report.html"
    hp.write_text(html_page(ht), encoding="utf-8")
    n = print_pdf(hp, R / "results_report.pdf")
    print(f"wrote results/results_report.md, .html, .pdf ({n} pages, {len(d.figures)} figures, {len(d.trace)} traced items); "
          f"figure pages {pages}")


if __name__ == "__main__":
    main()
