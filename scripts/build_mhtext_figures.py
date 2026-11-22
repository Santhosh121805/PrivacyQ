"""Figures for the Mental Health Text results report (results/mhtext/results_report_mhtext.md / .pdf).

Rules: every plotted value is read from a committed file under results/mhtext/ (NSCH values from the verbatim
copies in results/mhtext/phase7/nsch_reference/); nothing is recomputed or estimated here - the only operations are
selecting rows and sorting. Every column that is plotted passes through `use()`, which stops the build if any cell is
a suppressed small count (src/m9_audit/disclosure.py), so a "< 5 posts" cell can never become a point, bar or label.
No post text is read. Black-and-white safe: series differ by marker, line style or hatching, never by colour alone.

python scripts/build_mhtext_figures.py  -> results/mhtext/report_figures/*.png + figures.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.m9_audit.disclosure import is_suppressed  # noqa: E402

MH = ROOT / "results/mhtext"
FIG = MH / "report_figures"
NREF = MH / "phase7/nsch_reference"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "axes.prop_cycle": plt.cycler(color=["black"])})
MARKERS = ["o", "s", "^", "D", "v", "x", "P"]
LINES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1))]
HATCH = ["", "////", "....", "xxxx", "\\\\\\\\", "----"]
FIGS: list[dict] = []
CHECKED: list[str] = []


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def read(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.attrs["src"] = rel(p)
    return df


def use(df: pd.DataFrame, *cols: str) -> list[np.ndarray]:
    """The only way a column reaches a chart: refuse suppressed cells, return floats."""
    out = []
    for c in cols:
        s = df[c]
        bad = s.map(is_suppressed)
        if bad.any():
            raise SystemExit(f"refusing to plot suppressed cells: {df.attrs.get('src')} column {c}, rows {list(s.index[bad])}")
        CHECKED.append(f"{df.attrs.get('src')}:{c}:{len(s)}")
        out.append(pd.to_numeric(s).to_numpy(float))
    return out


def save(fig, name: str, box: int, caption: str, sources: list[str]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGS.append({"file": name, "box": box, "caption": caption, "sources": sources})


def eps_key(v) -> str:
    return "inf" if str(v) in ("inf", "nan", "None") else f"{float(v):g}"


# ---------------------------------------------------------------- Box 3
def fig_mi():
    mi = read(MH / "phase2/mi_scores_4class.csv")
    top = mi.sort_values("rank").head(20).iloc[::-1].reset_index(drop=True)
    fams = [("emb_pooled_", "MiniLM embedding, whole post"), ("emb_sent_mean_", "MiniLM embedding, sentence mean"),
            ("emb_sent_max_", "MiniLM embedding, sentence max"), ("emotion_", "Emotion probability"), ("", "Shipped column")]
    fam = [next(i for i, (p, _) in enumerate(fams) if top.column[k].startswith(p)) for k in range(len(top))]
    (v,) = use(top, "mutual_information")
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    for y in range(len(top)):
        ax.barh(y, v[y], color="white", edgecolor="black", hatch=HATCH[fam[y]], lw=0.8)
    ax.set_yticks(range(len(top)), top.column, fontsize=7.5)
    ax.set_xlabel("Mutual information with the 4-class label (nats), development rows")
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch=HATCH[i], label=l)
                       for i, (_, l) in enumerate(fams) if i in fam], loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2, fontsize=7.5, frameon=False)
    save(fig, "fig_mhtext_mi_ranking.png", 3,
         "Top 20 of 1,174 text features by mutual information with the 4-class label; hatching marks the feature family. "
         "Column names only; no post text.", [top.attrs["src"]])


# ---------------------------------------------------------------- Box 5
def fig_encodings():
    sw = read(MH / "phase4/binary/sweep_summary.csv")
    encs = ["angle", "reupload", "amplitude", "zz", "basis"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    w = 0.26
    for ax, model in zip(axes, ["qsvm", "vqc"]):
        for k, nq in enumerate([4, 6, 8]):
            d = sw[(sw.model == model) & (sw.feature_method == "pca") & (sw.n_qubits == nq)].set_index("encoding").loc[encs]
            d.attrs["src"] = sw.attrs["src"]
            m, s = use(d, "roc_auc", "roc_auc_std")
            ax.bar(np.arange(len(encs)) + (k - 1) * w, m, w, yerr=s, color="white", edgecolor="black", hatch=HATCH[k + 1],
                   lw=0.8, capsize=2, error_kw={"lw": 0.8}, label=f"{nq} qubits")
        ax.set_xticks(range(len(encs)), encs, fontsize=7.5, rotation=25)
        ax.set_title(f"{model.upper()} · PCA inputs", fontsize=9)
        ax.set_ylim(0.45, 0.85)
    axes[0].set_ylabel("ROC AUC (mean ± SD, 10 folds)")
    axes[1].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_mhtext_encodings.png", 5,
         "Encoding comparison on the binary Depression vs Suicidal view, 1,200 training rows, PCA inputs: ROC AUC per "
         "encoding at 4, 6 and 8 qubits (bars, hatched by qubit count) with SD error bars.", [sw.attrs["src"]])


# ---------------------------------------------------------------- Box 6
def fig_qubits():
    qp = read(MH / "phase7/pattern_checks/p3_qubit_pairs.csv")
    cells = (qp[qp.contrast == "8-4"].drop_duplicates(["model", "feature_method", "encoding"])
             [["model", "feature_method", "encoding"]].values.tolist())
    cells.sort(key=lambda c: (c[0], c[1], c[2]))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 6.4), sharey=True)
    for ax, ds in zip(axes, ["NSCH 2024", "Mental Health Text"]):
        for k, (con, mk) in enumerate((("6-4", "o"), ("8-6", "^"), ("8-4", "s"))):
            d = qp[(qp.dataset == ds) & (qp.contrast == con)].set_index(["model", "feature_method", "encoding"]).loc[
                [tuple(c) for c in cells]].reset_index()
            d.attrs["src"] = qp.attrs["src"]
            m, lo, hi = use(d, "mean_diff", "ci95_low", "ci95_high")
            y = np.arange(len(cells)) + (k - 1) * 0.25
            ax.hlines(y, lo, hi, color="black", lw=0.7, linestyle=LINES[k])
            sig = d.significant.to_numpy(bool)
            ax.plot(m[~sig], y[~sig], mk, color="black", mfc="white", ms=4, label=f"{con} qubits")
            ax.plot(m[sig], y[sig], mk, color="black", mfc="black", ms=4,
                    label=f"{con} qubits, Holm p < 0.05" if sig.any() else None)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("Paired change in ROC AUC (95 % CI)")
    axes[0].set_yticks(range(len(cells)), [f"{a.upper()} · {b} · {c}" for a, b, c in cells], fontsize=7)
    axes[0].invert_yaxis()
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.55, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_mhtext_qubit_pairs.png", 6,
         "Qubit count, paired: change in ROC AUC from 4 to 6, 6 to 8 and 4 to 8 qubits for every quantum model x feature "
         "method x encoding cell, NSCH beside Mental Health Text (paired NB-t over the same folds, 95 % CI; filled marker "
         "= Holm p < 0.05). Backs pattern P3.", [qp.attrs["src"]])


# ---------------------------------------------------------------- Box 11
def fig_headline():
    s = read(MH / "phase4/binary/summary.csv")
    sub = s[(s.regime == "subsample") & (s.feature_set != "length")].sort_values("roc_auc_mean").reset_index(drop=True)
    sub.attrs["src"] = s.attrs["src"]
    m, sd = use(sub, "roc_auc_mean", "roc_auc_std")
    names = {"qsvm": "QSVM", "vqc": "VQC", "hybrid": "Hybrid QNN", "logreg": "LogReg", "xgboost": "XGBoost", "mlp": "MLP",
             "svm": "SVM (RBF)"}
    inputs = {"pca4": "PCA-4", "pca8": "PCA-8", "all384": "all 384 dims"}
    quantum = sub.model.isin(["qsvm", "vqc", "hybrid"]).to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    y = np.arange(len(sub))
    for i in y:
        ax.barh(i, m[i], xerr=sd[i], color="white", edgecolor="black", hatch="////" if quantum[i] else "", lw=0.8,
                capsize=2, error_kw={"lw": 0.8})
    ax.set_yticks(y, [f"{names[a]} · {inputs[b]}" for a, b in zip(sub.model, sub.feature_set)], fontsize=7.5)
    refs = [(s[(s.model == "logreg") & (s.regime == "subsample") & (s.feature_set == "length")], ":", "length-only floor (same 1,200 rows)"),
            (s[(s.model == "svm") & (s.regime == "full") & (s.feature_set == "all384")], "--", "classical ceiling, 5,044 rows (SVM)"),
            (s[(s.model == "logreg") & (s.regime == "all_rows") & (s.feature_set == "all384")], "-.", "classical ceiling, all rows (LogReg)")]
    for d, ls, lab in refs:
        d.attrs["src"] = s.attrs["src"]
        (x,) = use(d, "roc_auc_mean")
        ax.axvline(x[0], color="black", lw=0.9, linestyle=ls, label=lab)
    ax.set_xlim(0.5, 0.86)
    ax.set_xlabel("ROC AUC (mean ± SD, 15 folds)")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [Patch(facecolor="white", edgecolor="black", hatch="////", label="quantum model"),
                                                           Patch(facecolor="white", edgecolor="black", label="classical model")],
              loc="upper center", bbox_to_anchor=(0.42, -0.1), ncol=2, fontsize=7, frameon=False)
    save(fig, "fig_mhtext_headline.png", 11,
         "Every quantum and classical model on the same 1,200 training rows and the same inputs (binary Depression vs "
         "Suicidal view), sorted by ROC AUC; reference lines: length-only floor and the two classical ceilings.",
         [s.attrs["src"]])


# ---------------------------------------------------------------- Box 12
def fig_dp():
    t = read(MH / "phase5/dp/summary.csv")
    n = read(NREF / "nsch_exp3_summary.csv")
    order = ["0.5", "1", "2", "4", "8", "inf"]
    arms = [("vqc_sub", "VQC, 1.2k rows, PCA-4"), ("logreg_sub", "LogReg, same 1.2k rows, PCA-4"),
            ("logreg_full", "LogReg, all training rows, all inputs")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
    for ax, (df, mcol, scol, title) in zip(axes, [(n, "roc_auc_mean", "roc_auc_std", "NSCH 2024"),
                                                   (t, "mean", "std", "Mental Health Text")]):
        for k, (arm, lab) in enumerate(arms):
            d = df[df.arm == arm].assign(e=lambda x: x.target_epsilon.map(eps_key)).set_index("e").loc[order]
            d.attrs["src"] = df.attrs["src"]
            m, s = use(d, mcol, scol)
            ax.errorbar(np.arange(6) + (k - 1) * 0.08, m, yerr=s, linestyle=LINES[k], marker=MARKERS[k], ms=4.5,
                        mfc="white" if k == 1 else "black", color="black", capsize=2, lw=1, label=lab)
        ax.set_xticks(range(6), ["0.5", "1", "2", "4", "8", "∞"])
        ax.set_xlabel("DP-SGD target ε (δ = 1e-5; ∞ = no DP)")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("ROC AUC (mean ± SD, 10 folds)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_mhtext_dp_utility.png", 12,
         "Utility against privacy budget for DP-SGD, NSCH beside Mental Health Text, same arms and settings (exp3 method).",
         [n.attrs["src"] + " (copy of " + n.source_file.iloc[0] + ")", t.attrs["src"]])


def fig_reid():
    fr = read(MH / "phase5/reid/fragment_risk.csv")
    ns = read(NREF / "nsch_reidentification_risk.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    series = [("dev", "as released", "development posts, as released"),
              ("dev", "scrubbed", "development posts, identifiers scrubbed"),
              ("dev+quarantine", "as released", "development + quarantined posts, as released")]
    for k, (corpus, ver, lab) in enumerate(series):
        d = fr[(fr.corpus == corpus) & fr.text.str.startswith(ver) & fr.attacker_knows.str.startswith("a ")].sort_values("span_words")
        d.attrs["src"] = fr.attrs["src"]
        x, u = use(d, "span_words", "uniqueness")
        ax.plot(x, u, linestyle=LINES[k], marker=MARKERS[k], mfc="white" if k == 1 else "black", color="black", ms=4.5, lw=1, label=lab)
    for k, (tab, lab) in enumerate([("7 demographic QIs, raw values + U.S. state", "NSCH: 7 demographics + state, raw"),
                                    ("7 demographic QIs, raw values", "NSCH: 7 demographics, raw"),
                                    ("7 demographic QIs after k-anonymity (k = 5, as used in the pipeline)", "NSCH: after k-anonymity (k = 5)")]):
        d = ns[ns.table == tab]
        d.attrs["src"] = ns.attrs["src"]
        (u,) = use(d, "uniqueness")
        ax.axhline(u[0], color="black", lw=0.8, linestyle=[(0, (1, 1)), (0, (4, 2)), (0, (6, 2, 1, 2))][k], label=lab)
    ax.set_xticks([3, 5, 8, 12, 20])
    ax.set_xlabel("Words in the fragment the attacker knows")
    ax.set_ylabel("Share of rows uniquely identified")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, frameon=False)
    save(fig, "fig_mhtext_reid_uniqueness.png", 12,
         "Re-identification: share of posts singled out by a fragment of k consecutive words, against k; horizontal lines: "
         "share of NSCH adolescents unique on their quasi-identifiers, before and after k-anonymity.",
         [fr.attrs["src"], ns.attrs["src"] + " (copy of " + ns.source_file.iloc[0] + ")"])


def fig_linkage():
    fr = read(MH / "phase5/reid/fragment_risk.csv")
    lk = read(MH / "phase5/reid/linkage.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    d = fr[(fr.corpus == "dev") & (fr.text == "as released") & fr.attacker_knows.str.startswith("a ")
           & fr.span_words.isin([8, 12, 20])].sort_values("span_words")
    d.attrs["src"] = fr.attrs["src"]
    x, r = use(d, "span_words", "avg_prosecutor_risk")
    ax.plot(x, r, "-o", color="black", ms=4.5, lw=1, label="released text (expected top-1 = average prosecutor risk)")
    for k, (rel_, lab) in enumerate([("MiniLM embedding (384 dims)", "released MiniLM vectors (384 dims)"),
                                     ("PCA-8 encoder input", "released 8 quantum inputs (PCA-8, projection known)")]):
        dd = lk[(lk.released == rel_) & (lk.epsilon_input.map(eps_key) == "inf")].sort_values("fragment_words")
        dd.attrs["src"] = lk.attrs["src"]
        x, t1 = use(dd, "fragment_words", "top1")
        ax.plot(x, t1, linestyle=LINES[k + 1], marker=MARKERS[k + 1], mfc="white", color="black", ms=4.5, lw=1, label=lab)
    ax.set_xticks([8, 12, 20])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Words in the fragment the attacker knows")
    ax.set_ylabel("Share of posts re-identified (top-1)")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "fig_mhtext_linkage.png", 12,
         "Linkage success against fragment length, without local DP, for three released forms of the same development "
         "posts. Cells under local DP that are suppressed (< 5 posts) are not plotted; the local-DP rows are in the table.",
         [fr.attrs["src"], lk.attrs["src"]])


# ---------------------------------------------------------------- Box 13
def fig_federated():
    t = read(MH / "phase6/summary.csv")
    n = read(NREF / "nsch_exp4_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3), sharey=True)
    for ax, (df, title, nonid, nonid_lab) in zip(axes, [(n, "NSCH 2024", "by_state", "non-IID by U.S. state"),
                                                         (t, "Mental Health Text", "by_length", "non-IID by post length")]):
        for a, (arm, alab) in enumerate((("vqc_sub", "VQC"), ("logreg_full", "LogReg"))):
            for p, (part, plab) in enumerate((("iid", "IID"), (nonid, nonid_lab))):
                d = df[(df.arm == arm) & (df.partition == part) & (df.setting == "federated")].sort_values("n_clients")
                d.attrs["src"] = df.attrs["src"]
                x, m, s = use(d, "n_clients", "roc_auc_mean", "roc_auc_std")
                ax.errorbar(x + (p - 0.5) * 0.15, m, yerr=s, linestyle=LINES[p], marker=MARKERS[a], ms=4.5,
                            mfc="black" if p == 0 else "white", color="black", capsize=2, lw=1, label=f"{alab}, {plab}")
        ax.set_xticks([3, 5, 10])
        ax.set_xlabel("Simulated clients")
        ax.set_title(f"{title} · client DP-SGD ε = 4", fontsize=9)
        ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    axes[0].set_ylabel("ROC AUC (mean ± SD, 10 folds)")
    fig.tight_layout()
    save(fig, "fig_mhtext_federated.png", 13,
         "Federated learning with client DP-SGD (ε = 4): ROC AUC against the number of simulated clients for the VQC and "
         "LogReg, NSCH beside Mental Health Text. Pattern P5: the VQC falls as clients increase, LogReg stays flat, on both.",
         [n.attrs["src"] + " (copy of " + n.source_file.iloc[0] + ")", t.attrs["src"]])


# ---------------------------------------------------------------- Box 15
def fig_replication():
    rs = read(MH / "phase7/pattern_checks/replication_summary.csv")
    panels = [("P1", "P1  QSVM ties classical on matched rows — replicates"),
              ("P2", "P2  VQC loses about 0.02, significantly — replicates"),
              ("P3", "P3  4 → 8 qubits: no significant gain — replicates"),
              ("P5", "P5  federated VQC falls with clients, LogReg flat — replicates"),
              ("P6", "P6  hybrid QNN lowest quantum model (ROC AUC) — replicates"),
              ("P4", "P4  PCA vs MI selection — the datasets DISAGREE")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.6, 8.6))
    style = {"NSCH 2024": ("o", "black"), "Mental Health Text": ("s", "white")}
    for ax, (pat, title) in zip(axes, panels):
        d = rs[rs.pattern == pat].reset_index(drop=True)
        d.attrs["src"] = rs.attrs["src"]
        if pat == "P4":
            (v,) = use(d, "estimate")
            labels = [f"{r.dataset}: {r.quantity}" for _, r in d.iterrows()]
            for i in range(len(d)):
                ax.barh(i, v[i], color="white", edgecolor="black", hatch="" if d.dataset[i] == "NSCH 2024" else "////", lw=0.8)
                ax.text(v[i] + 0.3, i, f"{int(v[i])} of {int(d.n_cells[i])}", va="center", fontsize=7)
            ax.set_yticks(range(len(d)), labels, fontsize=7)
            ax.set_xlim(0, 34)
            ax.set_xlabel("cells (of 33) with a significant difference", fontsize=7.5)
        else:
            qty = list(dict.fromkeys(d.quantity))
            for i, q in enumerate(qty):
                for j, ds in enumerate(["NSCH 2024", "Mental Health Text"]):
                    r = d[(d.quantity == q) & (d.dataset == ds)]
                    r.attrs["src"] = rs.attrs["src"]
                    (e,) = use(r, "estimate")
                    yy = i + (j - 0.5) * 0.3
                    if r.ci95_low.notna().all():
                        lo, hi = use(r, "ci95_low", "ci95_high")
                        ax.hlines(yy, lo[0], hi[0], color="black", lw=0.8)
                    mk, fc = style[ds]
                    ax.plot(e[0], yy, mk, color="black", mfc=fc, ms=5, label=ds if i == 0 else None)
            ax.axvline(0, color="black", lw=0.6)
            ax.set_yticks(range(len(qty)), [q.replace(", ROC AUC", "") for q in qty], fontsize=7)
            ax.set_ylim(len(qty) - 0.45, -0.55)
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax.tick_params(axis="x", labelsize=7.5)
            ax.set_xlabel("difference in ROC AUC (95 % CI where tested)", fontsize=7.5)
        if pat == "P4":
            ax.invert_yaxis()
        ax.set_title(title, fontsize=8.5, loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.6, 1.02))
    fig.tight_layout(h_pad=0.6, rect=(0, 0, 1, 0.98))
    save(fig, "fig_mhtext_replication.png", 15,
         "Replication summary: the five patterns that replicate across NSCH (filled circle) and Mental Health Text (open "
         "square), as differences in ROC AUC with 95 % CIs where a test exists, and the one disagreement (P4) as counts of "
         "significant cells.", [rs.attrs["src"]])


def main():
    for f in (fig_mi, fig_encodings, fig_qubits, fig_headline, fig_dp, fig_reid, fig_linkage, fig_federated, fig_replication):
        f()
    (FIG / "figures.json").write_text(json.dumps({"figures": FIGS, "plotted_columns_checked_for_suppression": CHECKED},
                                                 indent=1), encoding="utf-8")
    print(f"wrote {len(FIGS)} figures to {FIG}; {len(CHECKED)} plotted columns checked, none suppressed")


if __name__ == "__main__":
    main()
