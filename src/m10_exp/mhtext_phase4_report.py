"""Phase 4 report: quantum vs classical on the Mental Health Text dataset, beside the NSCH numbers.

Reads only finished files: results/mhtext/phase4/<view>/{compare,sweep}_fold_metrics.csv for this dataset, and the
existing NSCH exp2 run (summary.csv, tests_same_rows_features.csv) for the comparison column. Nothing from NSCH is
recomputed. Tests are the NSCH ones (exp2_report.compare: Nadeau-Bengio corrected t, Holm within metric).

python -m src.m10_exp.mhtext_phase4_report --view binary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.m8_eval.stats import add_holm
from src.m10_exp.exp2_report import HEAD, compare

ROOT = Path(__file__).resolve().parents[2]
NSCH_EXP2 = ROOT / "results" / "exp2-nsch2024-quantum-vs-classical-20261108-165426"
QUANTUM = {"qsvm": "pca8", "vqc": "pca4", "hybrid": "all384"}
CLASSICAL = ["logreg", "xgboost", "mlp", "svm"]
NAMES = {"qsvm": "QSVM (angle, 8 qubits)", "vqc": "VQC (re-upload, 4 qubits)", "hybrid": "Hybrid QNN (angle, 8 qubits)",
         "logreg": "Logistic regression", "xgboost": "XGBoost", "mlp": "MLP", "svm": "SVM (RBF)"}
SWEEP_NAMES = {"qsvm": "QSVM", "vqc": "VQC", "logreg": "Logistic regression (no encoding)"}
INPUTS = {"pca8": "PCA-8 of MiniLM", "pca4": "PCA-4 of MiniLM", "all384": "all 384 MiniLM dims",
          "length": "log words + log characters"}


def f3(x) -> str:
    return f"{x:.3f}"


def tests(df: pd.DataFrame, ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    same, floor, ceil = [], [], []
    full = df[(df.regime == "full") & (df.feature_set == "all384")]
    ceiling = full.groupby("model").roc_auc.mean().idxmax()
    for q, fs in QUANTUM.items():
        for c in CLASSICAL:
            same += compare(df, (q, "subsample", fs), (c, "subsample", fs), ratio)
        floor += compare(df, (q, "subsample", fs), ("logreg", "subsample", "length"), ratio)
        ceil += compare(df, (q, "subsample", fs), (ceiling, "full", "all384"), ratio)
    ceil_all, top = [], None
    if (df.regime == "all_rows").any():
        top = df[df.regime == "all_rows"].groupby("model").roc_auc.mean().idxmax()
        for q, fs in QUANTUM.items():
            ceil_all += compare(df, (q, "subsample", fs), (top, "all_rows", "all384"), ratio)
    return (add_holm(pd.DataFrame(same), ["metric"]), add_holm(pd.DataFrame(floor), ["metric"]),
            add_holm(pd.DataFrame(ceil), ["metric"]), ceiling,
            add_holm(pd.DataFrame(ceil_all), ["metric"]) if ceil_all else None, top)


def arm_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(["model", "regime", "feature_set"])
    out = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_std")).join(g.size().rename("n_folds"))
    return out.reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--view", default="binary")
    a = p.parse_args()
    d = ROOT / "results" / "mhtext" / "phase4" / a.view
    df = pd.read_csv(d / "compare_fold_metrics.csv")
    sw = pd.read_csv(d / "sweep_fold_metrics.csv")
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    ratio = float(run["meta"]["test_train_ratio"])
    cdir = d.parent / f"{a.view}_ceiling"                        # scripts/mhtext_binary_ceiling.py, if run
    crun = None
    if (cdir / "fold_metrics.csv").exists():
        df = pd.concat([df, pd.read_csv(cdir / "fold_metrics.csv")], ignore_index=True)
        crun = json.loads((cdir / "run.json").read_text(encoding="utf-8"))
    assert df.data_source.str.startswith("REAL:MHText").all() and sw.data_source.str.startswith("REAL:MHText").all()

    cols = HEAD + ["pr_auc", "accuracy", "f1", "extra_roc_auc", "extra_balanced_accuracy", "tuning_seconds"]
    summary = arm_table(df, cols)
    same, floor, ceil, ceiling, ceil_all, top = tests(df, ratio)
    summary.to_csv(d / "summary.csv", index=False)
    same.to_csv(d / "tests_same_rows_features.csv", index=False)
    floor.to_csv(d / "tests_vs_length_floor.csv", index=False)
    (d / "tests_vs_full_ceiling.csv").unlink(missing_ok=True)      # old name: it was never a full-data ceiling
    ceil.to_csv(d / "tests_vs_dev_rows_ceiling.csv", index=False)
    if ceil_all is not None:
        ceil_all.to_csv(d / "tests_vs_all_rows_ceiling.csv", index=False)
    sweep = sw.groupby(["model", "feature_method", "encoding", "n_qubits"]).agg(
        roc_auc=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"), extra_roc_auc=("extra_roc_auc", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"), n_folds=("roc_auc", "size")).reset_index()
    sweep.to_csv(d / "sweep_summary.csv", index=False)

    # NSCH numbers, read from its existing results (never recomputed)
    ns = pd.read_csv(NSCH_EXP2 / "summary.csv")
    nt = pd.read_csv(NSCH_EXP2 / "tests_same_rows_features.csv")

    def ns_auc(m, r, fs):
        row = ns[(ns.model == m) & (ns.regime == r) & (ns.feature_set == fs)]
        return float(row.roc_auc_mean.iloc[0]) if len(row) else float("nan")

    def ns_p(q, c):
        row = nt[(nt.quantum.str.startswith(q + "/")) & (nt.classical.str.startswith(c + "/")) & (nt.metric == "roc_auc")]
        return float(row.p_nb_t_holm.iloc[0]) if len(row) else float("nan")

    s = summary.set_index(["model", "regime", "feature_set"])
    L = [f"# Phase 4 — quantum models on Mental Health Text ({a.view} view)", ""]
    L += ["Source: REAL:MHText. Positive class Suicidal, negative Depression. "
          f"Development rows {run['meta']['dev_rows']:,} ({run['meta']['dev_positive']:,} positive); "
          f"quarantined block {run['meta']['quarantine_rows']:,} rows ({run['meta']['quarantine_positive']:,} positive), "
          "scored by every fold's model and never used for fitting or tuning. Locked hold-out not used.", "",
          "Method held fixed from NSCH: 5x3 repeated stratified K-fold; quantum and the same-rows classical arms train "
          "on the same 1,500-row subsample per fold (1,200 inner-train / 300 inner-validation), same PCA inputs, same "
          "per-fold seed; NB-t corrected resampled t-test, Holm within metric (primary). Input: frozen MiniLM sentence "
          "encoder (BERT family; sentence-transformers/all-MiniLM-L6-v2), 384 dims, reduced inside each fold.", "",
          f"Wall time: {run['wall_seconds_this_launch'] / 3600:.2f} h for the last launch "
          "(whole run 11:15-13:37 = 2.37 h including a restart; 655 tasks, 6 workers).", ""]

    L += ["## Headline: quantum vs classical on the same rows and inputs", "",
          "| Model | Input | ROC AUC | Bal. acc. | Recall | Accuracy | Quarantine AUC | best classical, same input | NB-t Holm p (AUC) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for q, fs in QUANTUM.items():
        r = s.loc[(q, "subsample", fs)]
        t = same[(same.quantum == f"{q}/subsample/{fs}") & (same.metric == "roc_auc")]
        best = t.loc[t.classical_mean.idxmax()]
        L.append(f"| {NAMES[q]} | {INPUTS[fs]} | {f3(r.roc_auc_mean)} ± {f3(r.roc_auc_std)} | {f3(r.balanced_accuracy_mean)} | "
                 f"{f3(r.recall_mean)} | {f3(r.accuracy_mean)} | {f3(r.extra_roc_auc_mean)} | "
                 f"{NAMES[best.classical.split('/')[0]]} {f3(best.classical_mean)} | {f3(best.p_nb_t_holm)} |")
    L += ["", "Every same-rows pair (ROC AUC; Holm over all 12 pairs within the metric):", "",
          "| Quantum | Classical | Quantum AUC | Classical AUC | Diff [95% CI] | Quantum wins /15 | NB-t Holm p | Significant |",
          "|---|---|---|---|---|---|---|---|"]
    for _, t in same[same.metric == "roc_auc"].iterrows():
        L.append(f"| {t.quantum} | {t.classical} | {f3(t.quantum_mean)} | {f3(t.classical_mean)} | "
                 f"{t.mean_diff:+.3f} [{t.ci95_low:+.3f}, {t.ci95_high:+.3f}] | {t.quantum_wins} | {f3(t.p_nb_t_holm)} | "
                 f"{'yes' if t.significant else 'no'} |")

    lf = s.loc[("logreg", "subsample", "length")]
    L += ["", "## Floor and ceiling (same folds)", "",
          f"- **Length-only floor** (logistic regression on log words + log characters, same 1,200 rows): ROC AUC "
          f"{f3(lf.roc_auc_mean)} ± {f3(lf.roc_auc_std)}, quarantine {f3(lf.extra_roc_auc_mean)}.",
          f"- **Classical ceiling, {int(df[df.regime == 'full'].n_train_rows.mean()):,} rows** (all 384 dims, every "
          f"development row of the fold's training part): best {NAMES[ceiling]} ROC AUC "
          f"{f3(s.loc[(ceiling, 'full', 'all384')].roc_auc_mean)}. This is NOT a full-data ceiling. The cap is not the "
          "SVM's n² cost: Phase 1 drew a 15,000-row stratified subsample of the 38,859 deduplicated ID-bearing rows "
          "(so that the frozen-encoder feature extraction and the 5x3 grids stay tractable on CPU) and locked 20 % of it "
          "as the hold-out, which leaves 6,306 Depression/Suicidal development rows in this view.",
          ]
    if crun is not None:
        L.append(f"- **Classical ceiling, all rows** ({NAMES[top]}, all 384 dims; the fold's development training rows "
                 f"plus the {crun['extra_rows']:,} deduplicated Depression/Suicidal ID rows outside the Phase 1 subsample, "
                 f"≈ {int(crun['mean_training_rows_per_fold']):,} training rows per fold; same validation folds and "
                 f"quarantine): ROC AUC {f3(s.loc[(top, 'all_rows', 'all384')].roc_auc_mean)} ± "
                 f"{f3(s.loc[(top, 'all_rows', 'all384')].roc_auc_std)}, quarantine "
                 f"{f3(s.loc[(top, 'all_rows', 'all384')].extra_roc_auc_mean)}. Extra rows were checked against every "
                 f"processed row (exact and TF-IDF cosine ≥ 0.90): {crun['dropped_exact_match_to_table']} exact and "
                 f"{crun['dropped_near_dup_to_table']} near duplicates dropped. This is the true binary ceiling "
                 "(SVM not re-run at this size).")
    L += ["", "| Quantum | AUC | vs length floor: diff, Holm p | vs ceiling (dev rows): diff, Holm p |"
          + (" vs ceiling (all rows): diff, Holm p |" if crun else ""), "|---|---|---|---|" + ("---|" if crun else "")]
    for q, fs in QUANTUM.items():
        a1 = floor[(floor.quantum == f"{q}/subsample/{fs}") & (floor.metric == "roc_auc")].iloc[0]
        a2 = ceil[(ceil.quantum == f"{q}/subsample/{fs}") & (ceil.metric == "roc_auc")].iloc[0]
        row = (f"| {NAMES[q]} | {f3(a1.quantum_mean)} | {a1.mean_diff:+.3f}, p {f3(a1.p_nb_t_holm)} | "
               f"{a2.mean_diff:+.3f}, p {f3(a2.p_nb_t_holm)} |")
        if ceil_all is not None:
            a3 = ceil_all[(ceil_all.quantum == f"{q}/subsample/{fs}") & (ceil_all.metric == "roc_auc")].iloc[0]
            row += f" {a3.mean_diff:+.3f}, p {f3(a3.p_nb_t_holm)} |"
        L.append(row)

    L += ["", "## All arms", "",
          "| Model | Regime | Input | ROC AUC | Bal. acc. | Recall | PR AUC | Accuracy | F1 | Quarantine AUC | Tuning s |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.sort_values(["regime", "feature_set", "roc_auc_mean"], ascending=[False, True, False]).iterrows():
        L.append(f"| {NAMES.get(r.model, r.model)} | {r.regime} | {INPUTS[r.feature_set]} | {f3(r.roc_auc_mean)} ± "
                 f"{f3(r.roc_auc_std)} | {f3(r.balanced_accuracy_mean)} | {f3(r.recall_mean)} | {f3(r.pr_auc_mean)} | "
                 f"{f3(r.accuracy_mean)} | {f3(r.f1_mean)} | {f3(r.extra_roc_auc_mean)} | {r.tuning_seconds_mean:.0f} |")

    L += ["", "## Beside NSCH (numbers read from " + NSCH_EXP2.relative_to(ROOT).as_posix() + ", not recomputed)", "",
          "| | NSCH 2024 (survey, 95 columns) | Mental Health Text binary (MiniLM) |", "|---|---|---|"]
    for q, fs_n, fs_m in [("qsvm", "pca8", "pca8"), ("vqc", "pca4", "pca4")]:
        L.append(f"| {NAMES[q]} AUC | {f3(ns_auc(q, 'subsample', fs_n))} | {f3(s.loc[(q, 'subsample', fs_m)].roc_auc_mean)} |")
        L.append(f"| Logistic regression, same rows + input | {f3(ns_auc('logreg', 'subsample', fs_n))} | "
                 f"{f3(s.loc[('logreg', 'subsample', fs_m)].roc_auc_mean)} |")
        mp = same[(same.quantum == f"{q}/subsample/{fs_m}") & (same.classical == f"logreg/subsample/{fs_m}") &
                  (same.metric == "roc_auc")].p_nb_t_holm.iloc[0]
        L.append(f"| {q.upper()} vs logistic regression, NB-t Holm p | {f3(ns_p(q, 'logreg'))} | {f3(mp)} |")
    L.append(f"| Hybrid QNN AUC (all inputs) | {f3(ns_auc('hybrid', 'subsample', 'all95'))} | "
             f"{f3(s.loc[('hybrid', 'subsample', 'all384')].roc_auc_mean)} |")
    ns_full = f3(ns[(ns.regime == "full") & (ns.feature_set == "all95")].roc_auc_mean.max())
    L.append(f"| Classical ceiling, development rows of the fold | {ns_full} (every NSCH development row) | "
             f"{f3(s.loc[(ceiling, 'full', 'all384')].roc_auc_mean)} (5,044 rows) |")
    if crun is not None:
        L.append(f"| Classical ceiling, all available rows | {ns_full} (same: NSCH had no subsample cap) | "
                 f"{f3(s.loc[(top, 'all_rows', 'all384')].roc_auc_mean)} (≈ {int(crun['mean_training_rows_per_fold']):,} rows) |")
    L.append("| Length-only floor | n/a | " + f3(lf.roc_auc_mean) + " |")
    L += ["", "NSCH Holm families had 9 same-rows pairs (3 classical models); this view has 12 (SVM added), so its "
          "Holm correction is slightly stricter.", "",
          "## Deliberately not run: four-class and three-class quantum views", "",
          "The multiclass quantum views were deliberately not run (decision recorded after the binary result). The "
          "binary Depression vs Suicidal view was chosen as the hard, length-free, tangled boundary - the fairest test "
          "quantum models could get on this dataset - and the QSVM tied classical there. A multiclass quantum run would "
          "be one-vs-rest, repeating that binary result four times and averaging it, and it would require extending the "
          "VQC and hybrid QNN beyond the NSCH implementations (three-class VQC, binary-only hybrid), which breaks the "
          "'method held fixed' design that makes the NSCH comparison valid. Multiclass results for this dataset are the "
          "Phase 3 classical and sequence baselines (results/mhtext/phase3/).", ""]

    L += ["## Encoding x qubit sweep (exp1 method; 1,200 rows; 2 repeats x 5 folds; ROC AUC, quarantine in brackets)", "",
          "| Model | Features | Encoding | 4 qubits | 6 qubits | 8 qubits |", "|---|---|---|---|---|---|"]
    for (m, fm, enc), g in sweep.groupby(["model", "feature_method", "encoding"]):
        cells = [f"{f3(r.roc_auc)} ({f3(r.extra_roc_auc)})" for _, r in g.sort_values("n_qubits").iterrows()]
        L.append(f"| {SWEEP_NAMES[m]} | {fm} | {enc} | " + " | ".join(cells) + " |")
    L += ["", "Every row reports ROC AUC beside balanced accuracy and recall in the CSVs; accuracy is never reported alone.",
          "Quantum results are simulated (statevector), not hardware; quantum models are not a privacy mechanism.", ""]
    (d / "phase4.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {d / 'phase4.md'}")


if __name__ == "__main__":
    main()
