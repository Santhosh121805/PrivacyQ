"""Audit of every significance decision: Wilcoxon-Holm (previous primary) vs Nadeau-Bengio-Holm (new primary).

Reads the regenerated test tables of every experiment (each now carries both ``p_nb_t_holm`` and
``wilcoxon_p_holm``, Holm-adjusted within the same family), plus the two families that only exist in
RESULTS.md (Experiment 2 without RandomForest; the Experiment 3 clipping sensitivity), and lists every
comparison with both decisions at alpha = 0.05. Writes results/significance/audit.csv and changed.md.

Usage: python scripts/significance_audit.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.m8_eval.stats import add_holm, corrected_resampled_ttest, test_train_ratio, wilcoxon_p  # noqa: E402
from src.m10_exp.common import load_config  # noqa: E402

R = ROOT / "results"
ALPHA = 0.05

# (experiment, file, identifying columns, effect column, [(primary col, sensitivity col, family label)])
SOURCES = [
    ("Exp 1 encodings", "exp1-nsch2024-encodings-final-20261108-160825/encoding_tests.csv",
     ["model", "feature_method", "n_qubits", "best", "vs"], "mean_diff",
     [("p_nb_t_holm", "wilcoxon_p_holm", "whole table"), ("p_nb_t_holm_in_config", "wilcoxon_p_holm_in_config", "within config")]),
    ("Exp 1 MI vs PCA", "exp1-nsch2024-encodings-final-20261108-160825/mi_vs_pca_tests.csv",
     ["model", "encoding", "n_qubits"], "mean_diff", [("p_nb_t_holm", "wilcoxon_p_holm", "whole table")]),
    ("Exp 1 encodings incl. basis", "exp1-nsch2024-encodings-with-basis-20261112/encoding_tests.csv",
     ["model", "feature_method", "n_qubits", "best", "vs"], "mean_diff",
     [("p_nb_t_holm", "wilcoxon_p_holm", "whole table"), ("p_nb_t_holm_in_config", "wilcoxon_p_holm_in_config", "within config")]),
    ("Exp 1 MI vs PCA incl. basis", "exp1-nsch2024-encodings-with-basis-20261112/mi_vs_pca_tests.csv",
     ["model", "encoding", "n_qubits"], "mean_diff", [("p_nb_t_holm", "wilcoxon_p_holm", "whole table")]),
    ("Exp 2 same rows/features (run report, incl. RF)", "exp2-nsch2024-quantum-vs-classical-20261108-165426/tests_same_rows_features.csv",
     ["quantum", "classical", "metric"], "mean_diff", [("p_nb_t_holm", "wilcoxon_p_holm", "within metric")]),
    ("Exp 2 vs full-data ceiling", "exp2-nsch2024-quantum-vs-classical-20261108-165426/tests_vs_full_data_ceiling.csv",
     ["quantum", "classical", "metric"], "mean_diff", [("p_nb_t_holm", "wilcoxon_p_holm", "within metric")]),
    ("Exp 3 private vs non-private", "exp3-nsch2024-privacy-utility-20261108-165618/tests_private_vs_nonprivate.csv",
     ["arm", "epsilon", "metric"], "change", [("p_nb_t_holm", "wilcoxon_p_holm", "arm x metric")]),
    ("Exp 3 VQC vs LogReg", "exp3-nsch2024-privacy-utility-20261108-165618/tests_quantum_vs_classical.csv",
     ["quantum", "classical", "epsilon"], "auc_diff",
     [("p_nb_t_holm", "wilcoxon_p_holm", "AUC difference"), ("drop_diff_p_nb_t_holm", "drop_diff_wilcoxon_p_holm", "drop difference")]),
    ("Exp 3 sensitivity: private vs non-private", "exp3-sensitivity-logreg-clipping-20261108-170152/tests_private_vs_nonprivate.csv",
     ["arm", "epsilon", "metric"], "change", [("p_nb_t_holm", "wilcoxon_p_holm", "arm x metric")]),
    ("Exp 4 federated vs centralised", "exp4-nsch2024-federated-dp-20261108-170320/tests_federated_vs_centralised.csv",
     ["arm", "epsilon", "partition", "n_clients", "vs"], "auc_change", [("p_nb_t_holm", "wilcoxon_p_holm", "arm x reference")]),
    ("Exp 7 federated vs centralised", "exp7-nsch2024-federated-cost-20261112-123743/tests_federated_vs_centralised.csv",
     ["arm", "epsilon", "partition", "n_clients", "vs"], "auc_change", [("p_nb_t_holm", "wilcoxon_p_holm", "arm x reference")]),
    ("Exp 7b federated vs centralised", "exp7b-nsch2024-federated-secure-aggregation-20261112-153004/tests_federated_vs_centralised.csv",
     ["arm", "epsilon", "partition", "n_clients", "vs"], "auc_change", [("p_nb_t_holm", "wilcoxon_p_holm", "arm x reference")]),
    ("Exp 2 SVM baseline", "exp2-nsch2024-svm-baseline-20261112-121554/comparison.csv", ["a", "b", "metric"], "diff",
     [("p_nb_t_holm", "wilcoxon_p_holm", "within metric")]),
    ("Exp 2 COBYLA", "exp2-nsch2024-vqc-cobyla-20261112-121718/comparison.csv", ["a", "b", "metric"], "diff",
     [("p_nb_t_holm", "wilcoxon_p_holm", "within metric")]),
    ("Exp 8 noise (angle, 8 qubits)", "exp8-nsch2024-hardware-noise-20261112-125925/comparison.csv", ["a", "b", "metric"], "diff",
     [("p_nb_t_holm", "wilcoxon_p_holm", "within metric")]),
    ("Exp 8b noise (angle vs ZZ, 4 qubits)", "exp8b-nsch2024-hardware-noise-zz-vs-angle-4q-20261112-144158/comparison.csv",
     ["a", "b", "metric"], "diff", [("p_nb_t_holm", "wilcoxon_p_holm", "within metric")]),
    ("Exp 9 multi-class (QSVM vs classical)", "exp9-nsch2024-multiclass-20261112-140242/paired_tests.csv", ["b", "metric"], "diff",
     [("p_nb_t_holm", "wilcoxon_p_holm", "within metric")]),
    ("Exp 9b multi-class (VQC vs classical)", "exp9b-nsch2024-multiclass-vqc-20261112-140937/paired_tests.csv", ["b", "metric"], "diff",
     [("p_nb_t_holm", "wilcoxon_p_holm", "within metric")]),
    ("Exp 10 UMAP / autoencoder", "exp10-nsch2024-umap-autoencoder-20261112-143551/comparison.csv", ["a", "b", "metric"], "diff",
     [("p_nb_t_holm", "wilcoxon_p_holm", "within metric")]),
]


def results_md_families() -> list[pd.DataFrame]:
    """Families that exist only in RESULTS.md (built by src/m10_exp/final_report.py)."""
    runs = {k: R.parent / v for k, v in load_config(ROOT / "configs/final_report.yaml")["runs"].items()}
    t = pd.read_csv(runs["exp2"] / "tests_same_rows_features.csv")
    t = add_holm(t[~t.classical.str.startswith("random_forest")], ["metric"])
    t = t.assign(experiment="RESULTS.md Table 4 (Exp 2 without RF)", family="within metric", effect=t.mean_diff,
                 comparison=t.quantum + " vs " + t.classical + " · " + t.metric)
    main, sen = pd.read_csv(runs["exp3"] / "fold_metrics.csv"), pd.read_csv(runs["exp3_sensitivity"] / "fold_metrics.csv")
    for f in (main, sen):
        f["eps"] = ["inf" if str(e) in ("inf", "nan") else f"{float(e):g}" for e in f.target_epsilon]
    rows, ratio = [], test_train_ratio(main)
    for e in ["0.5", "1", "2", "4", "8", "inf"]:
        a = main[(main.arm == "vqc_sub") & (main.eps == e)].set_index(["repeat", "fold"]).roc_auc.sort_index()
        b = sen[(sen.arm == "logreg_sub_clip10") & (sen.eps == e)].set_index(["repeat", "fold"]).roc_auc.sort_index()
        idx = a.index.intersection(b.index)
        x, y = a[idx].to_numpy(), b[idx].to_numpy()
        rows.append({"comparison": f"VQC vs LogReg clip 10 · ε {e}", "effect": (x - y).mean(),
                     "p_nb_t": corrected_resampled_ttest(x, y, ratio)["p"], "wilcoxon_p": wilcoxon_p(x, y)})
    s = add_holm(pd.DataFrame(rows), None).assign(experiment="RESULTS.md Table 7 (Exp 3 clipping sensitivity)",
                                                  family="whole table")
    cols = ["experiment", "family", "comparison", "effect", "p_nb_t", "p_nb_t_holm", "wilcoxon_p", "wilcoxon_p_holm"]
    return [t[cols], s[cols]]


def main():
    frames = []
    for exp, path, ids, eff, tests in SOURCES:
        df = pd.read_csv(R / path)
        label = df[ids].astype(str).agg(" · ".join, axis=1)
        for prim, sens, fam in tests:
            ok = df[prim].notna()
            raw_p = prim.replace("_holm_in_config", "").replace("_holm", "")
            raw_w = sens.replace("_holm_in_config", "").replace("_holm", "")
            frames.append(pd.DataFrame({"experiment": exp, "family": fam, "comparison": label[ok],
                                        "effect": df.loc[ok, eff] if fam != "drop difference" else
                                        df.loc[ok, "quantum_auc_drop"] - df.loc[ok, "classical_auc_drop"],
                                        "p_nb_t": df.loc[ok, raw_p], "p_nb_t_holm": df.loc[ok, prim],
                                        "wilcoxon_p": df.loc[ok, raw_w], "wilcoxon_p_holm": df.loc[ok, sens]}))
    frames += results_md_families()
    audit = pd.concat(frames, ignore_index=True)
    audit["significant_wilcoxon_holm (old)"] = audit.wilcoxon_p_holm < ALPHA
    audit["significant_nb_t_holm (new primary)"] = audit.p_nb_t_holm < ALPHA
    audit["changed"] = audit["significant_wilcoxon_holm (old)"] != audit["significant_nb_t_holm (new primary)"]
    out = R / "significance"
    out.mkdir(exist_ok=True)
    audit.to_csv(out / "audit.csv", index=False)
    ch = audit[audit.changed]
    by_exp = audit.groupby("experiment", sort=False).agg(comparisons=("changed", "size"), changed=("changed", "sum"),
                                                         sig_old=("significant_wilcoxon_holm (old)", "sum"),
                                                         sig_new=("significant_nb_t_holm (new primary)", "sum"))
    lost = int((ch["significant_wilcoxon_holm (old)"] & ~ch["significant_nb_t_holm (new primary)"]).sum())
    gained = int((~ch["significant_wilcoxon_holm (old)"] & ch["significant_nb_t_holm (new primary)"]).sum())
    n_sig_new = int(audit["significant_nb_t_holm (new primary)"].sum())
    lines = ["# Significance audit: what survives a correct test", "",
             f"The project makes {len(audit)} paired comparisons (the Experiment 1 tables with and without basis encoding, "
             "and the RESULTS.md re-adjustments, are counted separately), each over only 10–15 cross-validation folds. "
             "With so few folds most differences between configurations are not statistically resolvable: they are "
             "smaller than the fold-to-fold variation once the overlap between CV training sets is taken into account. "
             "The Wilcoxon test behind the earlier report labels ignores that overlap and so overstated what could be "
             "concluded; on 10 folds it also has a p-value floor (about 0.002) that decides some verdicts by the fold "
             f"count rather than the data. Under the Nadeau–Bengio corrected t with Holm adjustment, {n_sig_new} "
             "comparisons are significant: that subset is robust to a correct test and a multiplicity adjustment, and it "
             "is what the reports now state.", "",
             f"{len(ch)} of {len(audit)} decisions differ between the two tests (same Holm families, alpha = {ALPHA}): "
             f"{lost} that Wilcoxon called significant are not resolvable under the corrected test, and {gained} that "
             "Wilcoxon could not resolve (mostly because of its floor) are. Superseded reports are kept as "
             "`*.superseded-sig-20261112.*`.", "",
             "## Load-bearing comparisons", "",
             "The project's main conclusion (no quantum advantage on identical rows and features) rests on these "
             "comparisons (RESULTS.md Table 4, 15 folds). Each holds under both tests.", "",
             "| Comparison | Effect | NB-t p | NB-t Holm p (primary) | Wilcoxon p | Wilcoxon Holm p (sensitivity) | Conclusion |",
             "|---|---|---|---|---|---|---|"]
    t4 = audit[audit.experiment.str.startswith("RESULTS.md Table 4")]
    load = [("qsvm/subsample/pca8 vs logreg/subsample/pca8 · roc_auc", "QSVM vs LogReg (ROC-AUC)"),
            ("vqc/subsample/pca4 vs logreg/subsample/pca4 · roc_auc", "VQC vs LogReg (ROC-AUC)"),
            ("hybrid/subsample/all95 vs logreg/subsample/all95 · roc_auc", "Hybrid vs LogReg (ROC-AUC)"),
            ("hybrid/subsample/all95 vs logreg/subsample/all95 · balanced_accuracy", "Hybrid vs LogReg (balanced accuracy)")]
    for comp, name in load:
        r = t4[t4.comparison == comp].iloc[0]
        held = r["significant_wilcoxon_holm (old)"] == r["significant_nb_t_holm (new primary)"]
        verdict = "significant" if r["significant_nb_t_holm (new primary)"] else "not significant (tie)"
        lines.append(f"| {name} | {r.effect:+.3f} | {r.p_nb_t:.3g} | {r.p_nb_t_holm:.3g} | {r.wilcoxon_p:.3g} | "
                     f"{r.wilcoxon_p_holm:.3g} | {verdict}; {'same under both tests' if held else 'DIFFERS between tests'} |")
    lines += ["", "## Decisions by experiment", "",
              "| Experiment | Comparisons | Significant (Wilcoxon-Holm) | Significant (NB-t-Holm, primary) | Differ |",
              "|---|---|---|---|---|"]
    for e, r in by_exp.iterrows():
        lines.append(f"| {e} | {r.comparisons} | {r.sig_old} | {r.sig_new} | {r.changed} |")
    lines += ["", "## Every comparison whose decision differs (old and new p-values side by side)", "",
              "| Experiment | Family | Comparison | Effect | NB-t p | NB-t Holm p (new primary) | Wilcoxon p | "
              "Wilcoxon Holm p (old label) | Old → new |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in ch.iterrows():
        arrow = "significant → not resolvable" if r["significant_wilcoxon_holm (old)"] else "not resolvable → significant"
        lines.append(f"| {r.experiment} | {r.family} | {r.comparison} | {r.effect:+.3f} | {r.p_nb_t:.3g} | "
                     f"{r.p_nb_t_holm:.3g} | {r.wilcoxon_p:.3g} | {r.wilcoxon_p_holm:.3g} | {arrow} |")
    (out / "changed.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:16]))
    print(f"changed rows: {len(ch)}")


if __name__ == "__main__":
    main()
