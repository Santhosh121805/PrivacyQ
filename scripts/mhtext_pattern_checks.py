"""Phase 7 checks of two candidate cross-dataset patterns, from STORED per-fold results only (no model is run).

P3  qubit count: paired NB-t test of 8 vs 4 qubits for every (model, feature method, encoding) cell of the
    encoding x qubit sweep (exp1 method), Holm over the whole table, separately per dataset.
P4  PCA vs mutual-information selection: exp1_report.method_tests (the NSCH function) on the text sweep, with
    the same Holm families NSCH used (non-basis encodings in one run, basis in another). NSCH's own stored
    mi_vs_pca_tests.csv files are read, not recomputed.

python scripts/mhtext_pattern_checks.py  -> results/mhtext/phase7/pattern_checks/
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.m8_eval.stats import add_holm  # noqa: E402
from src.m10_exp.exp1_report import method_tests, paired  # noqa: E402

NSCH_SWEEP = ROOT / "results/exp1-nsch2024-encodings-with-basis-20261112/fold_metrics.csv"
NSCH_MI_PCA = [ROOT / "results/exp1-nsch2024-encodings-final-20261108-160825/mi_vs_pca_tests.csv",
               ROOT / "results/exp1-nsch2024-basis-encoding-20261112-121510/mi_vs_pca_tests.csv"]
TEXT_SWEEP = ROOT / "results/mhtext/phase4/binary/sweep_fold_metrics.csv"
OUT = ROOT / "results/mhtext/phase7/pattern_checks"


def qubit_tests(res: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for (model, method, enc), g in res.groupby(["model", "feature_method", "encoding"]):
        if not {4, 8} <= set(g.n_qubits):
            continue
        t = paired(res, {"model": model, "feature_method": method, "encoding": enc}, "n_qubits", 8, 4)
        m = g.groupby("n_qubits").roc_auc.mean()
        rows.append({"dataset": dataset, "model": model, "feature_method": method, "encoding": enc,
                     "auc_4": m[4], "auc_6": m.get(6), "auc_8": m[8],
                     "range_4_6_8": m.max() - m.min(), **t})
    return add_holm(pd.DataFrame(rows), None)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ns = pd.read_csv(NSCH_SWEEP)
    tx = pd.read_csv(TEXT_SWEEP)
    tx["n_fold_train"] = 6306 - tx["n_val"]          # binary-view development rows minus the fold's validation rows
    assert ns.data_source.str.startswith("REAL:NSCH").all() and tx.data_source.str.startswith("REAL:MHText").all()
    q = pd.concat([qubit_tests(ns, "NSCH 2024"), qubit_tests(tx, "Mental Health Text")])
    q.to_csv(OUT / "p3_qubits_8_vs_4.csv", index=False)
    m = pd.concat([method_tests(tx[tx.encoding != "basis"]).assign(family="non-basis"),
                   method_tests(tx[tx.encoding.isin(["basis", "none"]) & (tx.model != "logreg")]).assign(family="basis")])
    m.assign(dataset="Mental Health Text").to_csv(OUT / "p4_mi_vs_pca_text.csv", index=False)
    pd.concat([pd.read_csv(f).assign(family=fam) for f, fam in zip(NSCH_MI_PCA, ["non-basis", "basis"])]).assign(
        dataset="NSCH 2024").to_csv(OUT / "p4_mi_vs_pca_nsch_stored.csv", index=False)
    pd.concat([qubit_pairs(ns, "NSCH 2024"), qubit_pairs(tx, "Mental Health Text")]).to_csv(
        OUT / "p3_qubit_pairs.csv", index=False)
    nsch_reference()
    replication_summary()
    print(f"wrote {OUT}")


def qubit_pairs(res: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Paired NB-t for 6-4, 8-6 and 8-4 qubits in every quantum cell; Holm over the dataset's whole table."""
    rows = []
    q = res[res.model.isin(["qsvm", "vqc"])]
    for (model, method, enc), g in q.groupby(["model", "feature_method", "encoding"]):
        for a, b in ((6, 4), (8, 6), (8, 4)):
            t = paired(q, {"model": model, "feature_method": method, "encoding": enc}, "n_qubits", a, b)
            rows.append({"dataset": dataset, "model": model, "feature_method": method, "encoding": enc,
                         "contrast": f"{a}-{b}", **t})
    return add_holm(pd.DataFrame(rows), None)


NSCH_REF = {   # NSCH result files copied verbatim (numbers only) so the figures read committed files
    "exp2_summary": ROOT / "results/exp2-nsch2024-quantum-vs-classical-20261108-165426/summary.csv",
    "exp2_tests_same_rows_features": ROOT / "results/exp2-nsch2024-quantum-vs-classical-20261108-165426/tests_same_rows_features.csv",
    "exp3_summary": ROOT / "results/exp3-nsch2024-privacy-utility-20261108-165618/summary.csv",
    "exp4_summary": ROOT / "results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv",
    "reidentification_risk": ROOT / "results/reidentification/risk.csv",
}


def nsch_reference() -> None:
    d = OUT.parent / "nsch_reference"
    d.mkdir(parents=True, exist_ok=True)
    for name, src in NSCH_REF.items():
        df = pd.read_csv(src)
        df["source_file"] = src.relative_to(ROOT).as_posix()
        df.to_csv(d / f"nsch_{name}.csv", index=False)


def replication_summary() -> None:
    """One row per pattern x dataset: the estimate the replication table reports, with its CI and Holm p."""
    ref = OUT.parent / "nsch_reference"
    rows = []
    tt = pd.read_csv(ROOT / "results/mhtext/phase4/binary/tests_same_rows_features.csv")
    nt = pd.read_csv(ref / "nsch_exp2_tests_same_rows_features.csv")
    for pat, q, fs_t, fs_n in (("P1", "qsvm", "pca8", "pca8"), ("P2", "vqc", "pca4", "pca4")):
        for ds, t, fs in (("Mental Health Text", tt, fs_t), ("NSCH 2024", nt, fs_n)):
            r = t[(t.quantum == f"{q}/subsample/{fs}") & (t.classical == f"logreg/subsample/{fs}") & (t.metric == "roc_auc")].iloc[0]
            rows.append({"pattern": pat, "dataset": ds, "quantity": f"{q.upper()} minus LogReg, same rows, ROC AUC",
                         "estimate": r.mean_diff, "ci95_low": r.ci95_low, "ci95_high": r.ci95_high, "p_holm": r.p_nb_t_holm})
    p3 = pd.read_csv(OUT / "p3_qubits_8_vs_4.csv")
    for q, enc in (("qsvm", "angle"), ("vqc", "reupload")):
        for ds in ("Mental Health Text", "NSCH 2024"):
            r = p3[(p3.dataset == ds) & (p3.model == q) & (p3.feature_method == "pca") & (p3.encoding == enc)].iloc[0]
            rows.append({"pattern": "P3", "dataset": ds, "quantity": f"{q.upper()} {enc}: 8 minus 4 qubits, ROC AUC",
                         "estimate": r.mean_diff, "ci95_low": r.ci95_low, "ci95_high": r.ci95_high, "p_holm": r.p_nb_t_holm})
    rep = pd.read_csv(ROOT / "results/mhtext/phase6/tests_replication_clients.csv")
    for ds in ("Mental Health Text", "NSCH 2024"):
        for qty, lab in (("VQC: 10 clients minus 3", "VQC: 10 minus 3 clients, ROC AUC"),
                         ("LogReg: 10 clients minus 3", "LogReg: 10 minus 3 clients, ROC AUC")):
            r = rep[(rep.dataset == ds) & (rep.partition == "iid") & (rep.quantity == qty)].iloc[0]
            rows.append({"pattern": "P5", "dataset": ds, "quantity": lab, "estimate": r["mean"],
                         "ci95_low": r.ci95_low, "ci95_high": r.ci95_high, "p_holm": r.p_nb_t_holm})
    s_t = pd.read_csv(ROOT / "results/mhtext/phase4/binary/summary.csv").set_index(["model", "regime", "feature_set"])
    s_n = pd.read_csv(ref / "nsch_exp2_summary.csv").set_index(["model", "regime", "feature_set"])
    for ds, s, hy in (("Mental Health Text", s_t, "all384"), ("NSCH 2024", s_n, "all95")):
        others = {"qsvm": s.loc[("qsvm", "subsample", "pca8")].roc_auc_mean, "vqc": s.loc[("vqc", "subsample", "pca4")].roc_auc_mean}
        low = min(others, key=others.get)
        rows.append({"pattern": "P6", "dataset": ds, "quantity": f"Hybrid QNN minus lowest other quantum model ({low.upper()}), ROC AUC",
                     "estimate": s.loc[("hybrid", "subsample", hy)].roc_auc_mean - others[low],
                     "ci95_low": None, "ci95_high": None, "p_holm": None})
    for ds, f in (("Mental Health Text", OUT / "p4_mi_vs_pca_text.csv"), ("NSCH 2024", OUT / "p4_mi_vs_pca_nsch_stored.csv")):
        m = pd.read_csv(f)
        for lab, mask in (("cells where PCA is significantly better", (m.mean_diff < 0) & m.significant),
                          ("cells where MI is significantly better", (m.mean_diff > 0) & m.significant)):
            rows.append({"pattern": "P4", "dataset": ds, "quantity": lab, "estimate": int(mask.sum()),
                         "ci95_low": None, "ci95_high": None, "p_holm": None, "n_cells": len(m)})
    pd.DataFrame(rows).to_csv(OUT / "replication_summary.csv", index=False)


if __name__ == "__main__":
    main()
