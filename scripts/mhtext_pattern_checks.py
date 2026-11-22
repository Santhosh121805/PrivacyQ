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
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
