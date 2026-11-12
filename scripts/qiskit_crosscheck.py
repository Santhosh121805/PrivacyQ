"""Item 12: cross-check the Experiment 2 VQC between PennyLane and an independent Qiskit implementation.

For each of the 15 Experiment 2 folds: rebuild the fold's data exactly as Experiment 2 does
(``src.m10_exp.exp2_compare.build_tasks``: subsample, inner split, preprocessing and PCA-4 fitted
on inner-train), refit the PennyLane VQC with the hyperparameters Experiment 2 selected for that fold
(same seed, same early stopping on inner-val), check that it reproduces Experiment 2's ROC-AUC,
then score the outer-validation rows with PennyLane and with Qiskit (``src/m5_models/qiskit_vqc.py``)
and compare scores and metrics. Writes results/qiskit_crosscheck/.

Usage: python scripts/qiskit_crosscheck.py
"""

import ast
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.m8_eval.evaluation import binary_metrics  # noqa: E402
from src.m10_exp.common import (  # noqa: E402
    cv_folds, environment_record, holdout_split, limit_worker_threads, load_config, load_dataset, write_json,
)
from src.m10_exp.exp2_compare import build_tasks, feature_matrices  # noqa: E402

EXP2_CONFIG = ROOT / "configs/exp2_nsch2024.yaml"


def check(task: dict, best_params: dict, exp2_auc: float, exp2_threshold: float) -> dict:
    import torch

    from src.m5_models.qiskit_vqc import qiskit_decision_scores
    from src.m5_models.quantum import VQCClassifier

    torch.set_num_threads(1)
    seed = task["seed"]
    A, B, C = feature_matrices(task["feature_spec"], task["X_in"], task["y_in"], [task["X_iv"], task["X_ov"]], seed)
    fixed = {k: v for k, v in task["quantum"]["vqc"].items() if k not in ("feature_set", "grid")}
    model = VQCClassifier(seed=seed, **fixed, **best_params).fit(A, task["y_in"], eval_set=(B, task["y_iv"]))
    t0 = time.perf_counter()
    pl = model.decision_scores(C)
    pl_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    qk = qiskit_decision_scores(model, C)
    qk_s = time.perf_counter() - t0
    y = task["y_ov"]
    m_pl, m_qk = binary_metrics(y, pl, threshold=exp2_threshold), binary_metrics(y, qk, threshold=exp2_threshold)
    return {"repeat": task["repeat"], "fold": task["fold"], "n_val": len(y), "best_params": str(best_params),
            "exp2_roc_auc": exp2_auc, "pennylane_roc_auc": roc_auc_score(y, pl), "qiskit_roc_auc": roc_auc_score(y, qk),
            "max_abs_score_diff": float(np.max(np.abs(pl - qk))),
            "pennylane_balanced_accuracy": m_pl["balanced_accuracy"], "qiskit_balanced_accuracy": m_qk["balanced_accuracy"],
            "pennylane_recall": m_pl["recall"], "qiskit_recall": m_qk["recall"],
            "same_predictions": bool(np.array_equal(pl >= exp2_threshold, qk >= exp2_threshold)),
            "pennylane_seconds": pl_s, "qiskit_seconds": qk_s}


def main():
    limit_worker_threads()
    config = load_config(EXP2_CONFIG)
    config["experiment"]["classical_models"] = []
    config["experiment"]["quantum"] = {"vqc": config["experiment"]["quantum"]["vqc"]}
    config["experiment"]["regimes"] = ["subsample"]
    seed = int(config["project"]["seed"])
    loader, frame = load_dataset(config)
    labels = frame[loader.label_column].to_numpy(dtype=int)
    dev, _ = holdout_split(labels, float(config["split"]["holdout_fraction"]), seed)
    folds = cv_folds(dev, labels, int(config["cv"]["n_splits"]), int(config["cv"]["n_repeats"]), seed)
    exp2_dir = ROOT / load_config(ROOT / "configs/final_report.yaml")["runs"]["exp2"]
    exp2 = pd.read_csv(exp2_dir / "fold_metrics.csv")
    exp2 = exp2[(exp2.model == "vqc") & (exp2.regime == "subsample")].set_index(["repeat", "fold"])
    tasks = [t for t in build_tasks(config, loader, frame, folds, labels) if t["models"] == ["vqc"]]
    jobs = [(t, ast.literal_eval(exp2.loc[(t["repeat"], t["fold"]), "best_params"]),
             float(exp2.loc[(t["repeat"], t["fold"]), "roc_auc"]), float(exp2.loc[(t["repeat"], t["fold"]), "threshold"]))
            for t in tasks]
    rows = Parallel(n_jobs=5)(delayed(check)(*j) for j in jobs)
    out = pd.DataFrame(rows).sort_values(["repeat", "fold"])
    d = ROOT / "results/qiskit_crosscheck"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "crosscheck.csv", index=False)
    write_json(d / "environment.json", environment_record(config))
    repro = (out.pennylane_roc_auc - out.exp2_roc_auc).abs().max()
    lines = ["# Qiskit cross-check of the Experiment 2 VQC", "",
             f"{len(out)} folds; VQC re-uploading, 4 qubits, PCA-4, 1,500-row subsample; outer-validation rows "
             f"{out.n_val.min():,}–{out.n_val.max():,} per fold. PennyLane `default.qubit` vs Qiskit `Statevector` "
             "(independent implementation, `src/m5_models/qiskit_vqc.py`), same trained parameters.", "",
             f"- PennyLane refit reproduces Experiment 2 ROC-AUC: max |difference| {repro:.3g} over {len(out)} folds.",
             f"- Max |PennyLane score − Qiskit score| over all {int(out.n_val.sum()):,} validation rows: {out.max_abs_score_diff.max():.3g}.",
             f"- ROC-AUC PennyLane vs Qiskit: max |difference| {(out.pennylane_roc_auc - out.qiskit_roc_auc).abs().max():.3g}; "
             f"balanced accuracy max |difference| {(out.pennylane_balanced_accuracy - out.qiskit_balanced_accuracy).abs().max():.3g}; "
             f"identical thresholded predictions in {int(out.same_predictions.sum())}/{len(out)} folds.",
             f"- Scoring time per fold (one process): PennyLane {out.pennylane_seconds.median():.2f} s, "
             f"Qiskit {out.qiskit_seconds.median():.2f} s (median).", "",
             "| Repeat | Fold | Exp 2 AUC | PennyLane AUC | Qiskit AUC | Max abs. score diff | Same predictions |",
             "|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {r.repeat} | {r.fold} | {r.exp2_roc_auc:.4f} | {r.pennylane_roc_auc:.4f} | {r.qiskit_roc_auc:.4f} | "
                     f"{r.max_abs_score_diff:.2e} | {r.same_predictions} |")
    (d / "crosscheck.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
