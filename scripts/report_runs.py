"""Short real runs whose terminal output is captured for the report.

    python scripts/report_runs.py core      --config configs/report_runs.yaml   # load, clean/de-identify, 1 model, score
    python scripts/report_runs.py baselines --config configs/report_runs.yaml   # the 4 classical baselines, imbalance metrics
    python scripts/report_runs.py handoff   --config configs/report_runs.yaml   # Module 1 -> Module 2 for one hold-out record

core / baselines train on fold r0f0 of the development rows and score its validation rows (the experiments' own
split and folds). handoff runs one locked hold-out record through the stored transformer: inference only, no metric.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

NAMES = {"logreg": "Logistic regression", "xgboost": "XGBoost", "mlp": "MLP", "svm": "SVM (RBF)"}


def _setup(cfg):
    from src.m10_exp.common import cv_folds, holdout_split, load_dataset

    seed = int(cfg["project"]["seed"])
    t0 = time.perf_counter()
    loader, frame = load_dataset(cfg)
    y = frame[loader.label_column].to_numpy(dtype=int)
    dev, hold = holdout_split(y, float(cfg["split"]["holdout_fraction"]), seed)
    fold = cv_folds(dev, y, int(cfg["cv"]["n_splits"]), int(cfg["cv"]["n_repeats"]), seed)[0]
    print(f"[load]    {cfg['data']['paths'][0]}  ({time.perf_counter() - t0:.1f} s)")
    print(f"          {len(frame):,} records aged 12-17; {y.sum():,} with current depression ({y.mean():.2%})")
    print(f"[split]   locked hold-out {len(hold):,} (not used) | development {len(dev):,}")
    print(f"          fold r0f0: train {len(fold['train']):,}, validation {len(fold['val']):,} "
          f"({y[fold['val']].sum()} positive)")
    return loader, frame, y, fold, seed


def _prepare(cfg, loader, frame, y, fold, seed):
    from sklearn.preprocessing import StandardScaler

    from src.m1_data.nsch import FEATURES
    from src.m2_preprocess.preprocessing import TabularPreprocessor
    from src.m10_exp.tuning import inner_split

    tr, iv = inner_split(fold["train"], y, float(cfg["tuning"]["inner_val_fraction"]), seed)
    t0 = time.perf_counter()
    prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
    X_tr = prep.fit_transform(frame.iloc[tr], loader.label_column).to_numpy()
    X_iv = prep.transform(frame.iloc[iv], loader.label_column).to_numpy()
    X_va = prep.transform(frame.iloc[fold["val"]], loader.label_column).to_numpy()
    sc = StandardScaler().fit(X_tr)
    miss = frame[list(FEATURES)].iloc[tr].isna().to_numpy().mean()
    print(f"[clean]   de-identify: dropped {', '.join(loader.metadata_columns)} (never model inputs)")
    print(f"[clean]   {len(FEATURES)} survey answers; {miss:.2%} of training cells missing")
    print("          imputed with training-row median/mode; one-hot encoded; standardised on training rows")
    print(f"          -> {X_tr.shape[1]} columns  ({time.perf_counter() - t0:.1f} s)")
    print(f"[tune]    fold training rows split {len(tr):,} fit / {len(iv):,} for hyperparameters + threshold")
    return sc.transform(X_tr), y[tr], sc.transform(X_iv), y[iv], sc.transform(X_va), y[fold["val"]]


def _fit_score(name, spec, data, seed):
    from src.m5_models.classical import make_classical
    from src.m8_eval.evaluation import binary_metrics
    from src.m10_exp.tuning import scores_of, select_and_fit

    X_tr, y_tr, X_iv, y_iv, X_va, y_va = data
    t0 = time.perf_counter()
    best = select_and_fit(lambda **p: make_classical(name, y_tr, seed=seed, n_jobs=4, **{**spec.get("base", {}), **p}),
                          spec.get("grid"), X_tr, y_tr, X_iv, y_iv)
    m = binary_metrics(y_va, scores_of(best["model"], X_va), threshold=best["threshold"])
    return best, m, time.perf_counter() - t0


def core(cfg):
    loader, frame, y, fold, seed = _setup(cfg)
    data = _prepare(cfg, loader, frame, y, fold, seed)
    name = cfg["core_model"]
    best, m, secs = _fit_score(name, cfg["classical"][name], data, seed)
    print(f"[train]   {NAMES[name]}, class-weighted  ({secs:.1f} s)")
    print(f"          chosen {best['params']} by inner-validation ROC-AUC ({best['inner_val_auc']:.4f})")
    print(f"[score]   validation fold, {len(data[5]):,} records; threshold {best['threshold']:.4f} "
          f"(set on inner validation)")
    for k, lab in [("roc_auc", "ROC-AUC"), ("recall", "Recall (sensitivity)"), ("specificity", "Specificity"),
                   ("balanced_accuracy", "Balanced accuracy"), ("precision", "Precision"), ("f1", "F1"),
                   ("accuracy", "Accuracy")]:
        print(f"          {lab:<22}{m[k]:.4f}")


def baselines(cfg):
    loader, frame, y, fold, seed = _setup(cfg)
    data = _prepare(cfg, loader, frame, y, fold, seed)
    print(f"[metrics] positives are {data[5].mean():.2%} of validation rows, so an all-negative model")
    print(f"          would score accuracy {1 - data[5].mean():.4f} with recall 0.0000")
    rows = []
    for name, spec in cfg["classical"].items():
        best, m, secs = _fit_score(name, spec, data, seed)
        print(f"[train]   {NAMES[name]:<20} {str(best['params']):<42} {secs:7.1f} s")
        rows.append((NAMES[name], m))
    print()
    head = ["Model", "Accuracy", "Bal.acc", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    print(f"{head[0]:<21}" + "".join(f"{h:>10}" for h in head[1:]))
    print("-" * 91)
    for lab, m in sorted(rows, key=lambda r: -r[1]["roc_auc"]):
        print(f"{lab:<21}" + "".join(f"{m[k]:>10.4f}" for k in
                                     ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")))
    print("\nsorted by ROC-AUC; one fold (r0f0); thresholds set on inner-validation rows only")


def handoff(cfg):
    from src.m8_eval.state_viz import statevector
    from src.m10_exp.common import holdout_split, load_dataset

    h = cfg["handoff"]
    seed = int(cfg["project"]["seed"])
    loader, frame = load_dataset(cfg)
    y = frame[loader.label_column].to_numpy(dtype=int)
    _, hold = holdout_split(y, float(cfg["split"]["holdout_fraction"]), seed)
    pos = int(h["holdout_index"])
    row = int(hold[pos])
    cache = pickle.load(open(ROOT / h["models"], "rb"))
    prep, ft = cache["prep"], cache["qsvm"][(h["encoding"], int(h["n_qubits"]))]["transformer"]
    print(f"[record]  locked hold-out index {pos} = dataset row {row}")
    print("          never used in training; inference only, no metric")
    x95 = prep.transform(frame.iloc[[row]], loader.label_column).to_numpy()
    print(f"[module1] cleaned + one-hot: {x95.shape[1]} columns")
    print(f"          transformer fitted on {cache['n_train']:,} development rows ({h['models']})")
    z = ft.scaler.transform(x95)
    pcs = ft.pca.transform(z)
    ang = ft.angle_scaler.transform(pcs)
    assert np.allclose(ang, ft.transform(x95)), "step-by-step result differs from FeatureTransformer.transform"
    lo, hi = ft.angle_scaler.data_min_, ft.angle_scaler.data_max_
    print(f"[module1] standardise -> PCA to {ft.pca.n_components_} components "
          f"({ft.pca.explained_variance_ratio_.sum():.1%} of training variance)\n")
    print("MODULE 1 OUTPUT -> MODULE 2 INPUT (one row per feature = one qubit)")
    print(f"{'Feature':<9}{'PCA value':>11}{'train min':>11}{'train max':>11}{'angle (rad)':>13}   gate")
    print("-" * 70)
    for q in range(len(ang[0])):
        print(f"{'PC' + str(q + 1):<9}{pcs[0, q]:>11.4f}{lo[q]:>11.4f}{hi[q]:>11.4f}{ang[0, q]:>13.4f}   "
              f"RY({ang[0, q]:.4f}) on qubit {q}")
    print("\nangle = pi * (value - train min) / (train max - train min), clipped to [0, pi]")
    psi = statevector(h["encoding"], int(h["n_qubits"]), ang[0])
    top = np.argsort(-np.abs(psi) ** 2)[:3]
    print(f"[module2] {h['encoding']} encoding on {h['n_qubits']} qubits: |psi> = U(x)|0>, {psi.size} amplitudes, "
          f"norm {np.sum(np.abs(psi) ** 2):.6f}")
    print("[module2] largest amplitudes: " + ", ".join(f"|{k:0{int(h['n_qubits'])}b}> {psi[k].real:+.4f}" for k in top))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", choices=["core", "baselines", "handoff"])
    p.add_argument("--config", required=True)
    a = p.parse_args()
    from src.m10_exp.common import limit_worker_threads, load_config

    limit_worker_threads()
    {"core": core, "baselines": baselines, "handoff": handoff}[a.run](load_config(ROOT / a.config))


if __name__ == "__main__":
    main()
