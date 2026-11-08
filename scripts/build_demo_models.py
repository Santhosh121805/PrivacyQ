"""Fit the final configurations once for the web demo and cache them in models/.

Reproduces the training side of `src.m10_exp.holdout_eval` (same dev rows, seed,
inner split, grids and threshold rule) and checks that the selected
hyperparameters and thresholds equal those recorded in the hold-out run. The
hold-out rows are never scored here.

Also fits a QSVM for each of the four encodings on the same 1,200 rows (for the
"Inside the pipeline" view) and stores a pool of dev records that neither model
was trained on, for the "load a random real record" button.

Usage: python scripts/build_demo_models.py
"""

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.m1_data.nsch import FEATURES  # noqa: E402
from src.m2_preprocess.preprocessing import TabularPreprocessor  # noqa: E402
from src.m3_features.features import FeatureTransformer  # noqa: E402
from src.m5_models.classical import make_classical  # noqa: E402
from src.m5_models.quantum import QuantumKernelSVC  # noqa: E402
from src.m10_exp.common import holdout_split, load_config, load_dataset  # noqa: E402
from src.m10_exp.tuning import inner_split, select_and_fit, stratified_subsample  # noqa: E402

ENCODINGS = ("angle", "amplitude", "zz", "reupload")


def main() -> None:
    cfg = load_config(ROOT / "configs/holdout_final.yaml")
    final = load_config(ROOT / "configs/final_report.yaml")
    holdout_csv = ROOT / final["runs"]["holdout"] / "holdout_metrics.csv"
    seed = int(cfg["project"]["seed"])
    frac = float(cfg["tuning"]["inner_val_fraction"])
    loader, frame = load_dataset(cfg)
    labels = frame[loader.label_column].to_numpy(dtype=int)
    dev, _ = holdout_split(labels, float(cfg["split"]["holdout_fraction"]), seed)

    # --- LogReg, full dev rows, all 95 features (as in holdout_eval) ---
    tr, iv = inner_split(dev, labels, frac, seed)
    prep_lr = TabularPreprocessor(identifier_columns=loader.metadata_columns)
    X_in = prep_lr.fit_transform(frame.iloc[tr], loader.label_column).to_numpy()
    X_iv = prep_lr.transform(frame.iloc[iv], loader.label_column).to_numpy()
    scaler = StandardScaler().fit(X_in)
    grid = cfg["models"]["logreg_full_all95"]["grid"]
    lr = select_and_fit(lambda **p: make_classical("logreg", labels[tr], seed=seed, n_jobs=4, **p),
                        grid, scaler.transform(X_in), labels[tr], scaler.transform(X_iv), labels[iv])

    # --- QSVM, 1,500-row subsample of dev, PCA-8 (angle is the final model) ---
    sub = stratified_subsample(dev, labels, cfg["models"]["qsvm_angle_pca8_1500rows"]["subsample"], seed)
    tr_q, iv_q = inner_split(sub, labels, frac, seed)
    prep_q = TabularPreprocessor(identifier_columns=loader.metadata_columns)
    Xq_in = prep_q.fit_transform(frame.iloc[tr_q], loader.label_column).to_numpy()
    Xq_iv = prep_q.transform(frame.iloc[iv_q], loader.label_column).to_numpy()
    qsvm = {}
    for enc in ENCODINGS:
        if enc == "amplitude":  # up to 2**8 - 1 inputs: all 95 standardized PCA components
            ft = FeatureTransformer(d_prime=min(255, Xq_in.shape[1]), method="pca", output="standard", seed=seed)
        else:
            ft = FeatureTransformer(d_prime=8, method="pca", output="angle", seed=seed)
        ft.fit(Xq_in, labels[tr_q])
        best = select_and_fit(lambda enc=enc, **p: QuantumKernelSVC(encoding=enc, n_qubits=8, seed=seed, **p),
                              {"C": [0.1, 1.0, 10.0]}, ft.transform(Xq_in), labels[tr_q],
                              ft.transform(Xq_iv), labels[iv_q])
        qsvm[enc] = {"transformer": ft, "model": best["model"], "threshold": best["threshold"],
                     "params": best["params"], "inner_val_auc": best["inner_val_auc"]}
        print(f"QSVM {enc}: C={best['params']['C']} threshold={best['threshold']:.4f} inner-val AUC {best['inner_val_auc']:.3f}")

    # --- reproducibility check against the one-shot hold-out run (no re-scoring) ---
    h = pd.read_csv(holdout_csv).set_index("model")
    checks = {
        "logreg C": (str(lr["params"]), h.loc["logreg_full_all95", "params"]),
        "logreg threshold": (round(lr["threshold"], 10), round(float(h.loc["logreg_full_all95", "threshold"]), 10)),
        "qsvm C": (str(qsvm["angle"]["params"]), h.loc["qsvm_angle_pca8_1500rows", "params"]),
        "qsvm threshold": (round(qsvm["angle"]["threshold"], 10), round(float(h.loc["qsvm_angle_pca8_1500rows", "threshold"]), 10)),
    }
    for name, (got, want) in checks.items():
        print(f"check {name}: {got} vs hold-out run {want} -> {'OK' if got == want else 'MISMATCH'}")
        if got != want:
            raise SystemExit("refit does not reproduce the hold-out models; not caching")

    # --- demo record pool: dev rows used to train neither model ---
    unused = np.setdiff1d(dev, np.union1d(tr, tr_q))
    pool_rows = np.random.default_rng(seed).choice(unused, size=min(600, len(unused)), replace=False)
    pool = frame.iloc[np.sort(pool_rows)][list(FEATURES) + [loader.label_column]].copy()
    pool.index.name = "row"

    exp1 = pd.read_csv(ROOT / final["runs"]["exp1"] / "summary.csv")
    cv_auc = {}
    for enc in ENCODINGS:
        r = exp1[(exp1.model == "qsvm") & (exp1.encoding == enc) & (exp1.feature_method == "pca") & (exp1.n_qubits == 8)]
        cv_auc[enc] = {"mean": float(r.roc_auc_mean.iloc[0]), "std": float(r.roc_auc_std.iloc[0]), "folds": int(r.n_folds.iloc[0])}

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    bundle = {
        "logreg": {"prep": prep_lr, "scaler": scaler, "model": lr["model"], "threshold": lr["threshold"],
                   "params": lr["params"], "n_train": len(tr)},
        "qsvm": {"prep": prep_q, "by_encoding": qsvm, "n_train": len(tr_q)},
        "pool": pool, "qsvm_exp1_cv_auc_pca8_8q": cv_auc,
        "source": loader.source_tag(), "holdout_run": str(final["runs"]["holdout"]),
    }
    with open(out / "demo_models.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"cached {out / 'demo_models.pkl'} ({len(pool)} demo records, positives {int(pool.label.sum())})")


if __name__ == "__main__":
    main()
