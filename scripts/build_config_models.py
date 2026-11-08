"""Fit, once, the models the PrivacyQ "Inspect" page traces records through.

All models use the same 1,500-row stratified dev subsample (seed 2026; the
quantum training regime of the experiments): inner-train 1,200 rows, inner-val
300 rows for hyperparameters and threshold, exactly as in Experiment 1.

- QSVM for every encoding (angle, amplitude, ZZ, re-uploading) x {4, 6, 8} qubits, PCA features.
- Logistic regression on PCA-4/6/8 of the same rows (the Experiment 1 reference).

Nothing here is evaluated: the page reports measured numbers from results/.
Also stores a pool of dev records used to train none of these models.

Usage: python scripts/build_config_models.py
"""

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.m1_data.nsch import FEATURES  # noqa: E402
from src.m2_preprocess.preprocessing import TabularPreprocessor  # noqa: E402
from src.m3_features.features import FeatureTransformer  # noqa: E402
from src.m5_models.classical import make_classical  # noqa: E402
from src.m5_models.quantum import QuantumKernelSVC  # noqa: E402
from src.m10_exp.common import holdout_split, load_config, load_dataset  # noqa: E402
from src.m10_exp.tuning import inner_split, select_and_fit, stratified_subsample  # noqa: E402

ENCODINGS = ("angle", "amplitude", "zz", "reupload")
SIZES = (4, 6, 8)


def main() -> None:
    t0 = time.time()
    cfg = load_config(ROOT / "configs/holdout_final.yaml")
    seed = int(cfg["project"]["seed"])
    loader, frame = load_dataset(cfg)
    labels = frame[loader.label_column].to_numpy(dtype=int)
    dev, _ = holdout_split(labels, float(cfg["split"]["holdout_fraction"]), seed)
    sub = stratified_subsample(dev, labels, 1500, seed)
    tr, iv = inner_split(sub, labels, float(cfg["tuning"]["inner_val_fraction"]), seed)
    prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
    X_in = prep.fit_transform(frame.iloc[tr], loader.label_column).to_numpy()
    X_iv = prep.transform(frame.iloc[iv], loader.label_column).to_numpy()

    qsvm, logreg = {}, {}
    for n in SIZES:
        ft = FeatureTransformer(d_prime=n, method="pca", output="angle", seed=seed).fit(X_in, labels[tr])
        A, B = ft.transform(X_in), ft.transform(X_iv)
        best = select_and_fit(lambda **p: make_classical("logreg", labels[tr], seed=seed, n_jobs=1, **p),
                              {"C": [0.01, 0.1, 1.0, 10.0]}, A, labels[tr], B, labels[iv])
        logreg[n] = {"transformer": ft, "model": best["model"], "threshold": best["threshold"], "params": best["params"]}
        for enc in ENCODINGS:
            if enc == "amplitude":
                ft_e = FeatureTransformer(d_prime=min(2**n - 1, X_in.shape[1]), method="pca", output="standard", seed=seed)
                ft_e.fit(X_in, labels[tr])
                A_e, B_e = ft_e.transform(X_in), ft_e.transform(X_iv)
            else:
                ft_e, A_e, B_e = ft, A, B
            best = select_and_fit(lambda enc=enc, n=n, **p: QuantumKernelSVC(encoding=enc, n_qubits=n, seed=seed, **p),
                                  {"C": [0.1, 1.0, 10.0]}, A_e, labels[tr], B_e, labels[iv])
            qsvm[(enc, n)] = {"transformer": ft_e, "model": best["model"], "threshold": best["threshold"],
                              "params": best["params"]}
            print(f"QSVM {enc:9s} {n} qubits: C={best['params']['C']}")

    unused = np.setdiff1d(dev, sub)
    pool_rows = np.random.default_rng(seed).choice(unused, size=600, replace=False)
    pool = frame.iloc[np.sort(pool_rows)][list(FEATURES) + [loader.label_column]].copy()
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    with open(out / "config_models.pkl", "wb") as fh:
        pickle.dump({"prep": prep, "qsvm": qsvm, "logreg": logreg, "pool": pool, "n_train": len(tr),
                     "source": loader.source_tag()}, fh)
    print(f"cached models/config_models.pkl in {time.time() - t0:.0f} s "
          f"({len(qsvm)} QSVMs, {len(logreg)} LogRegs, {len(pool)} demo records)")


if __name__ == "__main__":
    main()
