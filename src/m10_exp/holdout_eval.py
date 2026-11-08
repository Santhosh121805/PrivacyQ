"""One-shot evaluation on the locked hold-out (4,358 rows, never used before).

Each final configuration is trained on the dev rows (all non-hold-out rows) with
exactly the protocol used in cross-validation: inner split of its training rows,
hyperparameters and threshold chosen on inner-val, then scored ONCE on the
hold-out. Nothing is tuned against the hold-out. The script refuses to run if a
hold-out result already exists, so the hold-out cannot be re-scored.

Usage: python -m src.m10_exp.holdout_eval --config configs/holdout_final.yaml
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.m2_preprocess.preprocessing import TabularPreprocessor
from src.m5_models.classical import make_classical
from src.m5_models.quantum import QuantumKernelSVC
from src.m7_privacy.dp_training import DPClassifier
from src.m8_eval.evaluation import binary_metrics
from src.m10_exp.common import environment_record, holdout_split, load_config, load_dataset, write_json
from src.m10_exp.exp2_compare import feature_matrices
from src.m10_exp.tuning import inner_split, select_and_fit, stratified_subsample, youden_threshold



def bootstrap_auc_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def run(config_path: str) -> Path:
    import torch

    torch.manual_seed(0)
    cfg = load_config(config_path)
    seed = int(cfg["project"]["seed"])
    loader, frame = load_dataset(cfg)
    lock = Path(cfg["project"]["output_dir"]) / f"HOLDOUT_EVALUATED_{loader.source_tag().replace(':', '_')}.lock"
    if lock.exists():
        raise SystemExit(f"hold-out already evaluated ({lock.read_text().strip()}); refusing to re-score")
    labels = frame[loader.label_column].to_numpy(dtype=int)
    dev, hold = holdout_split(labels, float(cfg["split"]["holdout_fraction"]), seed)
    prefix = "SYNTHETIC-" if loader.is_synthetic else ""
    out = Path(cfg["project"]["output_dir"]) / f"{prefix}holdout-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    shutil.copy(config_path, out / "config.yaml")
    write_json(out / "environment.json", environment_record(cfg))
    lock.write_text(f"{out}\n", encoding="utf-8")
    frac = float(cfg["tuning"]["inner_val_fraction"])
    rows = []
    for name, spec in cfg["models"].items():
        rows_train = stratified_subsample(dev, labels, spec.get("subsample"), seed)
        tr, iv = inner_split(rows_train, labels, frac, seed)
        prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
        X_in = prep.fit_transform(frame.iloc[tr], loader.label_column).to_numpy()
        X_iv = prep.transform(frame.iloc[iv], loader.label_column).to_numpy()
        X_ho = prep.transform(frame.iloc[hold], loader.label_column).to_numpy()
        A, B, C = feature_matrices(spec["features"], X_in, labels[tr], [X_iv, X_ho], seed)
        t0 = time.perf_counter()
        kind = spec["kind"]
        if kind == "dp":
            model = DPClassifier(seed=seed, **spec["params"]).fit(A, labels[tr])
            threshold, params, eps = youden_threshold(labels[iv], model.decision_scores(B)), spec["params"], model.privacy_.epsilon
        else:
            if kind == "qsvm":
                def factory(**p):
                    return QuantumKernelSVC(seed=seed, **spec["params"], **p)
            else:
                def factory(**p):
                    return make_classical(kind, labels[tr], seed=seed, n_jobs=4, **p)
            best = select_and_fit(factory, spec.get("grid"), A, labels[tr], B, labels[iv])
            model, threshold, params, eps = best["model"], best["threshold"], best["params"], None
        fit_seconds = time.perf_counter() - t0
        scores = (model.decision_scores(C) if hasattr(model, "decision_scores") else model.predict_proba(C)[:, 1])
        lo, hi = bootstrap_auc_ci(labels[hold], scores)
        rows.append({"model": name, **binary_metrics(labels[hold], scores, threshold=threshold),
                     "auc_ci95_low": lo, "auc_ci95_high": hi, "threshold": threshold, "params": str(params),
                     "epsilon_spent": eps, "n_train_rows": len(rows_train), "n_inner_train": len(tr),
                     "n_holdout": len(hold), "n_holdout_pos": int(labels[hold].sum()), "fit_seconds": fit_seconds,
                     "data_source": loader.source_tag()})
        print(f"{name}: hold-out AUC {rows[-1]['roc_auc']:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    pd.DataFrame(rows).to_csv(out / "holdout_metrics.csv", index=False)
    print(f"wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot hold-out evaluation")
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
