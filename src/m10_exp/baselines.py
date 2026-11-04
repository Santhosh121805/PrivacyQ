"""Phase 1: classical baselines under repeated stratified CV.

Protocol
--------
1. Load the dataset named in the YAML config.
2. Lock away a stratified hold-out set (``split.holdout_fraction``). It is not
   used in this phase.
3. On the remaining dev rows, run repeated stratified K-fold. Inside every fold,
   imputation, one-hot encoding, scaling, MI selection and PCA are fitted on the
   fold's training rows only.
4. For each feature setting (all features, or MI/PCA reduced to k) and each
   classical model, fit on the fold-train rows and score on the fold-val rows.

Usage: python -m src.m10_exp.baselines --config configs/phase1_nsch2024.yaml
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.m2_preprocess.preprocessing import TabularPreprocessor
from src.m3_features.features import FeatureTransformer
from src.m5_models.classical import count_parameters, make_classical
from src.m8_eval.evaluation import binary_metrics
from src.m10_exp.common import (
    cv_folds,
    environment_record,
    holdout_split,
    load_config,
    load_dataset,
    make_run_dir,
    markdown_table,
    save_splits,
    summarise,
    report_header,
    write_json,
)

METRICS = ["balanced_accuracy", "recall", "roc_auc", "pr_auc", "specificity", "f1", "accuracy"]


def feature_settings(cfg: dict) -> list[tuple[str, int | None]]:
    settings: list[tuple[str, int | None]] = []
    if cfg.get("include_full", True):
        settings.append(("full", None))
    for method in cfg.get("methods", ["mi", "pca"]):
        for k in cfg.get("ks", [4, 6, 8]):
            settings.append((method, int(k)))
    return settings


def dataset_profile(loader, frame: pd.DataFrame, dev, holdout) -> dict:
    label = loader.label_column
    features = frame.drop(columns=[label, *loader.metadata_columns], errors="ignore")
    numeric = features.select_dtypes(include=[np.number]).columns
    return {
        "source": loader.source_tag(),
        "loader": loader.describe(),
        "n_rows": int(len(frame)),
        "n_raw_features": int(features.shape[1]),
        "n_numeric_features": int(len(numeric)),
        "n_categorical_features": int(features.shape[1] - len(numeric)),
        "n_positive": int(frame[label].sum()),
        "positive_rate": float(frame[label].mean()),
        "missing_cell_rate": float(features.isna().to_numpy().mean()),
        "most_missing": features.isna().mean().sort_values(ascending=False).head(5).round(4).to_dict(),
        "n_dev": int(len(dev)),
        "n_holdout": int(len(holdout)),
        "dev_positive_rate": float(frame[label].to_numpy()[dev].mean()),
        "holdout_positive_rate": float(frame[label].to_numpy()[holdout].mean()),
    }


def run(config_path: str) -> Path:
    config = load_config(config_path)
    seed = int(config.get("project", {}).get("seed", 0))
    loader, frame = load_dataset(config)
    run_dir = make_run_dir(config, loader)
    shutil.copy(config_path, run_dir / "config.yaml")
    write_json(run_dir / "environment.json", environment_record(config))

    labels = frame[loader.label_column].to_numpy(dtype=int)
    split_cfg = config.get("split", {})
    dev, holdout = holdout_split(labels, float(split_cfg.get("holdout_fraction", 0.2)), seed)
    cv_cfg = config.get("cv", {})
    folds = cv_folds(dev, labels, int(cv_cfg.get("n_splits", 5)), int(cv_cfg.get("n_repeats", 3)), seed)
    save_splits(run_dir, holdout, folds)
    profile = dataset_profile(loader, frame, dev, holdout)
    write_json(run_dir / "dataset_profile.json", profile)
    print(f"[{loader.source_tag()}] rows={profile['n_rows']} positive_rate={profile['positive_rate']:.4f} "
          f"dev={len(dev)} holdout={len(holdout)} folds={len(folds)}", flush=True)

    settings = feature_settings(config.get("features", {}))
    models = config.get("models", ["logreg", "random_forest", "xgboost", "mlp"])
    model_params = config.get("model_params", {}) or {}
    n_jobs = int(config.get("project", {}).get("n_jobs", 4))
    records, selections = [], []

    for f in folds:
        fold_seed = seed + 1000 * f["repeat"] + f["fold"]
        train_frame, val_frame = frame.iloc[f["train"]], frame.iloc[f["val"]]
        y_tr, y_va = labels[f["train"]], labels[f["val"]]
        prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
        X_tr_full = prep.fit_transform(train_frame, loader.label_column)
        X_va_full = prep.transform(val_frame, loader.label_column)
        names = np.array(X_tr_full.columns)

        for method, k in settings:
            t0 = time.perf_counter()
            if method == "full":
                scaler = StandardScaler().fit(X_tr_full)
                X_tr, X_va = scaler.transform(X_tr_full), scaler.transform(X_va_full)
            else:
                transformer = FeatureTransformer(d_prime=k, method=method, seed=fold_seed)
                X_tr = transformer.fit_transform(X_tr_full, y_tr)
                X_va = transformer.transform(X_va_full)
                if method == "mi":
                    chosen = names[transformer.selector.get_support()]
                    selections.append({"repeat": f["repeat"], "fold": f["fold"], "k": k,
                                        "features": ";".join(chosen)})
            prep_seconds = time.perf_counter() - t0

            for model_name in models:
                model = make_classical(model_name, y_tr, seed=fold_seed, n_jobs=n_jobs,
                                       **model_params.get(model_name, {}))
                t0 = time.perf_counter()
                model.fit(X_tr, y_tr)
                fit_seconds = time.perf_counter() - t0
                scores = model.predict_proba(X_va)[:, 1]
                records.append({
                    "data_source": loader.source_tag(),
                    "repeat": f["repeat"], "fold": f["fold"],
                    "feature_method": method, "n_features": X_tr.shape[1],
                    "model": model_name, "seed": fold_seed,
                    **binary_metrics(y_va, scores),
                    "fit_seconds": fit_seconds, "feature_prep_seconds": prep_seconds,
                    "n_params": count_parameters(model),
                    "n_train": len(y_tr), "n_val": len(y_va),
                })
        done = pd.DataFrame(records)
        latest = done[(done.repeat == f["repeat"]) & (done.fold == f["fold"])]
        print(f"repeat {f['repeat']} fold {f['fold']}: "
              + ", ".join(f"{r.model}/{r.feature_method}{'' if r.feature_method == 'full' else r.n_features}"
                          f"={r.roc_auc:.3f}" for r in latest.itertuples() if r.feature_method in ("full", "mi")),
              flush=True)

    results = pd.DataFrame(records)
    results.to_csv(run_dir / "fold_metrics.csv", index=False)
    if selections:
        pd.DataFrame(selections).to_csv(run_dir / "mi_selected_features.csv", index=False)
    keys = ["feature_method", "n_features", "model"]
    summary = summarise(results, keys, METRICS + ["fit_seconds"])
    summary.insert(0, "data_source", loader.source_tag())
    summary.to_csv(run_dir / "summary.csv", index=False)

    lines = report_header("Phase 1 classical baselines", loader)
    lines += [
        f"Rows: {profile['n_rows']} (dev {profile['n_dev']}, locked hold-out {profile['n_holdout']}); "
        f"positive rate {profile['positive_rate']:.3%}. "
        f"CV: {cv_cfg.get('n_repeats', 3)} x {cv_cfg.get('n_splits', 5)}-fold stratified on dev rows; "
        "mean ± std over folds. Threshold 0.5 on class-weighted models.", "",
        markdown_table(summary, keys, ["balanced_accuracy", "recall", "roc_auc", "pr_auc"]), "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
