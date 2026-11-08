"""Experiment 2: best quantum configurations vs classical baselines (no privacy).

Quantum models (best configuration per family from Experiment 1, chosen by the
pre-declared rule "highest mean ROC-AUC"):
  QSVM   angle / PCA / 8 qubits   on feature set pca8
  VQC    re-uploading / PCA / 4   on feature set pca4
  Hybrid angle / 8 qubits         on feature set all95 (learned compression)

Classical models (LogReg, RandomForest, XGBoost, MLP) run on every feature set
in two data regimes:
  subsample - the identical 1,500-row stratified subsample the quantum models use
  full      - all of the fold's training rows
Every model, quantum or classical, goes through the same tuning protocol
(``src.m10_exp.tuning``): grid, early-stopping epoch and threshold chosen on an
inner-val split of the regime's training rows. Scored on the full outer fold.

Usage: python -m src.m10_exp.exp2_compare --config configs/exp2_nsch2024.yaml
       python -m src.m10_exp.exp2_compare --report-only results/<run_dir>
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from src.m2_preprocess.preprocessing import TabularPreprocessor
from src.m3_features.features import FeatureTransformer
from src.m5_models.classical import make_classical
from src.m5_models.quantum import HybridQNNClassifier, QuantumKernelSVC, VQCClassifier
from src.m8_eval.evaluation import binary_metrics
from src.m10_exp.common import (
    cv_folds, environment_record, holdout_split, limit_worker_threads, load_config,
    load_dataset, make_run_dir, save_splits, write_json,
)
from src.m10_exp.tuning import inner_split, scores_of, select_and_fit, stratified_subsample

QUANTUM = {"qsvm": QuantumKernelSVC, "vqc": VQCClassifier, "hybrid": HybridQNNClassifier}


def feature_matrices(spec: dict, X_in, y_in, others, seed):
    if spec["method"] == "all":
        scaler = StandardScaler().fit(X_in)
        return [scaler.transform(X_in)] + [scaler.transform(o) for o in others]
    ft = FeatureTransformer(d_prime=int(spec["k"]), method=spec["method"], output="angle", seed=seed)
    ft.fit(X_in, y_in)
    return [ft.transform(X_in)] + [ft.transform(o) for o in others]


def run_task(task: dict) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    seed = task["seed"]
    A, B, C = feature_matrices(task["feature_spec"], task["X_in"], task["y_in"], [task["X_iv"], task["X_ov"]], seed)
    rows = []
    for model_name in task["models"]:
        if model_name in QUANTUM:
            q = task["quantum"][model_name]
            cls = QUANTUM[model_name]
            fixed = {k: v for k, v in q.items() if k not in ("feature_set", "grid")}

            def factory(cls=cls, fixed=fixed, **p):
                return cls(seed=seed, **fixed, **p)

            grid = q.get("grid")
            family = "quantum"
        else:
            base = task["base_params"].get(model_name, {})

            def factory(name=model_name, base=base, **p):
                return make_classical(name, task["y_in"], seed=seed, n_jobs=1, **{**base, **p})

            grid = task["grids"].get(model_name)
            family = "classical"
        best = select_and_fit(factory, grid, A, task["y_in"], B, task["y_iv"])
        t0 = time.perf_counter()
        scores = scores_of(best["model"], C)
        predict_seconds = time.perf_counter() - t0
        model = best["model"]
        if hasattr(model, "n_parameters"):
            n_params = model.n_parameters()
        elif hasattr(model, "coef_"):
            n_params = int(model.coef_.size + model.intercept_.size)
        elif hasattr(model, "model_"):
            n_params = int(sum(p.numel() for p in model.model_.parameters()))
        else:
            n_params = None
        rows.append({
            "record_id": f"{model_name}|{task['regime']}|{task['feature_set']}|r{task['repeat']}f{task['fold']}",
            "data_source": task["source"], "repeat": task["repeat"], "fold": task["fold"],
            "model": model_name, "family": family, "regime": task["regime"],
            "feature_set": task["feature_set"], "n_features_in": A.shape[1],
            **binary_metrics(task["y_ov"], scores, threshold=best["threshold"]),
            "threshold": best["threshold"], "inner_val_auc": best["inner_val_auc"],
            "best_params": str(best["params"]), "best_epoch": getattr(model, "best_epoch_", None),
            "fit_seconds": best["fit_seconds"], "tuning_seconds": best["tuning_seconds"],
            "predict_seconds": predict_seconds, "n_params": n_params,
            **task["counts"], "_scores": scores.astype(np.float32),
        })
    return rows


def build_tasks(config, loader, frame, folds, labels):
    exp = config["experiment"]
    frac = float(config["tuning"]["inner_val_fraction"])
    feature_sets = exp["feature_sets"]
    tasks = []
    for f in folds:
        seed = int(config["project"]["seed"]) + 1000 * f["repeat"] + f["fold"]
        regimes = {"subsample": stratified_subsample(f["train"], labels, int(config["subsample"]["n"]), seed),
                   "full": f["train"]}
        for regime, rows in regimes.items():
            if regime not in exp["regimes"]:
                continue
            tr, iv = inner_split(rows, labels, frac, seed)
            prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
            X_in = prep.fit_transform(frame.iloc[tr], loader.label_column).to_numpy()
            X_iv = prep.transform(frame.iloc[iv], loader.label_column).to_numpy()
            X_ov = prep.transform(frame.iloc[f["val"]], loader.label_column).to_numpy()
            counts = {"n_train_rows": len(rows), "n_train_pos": int(labels[rows].sum()),
                      "n_inner_train": len(tr), "n_inner_train_pos": int(labels[tr].sum()),
                      "n_inner_val": len(iv), "n_inner_val_pos": int(labels[iv].sum()),
                      "n_val": len(f["val"]), "n_val_pos": int(labels[f["val"]].sum()),
                      "n_fold_train": len(f["train"])}
            common = dict(source=loader.source_tag(), repeat=f["repeat"], fold=f["fold"], seed=seed,
                          regime=regime, X_in=X_in, y_in=labels[tr], X_iv=X_iv, y_iv=labels[iv],
                          X_ov=X_ov, y_ov=labels[f["val"]], counts=counts, grids=config["grids"],
                          base_params=config.get("base_params", {}), quantum=exp["quantum"])
            for fs_name, spec in feature_sets.items():
                classical = [{**common, "feature_set": fs_name, "feature_spec": spec, "models": [m]}
                             for m in exp["classical_models"]]
                tasks.extend(classical)
                if regime == "subsample":
                    for qname, q in exp["quantum"].items():
                        if q["feature_set"] == fs_name:
                            tasks.append({**common, "feature_set": fs_name, "feature_spec": spec, "models": [qname]})
    cost = {"hybrid": 0, "vqc": 1, "mlp": 2, "random_forest": 3, "xgboost": 4, "qsvm": 5, "logreg": 6}
    return sorted(tasks, key=lambda t: (cost.get(t["models"][0], 9), t["regime"] != "full"))


def run(config_path: str) -> Path:
    limit_worker_threads()
    config = load_config(config_path)
    seed = int(config["project"]["seed"])
    loader, frame = load_dataset(config)
    run_dir = make_run_dir(config, loader)
    shutil.copy(config_path, run_dir / "config.yaml")
    write_json(run_dir / "environment.json", environment_record(config))
    labels = frame[loader.label_column].to_numpy(dtype=int)
    dev, holdout = holdout_split(labels, float(config["split"]["holdout_fraction"]), seed)
    folds = cv_folds(dev, labels, int(config["cv"]["n_splits"]), int(config["cv"]["n_repeats"]), seed)
    save_splits(run_dir, holdout, folds)
    tasks = build_tasks(config, loader, frame, folds, labels)
    records, scores = [], {}
    resume = config.get("resume")
    if resume:  # reuse finished rows of an interrupted run on the same folds
        old_dir = Path(resume["run_dir"])
        if not pd.read_csv(old_dir / "cv_val_rows.csv").equals(pd.read_csv(run_dir / "cv_val_rows.csv")):
            raise ValueError("resume run used different folds; refusing to merge")
        old = pd.read_csv(old_dir / "fold_metrics.partial.csv")
        old = old[old.model.isin(resume["models"])]
        records.extend(old.to_dict("records"))
        tasks = [t for t in tasks if t["models"][0] not in resume["models"]]
        write_json(run_dir / "resumed_from.json", {"run_dir": str(old_dir), "models": resume["models"], "rows": len(old)})
    print(f"[{loader.source_tag()}] {len(folds)} folds, {len(tasks)} tasks", flush=True)
    part = run_dir / "fold_metrics.partial.csv"
    started = time.time()
    for i, rows in enumerate(Parallel(n_jobs=int(config["project"]["n_jobs"]), return_as="generator_unordered")(
            delayed(run_task)(t) for t in tasks), start=1):
        for r in rows:
            scores[r["record_id"]] = r.pop("_scores")
        records.extend(rows)
        pd.DataFrame(rows).to_csv(part, mode="a", header=(i == 1), index=False)
        r = rows[0]
        print(f"[{i}/{len(tasks)} {time.time() - started:6.0f}s] {r['model']}/{r['regime']}/{r['feature_set']} "
              f"r{r['repeat']}f{r['fold']} auc={r['roc_auc']:.3f} bacc={r['balanced_accuracy']:.3f}", flush=True)
    results = pd.DataFrame(records)
    results.to_csv(run_dir / "fold_metrics.csv", index=False)
    part.unlink(missing_ok=True)
    np.savez_compressed(run_dir / "val_scores.npz", **{k.replace("|", "__"): v for k, v in scores.items()})
    if not config.get("skip_report"):
        from src.m10_exp.exp2_report import write_report

        write_report(run_dir, results, loader, config)
    print(f"wrote {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: quantum vs classical")
    parser.add_argument("--config")
    parser.add_argument("--report-only")
    args = parser.parse_args()
    if args.report_only:
        from src.m1_data import make_loader
        from src.m10_exp.exp2_report import write_report

        run_dir = Path(args.report_only)
        config = load_config(run_dir / "config.yaml")
        loader = make_loader(config["data"], seed=int(config["project"]["seed"]))
        write_report(run_dir, pd.read_csv(run_dir / "fold_metrics.csv"), loader, config)
        return
    run(args.config)


if __name__ == "__main__":
    main()
