"""Experiment 1: quantum encoding comparison (no privacy).

Encodings {angle, amplitude, ZZ, data re-uploading} x qubits {4, 6, 8} x
feature selection {MI top-k, PCA}, for the quantum kernel SVM and the VQC. The
hybrid QNN (learned linear compression of all features) is run per encoding and
qubit count. Logistic regression on the identical rows/features is included as
a reference line only; the full classical comparison is Experiment 2.

Protocol per outer fold (same folds as Phase 1):
- stratified subsample of the fold's training rows (``subsample.n``),
- stratified inner split into inner-train / inner-val (``tuning.inner_val_fraction``),
- imputation/encoding/scaling/selection fitted on inner-train only,
- hyperparameters, early-stopping epoch and threshold chosen on inner-val
  (``src.m10_exp.tuning``), identical for every model,
- scored once on the full outer validation fold.

Usage: python -m src.m10_exp.exp1_encodings --config configs/exp1_nsch2024.yaml
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.m2_preprocess.preprocessing import TabularPreprocessor
from src.m3_features.features import FeatureTransformer
from src.m4_encoding import ENCODING_NAMES, encoding_resources, make_encoding
from src.m5_models.quantum import HybridQNNClassifier, QuantumKernelSVC, VQCClassifier, vqc_resources
from src.m8_eval.evaluation import binary_metrics
from src.m10_exp.common import (
    cv_folds, environment_record, limit_worker_threads, holdout_split, load_config, load_dataset,
    make_run_dir, report_header, save_splits, write_json,
)
from src.m10_exp.tuning import inner_split, scores_of, select_and_fit, stratified_subsample

MODEL_CLASSES = {"qsvm": QuantumKernelSVC, "vqc": VQCClassifier, "hybrid": HybridQNNClassifier}


def _features(X_in, y_in, others, n_features, method, scale, seed):
    if method == "learned":
        scaler = StandardScaler().fit(X_in)
        return [scaler.transform(X_in)] + [scaler.transform(o) for o in others]
    ft = FeatureTransformer(d_prime=n_features, method=method, output=scale, seed=seed).fit(X_in, y_in)
    return [ft.transform(X_in)] + [ft.transform(o) for o in others]


def run_task(task: dict) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    torch.set_flush_denormal(True)  # guard against denormal slow paths
    rows = []
    X_in, y_in, X_iv, y_iv, X_ov, y_ov = (task[k] for k in ("X_in", "y_in", "X_iv", "y_iv", "X_ov", "y_ov"))
    for n_qubits in task["qubits"]:
        model_kind, encoding, method, seed = task["model"], task["encoding"], task["method"], task["seed"]
        if model_kind == "logreg":
            enc, scale, n_feat = None, "angle", n_qubits
        else:
            enc = make_encoding(encoding, n_qubits)
            scale = enc.input_scale
            n_feat = enc.n_features(available=X_in.shape[1])
        A, B, C = _features(X_in, y_in, [X_iv, X_ov], n_feat, method, scale, seed)

        if model_kind == "logreg":
            def factory(**p):
                return LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed, **p)
        else:
            base = dict(task["base_params"].get(model_kind, {}))
            cls = MODEL_CLASSES[model_kind]

            def factory(**p):
                return cls(encoding=encoding, n_qubits=n_qubits, seed=seed, **base, **p)

        best = select_and_fit(factory, task["grids"].get(model_kind), A, y_in, B, y_iv)
        t0 = time.perf_counter()
        scores = scores_of(best["model"], C)
        predict_seconds = time.perf_counter() - t0
        model = best["model"]
        n_params = (model.n_parameters() if hasattr(model, "n_parameters")
                    else int(model.coef_.size + model.intercept_.size))
        rows.append({
            "record_id": f"{model_kind}|{encoding}|{method}|{n_qubits}|r{task['repeat']}f{task['fold']}",
            "data_source": task["source"], "repeat": task["repeat"], "fold": task["fold"],
            "model": model_kind, "encoding": encoding, "feature_method": method,
            "n_qubits": n_qubits, "n_features_in": A.shape[1],
            **binary_metrics(y_ov, scores, threshold=best["threshold"]),
            "threshold": best["threshold"], "inner_val_auc": best["inner_val_auc"],
            "best_params": str(best["params"]), "best_epoch": getattr(model, "best_epoch_", None),
            "fit_seconds": best["fit_seconds"], "tuning_seconds": best["tuning_seconds"],
            "predict_seconds": predict_seconds, "n_params": n_params,
            **task["counts"], "_scores": scores.astype(np.float32),
        })
    return rows


def build_tasks(config, loader, frame, folds, labels):
    sub_cfg, tune_cfg = config.get("subsample", {}), config.get("tuning", {})
    exp = config["experiment"]
    tasks = []
    for f in folds:
        seed = int(config["project"]["seed"]) + 1000 * f["repeat"] + f["fold"]
        sub = stratified_subsample(f["train"], labels, sub_cfg.get("n"), seed)
        tr, iv = inner_split(sub, labels, float(tune_cfg.get("inner_val_fraction", 0.2)), seed)
        prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
        X_in = prep.fit_transform(frame.iloc[tr], loader.label_column).to_numpy()
        X_iv = prep.transform(frame.iloc[iv], loader.label_column).to_numpy()
        X_ov = prep.transform(frame.iloc[f["val"]], loader.label_column).to_numpy()
        counts = {
            "n_subsample": len(sub), "n_subsample_pos": int(labels[sub].sum()),
            "n_inner_train": len(tr), "n_inner_train_pos": int(labels[tr].sum()),
            "n_inner_val": len(iv), "n_inner_val_pos": int(labels[iv].sum()),
            "n_val": len(f["val"]), "n_val_pos": int(labels[f["val"]].sum()),
            "n_fold_train": len(f["train"]),
        }
        common = dict(source=loader.source_tag(), repeat=f["repeat"], fold=f["fold"], seed=seed,
                      X_in=X_in, y_in=labels[tr], X_iv=X_iv, y_iv=labels[iv], X_ov=X_ov,
                      y_ov=labels[f["val"]], counts=counts, qubits=exp["qubits"],
                      grids=config.get("grids", {}), base_params=config.get("base_params", {}))
        for model_kind in exp["models"]:
            if model_kind == "hybrid":
                for enc in exp["encodings"]:
                    tasks.append({**common, "model": "hybrid", "encoding": enc, "method": "learned"})
                continue
            for method in exp["feature_methods"]:
                if model_kind == "logreg":
                    tasks.append({**common, "model": "logreg", "encoding": "none", "method": method})
                    continue
                for enc in exp["encodings"]:
                    tasks.append({**common, "model": model_kind, "encoding": enc, "method": method})
    order = {"vqc": 0, "hybrid": 1, "qsvm": 2, "logreg": 3}
    return sorted(tasks, key=lambda t: order[t["model"]])


def resources_table(exp: dict, grids: dict) -> pd.DataFrame:
    rows = []
    layers = sorted(set(grids.get("vqc", {}).get("n_layers", [2])) | set(grids.get("hybrid", {}).get("n_layers", [2])))
    for enc in exp["encodings"]:
        for n in exp["qubits"]:
            fm = encoding_resources(make_encoding(enc, n))
            rows.append({"encoding": enc, "n_qubits": n, "circuit": "feature_map (QSVM kernel uses U(x')^dag U(x): ~2x)",
                         "n_layers": 0, **{k: fm[k] for k in ("depth", "n_gates", "n_cnot")}})
            for L in layers:
                r = vqc_resources(enc, n, L)
                rows.append({"encoding": enc, "n_qubits": n, "circuit": "variational (VQC / hybrid quantum layer)",
                             "n_layers": L, **{k: r[k] for k in ("depth", "n_gates", "n_cnot")},
                             "n_circuit_params": 3 * n * L})
    out = pd.DataFrame(rows)
    out["n_circuit_params"] = out["n_circuit_params"].astype("Int64")
    return out


def run(config_path: str) -> Path:
    config = load_config(config_path)
    seed = int(config["project"]["seed"])
    limit_worker_threads()
    loader, frame = load_dataset(config)
    run_dir = make_run_dir(config, loader)
    shutil.copy(config_path, run_dir / "config.yaml")
    write_json(run_dir / "environment.json", {**environment_record(config), "simulator": "default.qubit (torch backprop, broadcasting)"})
    labels = frame[loader.label_column].to_numpy(dtype=int)
    dev, holdout = holdout_split(labels, float(config["split"]["holdout_fraction"]), seed)
    folds = cv_folds(dev, labels, int(config["cv"]["n_splits"]), int(config["cv"]["n_repeats"]), seed)
    save_splits(run_dir, holdout, folds)

    resources = resources_table(config["experiment"], config.get("grids", {}))
    resources.to_csv(run_dir / "circuit_resources.csv", index=False)

    tasks = build_tasks(config, loader, frame, folds, labels)
    records, scores = [], {}
    resume = config.get("resume")
    if resume:
        # Reuse rows from an earlier run of the same code on the same folds
        # (checked: identical validation rows per fold).
        old_dir = Path(resume["run_dir"])
        old_val = pd.read_csv(old_dir / "cv_val_rows.csv")
        new_val = pd.read_csv(run_dir / "cv_val_rows.csv")
        keep_repeats = set(new_val.repeat)
        old_val = old_val[old_val.repeat.isin(keep_repeats)].reset_index(drop=True)
        if not old_val.equals(new_val):
            raise ValueError("resume run used different folds; refusing to merge")
        old = pd.read_csv(old_dir / resume.get("file", "fold_metrics.partial.csv"))
        old = old[old.model.isin(resume["models"]) & old.repeat.isin(keep_repeats)]
        records.extend(old.to_dict("records"))
        tasks = [t for t in tasks if t["model"] not in resume["models"]]
        write_json(run_dir / "resumed_from.json", {"run_dir": str(old_dir), "models": resume["models"],
                                                   "rows": len(old), "repeats": sorted(keep_repeats)})
        print(f"resumed {len(old)} rows for {resume['models']} from {old_dir}", flush=True)
    print(f"[{loader.source_tag()}] {len(folds)} folds, {len(tasks)} tasks", flush=True)
    part_path = run_dir / "fold_metrics.partial.csv"
    n_jobs = int(config["project"].get("n_jobs", 8))
    started = time.time()
    for i, rows in enumerate(Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
            delayed(run_task)(t) for t in tasks), start=1):
        for r in rows:
            scores[r["record_id"]] = r.pop("_scores")
        records.extend(rows)
        pd.DataFrame(rows).to_csv(part_path, mode="a", header=(i == 1), index=False)
        r0 = rows[0]
        print(f"[{i}/{len(tasks)} {time.time() - started:6.0f}s] {r0['model']}/{r0['encoding']}/{r0['feature_method']} "
              f"r{r0['repeat']}f{r0['fold']}: " + ", ".join(f"{r['n_qubits']}q auc={r['roc_auc']:.3f}" for r in rows), flush=True)

    results = pd.DataFrame(records)
    results.to_csv(run_dir / "fold_metrics.csv", index=False)
    part_path.unlink(missing_ok=True)
    np.savez_compressed(run_dir / "val_scores.npz", **{k.replace("|", "__"): v for k, v in scores.items()})
    from src.m10_exp.exp1_report import write_report

    write_report(run_dir, results, resources, loader, config)
    print(f"wrote {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: encoding comparison")
    parser.add_argument("--config", required=True)
    parser.add_argument("--report-only", help="rebuild the report of an existing run directory")
    args = parser.parse_args()
    if args.report_only:
        from src.m1_data import make_loader
        from src.m10_exp.exp1_report import write_report

        run_dir = Path(args.report_only)
        config = load_config(run_dir / "config.yaml")
        loader = make_loader(config["data"], seed=int(config["project"]["seed"]))
        write_report(run_dir, pd.read_csv(run_dir / "fold_metrics.csv"),
                     pd.read_csv(run_dir / "circuit_resources.csv"), loader, config)
        print(f"rebuilt report in {run_dir}")
        return
    run(args.config)


if __name__ == "__main__":
    main()
