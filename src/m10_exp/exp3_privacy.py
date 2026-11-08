"""Experiment 3: privacy-utility curve (DP-SGD via Opacus).

Each arm (model x data regime x feature set) is trained with DP-SGD at every
target epsilon, plus epsilon = infinity (same model and optimiser, no clipping
or noise). Epsilon is calibrated and reported by Opacus's RDP accountant only.

Fixed (not tuned under DP) per arm: learning rate, epochs, expected batch size,
clipping norm; no early stopping. The decision threshold is chosen on the
inner-val rows (Youden J) and is NOT covered by the accounted epsilon; ROC-AUC,
the curve's headline metric, does not depend on it.

QSVM is excluded: it is not gradient-trained, so DP-SGD does not apply.

Usage: python -m src.m10_exp.exp3_privacy --config configs/exp3_nsch2024.yaml
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.m2_preprocess.preprocessing import TabularPreprocessor
from src.m7_privacy.dp_training import DPClassifier
from src.m8_eval.evaluation import binary_metrics
from src.m10_exp.common import (
    cv_folds, environment_record, holdout_split, limit_worker_threads, load_config,
    load_dataset, make_run_dir, save_splits, write_json,
)
from src.m10_exp.exp2_compare import feature_matrices
from src.m10_exp.tuning import inner_split, stratified_subsample, youden_threshold


def eps_label(eps) -> str:
    return "inf" if eps is None else f"{float(eps):g}"


def run_task(task: dict) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    arm = task["arm"]
    A, B, C = feature_matrices(task["feature_spec"], task["X_in"], task["y_in"], [task["X_iv"], task["X_ov"]], task["seed"])
    rows = []
    for eps in task["epsilons"]:
        model = DPClassifier(kind=arm["kind"], epsilon=eps, delta=task["delta"],
                             max_grad_norm=arm.get("max_grad_norm", 1.0), epochs=arm["epochs"],
                             batch_size=arm["batch_size"], learning_rate=arm["learning_rate"],
                             seed=task["seed"], model_params=arm.get("model_params", {}))
        t0 = time.perf_counter()
        model.fit(A, task["y_in"])
        fit_seconds = time.perf_counter() - t0
        threshold = youden_threshold(task["y_iv"], model.decision_scores(B))
        scores = model.decision_scores(C)
        rep = model.privacy_
        rows.append({
            "record_id": f"{task['arm_name']}|eps{eps_label(eps)}|r{task['repeat']}f{task['fold']}",
            "data_source": task["source"], "repeat": task["repeat"], "fold": task["fold"],
            "arm": task["arm_name"], "kind": arm["kind"], "family": arm["family"],
            "regime": task["regime"], "feature_set": task["feature_set"],
            "target_epsilon": eps_label(eps), "epsilon_spent": rep.epsilon, "delta": rep.delta,
            "noise_multiplier": rep.noise_multiplier, "sample_rate": rep.sample_rate, "dp_steps": rep.steps,
            "accountant": rep.accountant, "dp_trained": rep.dp_trained,
            **binary_metrics(task["y_ov"], scores, threshold=threshold), "threshold": threshold,
            "fit_seconds": fit_seconds, "n_params": model.n_parameters(), **task["counts"],
            "_scores": scores.astype(np.float32),
        })
    return rows


def build_tasks(config, loader, frame, folds, labels):
    exp = config["experiment"]
    frac = float(config["tuning"]["inner_val_fraction"])
    tasks = []
    for f in folds:
        seed = int(config["project"]["seed"]) + 1000 * f["repeat"] + f["fold"]
        regimes = {"subsample": stratified_subsample(f["train"], labels, int(config["subsample"]["n"]), seed),
                   "full": f["train"]}
        prepared = {}
        for name, arm in exp["arms"].items():
            regime = arm["regime"]
            if regime not in prepared:
                rows = regimes[regime]
                tr, iv = inner_split(rows, labels, frac, seed)
                prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
                prepared[regime] = dict(
                    X_in=prep.fit_transform(frame.iloc[tr], loader.label_column).to_numpy(),
                    X_iv=prep.transform(frame.iloc[iv], loader.label_column).to_numpy(),
                    X_ov=prep.transform(frame.iloc[f["val"]], loader.label_column).to_numpy(),
                    y_in=labels[tr], y_iv=labels[iv], y_ov=labels[f["val"]],
                    counts={"n_inner_train": len(tr), "n_inner_train_pos": int(labels[tr].sum()),
                            "n_inner_val": len(iv), "n_inner_val_pos": int(labels[iv].sum()),
                            "n_val": len(f["val"]), "n_val_pos": int(labels[f["val"]].sum()),
                            "n_fold_train": len(f["train"])})
            for eps in exp["epsilons"]:
                tasks.append({**prepared[regime], "arm_name": name, "arm": arm, "regime": regime,
                              "feature_set": arm["feature_set"], "feature_spec": exp["feature_sets"][arm["feature_set"]],
                              "epsilons": [eps], "delta": float(exp["delta"]), "seed": seed,
                              "repeat": f["repeat"], "fold": f["fold"], "source": loader.source_tag()})
    heavy = {"hybrid": 0, "vqc": 1, "mlp": 2, "logreg": 3}
    return sorted(tasks, key=lambda t: heavy.get(t["arm"]["kind"], 9))


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
    print(f"[{loader.source_tag()}] {len(folds)} folds, {len(tasks)} tasks", flush=True)
    part = run_dir / "fold_metrics.partial.csv"
    records, scores = [], {}
    started = time.time()
    for i, rows in enumerate(Parallel(n_jobs=int(config["project"]["n_jobs"]), return_as="generator_unordered")(
            delayed(run_task)(t) for t in tasks), start=1):
        for r in rows:
            scores[r["record_id"]] = r.pop("_scores")
        records.extend(rows)
        pd.DataFrame(rows).to_csv(part, mode="a", header=(i == 1), index=False)
        r = rows[0]
        print(f"[{i}/{len(tasks)} {time.time() - started:6.0f}s] {r['arm']} eps={r['target_epsilon']} "
              f"spent={r['epsilon_spent']} r{r['repeat']}f{r['fold']} auc={r['roc_auc']:.3f}", flush=True)
    results = pd.DataFrame(records)
    results.to_csv(run_dir / "fold_metrics.csv", index=False)
    part.unlink(missing_ok=True)
    np.savez_compressed(run_dir / "val_scores.npz", **{k.replace("|", "__"): v for k, v in scores.items()})
    from src.m10_exp.exp3_report import write_report

    write_report(run_dir, results, loader, config)
    print(f"wrote {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: privacy-utility curve")
    parser.add_argument("--config")
    parser.add_argument("--report-only")
    args = parser.parse_args()
    if args.report_only:
        from src.m1_data import make_loader
        from src.m10_exp.exp3_report import write_report

        run_dir = Path(args.report_only)
        config = load_config(run_dir / "config.yaml")
        loader = make_loader(config["data"], seed=int(config["project"]["seed"]))
        write_report(run_dir, pd.read_csv(run_dir / "fold_metrics.csv"), loader, config)
        return
    run(args.config)


if __name__ == "__main__":
    main()
