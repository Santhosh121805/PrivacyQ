"""Experiment 4: federated learning + DP-SGD.

The inner-train rows of each outer fold are split across N simulated clients
("schools"):
  iid      - stratified random split (same label ratio in every client)
  by_state - non-IID: whole U.S. states (NSCH ``state_fips`` metadata) are
             assigned to clients, so clients differ in population and prevalence
Training: FedAvg for ``rounds`` rounds x ``local_epochs`` local epochs, every
client running Opacus DP-SGD at the target epsilon (or no DP at epsilon = inf).
Per-round global-model ROC-AUC on the outer fold is logged for convergence
plots only (never used for any choice); the final model is the last round.
Centralised DP and centralised non-private references come from the
Experiment 3 run on the same folds (``reference_run``).

Usage: python -m src.m10_exp.exp4_federated --config configs/exp4_nsch2024.yaml
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
from src.m8_eval.evaluation import binary_metrics
from src.m10_exp.common import (
    cv_folds, environment_record, holdout_split, limit_worker_threads, load_config,
    load_dataset, make_run_dir, save_splits, write_json,
)
from src.m10_exp.exp2_compare import feature_matrices
from src.m10_exp.exp3_privacy import eps_label
from src.m10_exp.tuning import inner_split, stratified_subsample, youden_threshold


def run_task(task: dict) -> dict:
    import torch
    from sklearn.metrics import roc_auc_score

    from src.m6_federated.fedavg import fedavg_train, group_partition, stratified_iid_partition
    from src.m5_models.quantum_dp import prepare_inputs

    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    arm, exp = task["arm"], task["exp"]
    A, B, C = feature_matrices(task["feature_spec"], task["X_in"], task["y_in"], [task["X_iv"], task["X_ov"]], task["seed"])
    params = arm.get("model_params", {})

    def prepare(X):
        return prepare_inputs(arm["kind"], params.get("encoding"), params.get("n_qubits"), X)

    def scores(module, X):
        module.eval()
        with torch.no_grad():
            return module(torch.as_tensor(prepare(X), dtype=torch.float64)).reshape(-1).numpy()

    def evaluate(module):
        return {"val_auc": float(roc_auc_score(task["y_ov"], scores(module, C)))}

    y = task["y_in"]
    if task["partition"] == "iid":
        parts = stratified_iid_partition(y, task["n_clients"], task["seed"])
    else:
        parts = group_partition(task["groups_in"], task["n_clients"], task["seed"])
    t0 = time.perf_counter()
    out = fedavg_train(arm["kind"], params, [(A[p], y[p]) for p in parts], rounds=exp["rounds"],
                       local_epochs=exp["local_epochs"], learning_rate=arm["learning_rate"],
                       sample_rate=arm["batch_size"] / len(y), epsilon=task["epsilon"],
                       delta=float(exp["delta"]), max_grad_norm=arm.get("max_grad_norm", 1.0),
                       seed=task["seed"], prepare=prepare, evaluate=evaluate,
                       secure_aggregation=bool(exp.get("secure_aggregation", False)))
    seconds = time.perf_counter() - t0
    threshold = youden_threshold(task["y_iv"], scores(out["module"], B))
    final = scores(out["module"], C)
    hist = out["history"].assign(arm=task["arm_name"], n_clients=task["n_clients"], partition=task["partition"],
                                 target_epsilon=eps_label(task["epsilon"]), repeat=task["repeat"], fold=task["fold"])
    sizes = np.array(out["client_sizes"])
    pos = np.array([int(y[p].sum()) for p in parts])
    return {"row": {
        "data_source": task["source"], "repeat": task["repeat"], "fold": task["fold"],
        "arm": task["arm_name"], "kind": arm["kind"], "family": arm["family"], "regime": arm["regime"],
        "feature_set": arm["feature_set"], "setting": "federated", "n_clients": task["n_clients"],
        "partition": task["partition"], "target_epsilon": eps_label(task["epsilon"]),
        "epsilon_spent": out["epsilon_spent_max"], "delta": float(exp["delta"]) if task["epsilon"] else None,
        "noise_multiplier_mean": (float(np.mean(out["noise_multipliers"])) if task["epsilon"] else None),
        **binary_metrics(task["y_ov"], final, threshold=threshold), "threshold": threshold,
        "rounds": exp["rounds"], "local_epochs": exp["local_epochs"], "n_params": out["n_params"],
        "secure_aggregation": out["secure_aggregation"], "secagg_setup_seconds": out["secagg_setup_seconds"],
        "bytes_per_round": int(out["history"]["bytes"].iloc[0]),
        "total_bytes": int(out["history"]["cumulative_bytes"].iloc[-1]),
        "client_size_min": int(sizes.min()), "client_size_max": int(sizes.max()),
        "client_pos_min": int(pos.min()), "client_pos_max": int(pos.max()),
        "client_prevalence_min": float((pos / sizes).min()), "client_prevalence_max": float((pos / sizes).max()),
        "fit_seconds": seconds, "n_val": len(task["y_ov"]), "n_val_pos": int(task["y_ov"].sum()),
        "n_inner_train": len(y), "n_fold_train": task["n_fold_train"],
    }, "history": hist}


def build_tasks(config, loader, frame, folds, labels):
    exp = config["experiment"]
    frac = float(config["tuning"]["inner_val_fraction"])
    groups = frame[loader.group_column].to_numpy()
    tasks = []
    for f in folds:
        seed = int(config["project"]["seed"]) + 1000 * f["repeat"] + f["fold"]
        regimes = {"subsample": stratified_subsample(f["train"], labels, int(config["subsample"]["n"]), seed),
                   "full": f["train"]}
        prepared = {}
        for name, arm in exp["arms"].items():
            regime = arm["regime"]
            if regime not in prepared:
                tr, iv = inner_split(regimes[regime], labels, frac, seed)
                prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
                prepared[regime] = dict(
                    X_in=prep.fit_transform(frame.iloc[tr], loader.label_column).to_numpy(),
                    X_iv=prep.transform(frame.iloc[iv], loader.label_column).to_numpy(),
                    X_ov=prep.transform(frame.iloc[f["val"]], loader.label_column).to_numpy(),
                    y_in=labels[tr], y_iv=labels[iv], y_ov=labels[f["val"]], groups_in=groups[tr],
                    n_fold_train=len(f["train"]))
            for n_clients in exp["clients"]:
                for partition in exp["partitions"]:
                    for eps in exp["epsilons"]:
                        tasks.append({**prepared[regime], "arm_name": name, "arm": arm, "exp": exp,
                                      "feature_spec": exp["feature_sets"][arm["feature_set"]],
                                      "n_clients": n_clients, "partition": partition, "epsilon": eps,
                                      "seed": seed, "repeat": f["repeat"], "fold": f["fold"],
                                      "source": loader.source_tag()})
    heavy = {"hybrid": 0, "vqc": 1, "mlp": 2, "logreg": 3}
    return sorted(tasks, key=lambda t: (heavy.get(t["arm"]["kind"], 9), -t["n_clients"]))


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
    rows, histories = [], []
    started = time.time()
    for i, res in enumerate(Parallel(n_jobs=int(config["project"]["n_jobs"]), return_as="generator_unordered")(
            delayed(run_task)(t) for t in tasks), start=1):
        rows.append(res["row"])
        histories.append(res["history"])
        pd.DataFrame([res["row"]]).to_csv(part, mode="a", header=(i == 1), index=False)
        r = res["row"]
        print(f"[{i}/{len(tasks)} {time.time() - started:6.0f}s] {r['arm']} N={r['n_clients']} {r['partition']} "
              f"eps={r['target_epsilon']} spent={r['epsilon_spent']} r{r['repeat']}f{r['fold']} auc={r['roc_auc']:.3f}",
              flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "fold_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(run_dir / "round_history.csv", index=False)
    part.unlink(missing_ok=True)
    from src.m10_exp.exp4_report import write_report

    write_report(run_dir, results, pd.read_csv(run_dir / "round_history.csv"), loader, config)
    print(f"wrote {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: federated + DP")
    parser.add_argument("--config")
    parser.add_argument("--report-only")
    args = parser.parse_args()
    if args.report_only:
        from src.m1_data import make_loader
        from src.m10_exp.exp4_report import write_report

        run_dir = Path(args.report_only)
        config = load_config(run_dir / "config.yaml")
        loader = make_loader(config["data"], seed=int(config["project"]["seed"]))
        write_report(run_dir, pd.read_csv(run_dir / "fold_metrics.csv"),
                     pd.read_csv(run_dir / "round_history.csv"), loader, config)
        return
    run(args.config)


if __name__ == "__main__":
    main()
