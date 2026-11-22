"""Experiment 6: membership inference risk at each privacy level.

For each outer fold and arm, the model is trained exactly as in Experiments 2-3 on its training
rows ("members"); the same number of rows is drawn from the outer validation fold, which the model
never saw ("non-members"). A loss-threshold attack tries to tell them apart.

Arms (all on real NSCH 2024 data):
  dp_logreg_full  LogReg, all 95 features, full training rows, DP-SGD at each ε   (Exp 3 classical)
  dp_vqc          VQC re-uploading, 4 qubits, 1.2k rows, DP-SGD at each ε        (Exp 3 quantum)
  qsvm            QSVM angle, 8 qubits, 1.2k rows, no DP (DP does not apply)
  mlp_overfit     MLP, all 95 features, 1.2k rows, no DP, no early stopping, 200 epochs:
                  a deliberately overfitting POSITIVE CONTROL showing the attack detects leakage

Usage: python -m src.m10_exp.exp6_membership --config configs/exp6_nsch2024.yaml
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
from src.m5_models.classical import TorchMLPClassifier
from src.m5_models.quantum import QuantumKernelSVC
from src.m7_privacy.dp_training import DPClassifier
from src.m9_audit.membership import loss_threshold_attack, margin_loss, per_example_loss
from src.m10_exp.common import (
    cv_folds, environment_record, holdout_split, limit_worker_threads, load_config, load_dataset,
    make_run_dir, report_header, write_json,
)
from src.m10_exp.tuning import inner_split, stratified_subsample


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def run_task(t: dict) -> list[dict]:
    import torch
    torch.set_num_threads(1)
    labels, seed, arm = t["labels"], t["seed"], t["arm"]
    rng = np.random.default_rng(seed)
    members = t["tr"]
    nonmembers = rng.choice(t["ov"], size=min(len(members), len(t["ov"])), replace=False)
    if "X" in t:      # a ready numeric matrix (text embeddings); the NSCH path below never passes one
        X_m, X_n = t["X"][members], t["X"][nonmembers]
    else:
        frame = t["frame"]
        prep = TabularPreprocessor(identifier_columns=t["metadata"])
        X_m = prep.fit_transform(frame.iloc[members], "label").to_numpy()
        X_n = prep.transform(frame.iloc[nonmembers], "label").to_numpy()
    y_m, y_n = labels[members], labels[nonmembers]
    if arm in ("dp_logreg_full", "mlp_overfit"):
        sc = StandardScaler().fit(X_m)
        A, B = sc.transform(X_m), sc.transform(X_n)
    else:
        n = 4 if arm == "dp_vqc" else 8
        ft = FeatureTransformer(d_prime=n, method="pca", output="angle", seed=seed).fit(X_m, y_m)
        A, B = ft.transform(X_m), ft.transform(X_n)
    out = []
    for eps in (t["epsilons"] if arm.startswith("dp_") else [None]):
        t0 = time.perf_counter()
        eps_spent = None
        if arm == "dp_logreg_full":
            m = DPClassifier(kind="logreg", epsilon=eps, delta=1e-5, epochs=30, batch_size=256, learning_rate=0.05,
                             seed=seed).fit(A, y_m)
            lm, ln = per_example_loss(y_m, sigmoid(m.decision_scores(A))), per_example_loss(y_n, sigmoid(m.decision_scores(B)))
            eps_spent = m.privacy_.epsilon
        elif arm == "dp_vqc":
            m = DPClassifier(kind="vqc", epsilon=eps, delta=1e-5, epochs=30, batch_size=128, learning_rate=0.05, seed=seed,
                             model_params={"encoding": "reupload", "n_qubits": 4, "n_layers": 2}).fit(A, y_m)
            lm, ln = per_example_loss(y_m, sigmoid(m.decision_scores(A))), per_example_loss(y_n, sigmoid(m.decision_scores(B)))
            eps_spent = m.privacy_.epsilon
        elif arm == "qsvm":
            m = QuantumKernelSVC(encoding="angle", n_qubits=8, C=0.1, seed=seed).fit(A, y_m)
            lm, ln = margin_loss(y_m, m.decision_scores(A)), margin_loss(y_n, m.decision_scores(B))
        else:  # mlp_overfit: positive control
            m = TorchMLPClassifier(hidden=(256, 128), epochs=200, batch_size=64, weight_decay=0.0, seed=seed).fit(A, y_m)
            lm, ln = per_example_loss(y_m, m.predict_proba(A)[:, 1]), per_example_loss(y_n, m.predict_proba(B)[:, 1])
        res = loss_threshold_attack(lm, ln)
        out.append({"data_source": t["source"], "repeat": t["repeat"], "fold": t["fold"], "arm": arm,
                    "target_epsilon": "inf" if eps is None else f"{float(eps):g}", "epsilon_spent": eps_spent,
                    **res, "fit_seconds": time.perf_counter() - t0})
    return out


def write_report(run_dir: Path, results: pd.DataFrame, loader) -> None:
    results = results.copy()
    results["target_epsilon"] = [("inf" if str(v) in ("inf", "nan") else f"{float(v):g}") for v in results.target_epsilon]
    g = results.groupby(["arm", "target_epsilon"], sort=False)
    s = g[["attack_auc", "tpr_at_1pct_fpr", "advantage"]].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["epsilon_spent"] = g["epsilon_spent"].mean()
    s["n_members"] = g["n_members"].first()
    s["n_folds"] = g.size()
    s = s.reset_index()
    s.insert(0, "data_source", loader.source_tag())
    order = {"0.5": 0, "1": 1, "2": 2, "4": 3, "8": 4, "inf": 5}
    s = s.sort_values(["arm", "target_epsilon"], key=lambda c: c.map(order) if c.name == "target_epsilon" else c)
    s.to_csv(run_dir / "summary.csv", index=False)
    lines = report_header("Experiment 6 — membership inference risk", loader)
    lines += [f"{results.groupby(['repeat', 'fold']).ngroups} outer folds. Loss-threshold attack on equal numbers of members "
              "(training rows) and non-members (unseen validation rows). Attack AUC 0.5 = no membership signal. "
              "Theory: ε-DP bounds the membership advantage by e^ε − 1 (meaningful only for small ε).", "",
              "| Arm | ε | Attack AUC | TPR @ 1% FPR | Advantage | ε spent | Members |", "|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r.arm} | {'∞' if r.target_epsilon == 'inf' else r.target_epsilon} | {r.attack_auc_mean:.3f} ± {r.attack_auc_std:.3f} | "
                     f"{r.tpr_at_1pct_fpr_mean:.3f} | {r.advantage_mean:.3f} | {'' if pd.isna(r.epsilon_spent) else f'{r.epsilon_spent:.2f}'} | {int(r.n_members)} |")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str) -> Path:
    limit_worker_threads()
    config = load_config(config_path)
    seed0 = int(config["project"]["seed"])
    loader, frame = load_dataset(config)
    labels = frame[loader.label_column].to_numpy(dtype=int)
    run_dir = make_run_dir(config, loader)
    shutil.copy(config_path, run_dir / "config.yaml")
    write_json(run_dir / "environment.json", environment_record(config))
    dev, _ = holdout_split(labels, 0.2, seed0)
    folds = cv_folds(dev, labels, int(config["cv"]["n_splits"]), int(config["cv"]["n_repeats"]), seed0)
    exp = config["experiment"]
    tasks = []
    for f in folds:
        seed = seed0 + 1000 * f["repeat"] + f["fold"]
        sub = stratified_subsample(f["train"], labels, 1500, seed)
        tr_sub, _ = inner_split(sub, labels, 0.2, seed)
        tr_full, _ = inner_split(f["train"], labels, 0.2, seed)
        for arm in exp["arms"]:
            tasks.append(dict(frame=frame, labels=labels, metadata=loader.metadata_columns, seed=seed, arm=arm,
                              tr=tr_full if arm == "dp_logreg_full" else tr_sub, ov=f["val"],
                              epsilons=exp["epsilons"], repeat=f["repeat"], fold=f["fold"], source=loader.source_tag()))
    order = {"dp_vqc": 0, "mlp_overfit": 1, "dp_logreg_full": 2, "qsvm": 3}
    tasks.sort(key=lambda t: order[t["arm"]])
    print(f"[{loader.source_tag()}] {len(folds)} folds, {len(tasks)} tasks", flush=True)
    records, t0 = [], time.time()
    for i, rows in enumerate(Parallel(n_jobs=int(config["project"]["n_jobs"]), return_as="generator_unordered")(
            delayed(run_task)(t) for t in tasks), start=1):
        records.extend(rows)
        print(f"[{i}/{len(tasks)} {time.time() - t0:5.0f}s] {rows[0]['arm']} r{rows[0]['repeat']}f{rows[0]['fold']} "
              + ", ".join(f"eps{r['target_epsilon']}: {r['attack_auc']:.3f}" for r in rows), flush=True)
    results = pd.DataFrame(records)
    results.to_csv(run_dir / "fold_metrics.csv", index=False)
    write_report(run_dir, results, loader)
    print(f"wrote {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 6: membership inference")
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
