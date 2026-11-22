"""Experiment 5: privacy at the input stage (before quantum encoding).

Pipeline per outer fold, every transform fitted on training rows only:
  raw rows -> KAnonymizer (k-anonymity on demographic quasi-identifiers, optional)
           -> imputation + one-hot -> PCA-8, angle-scaled to [0, pi]
           -> LaplaceLDP on the encoder input of training rows (ε_input, optional)
           -> model -> scored on the outer validation fold (anonymised the same way, no LDP noise)
The input-stage budget (ε_input, pure LDP) and the training-stage budget (ε_training, Opacus
DP-SGD, only for the DP arm) are recorded separately.

Usage: python -m src.m10_exp.exp5_input_privacy --config configs/exp5_nsch2024.yaml
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
from src.m3_features.features import FeatureTransformer
from src.m5_models.classical import make_classical
from src.m5_models.quantum import QuantumKernelSVC
from src.m7_privacy.dp_training import DPClassifier
from src.m7_privacy.input_privacy import KAnonymizer, LaplaceLDP
from src.m8_eval.evaluation import binary_metrics
from src.m10_exp.common import (
    cv_folds, environment_record, holdout_split, limit_worker_threads, load_config, load_dataset,
    make_run_dir, report_header, save_splits, write_json,
)
from src.m10_exp.tuning import inner_split, scores_of, select_and_fit, stratified_subsample, youden_threshold


def eps_label(e) -> str:
    return "inf" if e is None else f"{float(e):g}"


def run_task(t: dict) -> list[dict]:
    import torch
    torch.set_num_threads(1)
    labels, seed = t["labels"], t["seed"]
    tr, iv, ov = t["tr"], t["iv"], t["ov"]
    kinfo = {"k": None, "suppression_rate_train": 0.0, "suppression_rate_val": 0.0, "k_anonymous_train": None,
             "min_class_size_train": None}
    if "X" in t:      # a ready numeric matrix (text embeddings; no quasi-identifier columns, so no k-anonymity)
        X_tr, X_iv, X_ov = t["X"][tr], t["X"][iv], t["X"][ov]
    else:
        frame = t["frame"]
        f_tr, f_iv, f_ov = frame.iloc[tr], frame.iloc[iv], frame.iloc[ov]
        if t["k"]:
            anon = KAnonymizer(k=t["k"]).fit(f_tr)
            f_tr, f_iv = anon.transform(f_tr), anon.transform(f_iv)
            f_ov = anon.transform(f_ov)
            kinfo = {"k": t["k"], "suppression_rate_train": anon.report_["suppression_rate"],
                     "suppression_rate_val": anon.last_suppressed_ / len(ov),
                     "k_anonymous_train": anon.report_["k_anonymous"], "min_class_size_train": anon.report_["min_class_size"]}
        prep = TabularPreprocessor(identifier_columns=t["metadata"])
        X_tr = prep.fit_transform(f_tr, "label").to_numpy()
        X_iv, X_ov = prep.transform(f_iv, "label").to_numpy(), prep.transform(f_ov, "label").to_numpy()
    ft = FeatureTransformer(d_prime=8, method="pca", output="angle", seed=seed).fit(X_tr, labels[tr])
    A, B, C = ft.transform(X_tr), ft.transform(X_iv), ft.transform(X_ov)
    rows = []
    for eps_in in t["eps_inputs"]:
        ldp = LaplaceLDP(epsilon=eps_in, seed=seed)
        A_p = ldp.transform(A)
        B_p = LaplaceLDP(epsilon=eps_in, seed=seed + 1).transform(B)   # inner-val rows are training-side data
        y_tr, y_iv = labels[tr], labels[iv]
        for arm in t["arms"]:
            t0 = time.perf_counter()
            eps_train = None
            if arm == "qsvm":
                best = select_and_fit(lambda **p: QuantumKernelSVC(encoding="angle", n_qubits=8, seed=seed, **p),
                                      {"C": [0.1, 1.0, 10.0]}, A_p, y_tr, B_p, y_iv)
                model, thr = best["model"], best["threshold"]
            elif arm == "dp_logreg":
                model = DPClassifier(kind="logreg", epsilon=t["eps_training"], delta=1e-5, epochs=30,
                                     batch_size=128, learning_rate=0.05, seed=seed).fit(A_p, y_tr)
                thr = youden_threshold(y_iv, model.decision_scores(B_p))
                eps_train = model.privacy_.epsilon
            else:
                best = select_and_fit(lambda **p: make_classical("logreg", y_tr, seed=seed, n_jobs=1, **p),
                                      {"C": [0.01, 0.1, 1.0, 10.0]}, A_p, y_tr, B_p, y_iv)
                model, thr = best["model"], best["threshold"]
            scores = scores_of(model, C)
            rows.append({"data_source": t["source"], "repeat": t["repeat"], "fold": t["fold"], "arm": arm,
                         "regime": t["regime"], **kinfo, "epsilon_input": eps_label(eps_in),
                         "ldp_noise_scale": ldp.noise_scale(A.shape[1]),
                         "epsilon_training": eps_train, **binary_metrics(labels[ov], scores, threshold=thr),
                         "fit_seconds": time.perf_counter() - t0, "n_train": len(tr), "n_val": len(ov)})
    return rows


def write_report(run_dir: Path, results: pd.DataFrame, loader, config: dict) -> None:
    results = results.copy()
    results["epsilon_input"] = ["inf" if str(v) in ("inf", "nan", "None") else f"{float(v):g}" for v in results["epsilon_input"]]
    results["k_anon"] = results["k"].map(lambda v: "off" if pd.isna(v) else f"k={int(v)}")
    g = results.groupby(["arm", "regime", "k_anon", "epsilon_input"], sort=False)
    s = g[["roc_auc", "balanced_accuracy", "recall"]].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["suppression_rate_train"] = g["suppression_rate_train"].mean()
    s["suppression_rate_val"] = g["suppression_rate_val"].mean()
    s["k_anonymous_all_folds"] = g["k_anonymous_train"].apply(lambda v: None if v.isna().all() else bool(v.all()))
    s["epsilon_training"] = g["epsilon_training"].mean()
    s["n_folds"] = g.size()
    s = s.reset_index()
    s.insert(0, "data_source", loader.source_tag())
    s.to_csv(run_dir / "summary.csv", index=False)
    order = ["inf"] + [eps_label(e) for e in config["experiment"]["eps_inputs"] if e is not None]
    lines = report_header("Experiment 5 — input-stage privacy (k-anonymity and local DP before encoding)", loader)
    lines += [f"{results.groupby(['repeat', 'fold']).ngroups} outer folds. ROC-AUC mean ± std. ε_input = local DP on the "
              "8 encoder-input features of training rows (Laplace, pure ε-LDP, labels not perturbed); ε_training = "
              "Opacus DP-SGD (only for dp_logreg). Validation rows are anonymised but not noised.",
              "AUC below 0.5 means the model learned no usable signal: under strong input noise plus DP-SGD the weights act "
              "as an intercept along the (uncentred, [0, π]) mean-feature direction, which anti-correlates with the true "
              "signal on clean validation rows. Treat such cells as 'no useful model', not as a meaningful ranking.", ""]
    for (arm, regime), d in s.groupby(["arm", "regime"], sort=False):
        lines += [f"### {arm} ({regime})", "", "| k-anonymity | " + " | ".join(f"ε_input = {'∞' if e == 'inf' else e}" for e in order)
                  + " | suppressed (train) |", "|" + "---|" * (len(order) + 2)]
        for kan, dd in d.groupby("k_anon", sort=False):
            dd = dd.set_index("epsilon_input")
            cells = [f"{dd.loc[e, 'roc_auc_mean']:.3f} ± {dd.loc[e, 'roc_auc_std']:.3f}" if e in dd.index else "–" for e in order]
            lines.append(f"| {kan} | " + " | ".join(cells) + f" | {dd['suppression_rate_train'].iloc[0]:.1%} |")
        lines.append("")
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
    dev, holdout = holdout_split(labels, float(config["split"]["holdout_fraction"]), seed0)
    folds = cv_folds(dev, labels, int(config["cv"]["n_splits"]), int(config["cv"]["n_repeats"]), seed0)
    save_splits(run_dir, holdout, folds)
    exp = config["experiment"]
    tasks = []
    for f in folds:
        seed = seed0 + 1000 * f["repeat"] + f["fold"]
        for regime, arms in exp["regimes"].items():
            rows = f["train"] if regime == "full" else stratified_subsample(f["train"], labels, int(config["subsample"]["n"]), seed)
            tr, iv = inner_split(rows, labels, float(config["tuning"]["inner_val_fraction"]), seed)
            for k in exp["k_values"]:
                tasks.append(dict(frame=frame, labels=labels, metadata=loader.metadata_columns, seed=seed, tr=tr, iv=iv,
                                  ov=f["val"], k=k, eps_inputs=exp["eps_inputs"], arms=arms, regime=regime,
                                  eps_training=float(exp["eps_training"]), repeat=f["repeat"], fold=f["fold"],
                                  source=loader.source_tag()))
    print(f"[{loader.source_tag()}] {len(folds)} folds, {len(tasks)} tasks", flush=True)
    records = []
    t0 = time.time()
    for i, rows in enumerate(Parallel(n_jobs=int(config["project"]["n_jobs"]), return_as="generator_unordered")(
            delayed(run_task)(t) for t in tasks), start=1):
        records.extend(rows)
        print(f"[{i}/{len(tasks)} {time.time() - t0:5.0f}s] {rows[0]['regime']} k={rows[0]['k']} r{rows[0]['repeat']}f{rows[0]['fold']}", flush=True)
    results = pd.DataFrame(records)
    results.to_csv(run_dir / "fold_metrics.csv", index=False)
    write_report(run_dir, results, loader, config)
    print(f"wrote {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 5: input-stage privacy")
    parser.add_argument("--config")
    parser.add_argument("--report-only")
    a = parser.parse_args()
    if a.report_only:
        from src.m1_data import make_loader
        d = Path(a.report_only)
        cfg = load_config(d / "config.yaml")
        write_report(d, pd.read_csv(d / "fold_metrics.csv"), make_loader(cfg["data"], seed=int(cfg["project"]["seed"])), cfg)
        return
    run(a.config)


if __name__ == "__main__":
    main()
