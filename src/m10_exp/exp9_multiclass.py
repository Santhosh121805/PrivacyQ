"""Experiment 9: three-class task (none / anxiety only / depression with or without anxiety).

Labels: K2Q32B (depression, currently) and K2Q33B (anxiety, currently), both the "currently has"
variant, combined by ``src.m1_data.nsch.three_class_labels``. Merging classes 0 and 1 gives the
binary depression_current target exactly, so results stay comparable with Experiments 1-8.

Population and splits: the same 21,789 rows as every other experiment. The locked hold-out and
the 15 outer-fold assignments are generated exactly as before (stratified on the binary label, same
seed) and checked against the Experiment 2 run. Of the 165 depression-negative rows whose anxiety
status is undetermined, those in the development set (141) are excluded from each fold's training
and validation rows (never imputed or counted as "no"); the 24 in the hold-out are untouched, and
the hold-out is not used by this experiment.

Protocol (as Experiment 2): inner-train / inner-val split of the fold's training rows; grid
point chosen by inner-val macro one-vs-rest ROC-AUC; the chosen inner-train model scored once on
the outer validation fold. All models class-weighted ("balanced"). Headline metrics: macro-F1 and
macro one-vs-rest ROC-AUC; balanced accuracy and per-class recall also reported; plain accuracy
is not reported. The QSVM provides no privacy.

Usage: python -m src.m10_exp.exp9_multiclass --config configs/exp9_multiclass_nsch2024.yaml
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.m1_data.nsch import three_class_labels
from src.m2_preprocess.preprocessing import TabularPreprocessor
from src.m5_models.multiclass import CLASS_NAMES, MULTICLASS_MODELS
from src.m8_eval import multiclass_metrics
from src.m8_eval.stats import add_holm, corrected_resampled_ttest, wilcoxon_p
from src.m10_exp.common import (
    cv_folds, environment_record, holdout_split, limit_worker_threads, load_config, load_dataset,
    make_run_dir, save_splits, write_json,
)
from src.m10_exp.exp2_compare import feature_matrices
from src.m10_exp.tuning import grid_points, inner_split, stratified_subsample

METRICS = ["macro_f1", "macro_ovr_auc", "balanced_accuracy"]

# Pre-specified hold-out rule (committed before any multi-class hold-out scoring; also in the Exp 9 config and
# CONFORMANCE.md). The 24 hold-out rows whose anxiety status is undetermined are all depression-negative, so:
#   - three-class metrics are scored on hold-out rows with a determined three-class label only;
#   - binary depression metrics (class "depression" vs rest) are scored on ALL hold-out rows.
HOLDOUT_POLICY = {"three_class_metrics": "exclude_undetermined", "binary_depression_metrics": "all_rows"}


def holdout_scoring_rows(holdout: np.ndarray, y3: np.ndarray, family: str) -> np.ndarray:
    """Hold-out row positions to score for a metric family under ``HOLDOUT_POLICY``."""
    holdout = np.asarray(holdout)
    if family == "three_class_metrics":
        return holdout[~np.isnan(y3[holdout])]
    if family == "binary_depression_metrics":
        return holdout
    raise ValueError(f"unknown metric family {family!r}; choose from {sorted(HOLDOUT_POLICY)}")


def check_holdout_policy(config: dict) -> None:
    """Refuse to run if the config's hold-out policy differs from the pre-specified one."""
    declared = config.get("holdout_policy")
    if declared != HOLDOUT_POLICY:
        raise ValueError(f"holdout_policy in config ({declared}) differs from the pre-specified {HOLDOUT_POLICY}")


def _macro_auc(model, X, y) -> float:
    return multiclass_metrics(y, model.class_scores(X), model.predict(X), CLASS_NAMES)["macro_ovr_auc"]


def run_task(task: dict) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    seed = task["seed"]
    A, B, C = feature_matrices(task["feature_spec"], task["X_in"], task["y_in"], [task["X_iv"], task["X_ov"]], seed)
    rows = []
    for name in task["models"]:
        spec = task["models_spec"][name]
        cls = MULTICLASS_MODELS[spec.get("class", name)]
        fixed = {k: v for k, v in spec.items() if k not in ("grid", "class", "feature_sets", "regimes")}
        best, t_tune = None, time.perf_counter()
        for params in grid_points(spec.get("grid")):
            t0 = time.perf_counter()
            model = cls(seed=seed, **fixed, **params)
            if getattr(model, "supports_eval_set", False):      # VQC: early stopping on inner-val, as in Exp 2
                model.fit(A, task["y_in"], eval_set=(B, task["y_iv"]))
            else:
                model.fit(A, task["y_in"])
            fit_s = time.perf_counter() - t0
            auc = _macro_auc(model, B, task["y_iv"])
            if best is None or auc > best["auc"]:
                best = {"model": model, "params": params, "auc": auc, "fit_seconds": fit_s}
        model = best["model"]
        scores, pred = model.class_scores(C), model.predict(C)
        rows.append({
            "model": name, "regime": task["regime"], "feature_set": task["feature_set"],
            "repeat": task["repeat"], "fold": task["fold"], "n_features_in": A.shape[1],
            **multiclass_metrics(task["y_ov"], scores, pred, CLASS_NAMES),
            "inner_val_macro_ovr_auc": best["auc"], "best_params": str(best["params"]),
            "fit_seconds": best["fit_seconds"], "tuning_seconds": time.perf_counter() - t_tune,
            "n_params": model.n_parameters(), **task["counts"],
        })
    return rows


def build_tasks(config, loader, frame, folds, y3):
    exp = config["experiment"]
    frac = float(config["tuning"]["inner_val_fraction"])
    keep = ~np.isnan(y3)
    y = np.where(keep, y3, -1).astype(int)
    tasks = []
    for f in folds:
        seed = int(config["project"]["seed"]) + 1000 * f["repeat"] + f["fold"]
        train, val = f["train"][keep[f["train"]]], f["val"][keep[f["val"]]]
        regimes = {"subsample": stratified_subsample(train, y, int(config["subsample"]["n"]), seed), "full": train}
        for regime, rows in regimes.items():
            if regime not in exp["regimes"]:
                continue
            tr, iv = inner_split(rows, y, frac, seed)
            prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
            X_in = prep.fit_transform(frame.iloc[tr], loader.label_column).to_numpy()
            X_iv = prep.transform(frame.iloc[iv], loader.label_column).to_numpy()
            X_ov = prep.transform(frame.iloc[val], loader.label_column).to_numpy()
            counts = {"n_inner_train": len(tr), "n_inner_val": len(iv), "n_val": len(val),
                      "n_fold_train": len(train),
                      **{f"n_val_{c}": int((y[val] == k).sum()) for k, c in enumerate(CLASS_NAMES)},
                      "n_removed_undetermined_train": int((~keep[f["train"]]).sum()),
                      "n_removed_undetermined_val": int((~keep[f["val"]]).sum())}
            common = dict(repeat=f["repeat"], fold=f["fold"], seed=seed, regime=regime, X_in=X_in, y_in=y[tr],
                          X_iv=X_iv, y_iv=y[iv], X_ov=X_ov, y_ov=y[val], counts=counts, models_spec=exp["models"])
            for fs_name, spec in exp["feature_sets"].items():
                names = [m for m, s in exp["models"].items()
                         if fs_name in s["feature_sets"] and regime in s["regimes"]]
                for m in names:
                    tasks.append({**common, "feature_set": fs_name, "feature_spec": spec, "models": [m]})
    order = {"vqc": -1, "svm": 0, "qsvm": 1, "xgboost": 2, "logreg": 3}
    return sorted(tasks, key=lambda t: (t["regime"] != "full", order.get(t["models"][0], 9)))


def write_report(run_dir: Path, results: pd.DataFrame, config: dict, population: dict) -> None:
    keys = ["model", "regime", "feature_set"]
    cols = METRICS + [f"recall_{c}" for c in CLASS_NAMES] + ["auc_depression_vs_rest"]
    summary = results.groupby(keys)[cols].agg(["mean", "std"])
    lines = ["# Experiment 9: three-class task (none / anxiety only / depression)", "",
             "Labels: K2Q32B depression (currently) and K2Q33B anxiety (currently). Class 2 = depression with or "
             "without anxiety; merging classes 0 and 1 gives the binary `depression_current` target exactly.", "",
             f"Population: {population['n_rows']:,} adolescents aged 12–17 with a determinable depression label "
             f"(the same rows as Experiments 1–8). {population['n_undetermined']} depression-negative rows have "
             f"undetermined anxiety (K2Q33A missing, or K2Q33A = yes with K2Q33B missing): not imputed and not "
             f"counted as 'no'. Class counts of the rows with a determined class: "
             + ", ".join(f"{c} {population['counts'][c]:,} ({population['shares'][c]:.1f}%)" for c in CLASS_NAMES)
             + ".", "",
             f"Hold-out: the locked 20% hold-out is the same set of rows as in every other experiment; "
             f"{population['n_undetermined_in_holdout']} of the {population['n_undetermined']} undetermined rows "
             f"fall in it. The hold-out is not used in this experiment at all (no training, tuning or scoring), "
             f"so nothing is removed from it here. The other "
             f"{population['n_undetermined'] - population['n_undetermined_in_holdout']} undetermined rows are in "
             f"the development set and are excluded from each fold's training and validation rows.", "",
             (f"Folds: the outer-fold assignments are identical to `{config['check_folds_against']}` (checked), "
              f"but the {population['n_undetermined'] - population['n_undetermined_in_holdout']} undetermined "
              f"development rows are excluded from training and validation, so each fold has slightly fewer rows "
              f"than in Experiment 2. "
              if config.get("check_folds_against") else "Folds NOT checked against Experiment 2. ")
             + "All models class-weighted. Plain accuracy is not reported.", "",
             "| Model | Regime | Features | Macro-F1 | Macro OvR AUC | Balanced acc. | Recall none | Recall anxiety only | Recall depression | Depression-vs-rest AUC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for idx, r in summary.iterrows():
        cells = [f"{r[(c, 'mean')]:.3f} ± {r[(c, 'std')]:.3f}" for c in cols]
        lines.append("| " + " | ".join(idx) + " | " + " | ".join(cells) + " |")
    # paired tests: the quantum model vs each classical model on the identical subsample folds
    pt = config.get("paired_test", {"model": "qsvm", "feature_set": "pca8"})
    qn, qfs = pt["model"], pt["feature_set"]
    sub = results[(results.regime == "subsample") & (results.feature_set == qfs)]
    q = sub[sub.model == qn].set_index(["repeat", "fold"]).sort_index()
    rows = []
    for other in (sorted(set(sub.model) - {qn}) if len(q) >= 2 else []):
        o = sub[sub.model == other].set_index(["repeat", "fold"]).sort_index()
        idx = q.index.intersection(o.index)
        ratio = float(q.loc[idx, "n_val"].mean() / q.loc[idx, "n_fold_train"].mean())
        for m in METRICS:
            x, z = q.loc[idx, m].to_numpy(), o.loc[idx, m].to_numpy()
            rows.append({"b": other, "metric": m, "diff": float((x - z).mean()), "wins": int((x > z).sum()),
                         "k": len(idx), "p_nb_t": corrected_resampled_ttest(x, z, ratio)["p"],
                         "wilcoxon_p": wilcoxon_p(x, z)})
    if rows:
        tests = add_holm(pd.DataFrame(rows), ["metric"])
        tests.to_csv(run_dir / "paired_tests.csv", index=False)
        lines += ["", f"Paired tests, {qn} vs classical on the identical subsample {qfs} folds, Holm within metric. "
                  "Primary: Nadeau–Bengio corrected resampled t (significant = Holm p < 0.05). Sensitivity check: "
                  "Wilcoxon signed-rank, Holm-adjusted the same way.", "",
                  f"| {qn} vs | Metric | {qn} − other | {qn} wins | NB-t p | NB-t Holm p (primary) | Significant | "
                  "Wilcoxon p | Wilcoxon Holm p (sensitivity) |", "|---|---|---|---|---|---|---|---|---|"]
        for _, t in tests.iterrows():
            lines.append(f"| {t.b} | {t.metric} | {t['diff']:+.3f} | {t.wins}/{t.k} | {t.p_nb_t:.3g} | "
                         f"{t.p_nb_t_holm:.3g} | {'yes' if t.significant else 'no'} | {t.wilcoxon_p:.3g} | "
                         f"{t.wilcoxon_p_holm:.3g} |")
    (run_dir / "multiclass_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def run(config_path: str) -> Path:
    limit_worker_threads()
    config = load_config(config_path)
    check_holdout_policy(config)
    seed = int(config["project"]["seed"])
    loader, frame = load_dataset(config)
    labels = frame[loader.label_column].to_numpy(dtype=int)
    y3 = three_class_labels(loader)
    if len(y3) != len(labels):
        raise ValueError("three-class labels are not aligned with the loaded frame")
    det = ~np.isnan(y3)
    if not (np.array_equal((y3[det] == 2).astype(int), labels[det]) and (labels[~det] == 0).all()):
        raise ValueError("merging classes 0 and 1 does not reproduce depression_current")
    run_dir = make_run_dir(config, loader)
    shutil.copy(config_path, run_dir / "config.yaml")
    write_json(run_dir / "environment.json", environment_record(config))
    dev, holdout = holdout_split(labels, float(config["split"]["holdout_fraction"]), seed)
    folds = cv_folds(dev, labels, int(config["cv"]["n_splits"]), int(config["cv"]["n_repeats"]), seed)
    save_splits(run_dir, holdout, folds)
    ref = config.get("check_folds_against")
    if ref and not pd.read_csv(Path(ref) / "cv_val_rows.csv").equals(pd.read_csv(run_dir / "cv_val_rows.csv")):
        raise ValueError(f"folds differ from {ref}; refusing to run")
    counts = {c: int((y3[det] == k).sum()) for k, c in enumerate(CLASS_NAMES)}
    population = {"n_rows": len(labels), "n_undetermined": int((~det).sum()), "counts": counts,
                  "shares": {c: 100 * v / det.sum() for c, v in counts.items()},
                  "n_undetermined_in_holdout": int((~det[holdout]).sum())}
    write_json(run_dir / "population.json", population)
    tasks = build_tasks(config, loader, frame, folds, y3)
    print(f"{len(folds)} folds, {len(tasks)} tasks; population {population}", flush=True)
    records, started = [], time.time()
    for i, rows in enumerate(Parallel(n_jobs=int(config["project"]["n_jobs"]), return_as="generator_unordered")(
            delayed(run_task)(t) for t in tasks), start=1):
        records.extend(rows)
        r = rows[0]
        print(f"[{i}/{len(tasks)} {time.time() - started:6.0f}s] {r['model']}/{r['regime']}/{r['feature_set']} "
              f"r{r['repeat']}f{r['fold']} macroF1={r['macro_f1']:.3f} macroAUC={r['macro_ovr_auc']:.3f}", flush=True)
    results = pd.DataFrame(records)
    results.to_csv(run_dir / "fold_metrics.csv", index=False)
    write_report(run_dir, results, config, population)
    print(f"wrote {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
