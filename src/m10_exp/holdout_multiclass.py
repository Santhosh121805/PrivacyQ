"""One-shot evaluation of the three-class models on the locked hold-out, under the pre-specified rule.

Pre-specified (committed before this script was ever run; ``HOLDOUT_POLICY`` in exp9_multiclass.py, the
``holdout_policy`` block of the config, and CONFORMANCE.md):
- three-class metrics on hold-out rows with a determined three-class label (4,334 rows);
- binary depression-vs-rest metrics on ALL hold-out rows (4,358), using the model's depression score for
  ROC-AUC and "predicted class = depression" for balanced accuracy and recall.

Each configuration listed in the config is trained on the development rows with a determined three-class
label (the 141 undetermined development rows are excluded, as in cross-validation), with exactly the
cross-validation protocol: inner-train / inner-val split, grid point chosen by inner-val macro one-vs-rest
ROC-AUC, VQC early stopping on inner-val. It is then scored ONCE on the hold-out. Nothing is tuned against
the hold-out, and the script refuses to run again once a result exists (lock file).

Usage: python -m src.m10_exp.holdout_multiclass --config configs/holdout_multiclass_nsch2024.yaml
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from src.m1_data.nsch import three_class_labels
from src.m2_preprocess.preprocessing import TabularPreprocessor
from src.m5_models.multiclass import CLASS_NAMES, MULTICLASS_MODELS
from src.m8_eval import multiclass_metrics
from src.m10_exp.common import environment_record, holdout_split, load_config, load_dataset, write_json
from src.m10_exp.exp2_compare import feature_matrices
from src.m10_exp.exp9_multiclass import check_holdout_policy, holdout_scoring_rows
from src.m10_exp.tuning import grid_points, inner_split, stratified_subsample

DEP = CLASS_NAMES.index("depression")


def bootstrap_ci(y, scores, stat, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        try:
            vals.append(stat(y[i], scores[i]))
        except ValueError:                       # a resample missing a class
            continue
    return np.percentile(vals, [2.5, 97.5])


def macro_ovr_auc(y, s):
    return float(np.mean([roc_auc_score(y == k, s[:, k]) for k in range(len(CLASS_NAMES))]))


def fit_selected(spec, seed, A, y_tr, B, y_iv):
    cls = MULTICLASS_MODELS[spec["model"]]
    best = None
    for params in grid_points(spec.get("grid")):
        model = cls(seed=seed, **spec.get("params", {}), **params)
        if getattr(model, "supports_eval_set", False):
            model.fit(A, y_tr, eval_set=(B, y_iv))
        else:
            model.fit(A, y_tr)
        auc = macro_ovr_auc(y_iv, model.class_scores(B))
        if best is None or auc > best[0]:
            best = (auc, model, params)
    return best


def run(config_path: str, dry_run_dir: str | None = None) -> Path:
    """``dry_run_dir``: exercise every code path WITHOUT touching the hold-out: a pseudo hold-out is carved
    from the development rows, grids are cut to their first point, the VQC trains 2 epochs, no lock is written."""
    import torch

    torch.manual_seed(0)
    cfg = load_config(config_path)
    check_holdout_policy(cfg)
    seed = int(cfg["project"]["seed"])
    loader, frame = load_dataset(cfg)
    lock = Path(cfg["project"]["output_dir"]) / f"HOLDOUT_EVALUATED_MULTICLASS_{loader.source_tag().replace(':', '_')}.lock"
    if lock.exists() and dry_run_dir is None:
        raise SystemExit(f"multi-class hold-out already evaluated ({lock.read_text().strip()}); refusing to re-score")
    labels = frame[loader.label_column].to_numpy(dtype=int)
    y3 = three_class_labels(loader)
    det = ~np.isnan(y3)
    y = np.where(det, y3, -1).astype(int)
    dev, hold = holdout_split(labels, float(cfg["split"]["holdout_fraction"]), seed)
    if dry_run_dir is not None:                               # pseudo hold-out from development rows only
        rng = np.random.default_rng(seed + 1)
        pseudo = np.sort(rng.choice(dev, size=len(dev) // 5, replace=False))
        dev, hold = np.setdiff1d(dev, pseudo), pseudo
        for spec in cfg["models"].values():
            spec["grid"] = {k: v[:1] for k, v in (spec.get("grid") or {}).items()}
            if spec["model"] == "vqc":
                spec.setdefault("params", {})["epochs"] = 2
    dev_det = dev[det[dev]]
    rows3 = holdout_scoring_rows(hold, y3, "three_class_metrics")
    rows_bin = holdout_scoring_rows(hold, y3, "binary_depression_metrics")
    if dry_run_dir is not None:
        out = Path(dry_run_dir) / f"DRYRUN-holdout-multiclass-{time.strftime('%Y%m%d-%H%M%S')}"
    else:
        out = Path(cfg["project"]["output_dir"]) / f"holdout-multiclass-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    shutil.copy(config_path, out / "config.yaml")
    write_json(out / "environment.json", environment_record(cfg))
    if dry_run_dir is None:
        lock.write_text(f"{out}\n", encoding="utf-8")
    frac = float(cfg["tuning"]["inner_val_fraction"])
    pos3 = np.searchsorted(hold, rows3)                      # positions of the scored rows inside the hold-out
    results = []
    for name, spec in cfg["models"].items():
        rows_train = stratified_subsample(dev_det, y, spec.get("subsample"), seed)
        tr, iv = inner_split(rows_train, y, frac, seed)
        prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
        X_in = prep.fit_transform(frame.iloc[tr], loader.label_column).to_numpy()
        X_iv = prep.transform(frame.iloc[iv], loader.label_column).to_numpy()
        X_ho = prep.transform(frame.iloc[hold], loader.label_column).to_numpy()
        A, B, C = feature_matrices(spec["features"], X_in, y[tr], [X_iv, X_ho], seed)
        t0 = time.perf_counter()
        inner_auc, model, params = fit_selected(spec, seed, A, y[tr], B, y[iv])
        fit_seconds = time.perf_counter() - t0
        scores, pred = model.class_scores(C), model.predict(C)
        m3 = multiclass_metrics(y[rows3], scores[pos3], pred[pos3], CLASS_NAMES)
        lo3, hi3 = bootstrap_ci(y[rows3], scores[pos3], macro_ovr_auc)
        yb, sb = labels[rows_bin], scores[:, DEP]
        pb = (pred == DEP).astype(int)
        lob, hib = bootstrap_ci(yb, sb, roc_auc_score)
        results.append({
            "model": name, "params": str(params), "inner_val_macro_ovr_auc": inner_auc,
            **{f"three_class_{k}": v for k, v in m3.items()},
            "three_class_macro_ovr_auc_ci95_low": lo3, "three_class_macro_ovr_auc_ci95_high": hi3,
            "binary_depression_roc_auc": roc_auc_score(yb, sb), "binary_depression_auc_ci95_low": lob,
            "binary_depression_auc_ci95_high": hib,
            "binary_depression_balanced_accuracy": recall_score(yb, pb, average="macro"),
            "binary_depression_recall": recall_score(yb, pb),
            "n_train_rows": len(rows_train), "n_inner_train": len(tr), "n_holdout": len(hold),
            "n_scored_three_class": len(rows3), "n_scored_binary": len(rows_bin), "n_binary_pos": int(yb.sum()),
            **{f"n_three_class_{c}": int((y[rows3] == k).sum()) for k, c in enumerate(CLASS_NAMES)},
            "fit_seconds": fit_seconds, "data_source": loader.source_tag()})
        r = results[-1]
        print(f"{name}: macro-F1 {r['three_class_macro_f1']:.3f}, macro AUC {r['three_class_macro_ovr_auc']:.3f} "
              f"[{lo3:.3f}, {hi3:.3f}], depression-vs-rest AUC {r['binary_depression_roc_auc']:.3f} [{lob:.3f}, {hib:.3f}]",
              flush=True)
    pd.DataFrame(results).to_csv(out / "holdout_multiclass_metrics.csv", index=False)
    print(f"wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot multi-class hold-out evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run-dir", help="exercise the code on a pseudo hold-out from dev rows; no lock")
    args = parser.parse_args()
    run(args.config, args.dry_run_dir)


if __name__ == "__main__":
    main()
