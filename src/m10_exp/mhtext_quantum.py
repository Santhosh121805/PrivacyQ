"""Phase 4: quantum models on the Mental Health Text dataset with the NSCH method held fixed.

The task functions are the NSCH ones, unchanged: ``exp1_encodings.run_task`` (encoding x qubit sweep) and
``exp2_compare.run_task`` (quantum vs classical on the same rows and inputs). Only the input differs: the frozen
MiniLM sentence-encoder (BERT family) embedding of each post (384 dims) instead of the 95 survey columns, reduced
inside each fold by the same PCA / mutual-information paths, on the same 1,500-row subsample (1,200 / 300).
Each fold's model also scores the quarantined block (``extra_*`` columns). The locked hold-out is not used.

Every finished task is written to <output>/<view>/tasks/ at once; a restart skips finished tasks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.m10_exp.common import cv_folds, environment_record, limit_worker_threads, load_config
from src.m10_exp.tuning import inner_split, stratified_subsample

ROOT = Path(__file__).resolve().parents[2]


def view_data(cfg: dict, view: str):
    table = pd.read_csv(ROOT / cfg["data"]["processed"])
    v = cfg["views"][view]
    rows = table.status.isin([v["negative"], v["positive"]]).to_numpy()
    y = (table.status == v["positive"]).astype(int).to_numpy()
    emb = np.load(ROOT / cfg["data"]["features_dir"] / "pooled.npy")
    t = table.text.astype(str)
    length = np.log1p(np.c_[t.str.split().str.len().to_numpy(float), t.str.len().to_numpy(float)])
    dev = np.where(rows & (table.split == "dev").to_numpy())[0]
    quar = np.where(rows & (table.split == "quarantine").to_numpy())[0]
    return table, y, {"embedding": emb, "length": length}, dev, quar


def build_tasks(cfg: dict, view: str) -> tuple[list[dict], dict]:
    table, y, X, dev, quar = view_data(cfg, view)
    seed0 = int(cfg["project"]["seed"])
    folds = cv_folds(dev, y, int(cfg["cv"]["n_splits"]), int(cfg["cv"]["n_repeats"]), seed0)
    frac, n_sub = float(cfg["tuning"]["inner_val_fraction"]), int(cfg["subsample"]["n"])
    sw, cp = cfg["sweep"], cfg["compare"]
    source = f"REAL:MHText-{view}"
    tasks = []
    for f in folds:
        seed = seed0 + 1000 * f["repeat"] + f["fold"]                     # the NSCH per-fold seed
        sub = stratified_subsample(f["train"], y, n_sub, seed)
        regimes = {"subsample": inner_split(sub, y, frac, seed), "full": inner_split(f["train"], y, frac, seed)}
        for regime, (tr, iv) in regimes.items():
            counts = {"n_train_rows": len(tr) + len(iv), "n_inner_train": len(tr), "n_inner_train_pos": int(y[tr].sum()),
                      "n_inner_val": len(iv), "n_inner_val_pos": int(y[iv].sum()), "n_val": len(f["val"]),
                      "n_val_pos": int(y[f["val"]].sum()), "n_extra": len(quar), "n_extra_pos": int(y[quar].sum())}

            def arrays(src):
                A = X[src]
                return dict(X_in=A[tr], y_in=y[tr], X_iv=A[iv], y_iv=y[iv], X_ov=A[f["val"]], y_ov=y[f["val"]],
                            X_extra=A[quar], y_extra=y[quar])
            common = dict(source=source, repeat=f["repeat"], fold=f["fold"], seed=seed, counts=counts)
            # part 1: encoding x qubit sweep (exp1), 1,200-row regime, first `repeats` repeats only
            if regime == "subsample" and f["repeat"] < int(sw["repeats"]):
                base = {**common, **arrays("embedding"), "qubits": sw["qubits"], "grids": sw["grids"],
                        "base_params": sw["base_params"], "part": "sweep"}
                for model in sw["models"]:
                    for method in sw["feature_methods"]:
                        for enc in (["none"] if model == "logreg" else sw["encodings"]):
                            tasks.append({**base, "model": model, "encoding": enc, "method": method,
                                          "key": f"sweep_{model}_{enc}_{method}_r{f['repeat']}f{f['fold']}"})
            # part 2: quantum vs classical on the same rows and inputs (exp2); full regime = classical ceiling
            if regime not in cp["regimes"]:
                continue
            for fs, spec in cp["feature_sets"].items():
                src = spec.get("source", "embedding")
                base = {**common, **arrays(src), "regime": regime, "feature_set": fs, "feature_spec": spec,
                        "grids": cp["grids"], "base_params": cp["base_params"], "quantum": cp["quantum"], "part": "compare"}
                models = cp["length_models"] if src == "length" else list(cp["classical_models"])
                if regime == "subsample" and src != "length":
                    models += [q for q, s in cp["quantum"].items() if s["feature_set"] == fs]
                for m in models:
                    tasks.append({**base, "models": [m], "key": f"compare_{regime}_{fs}_{m}_r{f['repeat']}f{f['fold']}"})
    cost = {"vqc": 0, "hybrid": 1, "mlp": 2, "svm": 3, "xgboost": 4, "qsvm": 5}
    # headline comparison first (so a time cap can only cut the sweep), heaviest first within each part
    tasks.sort(key=lambda t: (t["part"] != "compare", cost.get(t.get("model") or t["models"][0], 9)))
    meta = {"view": view, "dev_rows": int(len(dev)), "dev_positive": int(y[dev].sum()), "quarantine_rows": int(len(quar)),
            "quarantine_positive": int(y[quar].sum()), "folds": len(folds),
            "test_train_ratio": float(np.mean([len(f["val"]) / len(f["train"]) for f in folds]))}
    return tasks, meta


def run_keyed(task: dict) -> tuple[str, list[dict]]:
    if task["part"] == "sweep":
        from src.m10_exp.exp1_encodings import run_task
    else:
        from src.m10_exp.exp2_compare import run_task
    rows = run_task(task)
    for r in rows:
        r.pop("_scores", None)
        r["part"] = task["part"]
    return task["key"], rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--view", required=True)
    a = p.parse_args()
    cfg = load_config(ROOT / a.config)
    out = ROOT / cfg["project"]["output_dir"] / a.view
    ckpt = out / "tasks"
    ckpt.mkdir(parents=True, exist_ok=True)
    limit_worker_threads()
    tasks, meta = build_tasks(cfg, a.view)
    todo = [t for t in tasks if not (ckpt / f"{t['key']}.json").exists()]
    print(f"{a.view}: {len(tasks)} tasks, {len(tasks) - len(todo)} already done; meta {json.dumps(meta)}", flush=True)
    from joblib import Parallel, delayed

    t0 = time.time()
    for i, (key, rows) in enumerate(Parallel(n_jobs=int(cfg["project"]["n_jobs"]), return_as="generator_unordered")(
            delayed(run_keyed)(t) for t in todo), 1):
        tmp = ckpt / f"{key}.json.tmp"
        tmp.write_text(json.dumps(rows, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
        tmp.replace(ckpt / f"{key}.json")
        r = rows[0]
        print(f"[{i}/{len(todo)} {time.time() - t0:7.0f}s] {key}: AUC {r['roc_auc']:.3f} "
              f"(quarantine {r.get('extra_roc_auc', float('nan')):.3f}) tuning {r['tuning_seconds']:.0f}s", flush=True)
    rows = [r for t in tasks for r in json.loads((ckpt / f"{t['key']}.json").read_text(encoding="utf-8"))]
    df = pd.DataFrame(rows)
    df[df.part == "sweep"].to_csv(out / "sweep_fold_metrics.csv", index=False)
    df[df.part == "compare"].to_csv(out / "compare_fold_metrics.csv", index=False)
    (out / "run.json").write_text(json.dumps({"meta": meta, "wall_seconds_this_launch": round(time.time() - t0, 1),
                                              "tasks": len(tasks), "environment": environment_record(cfg)},
                                             indent=2, default=str), encoding="utf-8")
    print(f"done: {len(tasks)} tasks; wall {time.time() - t0:.0f}s this launch", flush=True)


if __name__ == "__main__":
    main()
