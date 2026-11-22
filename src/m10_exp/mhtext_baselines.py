"""Phase 3 (box 10): classical, frozen-MiniLM-embedding (BERT family) and sequence baselines on the Mental Health Text dataset.

For each view (4-class; 3-class with Normal dropped) every model runs on the SAME repeated stratified folds of the
development rows. Inside each fold: inner split of the fold's training rows for hyperparameters / early stopping,
scalers fitted on inner-train rows only, then the fold's validation rows are scored. The locked hold-out is not
touched. Each fold's model also scores the quarantined block (different provenance; its Normal posts are longer),
a sensitivity check on the length shortcut.

Metrics: accuracy beside balanced accuracy, macro precision / recall / F1 / one-vs-rest ROC-AUC, and the same per
class. Every model is compared with the length-only baseline by the Nadeau-Bengio corrected resampled t-test,
Holm-adjusted within view x metric.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.m1_data.mhtext import CLASSES_3, CLASSES_4, SHIPPED_FEATURES, assert_no_blocked
from src.m10_exp.common import cv_folds, environment_record, limit_worker_threads, load_config

ROOT = Path(__file__).resolve().parents[2]
SEQ = ("lstm", "cnn")


# ------------------------------------------------------------------ data
def token_ids(cfg: dict, texts: list[str]) -> np.ndarray:
    d = cfg["data"]
    path = ROOT / d["features_dir"] / f"token_ids_{d['max_tokens']}.npz"
    if path.exists():
        z = np.load(path)
        flat, off = z["flat"], z["offsets"]
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(d["tokenizer"], revision=d["tokenizer_revision"])
        ids = tok(texts, truncation=True, max_length=int(d["max_tokens"]), add_special_tokens=True)["input_ids"]
        off = np.cumsum([0] + [len(i) for i in ids])
        flat = np.concatenate([np.asarray(i, dtype=np.int32) for i in ids])
        np.savez(path, flat=flat, offsets=off)
    seqs = np.empty(len(off) - 1, dtype=object)
    for i in range(len(seqs)):
        seqs[i] = flat[off[i]:off[i + 1]]
    return seqs


def feature_matrix(fdir: Path, table: pd.DataFrame, name: str) -> tuple[np.ndarray, list[str]]:
    pooled = np.load(fdir / "pooled.npy", mmap_mode="r")
    sent = np.load(fdir / "sentence_mean_max.npy", mmap_mode="r")
    emo = np.load(fdir / "emotion.npy", mmap_mode="r")
    shipped = [c for c in SHIPPED_FEATURES if c in table.columns]
    if name == "length":                                         # from the text itself, so every row has it
        t = table.text.astype(str)
        return np.log1p(np.c_[t.str.split().str.len().to_numpy(float), t.str.len().to_numpy(float)]), ["log_words", "log_chars"]
    if name == "bert_pooled":
        return np.asarray(pooled), [f"emb_pooled_{i}" for i in range(pooled.shape[1])]
    if name == "bert_sentence":
        return np.asarray(sent), [f"emb_sent_{i}" for i in range(sent.shape[1])]
    if name == "all_pooled":
        cols = ([f"emb_pooled_{i}" for i in range(pooled.shape[1])] + [f"emb_sent_{i}" for i in range(sent.shape[1])]
                + [f"emotion_{i}" for i in range(emo.shape[1])] + shipped)
        return np.concatenate([pooled, sent, emo, table[shipped].to_numpy(float)], axis=1), cols
    raise ValueError(name)


def make_model(kind: str, seed: int, **p):
    from src.m5_models.multiclass import MulticlassLogReg, MulticlassSVM, MulticlassXGB
    from src.m5_models.text_models import MulticlassMLP, SequenceClassifier

    return {"logreg": lambda: MulticlassLogReg(seed=seed, **p), "xgboost": lambda: MulticlassXGB(seed=seed, **p),
            "svm": lambda: MulticlassSVM(seed=seed, **p), "mlp": lambda: MulticlassMLP(seed=seed, **p),
            "lstm": lambda: SequenceClassifier(kind="lstm", seed=seed, **p),
            "cnn": lambda: SequenceClassifier(kind="cnn", seed=seed, **p)}[kind]()


# ------------------------------------------------------------------ one task
def run_task(task: dict) -> list[dict]:
    import torch
    from sklearn.preprocessing import StandardScaler

    from src.m8_eval.evaluation import multiclass_full_metrics
    from src.m10_exp.tuning import grid_points, inner_split

    torch.set_num_threads(1)
    cfg, spec, seed = task["cfg"], task["spec"], task["seed"]
    table = pd.read_csv(ROOT / cfg["data"]["processed"])
    classes = CLASSES_4 if task["view"] == 4 else CLASSES_3
    names = tuple(classes)
    y_all = table.status.map(classes).fillna(-1).astype(int).to_numpy()   # -1 = out of view, never indexed
    tr, iv = inner_split(task["train"], y_all, float(cfg["tuning"]["inner_val_fraction"]), seed)
    va, qu = task["val"], task["quarantine"]
    if spec["features"] == "tokens":
        X = token_ids(cfg, table.text.astype(str).tolist())
        prep = lambda idx: X[idx]  # noqa: E731
    else:
        F, cols = feature_matrix(ROOT / cfg["data"]["features_dir"], table, spec["features"])
        assert_no_blocked(cols)
        sc = StandardScaler().fit(F[tr])                         # fitted on inner-train rows only
        prep = lambda idx: np.nan_to_num(sc.transform(F[idx]))  # noqa: E731
    Xtr, Xiv, Xva, Xqu = prep(tr), prep(iv), prep(va), prep(qu)
    best, t0 = None, time.perf_counter()
    for p in grid_points(spec.get("grid")):
        m = make_model(spec["class"], seed, **p)
        if getattr(m, "supports_eval_set", False):
            m.fit(Xtr, y_all[tr], eval_set=(Xiv, y_all[iv]))
        else:
            m.fit(Xtr, y_all[tr])
        auc = multiclass_full_metrics(y_all[iv], m.class_scores(Xiv), m.predict(Xiv), names)["macro_auc"]
        if best is None or auc > best[0]:
            best = (auc, p, m)
    secs = time.perf_counter() - t0
    auc_iv, params, model = best
    base = {"view": f"{task['view']}-class", "model": task["name"], "features": spec["features"], "repeat": task["repeat"],
            "fold": task["fold"], "params": json.dumps(params), "inner_val_macro_auc": auc_iv, "fit_seconds": secs,
            "best_epoch": getattr(model, "best_epoch_", None), "n_inner_train": len(tr), "n_inner_val": len(iv)}
    rows = []
    for part, idx, Xp in (("cv", va, Xva), ("quarantine", qu, Xqu)):
        rows.append({**base, "set": part, "n_scored": len(idx),
                     **multiclass_full_metrics(y_all[idx], model.class_scores(Xp), model.predict(Xp), names)})
    return rows


def task_key(t: dict) -> str:
    return f"{t['view']}class_{t['name']}_r{t['repeat']}f{t['fold']}"


def run_task_keyed(t: dict) -> tuple[str, list[dict]]:
    return task_key(t), run_task(t)


# ------------------------------------------------------------------ runner
def build_tasks(cfg: dict) -> tuple[list[dict], dict]:
    table = pd.read_csv(ROOT / cfg["data"]["processed"])
    seed = int(cfg["project"]["seed"])
    tasks, meta = [], {}
    for view in cfg["views"]:
        classes = CLASSES_4 if view == 4 else CLASSES_3
        inview = table.status.isin(classes).to_numpy()
        dev = np.where(inview & (table.split == "dev").to_numpy())[0]
        quar = np.where(inview & (table.split == "quarantine").to_numpy())[0]
        y = table.status.map(classes).fillna(-1).astype(int).to_numpy()
        folds = cv_folds(dev, y, int(cfg["cv"]["n_splits"]), int(cfg["cv"]["n_repeats"]), seed)
        meta[view] = {"dev_rows": int(len(dev)), "quarantine_rows": int(len(quar)), "folds": len(folds),
                      "test_train_ratio": float(np.mean([len(f["val"]) / len(f["train"]) for f in folds]))}
        for f in folds:
            for name, spec in cfg["models"].items():
                tasks.append({"cfg": cfg, "view": int(view), "name": name, "spec": spec, "repeat": f["repeat"],
                              "fold": f["fold"], "train": f["train"], "val": f["val"], "quarantine": quar,
                              "seed": seed + 1000 * f["repeat"] + f["fold"]})
    cost = {"lstm": 0, "cnn": 1, "svm": 2, "xgboost": 3, "mlp": 4}
    tasks.sort(key=lambda t: cost.get(t["spec"]["class"], 9))       # heaviest first for load balance
    return tasks, meta


def significance(fold: pd.DataFrame, ref: str, meta: dict) -> pd.DataFrame:
    from src.m8_eval.stats import corrected_resampled_ttest, holm

    out = []
    cv = fold[fold.set == "cv"]
    for view, part in cv.groupby("view"):
        ratio = meta[int(view[0])]["test_train_ratio"]
        r = part[part.model == ref].sort_values(["repeat", "fold"])
        for metric in ("macro_auc", "macro_f1"):
            rows = []
            for m, g in part[part.model != ref].groupby("model"):
                g = g.sort_values(["repeat", "fold"])
                t = corrected_resampled_ttest(g[metric].to_numpy(), r[metric].to_numpy(), ratio)
                rows.append({"view": view, "metric": metric, "model": m, "reference": ref,
                             "model_mean": g[metric].mean(), "reference_mean": r[metric].mean(), **t})
            df = pd.DataFrame(rows)
            df["p_holm"] = holm(df.p.to_numpy())
            df["significant"] = df.p_holm < 0.05
            out.append(df)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    cfg = load_config(ROOT / p.parse_args().config)
    out = ROOT / cfg["project"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    limit_worker_threads()
    table = pd.read_csv(ROOT / cfg["data"]["processed"])
    token_ids(cfg, table.text.astype(str).tolist())                  # cache once, before workers start
    tasks, meta = build_tasks(cfg)
    print(f"{len(tasks)} tasks ({len(cfg['models'])} models x {sum(m['folds'] for m in meta.values())} view-folds); "
          f"meta {json.dumps(meta)}", flush=True)
    from joblib import Parallel, delayed

    # checkpoint: one file per finished task; a restart skips every task already on disk
    ckpt = out / "tasks"
    ckpt.mkdir(exist_ok=True)
    key = task_key
    todo = [t for t in tasks if not (ckpt / f"{key(t)}.json").exists()]
    print(f"{len(tasks) - len(todo)} tasks already done (checkpoints), {len(todo)} to run", flush=True)
    t0 = time.time()
    for i, (k, res) in enumerate(Parallel(n_jobs=int(cfg["project"]["n_jobs"]), return_as="generator_unordered")(
            delayed(run_task_keyed)(t) for t in todo), 1):
        tmp = ckpt / f"{k}.json.tmp"
        tmp.write_text(json.dumps(res, default=float), encoding="utf-8")
        tmp.replace(ckpt / f"{k}.json")                          # atomic: no half-written checkpoint
        r = res[0]
        print(f"[{i}/{len(todo)} {time.time() - t0:6.0f}s] {r['view']} {r['model']:<22} r{r['repeat']}f{r['fold']} "
              f"macro AUC {r['macro_auc']:.3f} F1 {r['macro_f1']:.3f} ({r['fit_seconds']:.0f}s)", flush=True)
    rows = [row for t in tasks for row in json.loads((ckpt / f"{key(t)}.json").read_text(encoding="utf-8"))]
    fold = pd.DataFrame(rows).sort_values(["view", "set", "model", "repeat", "fold"])
    fold.to_csv(out / "fold_metrics.csv", index=False)
    metrics = [c for c in fold.columns if c in ("accuracy", "balanced_accuracy") or c.startswith(("macro_", "precision_",
                                                                                                    "recall_", "f1_", "auc_"))]
    summ = fold.groupby(["view", "set", "model", "features"])[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(out / "summary.csv", index=False)
    sig = significance(fold, cfg["reference_model"], meta)
    sig.to_csv(out / "tests_vs_length_only.csv", index=False)
    (out / "run.json").write_text(json.dumps({"meta": meta, "seconds": round(time.time() - t0, 1),
                                              "environment": environment_record(cfg)}, indent=2, default=str))
    from src.m10_exp.mhtext_report import write_phase3_report

    write_phase3_report(out, cfg)


if __name__ == "__main__":
    main()
