"""True classical ceiling for the binary Depression vs Suicidal view: every deduplicated ID row of those classes.

The Phase 4 "full" regime used every development row of each fold (~5,044), but Phase 1 only embedded the
15,000-row stratified subsample. Here the remaining deduplicated ID-bearing Depression / Suicidal rows ("extra")
are added to each fold's inner-training rows; validation rows, inner-validation rows and the quarantined block
are exactly the Phase 4 ones, so the numbers sit on the same folds.

Leakage guard: Phase 1 deduplication is re-run (deterministic), then any extra row with normalised-text match or
TF-IDF cosine >= 0.90 (the Phase 1 rule) to ANY row of the processed table (dev, hold-out, quarantine) is dropped.
The locked hold-out is never scored. Models: XGBoost and logistic regression on all 384 MiniLM dims, Phase 4 grids.

    python scripts/mhtext_binary_ceiling.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.m1_data.mhtext import KEYWORD_LEAKAGE, dedup, normalise, read_raw  # noqa: E402
from src.m10_exp.common import limit_worker_threads, load_config  # noqa: E402
from src.m10_exp.mhtext_quantum import build_tasks, run_keyed  # noqa: E402

OUT = ROOT / "results/mhtext/phase4/binary_ceiling"
CACHE = ROOT / "data/processed/mhtext/features/binary_extra"


def near_dup_mask(extra: pd.Series, table: pd.Series, threshold: float, batch: int = 1500) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, dtype=np.float32)
    vec.fit(pd.concat([extra, table]))
    E, T = vec.transform(extra), vec.transform(table)
    hit = np.zeros(E.shape[0], bool)
    for s in range(0, E.shape[0], batch):
        hit[s:s + batch] = (E[s:s + batch] @ T.T).max(axis=1).toarray().ravel() >= threshold
    return hit


def extra_rows(p1: dict, p2: dict, p4: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "pooled.npy").exists():
        return (pd.read_csv(CACHE / "rows.csv"), np.load(CACHE / "pooled.npy"),
                json.loads((CACHE / "log.json").read_text(encoding="utf-8")))
    v = p4["views"]["binary"]
    t0 = time.time()
    raw = read_raw(ROOT / p1["data"]["raw_root"]).drop(columns=list(KEYWORD_LEAKAGE))
    kept, _ = dedup(raw, threshold=float(p1["dedup"]["threshold"]))
    table = pd.read_csv(ROOT / p1["data"]["processed"])
    ids = kept[(kept.provenance == "id") & kept.status.isin([v["negative"], v["positive"]])]
    cand = ids[~ids.Unique_ID.isin(table.Unique_ID.dropna())].reset_index(drop=True)
    tnorm = table.text.astype(str).map(normalise)
    exact = cand.text.astype(str).map(normalise).isin(set(tnorm))
    near = near_dup_mask(cand.text.astype(str).map(normalise), tnorm, float(p1["dedup"]["threshold"]))
    extra = cand[~(exact.to_numpy() | near)].reset_index(drop=True)
    log = {"dedup_id_rows_binary_classes": int(len(ids)), "already_in_processed_table": int(len(ids) - len(cand)),
           "candidates": int(len(cand)), "dropped_exact_match_to_table": int(exact.sum()),
           "dropped_near_dup_to_table": int((near & ~exact.to_numpy()).sum()), "extra_rows": int(len(extra)),
           "extra_positive": int((extra.status == v["positive"]).sum()), "dedup_and_check_seconds": round(time.time() - t0, 1)}
    print(json.dumps(log), flush=True)
    from sentence_transformers import SentenceTransformer
    from src.m3_features.text import pooled_embeddings
    e = p2["embedding"]
    st = SentenceTransformer(e["model"], revision=e["revision"], device="cpu")
    st.max_seq_length = int(e["max_tokens"])
    t1 = time.time()
    emb = pooled_embeddings(st, extra.text.astype(str).tolist(), batch_size=int(e["batch_size"]))
    log["embedding_seconds"] = round(time.time() - t1, 1)
    extra[["Unique_ID", "status"]].to_csv(CACHE / "rows.csv", index=False)
    np.save(CACHE / "pooled.npy", emb)
    (CACHE / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return extra[["Unique_ID", "status"]], emb, log


def main() -> None:
    p1, p2 = load_config(ROOT / "configs/mhtext_phase1.yaml"), load_config(ROOT / "configs/mhtext_phase2.yaml")
    p4 = load_config(ROOT / "configs/mhtext_phase4.yaml")
    rows, emb, log = extra_rows(p1, p2, p4)
    y_extra = (rows.status == p4["views"]["binary"]["positive"]).astype(int).to_numpy()
    tasks, meta = build_tasks(p4, "binary")
    tasks = [t for t in tasks if t["part"] == "compare" and t["regime"] == "full" and t["feature_set"] == "all384"
             and t["models"][0] in ("xgboost", "logreg")]
    for t in tasks:
        t["X_in"] = np.vstack([t["X_in"], emb])
        t["y_in"] = np.concatenate([t["y_in"], y_extra])
        t["regime"] = "all_rows"
        t["counts"] = {**t["counts"], "n_inner_train": len(t["y_in"]), "n_inner_train_pos": int(t["y_in"].sum()),
                       "n_train_rows": len(t["y_in"]) + t["counts"]["n_inner_val"]}
    OUT.mkdir(parents=True, exist_ok=True)
    limit_worker_threads()
    from joblib import Parallel, delayed
    t0 = time.time()
    res = Parallel(n_jobs=int(p4["project"]["n_jobs"]))(delayed(run_keyed)(t) for t in tasks)
    df = pd.DataFrame([r for _, rs in res for r in rs])
    df.to_csv(OUT / "fold_metrics.csv", index=False)
    log.update(tasks=len(tasks), wall_seconds=round(time.time() - t0, 1),
               mean_training_rows_per_fold=float(df.n_train_rows.mean()))
    (OUT / "run.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(df.groupby("model")[["roc_auc", "balanced_accuracy", "recall", "accuracy", "extra_roc_auc"]].agg(["mean", "std"]).round(3))
    print(json.dumps(log))


if __name__ == "__main__":
    main()
