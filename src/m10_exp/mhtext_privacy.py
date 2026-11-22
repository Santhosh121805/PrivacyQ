"""Phase 5: privacy on the Mental Health Text dataset (boxes 4 and 12), with the NSCH method held fixed.

Parts (each resumable; finished tasks are written to <output>/<part>/tasks/ at once):
  reid  re-identification risk of the released posts (fragment prosecutor risk, before/after scrubbing),
        self-disclosed quasi-identifiers, and fragment -> embedding linkage with / without local DP
  mia   membership inference      = exp6_membership.run_task, unchanged (binary view, MiniLM embedding)
  dp    DP-SGD privacy-utility    = exp3_privacy.run_task, unchanged
  ldp   input-stage local DP      = exp5_input_privacy.run_task, unchanged (k-anonymity not applicable: no
        quasi-identifier columns exist in free text, see the reid part for what replaces it)
Privacy parameters are read from the NSCH configs (configs/exp3|5|6_nsch2024.yaml). No NSCH data is loaded.
The locked hold-out is not used.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.m10_exp.common import cv_folds, environment_record, limit_worker_threads, load_config
from src.m10_exp.mhtext_quantum import view_data
from src.m10_exp.tuning import inner_split, stratified_subsample

ROOT = Path(__file__).resolve().parents[2]


def _all384(d):
    """NSCH feature-set name all95 (every survey column) -> all384 (every embedding dimension)."""
    if isinstance(d, dict):
        return {("all384" if k == "all95" else k): _all384(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_all384(v) for v in d]
    return "all384" if d == "all95" else d


def folds_for(p4: dict, dev, y, n_repeats: int):
    return cv_folds(dev, y, int(p4["cv"]["n_splits"]), n_repeats, int(p4["project"]["seed"]))


# ---------------------------------------------------------------- tasks for the three model-based parts
def tasks_mia(cfg, p4, y, X, dev, source):
    nm = load_config(ROOT / cfg["nsch_method_configs"]["mia"])
    seed0, n_sub = int(p4["project"]["seed"]), int(p4["subsample"]["n"])
    frac = float(p4["tuning"]["inner_val_fraction"])
    out = []
    for f in folds_for(p4, dev, y, int(nm["cv"]["n_repeats"])):
        seed = seed0 + 1000 * f["repeat"] + f["fold"]
        tr_sub, _ = inner_split(stratified_subsample(f["train"], y, n_sub, seed), y, frac, seed)
        tr_full, _ = inner_split(f["train"], y, frac, seed)
        for arm in nm["experiment"]["arms"]:
            out.append(dict(X=X, labels=y, seed=seed, arm=arm, tr=tr_full if arm == "dp_logreg_full" else tr_sub,
                            ov=f["val"], epsilons=nm["experiment"]["epsilons"], repeat=f["repeat"], fold=f["fold"],
                            source=source, key=f"{arm}_r{f['repeat']}f{f['fold']}"))
    order = {"dp_vqc": 0, "mlp_overfit": 1, "dp_logreg_full": 2, "qsvm": 3}
    return sorted(out, key=lambda t: order[t["arm"]]), nm


def tasks_dp(cfg, p4, y, X, dev, source):
    nm = _all384(load_config(ROOT / cfg["nsch_method_configs"]["dp"]))
    exp, seed0 = nm["experiment"], int(p4["project"]["seed"])
    frac, n_sub = float(nm["tuning"]["inner_val_fraction"]), int(nm["subsample"]["n"])
    out = []
    for f in folds_for(p4, dev, y, int(nm["cv"]["n_repeats"])):
        seed = seed0 + 1000 * f["repeat"] + f["fold"]
        regimes = {"subsample": stratified_subsample(f["train"], y, n_sub, seed), "full": f["train"]}
        prepared = {}
        for name, arm in exp["arms"].items():
            regime = arm["regime"]
            if regime not in prepared:
                tr, iv = inner_split(regimes[regime], y, frac, seed)
                prepared[regime] = dict(X_in=X[tr], X_iv=X[iv], X_ov=X[f["val"]], y_in=y[tr], y_iv=y[iv], y_ov=y[f["val"]],
                                        counts={"n_inner_train": len(tr), "n_inner_train_pos": int(y[tr].sum()),
                                                "n_inner_val": len(iv), "n_inner_val_pos": int(y[iv].sum()),
                                                "n_val": len(f["val"]), "n_val_pos": int(y[f["val"]].sum()),
                                                "n_fold_train": len(f["train"])})
            for eps in exp["epsilons"]:
                e = "inf" if eps is None else f"{float(eps):g}"
                out.append({**prepared[regime], "arm_name": name, "arm": arm, "regime": regime,
                            "feature_set": arm["feature_set"], "feature_spec": exp["feature_sets"][arm["feature_set"]],
                            "epsilons": [eps], "delta": float(exp["delta"]), "seed": seed, "repeat": f["repeat"],
                            "fold": f["fold"], "source": source, "key": f"{name}_eps{e}_r{f['repeat']}f{f['fold']}"})
    heavy = {"hybrid": 0, "vqc": 1, "mlp": 2, "logreg": 3}
    return sorted(out, key=lambda t: heavy.get(t["arm"]["kind"], 9)), nm


def tasks_ldp(cfg, p4, y, X, dev, source):
    nm = load_config(ROOT / cfg["nsch_method_configs"]["ldp"])
    exp, seed0 = nm["experiment"], int(p4["project"]["seed"])
    out = []
    for f in folds_for(p4, dev, y, int(nm["cv"]["n_repeats"])):
        seed = seed0 + 1000 * f["repeat"] + f["fold"]
        for regime, arms in exp["regimes"].items():
            rows = f["train"] if regime == "full" else stratified_subsample(f["train"], y, int(nm["subsample"]["n"]), seed)
            tr, iv = inner_split(rows, y, float(nm["tuning"]["inner_val_fraction"]), seed)
            out.append(dict(X=X, labels=y, seed=seed, tr=tr, iv=iv, ov=f["val"], k=None, eps_inputs=exp["eps_inputs"],
                            arms=arms, regime=regime, eps_training=float(exp["eps_training"]), repeat=f["repeat"],
                            fold=f["fold"], source=source, n_fold_train=len(f["train"]),
                            key=f"{regime}_r{f['repeat']}f{f['fold']}"))
    return sorted(out, key=lambda t: t["regime"] != "subsample"), nm


def run_keyed(part: str, task: dict):
    if part == "mia":
        from src.m10_exp.exp6_membership import run_task
    elif part == "dp":
        from src.m10_exp.exp3_privacy import run_task
    else:
        from src.m10_exp.exp5_input_privacy import run_task
    rows = run_task(task)
    for r in rows:
        r.pop("_scores", None)
        if part == "ldp":
            r["n_fold_train"] = task["n_fold_train"]
    return task["key"], rows


def run_models(part: str, cfg: dict, out: Path) -> None:
    p4 = load_config(ROOT / cfg["phase4_config"])
    view = cfg["view"]
    _, y, feats, dev, _ = view_data(p4, view)
    source = f"REAL:MHText-{view}"
    tasks, nm = {"mia": tasks_mia, "dp": tasks_dp, "ldp": tasks_ldp}[part](cfg, p4, y, feats["embedding"], dev, source)
    ckpt = out / part / "tasks"
    ckpt.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (ckpt / f"{t['key']}.json").exists()]
    print(f"{part}: {len(tasks)} tasks, {len(tasks) - len(todo)} already done (source {source})", flush=True)
    from joblib import Parallel, delayed
    t0 = time.time()
    for i, (key, rows) in enumerate(Parallel(n_jobs=int(cfg["project"]["n_jobs"]), return_as="generator_unordered")(
            delayed(run_keyed)(part, t) for t in todo), 1):
        tmp = ckpt / f"{key}.json.tmp"
        tmp.write_text(json.dumps(rows, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
        tmp.replace(ckpt / f"{key}.json")
        print(f"[{i}/{len(todo)} {time.time() - t0:6.0f}s] {key}", flush=True)
    rows = [r for t in tasks for r in json.loads((ckpt / f"{t['key']}.json").read_text(encoding="utf-8"))]
    pd.DataFrame(rows).to_csv(out / part / "fold_metrics.csv", index=False)
    (out / part / "run.json").write_text(json.dumps({
        "part": part, "tasks": len(tasks), "wall_seconds_this_launch": round(time.time() - t0, 1),
        "nsch_method_config": cfg["nsch_method_configs"][part], "method_experiment_section": nm["experiment"],
        "dev_rows": int(len(dev)), "environment": environment_record(cfg)}, indent=2, default=str), encoding="utf-8")
    print(f"{part} done: wall {time.time() - t0:.0f}s this launch", flush=True)


# ---------------------------------------------------------------- re-identification
def run_reid(cfg: dict, out: Path) -> None:
    from src.m9_audit import text_reidentification as tr
    from src.m3_features.features import FeatureTransformer
    from src.m7_privacy.input_privacy import LaplaceLDP

    p4 = load_config(ROOT / cfg["phase4_config"])
    r = cfg["reid"]
    seed = int(cfg["project"]["seed"])
    table = pd.read_csv(ROOT / p4["data"]["processed"])
    emb = np.load(ROOT / p4["data"]["features_dir"] / "pooled.npy")
    d = out / "reid"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    risk, disc = [], []
    for corpus in r["corpora"]:
        idx = np.where(table.split.isin(corpus.split("+")).to_numpy())[0]
        texts = table.text.iloc[idx].astype(str).tolist()
        for version, tx in (("as released", texts), ("scrubbed (URLs, e-mails, handles, communities, numbers removed)",
                                                     [tr.scrub(t) for t in texts])):
            toks = tr.tokens(tx)
            for w in r["span_words"]:
                risk.append({"corpus": corpus, "text": version, "attacker_knows": f"a {w}-word fragment",
                             **tr.fragment_risk(toks, w, seed, int(r["k"]))})
            for n in r["unique_ngram_n"]:
                risk.append({"corpus": corpus, "text": version, "attacker_knows": f"any {n}-word span",
                             "n_rows": len(toks), "uniqueness": tr.any_unique_ngram_rate(toks, n), "span_words": n})
            print(f"reid {corpus} / {version[:11]}: {time.time() - t0:.0f}s", flush=True)
        disc.append({"corpus": corpus, **tr.self_disclosures(texts)})
    from src.m9_audit.disclosure import suppress_shares
    fr_df = pd.DataFrame(risk)
    fr_df, _ = suppress_shares(fr_df, "n_rows", ["uniqueness", "k_violation_rate"])
    fr_df.to_csv(d / "fragment_risk.csv", index=False)
    sd = pd.DataFrame(disc)
    sd, _ = suppress_shares(sd, "n_rows", [c for c in sd.columns if c not in ("corpus", "n_rows")])
    sd.to_csv(d / "self_disclosures.csv", index=False)

    # linkage: a fragment of a dev post, embedded with the public MiniLM, matched against released dev vectors
    p2 = load_config(ROOT / "configs/mhtext_phase2.yaml")["embedding"]
    from sentence_transformers import SentenceTransformer
    from src.m3_features.text import pooled_embeddings
    st = SentenceTransformer(p2["model"], revision=p2["revision"], device="cpu")
    st.max_seq_length = int(p2["max_tokens"])
    dev = np.where((table.split == "dev").to_numpy())[0]
    toks = tr.tokens(table.text.iloc[dev].astype(str).tolist())
    released = emb[dev]
    y4 = table.status.iloc[dev].to_numpy()
    ft = FeatureTransformer(d_prime=8, method="pca", output="angle", seed=seed).fit(released, y4)
    enc = ft.transform(released)                                  # the 8 encoder inputs the circuit receives
    link = []
    for w in r["linkage_span_words"]:
        q = pooled_embeddings(st, [" ".join(s) for s in tr.choose_spans(toks, w, seed)], batch_size=int(p2["batch_size"]))
        link.append({"released": "MiniLM embedding (384 dims)", "epsilon_input": "inf", "fragment_words": w,
                     **tr.linkage(q, released, "cosine")})
        qe = np.clip(ft.transform(q), 0.0, np.pi)
        for eps in r["linkage_eps_inputs"]:
            rel = LaplaceLDP(epsilon=eps, seed=seed).transform(enc)
            link.append({"released": "PCA-8 encoder input", "epsilon_input": "inf" if eps is None else f"{float(eps):g}",
                         "fragment_words": w, **tr.linkage(qe, rel, "euclidean")})
        print(f"linkage {w} words: {time.time() - t0:.0f}s", flush=True)
    lk_df, _ = suppress_shares(pd.DataFrame(link), "n_queries", ["top1", "top5", "top10"])
    lk_df.to_csv(d / "linkage.csv", index=False)
    (d / "run.json").write_text(json.dumps({"wall_seconds": round(time.time() - t0, 1), "config": r,
                                            "environment": environment_record(cfg)}, indent=2, default=str), encoding="utf-8")
    print(f"reid done: {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--part", required=True, choices=["reid", "mia", "dp", "ldp"])
    a = p.parse_args()
    cfg = load_config(ROOT / a.config)
    out = ROOT / cfg["project"]["output_dir"]
    limit_worker_threads()
    run_reid(cfg, out) if a.part == "reid" else run_models(a.part, cfg, out)


if __name__ == "__main__":
    main()
