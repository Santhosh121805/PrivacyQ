"""Phase 2 (box 3): text features for the Mental Health Text dataset, from frozen pretrained models.

    python scripts/build_mhtext_features.py --config configs/mhtext_phase2.yaml

Stages (each cached in data/processed/mhtext/features/, so a rerun resumes):
  1. token lengths under the embedding model's own tokenizer -> confirms or refutes the 256-token truncation
  2. pooled sentence embedding of each whole text (truncated at max_tokens)
  3. per-sentence embeddings, pooled by mean and by max over each text's sentences (no text truncation)
  4. emotion probabilities from a frozen emotion classifier (truncated at max_tokens)
Shipped features (polarity, subjectivity and the other numeric columns) come with the dataset; they are not
computed here. Mutual-information rankings use development rows only, for the 4-class and the 3-class view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.m1_data.mhtext import CLASSES_3, CLASSES_4, SHIPPED_FEATURES, assert_no_blocked  # noqa: E402
from src.m3_features.text import (  # noqa: E402
    emotion_probabilities, mi_ranking, pooled_embeddings, sentence_pooled_embeddings, token_lengths)
from src.m10_exp.common import environment_record, load_config  # noqa: E402

T0 = time.time()


def say(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def cached(path: Path, fn, timings: dict, key: str):
    if path.exists():
        say(f"{key}: cached ({path.name})")
        return np.load(path)
    t = time.time()
    arr = fn()
    np.save(path, arr)
    timings[key] = round(time.time() - t, 1)
    say(f"{key}: {arr.shape} in {timings[key]:.0f}s")
    return arr


def stats(x: np.ndarray, cap: int) -> dict:
    return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "max": int(x.max()), "share_over_cap": float((x > cap).mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    cfg = load_config(ROOT / p.parse_args().config)
    seed = int(cfg["project"]["seed"])
    out = ROOT / cfg["project"]["output_dir"]
    fdir = ROOT / cfg["data"]["features_dir"]
    out.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    proc = ROOT / cfg["data"]["processed"]
    table = pd.read_csv(proc)
    texts = table.text.astype(str).tolist()
    proc_sha = sha256(proc)
    stamp = fdir / "source.json"
    if stamp.exists() and json.loads(stamp.read_text())["processed_sha256"] != proc_sha:
        raise SystemExit("cached features were built from a different processed table; delete the features folder")
    stamp.write_text(json.dumps({"processed_sha256": proc_sha, "rows": len(table)}), encoding="utf-8")
    table[["Unique_ID", "split", "status"]].to_csv(fdir / "rows.csv", index_label="row")
    timings: dict = {}

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    e, m = cfg["embedding"], cfg["emotion"]
    st = SentenceTransformer(e["model"], revision=e["revision"], device="cpu")
    st.max_seq_length = int(e["max_tokens"])
    say(f"embedding model {e['model']}@{e['revision'][:8]}, max_seq_length {st.max_seq_length}, "
        f"dim {st.get_sentence_embedding_dimension()}, torch threads {torch.get_num_threads()}")

    tl = cached(fdir / "token_lengths.npy", lambda: token_lengths(st.tokenizer, texts), timings, "token_lengths")
    pooled = cached(fdir / "pooled.npy", lambda: pooled_embeddings(st, texts, int(e["batch_size"])), timings, "pooled")

    def sent():
        mean, mx, n = sentence_pooled_embeddings(st, texts)
        np.save(fdir / "sent_count.npy", n)
        return np.concatenate([mean, mx], axis=1)
    sentpool = cached(fdir / "sentence_mean_max.npy", sent, timings, "sentence_mean_max")
    n_sent = np.load(fdir / "sent_count.npy")
    d = pooled.shape[1]

    etok = AutoTokenizer.from_pretrained(m["model"], revision=m["revision"])
    emod = AutoModelForSequenceClassification.from_pretrained(m["model"], revision=m["revision"]).eval()
    emo_names = [emod.config.id2label[i] for i in range(emod.config.num_labels)]
    emo = cached(fdir / "emotion.npy",
                 lambda: emotion_probabilities(etok, emod, texts, int(m["max_tokens"]), int(m["batch_size"]),
                                               progress=lambda a, b: say(f"  emotion {a:,}/{b:,}")),
                 timings, "emotion")

    # feature table (all rows, processed-table order); blocked columns can never be part of it
    shipped = [c for c in SHIPPED_FEATURES if c in table.columns]
    cols = ([f"emb_pooled_{i:03d}" for i in range(d)] + [f"emb_sent_mean_{i:03d}" for i in range(d)]
            + [f"emb_sent_max_{i:03d}" for i in range(d)] + [f"emotion_{n}" for n in emo_names] + shipped)
    labels = ([f"Pooled embedding dim {i} ({e['model'].split('/')[-1]}, whole text, {e['max_tokens']} tokens)" for i in range(d)]
              + [f"Per-sentence embedding dim {i}, mean over sentences" for i in range(d)]
              + [f"Per-sentence embedding dim {i}, max over sentences" for i in range(d)]
              + [f"Emotion probability: {n} ({m['model'].split('/')[-1]})" for n in emo_names]
              + [f"{c} (shipped with the dataset, not computed here)" for c in shipped])
    assert_no_blocked(cols)
    X = np.concatenate([pooled, sentpool, emo, table[shipped].to_numpy(dtype=float)], axis=1)
    family = (["pooled embedding"] * d + ["sentence mean"] * d + ["sentence max"] * d + ["emotion"] * len(emo_names)
              + ["shipped: " + ("sentiment" if c in ("polarity", "subjectivity") else "length/style") for c in shipped])

    main_rows = table.split.isin(["dev", "holdout"]).to_numpy()
    if np.isnan(X[main_rows]).any():
        bad = sorted({cols[j] for j in np.where(np.isnan(X[main_rows]).any(axis=0))[0]})
        raise SystemExit(f"missing feature values in main rows: {bad[:10]}")

    results, fam_rows = {}, []
    for view, classes in (("4class", CLASSES_4), ("3class", CLASSES_3)):
        dev = (table.split == "dev").to_numpy() & table.status.isin(classes).to_numpy()
        y = table.status[dev].map(classes).to_numpy()
        t = time.time()
        rank = mi_ranking(X[dev], y, cols, labels, seed, top=int(cfg["mi"]["top"]))
        timings[f"mi_{view}"] = round(time.time() - t, 1)
        rank.to_csv(out / f"mi_scores_{view}.csv", index=False)
        results[view] = {"dev_rows": int(dev.sum()), "top10": rank.head(10)[["column", "mutual_information"]].values.tolist()}
        fam = pd.Series(family, index=cols)
        rk = rank.set_index("column")
        for f in dict.fromkeys(family):
            s = rk.loc[fam[fam == f].index, "mutual_information"]
            fam_rows.append({"view": view, "family": f, "columns": len(s), "max_mi": float(s.max()),
                             "mean_mi": float(s.mean()), "best_column": s.idxmax(), "best_rank": int(rk.loc[s.idxmax(), "rank"])})
        say(f"MI {view}: {dev.sum():,} dev rows, {len(cols)} columns in {timings[f'mi_{view}']:.0f}s")
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(out / "mi_by_family.csv", index=False)

    tok = {s: stats(tl[(table.split == s).to_numpy()], int(e["max_tokens"])) for s in ("dev", "holdout", "quarantine")}
    tok["main_by_class"] = {c: stats(tl[main_rows & (table.status == c).to_numpy()], int(e["max_tokens"])) for c in CLASSES_4}
    sent_stats = {"total_sentences": int(n_sent.sum()), "median_per_text": float(np.median(n_sent[main_rows])),
                  "share_single_sentence_main": float((n_sent[main_rows] == 1).mean())}
    manifest = {"seed": seed, "processed_sha256": proc_sha, "embedding": e, "emotion": {**m, "labels": emo_names},
                "feature_columns": len(cols), "shipped_columns": shipped, "token_lengths": tok, "sentences": sent_stats,
                "timings_seconds": timings, "mi": results, "torch_threads": torch.get_num_threads(),
                "environment": environment_record(cfg)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    L = ["# Mental Health Text — Phase 2 (box 3): text features", "",
         f"Frozen models, nothing fine-tuned. Embeddings: `{e['model']}` (revision {e['revision'][:8]}), "
         f"{d} dimensions, truncation {e['max_tokens']} tokens. Emotion: `{m['model']}` (revision {m['revision'][:8]}), "
         f"{len(emo_names)} classes ({', '.join(emo_names)}), truncation {m['max_tokens']} tokens.",
         f"Shipped with the dataset (not computed here): {', '.join(shipped)}; polarity and subjectivity are the "
         "sentiment features.", "",
         "## Token lengths under the embedding model's tokenizer (untruncated, special tokens included)", "",
         "| Rows | Median | p95 | p99 | Max | Share over 256 |", "|---|---|---|---|---|---|"]
    for k in ("dev", "holdout", "quarantine"):
        s = tok[k]
        L.append(f"| {k} | {s['median']:.0f} | {s['p95']:.0f} | {s['p99']:.0f} | {s['max']:,} | {s['share_over_cap']:.2%} |")
    for c, s in tok["main_by_class"].items():
        L.append(f"| main, {c} | {s['median']:.0f} | {s['p95']:.0f} | {s['p99']:.0f} | {s['max']:,} | {s['share_over_cap']:.2%} |")
    L += ["", f"Per-sentence variant: {sent_stats['total_sentences']:,} sentences; median {sent_stats['median_per_text']:.0f} per "
          f"main-population text; {sent_stats['share_single_sentence_main']:.1%} of main texts are a single sentence. "
          "Mean and max pooling over sentences cover the whole text (no truncation of the text; sentences longer than "
          f"{e['max_tokens']} tokens are themselves truncated).", "",
          "## Mutual information with the label (development rows only), by feature family", "",
          "| View | Family | Columns | Max MI | Mean MI | Best column (overall rank) |", "|---|---|---|---|---|---|"]
    for _, r in fam.iterrows():
        L.append(f"| {r['view']} | {r['family']} | {r['columns']} | {r['max_mi']:.4f} | {r['mean_mi']:.4f} | "
                 f"{r['best_column']} ({r['best_rank']}) |")
    for view in ("4class", "3class"):
        L += ["", f"Top 10, {view} ({results[view]['dev_rows']:,} dev rows): "
              + "; ".join(f"{c} {v:.4f}" for c, v in results[view]["top10"])]
    L += ["", "Timings (s): " + ", ".join(f"{k} {v}" for k, v in timings.items()) + f". Total {time.time() - T0:.0f} s.",
          "Files: mi_scores_4class.csv and mi_scores_3class.csv (same format as results/feature_importance/mi_scores.csv), "
          "mi_by_family.csv, manifest.json. Feature arrays: data/processed/mhtext/features/ (git-ignored)."]
    (out / "phase2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
