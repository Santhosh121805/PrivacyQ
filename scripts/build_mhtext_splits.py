"""Phase 1 build for the Mental Health Text dataset: verify, quarantine, deduplicate, subsample, lock hold-out.

    python scripts/build_mhtext_splits.py --config configs/mhtext_phase1.yaml

Order matters and is fixed: deduplicate (exact, then near >= 0.90, conflicting-label groups dropped) BEFORE
any split; then take the stratified 15,000-row subsample of the ID-bearing rows; then lock the 20 % hold-out with
the same stratified split function NSCH uses. The 9,600-row block without Unique_ID is quarantined (deduplicated
with the rest, so none of its rows duplicates a kept main row) and written with split = "quarantine".
Writes data/processed/mhtext/mhtext_phase1.csv (git-ignored; holds text) and results/mhtext/phase1/
(counts and checks only, no text).
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
from sklearn.model_selection import train_test_split  # noqa: E402

from src.m1_data.mhtext import (  # noqa: E402
    CLASSES_3, CLASSES_4, KEYWORD_LEAKAGE, SHIPPED_FEATURES, assert_no_blocked, dedup, normalise, read_raw)
from src.m10_exp.common import environment_record, holdout_split, load_config  # noqa: E402

RAW_SHA256 = {  # recorded in data/README.md at Phase 0
    "mental_heath_unbanlanced.csv": "d41cdc3ec3b8dde8f5553accaccd1a01a59c1440d83d7af6856fab77605c9471",
    "mental_health_combined_test.csv": "2806f9518b92915b11ea35b16955ac3826fb5d981fc5f9152ee2282f7cffa418",
    "mental_health_feature_engineered.csv": "122f1e702d2971d4c80c90fb3329dfa1485b56e1ccd0f4c5266604299858887e",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def balance(s: pd.Series, classes: dict) -> dict:
    vc = s.value_counts()
    return {c: {"n": int(vc.get(c, 0)), "share": float(vc.get(c, 0) / len(s)) if len(s) else 0.0} for c in classes}


def words(s: pd.Series) -> pd.Series:
    return s.astype(str).str.split().str.len()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    cfg = load_config(ROOT / p.parse_args().config)
    seed = int(cfg["project"]["seed"])
    raw_root = ROOT / cfg["data"]["raw_root"]
    out = ROOT / cfg["project"]["output_dir"]
    proc = ROOT / cfg["data"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    proc.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for name, want in RAW_SHA256.items():
        got = sha256(raw_root / name)
        if got != want:
            raise SystemExit(f"{name}: SHA-256 {got} differs from the Phase 0 record {want}")
    print("raw files match the Phase 0 SHA-256 record")
    print("label mapping, 4-class:", CLASSES_4)
    print("label mapping, 3-class (Normal dropped):", CLASSES_3)

    raw = read_raw(raw_root)
    raw = raw.drop(columns=list(KEYWORD_LEAKAGE))                 # label leakage: never kept
    ids, quar = raw[raw.provenance == "id"], raw[raw.provenance == "quarantine"]
    stages = {"raw_all": balance(raw.status, CLASSES_4), "raw_id_rows": balance(ids.status, CLASSES_4),
              "raw_quarantine": balance(quar.status, CLASSES_4)}
    print(f"raw rows {len(raw):,}: ID-bearing {len(ids):,}, quarantined block {len(quar):,}")

    kept, log = dedup(raw, threshold=float(cfg["dedup"]["threshold"]))
    kept_ids = kept[kept.provenance == "id"]
    kept_q = kept[kept.provenance == "quarantine"]
    log["id_rows_kept"], log["quarantine_rows_kept"] = int(len(kept_ids)), int(len(kept_q))
    log["id_rows_dropped"] = int(len(ids) - len(kept_ids))
    log["quarantine_rows_dropped"] = int(len(quar) - len(kept_q))
    stages["dedup_id_rows"] = balance(kept_ids.status, CLASSES_4)
    stages["dedup_quarantine"] = balance(kept_q.status, CLASSES_4)
    print(f"dedup ({time.time() - t0:.0f} s): {json.dumps(log)}")

    n = int(cfg["subsample"]["n"])
    if n > len(kept_ids):
        raise SystemExit(f"subsample {n} exceeds the {len(kept_ids)} deduplicated ID rows")
    sub, _ = train_test_split(kept_ids, train_size=n, stratify=kept_ids.status, random_state=seed)
    sub = sub.sort_values("Unique_ID").reset_index(drop=True)
    # confirmation pass on exactly the rows that will be split (IDF refitted on them), before the hold-out exists
    pool, log2 = dedup(pd.concat([sub, kept_q], ignore_index=True), threshold=float(cfg["dedup"]["threshold"]))
    log["confirm_pass_on_subsample_plus_quarantine"] = {k: v for k, v in log2.items() if k != "similarity"}
    log["confirm_pass_main_rows_dropped"] = int(len(sub) - (pool.provenance == "id").sum())
    log["confirm_pass_quarantine_rows_dropped"] = int(len(kept_q) - (pool.provenance == "quarantine").sum())
    sub = pool[pool.provenance == "id"].reset_index(drop=True)
    kept_q = pool[pool.provenance == "quarantine"].reset_index(drop=True)
    print(f"confirmation pass: {json.dumps(log['confirm_pass_on_subsample_plus_quarantine'])}")
    y4 = sub.status.map(CLASSES_4).to_numpy()
    dev_pos, hold_pos = holdout_split(y4, float(cfg["split"]["holdout_fraction"]), seed)
    sub["split"] = "dev"
    sub.loc[hold_pos, "split"] = "holdout"
    kept_q = kept_q.assign(split="quarantine")
    table = pd.concat([sub, kept_q], ignore_index=True)
    cols = ["Unique_ID", "provenance", "split", "status", "text", *[c for c in SHIPPED_FEATURES if c in table.columns]]
    assert_no_blocked([c for c in cols if c in SHIPPED_FEATURES])
    table[cols].to_csv(proc, index=False)

    # checks on what was written
    norm = table.text.map(normalise)
    for a, b in [("dev", "holdout"), ("dev", "quarantine"), ("holdout", "quarantine")]:
        shared = set(norm[table.split == a]) & set(norm[table.split == b])
        if shared:
            raise SystemExit(f"{len(shared)} texts appear in both {a} and {b}")
    if table.loc[table.split != "quarantine", "Unique_ID"].isna().any():
        raise SystemExit("main population row without Unique_ID")

    views = {}
    for k, classes in ((4, CLASSES_4), (3, CLASSES_3)):
        v = {}
        for s in ("dev", "holdout", "quarantine"):
            part = table[(table.split == s) & table.status.isin(classes)]
            w = words(part.text)
            v[s] = {"rows": int(len(part)), "balance": balance(part.status, classes),
                    "median_words_by_class": {c: float(w[part.status == c].median()) for c in classes},
                    "p95_words": float(w.quantile(0.95))}
        views[f"{k}-class"] = v
    manifest = {"seed": seed, "label_mapping_4": CLASSES_4, "label_mapping_3": CLASSES_3,
                "raw_sha256": RAW_SHA256, "blocked_columns_removed": list(KEYWORD_LEAKAGE),
                "shipped_feature_columns_kept": [c for c in SHIPPED_FEATURES if c in table.columns],
                "dedup": log, "subsample_n": n, "holdout_fraction": float(cfg["split"]["holdout_fraction"]),
                "stages": stages, "views": views, "processed_file": str(proc.relative_to(ROOT)).replace("\\", "/"),
                "processed_sha256": sha256(proc), "shipped_test_file": "not used: 496 of 992 rows occur in the main file",
                "environment": environment_record(cfg)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    L = ["# Mental Health Text — Phase 1 build", "",
         f"Seed {seed}. Raw files verified against the Phase 0 SHA-256 record. Keyword flags {list(KEYWORD_LEAKAGE)} "
         "removed (label leakage). Shipped test file not used (496 of its 992 rows are in the main file).", "",
         "Label mapping, 4-class: " + ", ".join(f"{k} = {v}" for k, v in CLASSES_4.items()),
         "Label mapping, 3-class: " + ", ".join(f"{k} = {v}" for k, v in CLASSES_3.items()), "",
         "## Class balance before deduplication", "", "| Rows | n | Normal | Anxiety | Depression | Suicidal |",
         "|---|---|---|---|---|---|"]
    for key, label in (("raw_all", "All rows"), ("raw_id_rows", "ID-bearing (main)"), ("raw_quarantine", "Quarantined block")):
        b = stages[key]
        tot = sum(x["n"] for x in b.values())
        L.append(f"| {label} | {tot:,} | " + " | ".join(f"{b[c]['n']:,} ({b[c]['share']:.2%})" for c in CLASSES_4) + " |")
    L += ["", "## Deduplication (before any split)", "",
          f"- Exact duplicates (normalised text): {log['exact_duplicate_rows_dropped']:,} rows dropped; "
          f"{log['exact_conflict_groups']:,} groups with conflicting labels → {log['exact_conflict_rows_dropped']:,} rows dropped entirely.",
          f"- Near-duplicates (TF-IDF cosine ≥ {log['threshold']:.2f}): {log['near_duplicate_pairs']:,} pairs; "
          f"{log['near_duplicate_rows_dropped']:,} rows dropped; {log['near_conflict_groups']:,} conflicting-label groups → "
          f"{log['near_conflict_rows_dropped']:,} rows dropped entirely.",
          f"- Rows {log['rows_in']:,} → {log['rows_out']:,}. ID-bearing kept {log['id_rows_kept']:,} (dropped {log['id_rows_dropped']:,}); "
          f"quarantined block kept {log['quarantine_rows_kept']:,} (dropped {log['quarantine_rows_dropped']:,}). "
          f"Near-duplicate removal repeated with IDF refitted on the survivors until no pair ≥ {log['threshold']:.2f} remained "
          f"({log['near_passes']} passes).",
          f"- Confirmation pass on the {n:,}-row subsample plus the quarantined block (IDF refitted on exactly those rows, "
          f"before the hold-out was drawn): {log['confirm_pass_main_rows_dropped']} main and "
          f"{log['confirm_pass_quarantine_rows_dropped']} quarantined rows dropped, so the main population is "
          f"{n - log['confirm_pass_main_rows_dropped']:,} rows.", "",
          "Conflicting-label groups (the same post filed under two classes, mostly Depression vs Suicidal) show that "
          "boundary is soft in the source data. They are dropped, not resolved, and they cap achievable accuracy "
          "between those classes.", "",
          "| After dedup | n | Normal | Anxiety | Depression | Suicidal |", "|---|---|---|---|---|---|"]
    for key, label in (("dedup_id_rows", "ID-bearing (main)"), ("dedup_quarantine", "Quarantined block")):
        b = stages[key]
        tot = sum(x["n"] for x in b.values())
        L.append(f"| {label} | {tot:,} | " + " | ".join(f"{b[c]['n']:,} ({b[c]['share']:.2%})" for c in CLASSES_4) + " |")
    for view, classes in (("4-class", CLASSES_4), ("3-class", CLASSES_3)):
        L += ["", f"## {view} view (subsample {n:,} → {len(sub):,} after the confirmation pass, seed {seed}; hold-out {cfg['split']['holdout_fraction']:.0%})", "",
              "| Split | Rows | " + " | ".join(classes) + " | Median words by class | p95 words |",
              "|---|---|" + "---|" * len(classes) + "---|---|"]
        for s, d in views[view].items():
            L.append(f"| {s} | {d['rows']:,} | " + " | ".join(f"{d['balance'][c]['n']:,} ({d['balance'][c]['share']:.2%})" for c in classes)
                     + " | " + ", ".join(f"{c} {d['median_words_by_class'][c]:.0f}" for c in classes) + f" | {d['p95_words']:.0f} |")
    L += ["", "The 3-class view is the 4-class rows with Normal removed and the same split assignment (nested).",
          "Checks passed: no normalised text in two splits; every main-population row has a Unique_ID.",
          f"Processed file: {manifest['processed_file']} (SHA-256 {manifest['processed_sha256'][:16]}…). Total {time.time() - t0:.0f} s."]
    (out / "phase1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
