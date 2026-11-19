"""Phase 1 report for AdoDAS: real class balance, schools, sessions, DASS-21 item distributions, split checks.

    python scripts/adodas_profile.py --config configs/adodas_profile.yaml

Reads data/raw/adodas/ only; writes results/adodas/profile/. Every number is computed from the release files;
nothing is assumed (the paper's prevalence figures are printed beside ours for comparison, labelled as such).
Test labels are withheld by the organisers, so validation is the reporting set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.m1_data.adodas import (  # noqa: E402
    AdoDASLoader, assert_sessions_together, assert_subject_disjoint, class_balance, item_distribution,
    school_table, subject_folds)
from src.m10_exp.common import environment_record, load_config  # noqa: E402

PAPER_PREVALENCE = {"depression": 0.2158, "anxiety": 0.2825, "stress": 0.1290}   # arXiv:2609.07038, section 2.1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    cfg = load_config(ROOT / p.parse_args().config)
    d = cfg["data"]
    out = ROOT / cfg["project"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    loader = AdoDASLoader(root=str(ROOT / d["root"]), target=d.get("target", "depression"),
                          splits=tuple(d.get("splits", ["train", "val", "test"])))
    frame, sessions = loader.load(), loader.sessions()
    labelled = frame.dropna(subset=["depression", "anxiety", "stress"], how="all")

    # split integrity, on the real data
    assert_subject_disjoint({s: frame.person_id[frame.split == s] for s in frame.split.unique()})
    assert_sessions_together(sessions, {}, column="split")
    train = labelled[labelled.split == "train"].reset_index(drop=True)
    seed = int(cfg["project"]["seed"])
    n_folds = 0
    for t in ("depression", "anxiety", "stress"):
        for f in subject_folds(train, t, int(cfg["cv"]["n_splits"]), int(cfg["cv"]["n_repeats"]), seed):
            part = {**{q: "tr" for q in train.person_id.iloc[f["train"]]}, **{q: "va" for q in train.person_id.iloc[f["val"]]}}
            assert_sessions_together(sessions[sessions.person_id.isin(part)], part)
            n_folds += 1

    bal = class_balance(labelled)
    bal["paper_prevalence_all_splits"] = bal.target.map(PAPER_PREVALENCE).where(bal.split == "all")
    items = item_distribution(labelled)
    schools = school_table(frame) if frame.school_id.notna().any() else pd.DataFrame()
    per_pid = sessions.groupby("person_id").agg(sessions_with_features=("n_feature_groups", lambda s: int((s > 0).sum())))
    coverage = sessions.groupby(["split", "session"]).n_feature_groups.agg(["count", "min", "median", "max"]).reset_index()

    bal.to_csv(out / "class_balance.csv", index=False)
    items.to_csv(out / "item_distribution.csv")
    schools.to_csv(out / "schools.csv", index=False)
    coverage.to_csv(out / "session_coverage.csv", index=False)
    counts = {s: int((frame.split == s).sum()) for s in ("train", "val", "test")}
    summary = {"participants_by_split": counts,
               "participants_in_feature_tree": int(sessions.person_id.nunique()),
               "session_rows": int(len(sessions)),
               "participants_with_4_sessions_with_features": int((per_pid.sessions_with_features == 4).sum()),
               "schools": int(frame.school_id.nunique()), "classes": int(frame.class_id.nunique()),
               "test_labels_released": bool((frame.split == "test").any() and frame.loc[frame.split == "test", "depression"].notna().any()),
               "label_csv_extra_columns": getattr(loader, "extra_columns_", {}),
               "subject_disjoint_folds_checked": n_folds,
               "environment": environment_record(cfg)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = ["# AdoDAS profile (Phase 1)", "",
             f"Participants: train {counts['train']:,}, validation {counts['val']:,}, test {counts['test']:,} "
             f"({'test labels released' if summary['test_labels_released'] else 'test labels withheld: validation is the reporting set'}).",
             f"Schools {summary['schools']}, classes {summary['classes']}. Session rows {summary['session_rows']:,}; "
             f"{summary['participants_with_4_sessions_with_features']:,} participants have features for all 4 sessions.",
             f"Split checks passed: no participant in two splits; all sessions of a participant in one split; "
             f"{n_folds} subject-grouped CV folds (5 x {cfg['cv']['n_repeats']} per target) subject-disjoint.", "",
             "## Class balance (Mild-or-above, from the DASS-21 items)", "",
             "| Split | Target | n | Positives | Rate | Paper (all splits) |", "|---|---|---|---|---|---|"]
    for _, r in bal.iterrows():
        paper = "" if pd.isna(r.paper_prevalence_all_splits) else f"{r.paper_prevalence_all_splits:.2%}"
        lines.append(f"| {r.split} | {r.target} | {r.n:,} | {r.positives:,} | {r.positive_rate:.2%} | {paper} |")
    if not schools.empty:
        lines += ["", "## Schools (natural federated clients)", "",
                  "| School | Participants | Classes | Train | Val | Test | Depression | Anxiety | Stress |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in schools.iterrows():
            lines.append(f"| {r.school_id} | {r.participants:,} | {r.classes} | {r.n_train:,} | {r.n_val:,} | {r.n_test:,} | "
                         f"{r.depression_rate:.2%} | {r.anxiety_rate:.2%} | {r.stress_rate:.2%} |")
    lines += ["", "Rates per school exclude test participants (labels withheld).", "",
              "## DASS-21 item responses (share of 0 / 1 / 2 / 3)", "", "| Item | 0 | 1 | 2 | 3 |", "|---|---|---|---|---|"]
    for i, r in items.iterrows():
        lines.append(f"| {i} | {r[0]:.2%} | {r[1]:.2%} | {r[2]:.2%} | {r[3]:.2%} |")
    (out / "profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
