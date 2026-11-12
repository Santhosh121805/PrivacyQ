"""Pair a supplementary run with the Experiment 2 run (same 15 folds, same seeds) and test it.

Writes <new_run>/comparison.csv and comparison.md: mean ± std of each arm and paired Wilcoxon
(Holm-adjusted within the table) plus the Nadeau-Bengio corrected t on ROC-AUC, balanced accuracy
and recall.

Usage:
  python scripts/compare_with_exp2.py <new_run_dir> "<a_model>/<regime>/<features>:<b_model>/<regime>/<features>" ...
Arms are looked up in the new run first, then in the Experiment 2 run of configs/final_report.yaml.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.m8_eval.stats import corrected_resampled_ttest, holm, wilcoxon_p  # noqa: E402
from src.m10_exp.common import load_config  # noqa: E402

METRICS = ["roc_auc", "balanced_accuracy", "recall"]


def main(new_dir: str, pairs: list[str]) -> None:
    new = pd.read_csv(Path(new_dir) / "fold_metrics.csv")
    exp2 = pd.read_csv(ROOT / load_config(ROOT / "configs/final_report.yaml")["runs"]["exp2"] / "fold_metrics.csv")

    def arm(spec):
        model, regime, fs = spec.split("/")
        for df in (new, exp2):
            sel = df[(df.model == model) & (df.regime == regime) & (df.feature_set == fs)]
            if len(sel):
                return sel.set_index(["repeat", "fold"]).sort_index()
        raise SystemExit(f"arm {spec} not found")

    rows = []
    for pair in pairs:
        a_spec, b_spec = pair.split(":")
        a, b = arm(a_spec), arm(b_spec)
        idx = a.index.intersection(b.index)
        ratio = float(a.loc[idx, "n_val"].mean() / a.loc[idx, "n_fold_train"].mean())
        for m in METRICS:
            x, y = a.loc[idx, m].to_numpy(), b.loc[idx, m].to_numpy()
            t = corrected_resampled_ttest(x, y, ratio)
            rows.append({"a": a_spec, "b": b_spec, "metric": m, "a_mean": x.mean(), "a_std": x.std(ddof=1),
                         "b_mean": y.mean(), "b_std": y.std(ddof=1), "diff": t["mean_diff"], "a_wins": int((x > y).sum()),
                         "k": len(idx), "wilcoxon_p": wilcoxon_p(x, y), "p_nb_t": t["p"]})
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = out.groupby("metric")["wilcoxon_p"].transform(lambda p: holm(p.to_numpy()))
    out.to_csv(Path(new_dir) / "comparison.csv", index=False)
    lines = ["| A | B | Metric | A (mean ± std) | B (mean ± std) | A − B | A wins | Wilcoxon p | Holm p | NB-t p |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {r.a} | {r.b} | {r.metric} | {r.a_mean:.3f} ± {r.a_std:.3f} | {r.b_mean:.3f} ± {r.b_std:.3f} | "
                     f"{r['diff']:+.3f} | {r.a_wins}/{r.k} | {r.wilcoxon_p:.3g} | {r.wilcoxon_p_holm:.3g} | {r.p_nb_t:.3g} |")
    (Path(new_dir) / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
