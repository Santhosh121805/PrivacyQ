"""Mutual-information ranking of the 95 model columns (real run; development rows only, hold-out excluded).

Uses the project's own selector (FeatureTransformer, method="mi", d_prime=8: StandardScaler + SelectKBest with
mutual_info_classif, seed 2026) fitted on the 17,431 development rows after the project's preprocessing. Labels are
the official NSCH codebook labels (src/m1_data/codebook.py). Writes results/feature_importance/mi_scores.csv and
mi_ranking.png (top 20; the 8 columns the selector keeps are marked).

Usage: python scripts/build_feature_importance.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.m1_data.codebook import parse_codebook, plain_label  # noqa: E402
from src.m2_preprocess.preprocessing import TabularPreprocessor  # noqa: E402
from src.m3_features.features import FeatureTransformer  # noqa: E402
from src.m10_exp.common import holdout_split, load_config, load_dataset  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs/holdout_final.yaml")
    seed = int(cfg["project"]["seed"])
    loader, frame = load_dataset(cfg)
    y = frame[loader.label_column].to_numpy(dtype=int)
    dev, _ = holdout_split(y, float(cfg["split"]["holdout_fraction"]), seed)
    prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
    Xdf = prep.fit_transform(frame.iloc[dev], loader.label_column)
    ft = FeatureTransformer(d_prime=8, method="mi", output="standard", seed=seed).fit(Xdf.to_numpy(), y[dev])
    var, vals = parse_codebook(ROOT / cfg["data"]["paths"][0].replace("nsch_2024e_topical.dta", "nsch_2024_topical.do"))
    scores = ft.selector.scores_
    keep = ft.selector.get_support()
    out = pd.DataFrame({"column": Xdf.columns, "label": [plain_label(c, var, vals) for c in Xdf.columns],
                        "mutual_information": scores, "selected_top8": keep}).sort_values("mutual_information", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    d = ROOT / "results/feature_importance"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "mi_scores.csv", index=False)

    top = out.head(20).iloc[::-1]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = ["#4269d0" if s else "#b9bcc2" for s in top.selected_top8]
    ax.barh([f"{l[:62]}" for l in top.label], top.mutual_information, color=colors)
    ax.set_xlabel("Mutual information with current depression diagnosis (nats)")
    ax.set_title(f"Top 20 of {len(out)} columns by mutual information (blue = the 8 kept by MI selection)",
                 loc="left", fontsize=14)
    ax.grid(axis="x", color="#e4e4e7")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(d / "mi_ranking.png", dpi=150, facecolor="white")
    print(out.head(10)[["rank", "column", "label", "mutual_information", "selected_top8"]].to_string(index=False))


if __name__ == "__main__":
    main()
