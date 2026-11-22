"""Text features (box 3): sentence splitting, MI output format, and alignment with the processed table.

Real-data checks are skipped until scripts/build_mhtext_features.py has written its outputs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.m1_data.mhtext import BLOCKED_COLUMNS
from src.m3_features.text import mi_ranking, split_sentences

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data/processed/mhtext/features"
PROC = ROOT / "data/processed/mhtext/mhtext_phase1.csv"
OUT = ROOT / "results/mhtext/phase2"
BUILT = pytest.mark.skipif(not (OUT / "mi_scores_4class.csv").exists(), reason="run scripts/build_mhtext_features.py")


def test_split_sentences_on_punctuation_and_newlines():
    assert split_sentences("I can't sleep. Nothing helps! Why?") == ["I can't sleep.", "Nothing helps!", "Why?"]
    assert split_sentences("first line\nsecond line") == ["first line", "second line"]
    assert split_sentences("no boundary here") == ["no boundary here"]
    assert split_sentences("   ") == [""]


def test_mi_ranking_has_the_nsch_file_format():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    X = np.c_[y + rng.normal(0, 0.1, 200), rng.normal(size=200)]
    r = mi_ranking(X, y, ["signal", "noise"], ["Signal", "Noise"], seed=1, top=1)
    assert list(r.columns) == ["column", "label", "mutual_information", "selected_top8", "rank"]
    assert r.column.tolist() == ["signal", "noise"] and r.selected_top8.tolist() == [True, False]
    nsch = ROOT / "results/feature_importance/mi_scores.csv"
    if nsch.exists():
        assert list(pd.read_csv(nsch, nrows=1).columns) == list(r.columns)


@BUILT
def test_features_align_with_the_processed_table():
    table = pd.read_csv(PROC)
    rows = pd.read_csv(FEAT / "rows.csv")
    assert len(rows) == len(table) and (rows.split.to_numpy() == table.split.to_numpy()).all()
    for name in ("pooled.npy", "sentence_mean_max.npy", "emotion.npy", "token_lengths.npy"):
        assert np.load(FEAT / name).shape[0] == len(table), name
    emo = np.load(FEAT / "emotion.npy")
    assert np.allclose(emo.sum(axis=1), 1.0, atol=1e-4)


@BUILT
@pytest.mark.parametrize("view", ["4class", "3class"])
def test_no_blocked_column_was_ranked(view):
    r = pd.read_csv(OUT / f"mi_scores_{view}.csv")
    assert not set(r.column) & BLOCKED_COLUMNS
    assert r.selected_top8.sum() == 8 and (r["rank"] == np.arange(1, len(r) + 1)).all()
