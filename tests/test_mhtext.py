"""Mental Health Text: split integrity and leakage guards.

Real-data tests run on data/processed/mhtext/mhtext_phase1.csv (built by scripts/build_mhtext_splits.py) and are
skipped until it exists. Guard tests check that the guards themselves fail; they use column names only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.m1_data.mhtext import (
    BLOCKED_COLUMNS,
    CLASSES_3,
    CLASSES_4,
    KEYWORD_LEAKAGE,
    MHTextDataError,
    MHTextLoader,
    assert_no_blocked,
    model_inputs,
    normalise,
)

PROCESSED = Path(__file__).resolve().parents[1] / "data/processed/mhtext/mhtext_phase1.csv"
REAL = pytest.mark.skipif(not PROCESSED.exists(), reason="run scripts/build_mhtext_splits.py first")
ALL = ("dev", "holdout", "quarantine")


@pytest.fixture(scope="module")
def four():
    return MHTextLoader(processed=str(PROCESSED), n_classes=4, splits=ALL).load()


@pytest.fixture(scope="module")
def three():
    return MHTextLoader(processed=str(PROCESSED), n_classes=3, splits=ALL).load()


# ------------------------------------------------------------------ the required split test
@REAL
@pytest.mark.parametrize("view", ["four", "three"])
def test_no_text_string_in_two_splits(view, request):
    frame = request.getfixturevalue(view)
    for raw_or_norm in (frame.text.astype(str), frame.text.map(normalise)):
        sets = {s: set(raw_or_norm[frame.split == s]) for s in ALL}
        for a, b in [("dev", "holdout"), ("dev", "quarantine"), ("holdout", "quarantine")]:
            shared = sets[a] & sets[b]
            assert not shared, f"{len(shared)} texts in both {a} and {b}"


@REAL
def test_no_near_duplicate_across_splits(four):
    from sklearn.feature_extraction.text import TfidfVectorizer

    norm = four.text.map(normalise)
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, dtype=np.float32).fit_transform(norm)
    split = four.split.to_numpy()
    for s in range(0, X.shape[0], 2000):
        S = (X[s:s + 2000] @ X.T).tocoo()
        hit = (S.data >= 0.90) & (S.col != S.row + s)
        cross = split[S.row[hit] + s] != split[S.col[hit]]
        assert not cross.any(), f"{int(cross.sum())} near-duplicate pairs (>= 0.90) cross splits"


# ------------------------------------------------------------------ blocked columns never reach a model
def test_guard_rejects_every_blocked_column():
    for col in sorted(BLOCKED_COLUMNS):
        with pytest.raises(MHTextDataError):
            assert_no_blocked(["polarity", col])
    assert_no_blocked(["polarity", "subjectivity", "word_count"])


def test_model_inputs_refuses_a_blocked_column():
    frame = pd.DataFrame({"polarity": [0.1], "has_suicidal_keyword": [1], "status": ["Suicidal"]})
    with pytest.raises(MHTextDataError):
        model_inputs(frame, ["polarity", "has_suicidal_keyword"])
    with pytest.raises(MHTextDataError):
        model_inputs(frame, ["status"])
    assert list(model_inputs(frame).columns) == ["polarity"]


@REAL
def test_real_model_inputs_carry_no_blocked_column(four):
    assert not set(KEYWORD_LEAKAGE) & set(four.columns), "keyword flags must not be in the processed table"
    X = model_inputs(four)
    assert X.shape[1] > 0 and not set(X.columns) & BLOCKED_COLUMNS


# ------------------------------------------------------------------ mapping, nesting, quarantine
@REAL
def test_label_mapping_is_explicit(four, three):
    assert CLASSES_4 == {"Normal": 0, "Anxiety": 1, "Depression": 2, "Suicidal": 3}
    assert CLASSES_3 == {"Anxiety": 0, "Depression": 1, "Suicidal": 2}
    assert (four.label == four.status.map(CLASSES_4)).all()
    assert (three.label == three.status.map(CLASSES_3)).all() and "Normal" not in set(three.status)


@REAL
def test_three_class_is_nested_in_four_class(four, three):
    key = ["Unique_ID", "split"]
    main4 = four[four.split != "quarantine"].set_index("Unique_ID").split
    main3 = three[three.split != "quarantine"]
    assert main3.Unique_ID.isin(main4.index).all()
    assert (main4.loc[main3.Unique_ID].to_numpy() == main3.split.to_numpy()).all(), key


@REAL
def test_quarantine_is_separate(four):
    default = MHTextLoader(processed=str(PROCESSED)).load()
    assert set(default.split) == {"dev", "holdout"}
    assert default.Unique_ID.notna().all()
    assert four.loc[four.split == "quarantine", "Unique_ID"].isna().all()
