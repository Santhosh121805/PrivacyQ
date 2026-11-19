"""AdoDAS split integrity.

The two required leakage tests run on the real release under data/raw/adodas/ and are skipped until it is
there. The guard tests at the bottom check that the guards themselves fail; they use participant-id strings
only, never data values.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.m1_data.adodas import (
    ITEMS,
    MILD_OR_ABOVE,
    SUBSCALE_ITEMS,
    TARGETS,
    AdoDASDataError,
    AdoDASLoader,
    assert_sessions_together,
    assert_subject_disjoint,
    subject_folds,
)

ROOT = Path(__file__).resolve().parents[1] / "data/raw/adodas"
REAL = pytest.mark.skipif(not (ROOT / "train.csv").exists(), reason="AdoDAS release not present in data/raw/adodas")


@pytest.fixture(scope="module")
def real():
    loader = AdoDASLoader(root=str(ROOT))
    return loader.load(), loader.sessions()


# ------------------------------------------------------------------ the two required leakage tests (real data)
@REAL
def test_no_participant_in_two_splits(real):
    frame, sessions = real
    assert_subject_disjoint({s: frame.person_id[frame.split == s] for s in frame.split.unique()})
    assert_subject_disjoint({s: sessions.person_id[sessions.split == s].unique() for s in sessions.split.unique()})
    for f in subject_folds(frame[frame.split == "train"].reset_index(drop=True), "depression", 5, 1, 2026):
        tr = frame[frame.split == "train"].reset_index(drop=True)
        assert set(tr.person_id.iloc[f["train"]]).isdisjoint(tr.person_id.iloc[f["val"]])


@REAL
def test_sessions_of_a_participant_stay_together(real):
    frame, sessions = real
    assert_sessions_together(sessions, {}, column="split")
    assert_sessions_together(sessions, {}, column="school_id")
    tr = frame[frame.split == "train"].reset_index(drop=True)
    for f in subject_folds(tr, "depression", 5, 1, 2026):
        part = {**{p: "train" for p in tr.person_id.iloc[f["train"]]}, **{p: "val" for p in tr.person_id.iloc[f["val"]]}}
        assert_sessions_together(sessions[sessions.person_id.isin(part)], part)


@REAL
def test_labels_follow_the_documented_dass21_rule(real):
    frame, _ = real
    for t, items in SUBSCALE_ITEMS.items():
        ok = frame[list(items)].notna().all(axis=1)
        expected = (2 * frame.loc[ok, list(items)].sum(axis=1) >= MILD_OR_ABOVE[t]).astype(float)
        assert np.array_equal(frame.loc[ok, t].to_numpy(), expected.to_numpy())


@REAL
def test_every_participant_has_four_sessions(real):
    _, sessions = real
    per = sessions.groupby("person_id").session.nunique()
    assert (per == 4).all(), f"{(per != 4).sum()} participants without exactly 4 session rows"


# ------------------------------------------------------------------ the guards must fail when they should
def test_disjoint_guard_raises_on_shared_id():
    with pytest.raises(AdoDASDataError):
        assert_subject_disjoint({"train": ["p1", "p2"], "val": ["p2", "p3"]})
    assert_subject_disjoint({"train": ["p1", "p2"], "val": ["p3"]})


def test_session_guard_raises_when_sessions_are_separated():
    sessions = pd.DataFrame({"person_id": ["p1"] * 4 + ["p2"] * 4,
                             "session": ["A01", "B01", "B02", "B03"] * 2,
                             "split": ["train"] * 3 + ["val"] + ["val"] * 4})
    with pytest.raises(AdoDASDataError):
        assert_sessions_together(sessions, {}, column="split")
    assert_sessions_together(sessions[sessions.person_id == "p2"], {}, column="split")


def test_item_and_subscale_definitions_match_the_project_page():
    assert len(ITEMS) == 21 and set(TARGETS) == {"depression", "anxiety", "stress"}
    used = sorted(i for items in SUBSCALE_ITEMS.values() for i in items)
    assert used == sorted(ITEMS)                                   # each item in exactly one subscale
    assert MILD_OR_ABOVE == {"depression": 10, "anxiety": 8, "stress": 15}


def test_loader_fails_loudly_without_data(tmp_path):
    with pytest.raises(AdoDASDataError):
        AdoDASLoader(root=str(tmp_path / "absent")).load()
