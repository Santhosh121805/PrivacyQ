import numpy as np
import pytest

from src.m8_eval.screening import band_summary, risk_band, risk_cutpoints


def test_cutpoints_meet_the_stated_sensitivity_and_specificity():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(40), np.zeros(260)].astype(int)
    p = np.r_[rng.beta(5, 2, 40), rng.beta(2, 5, 260)]
    cuts = risk_cutpoints(p, y, 0.90, 0.90)
    assert np.mean(p[y == 1] >= cuts["low_moderate"]) >= 0.90           # at most 10 % of positives called "low"
    assert np.mean(p[y == 0] < cuts["moderate_high"]) >= 0.90           # at most 10 % of negatives called "high"
    assert cuts["low_moderate"] <= cuts["moderate_high"]


def test_band_assignment_and_summary():
    cuts = {"low_moderate": 0.3, "moderate_high": 0.7}
    assert [risk_band(v, cuts) for v in (0.1, 0.3, 0.69, 0.7, 0.95)] == ["low", "moderate", "moderate", "high", "high"]
    s = band_summary([0.1, 0.5, 0.9, 0.95], [0, 0, 1, 1], cuts)
    assert s == {"low": {"rows": 1, "positives": 0}, "moderate": {"rows": 1, "positives": 0},
                 "high": {"rows": 2, "positives": 2}}


def test_overlapping_rules_collapse_the_moderate_band_and_one_class_is_refused():
    p, y = np.array([0.9, 0.95, 0.1, 0.99]), np.array([1, 1, 0, 0])       # a negative scores highest
    cuts = risk_cutpoints(p, y, 0.90, 0.90)
    assert cuts["moderate_high"] >= cuts["low_moderate"]
    with pytest.raises(ValueError):
        risk_cutpoints([0.2, 0.3], [0, 0])
