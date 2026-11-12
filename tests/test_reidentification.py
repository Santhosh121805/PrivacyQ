import pandas as pd
import pytest

from src.m9_audit.reidentification import reidentification_risk


def test_reidentification_risk_counts_classes():
    df = pd.DataFrame({"a": [1, 1, 1, 2, 3], "b": ["x", "x", "x", "y", "y"]})
    r = reidentification_risk(df, ["a", "b"], k=2)
    assert r["n_classes"] == 3 and r["uniqueness"] == pytest.approx(2 / 5)
    assert r["k_violation_rate"] == pytest.approx(2 / 5) and r["min_class_size"] == 1
    assert r["avg_prosecutor_risk"] == pytest.approx((3 * (1 / 3) + 1 + 1) / 5) and r["max_prosecutor_risk"] == 1.0


def test_missing_values_form_their_own_class_and_bad_columns_fail():
    df = pd.DataFrame({"a": [None, None, 1.0]})
    assert reidentification_risk(df, ["a"], k=2)["n_classes"] == 2
    with pytest.raises(ValueError):
        reidentification_risk(df, ["zzz"])
