import numpy as np
import pandas as pd
import pytest

from src.m2_preprocess.preprocessing import k_anonymity_report
from src.m7_privacy.input_privacy import KAnonymizer, LaplaceLDP

QI = {"age": [11, 13, 15, 17], "sex": None, "race": None}


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(12, 18, n).astype(float),
        "sex": rng.integers(1, 3, n).astype(float),
        "race": rng.choice(["c1", "c2", "c3", "c4", "c5", "c6", "c7"], n, p=[.6, .2, .1, .05, .03, .015, .005]),
        "x": rng.normal(size=n),
    })


def test_kanonymizer_makes_training_rows_k_anonymous():
    df = _frame()
    anon = KAnonymizer(k=5, quasi_identifiers=QI).fit(df)
    out = anon.transform(df)
    assert k_anonymity_report(out, list(QI), 5)["compliant"] or anon.report_["n_suppressed"] < 5
    assert anon.report_["k_anonymous"] == k_anonymity_report(out, list(QI), 5)["compliant"]
    assert set(out["age"].dropna().unique()) <= {1.0, 2.0, 3.0}          # generalised into bands
    pd.testing.assert_series_equal(out["x"], df["x"])                     # non-QI columns untouched
    assert len(out) == len(df)                                            # local suppression, no rows dropped


def test_kanonymizer_suppresses_rare_combinations_only():
    df = _frame()
    anon = KAnonymizer(k=5, quasi_identifiers=QI).fit(df)
    out = anon.transform(df)
    suppressed = out[list(QI)].isna().all(axis=1)
    assert suppressed.sum() == anon.report_["n_suppressed"] > 0
    keys = anon._keys(anon._generalise(df))
    counts = keys.value_counts()
    assert (counts[keys[suppressed]] < 5).all() and (counts[keys[~suppressed]] >= 5).all()


def test_kanonymizer_rule_is_fitted_on_training_rows_only():
    train, test = _frame(seed=1), _frame(n=50, seed=2)
    anon = KAnonymizer(k=5, quasi_identifiers=QI).fit(train)
    safe_before = set(anon.safe_keys_)
    anon.transform(test)
    assert anon.safe_keys_ == safe_before


def test_ldp_noise_scale_and_bounds():
    ldp = LaplaceLDP(epsilon=8.0, seed=0)
    assert ldp.noise_scale(8) == pytest.approx(8 * np.pi / 8.0)
    X = np.full((20000, 8), np.pi / 2)
    noisy = ldp.transform(X)
    assert noisy.min() >= 0 and noisy.max() <= np.pi
    unclipped = X + np.random.default_rng(0).laplace(0, ldp.noise_scale(8), size=X.shape)
    assert np.mean(np.abs(unclipped - X)) == pytest.approx(ldp.noise_scale(8), rel=0.05)  # Laplace mean |noise| = b


def test_ldp_infinite_epsilon_is_identity_and_smaller_epsilon_is_noisier():
    X = np.random.default_rng(3).uniform(0, np.pi, (500, 4))
    np.testing.assert_allclose(LaplaceLDP(epsilon=None).transform(X), X)
    err_hi = np.abs(LaplaceLDP(epsilon=64, seed=1).transform(X) - X).mean()
    err_lo = np.abs(LaplaceLDP(epsilon=4, seed=1).transform(X) - X).mean()
    assert err_lo > err_hi
    with pytest.raises(ValueError):
        LaplaceLDP(epsilon=0).transform(X)
