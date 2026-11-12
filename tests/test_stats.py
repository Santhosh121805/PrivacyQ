import numpy as np
import pandas as pd
import pytest

from src.m8_eval.stats import PRIMARY, SENSITIVITY, add_holm, corrected_resampled_ttest, holm, wilcoxon_p


def test_primary_decision_follows_the_corrected_t_not_wilcoxon():
    df = pd.DataFrame({"metric": ["auc", "auc"], "p_nb_t": [0.30, 0.001], "wilcoxon_p": [0.0001, 0.40]})
    out = add_holm(df, ["metric"])
    assert PRIMARY == "p_nb_t_holm" and SENSITIVITY == "wilcoxon_p_holm"
    assert list(out.significant) == [False, True]                    # Wilcoxon would say the opposite
    assert list(out.p_nb_t_holm) == pytest.approx(list(holm([0.30, 0.001])))
    assert list(out.wilcoxon_p_holm) == pytest.approx(list(holm([0.0001, 0.40])))


def test_holm_is_applied_within_families_only():
    df = pd.DataFrame({"metric": ["a", "a", "b"], "p_nb_t": [0.02, 0.03, 0.02], "wilcoxon_p": [0.5, 0.5, 0.5]})
    out = add_holm(df, ["metric"])
    assert out.p_nb_t_holm.tolist() == pytest.approx([0.04, 0.04, 0.02])   # family "b" has one test
    whole = add_holm(df, None)
    assert whole.p_nb_t_holm.tolist() == pytest.approx([0.06, 0.06, 0.06])


def test_correlated_fold_scores_are_more_conservative_under_the_corrected_t():
    # Same consistent small difference on all 15 folds: Wilcoxon reaches its floor, the corrected t (which
    # inflates the variance for overlapping training sets) is far less certain.
    rng = np.random.default_rng(0)
    b = rng.normal(0.85, 0.01, 15)
    a = b + 0.004 + rng.normal(0, 0.003, 15)
    assert wilcoxon_p(a, b) < 0.01
    assert corrected_resampled_ttest(a, b, test_train_ratio=0.25)["p"] > wilcoxon_p(a, b)
