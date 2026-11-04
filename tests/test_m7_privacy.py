import importlib.util

import pytest

from src.m7_privacy import PrivacyReport, epsilon_for, noise_for


def test_hand_written_accountant_is_gone():
    assert importlib.util.find_spec("src.m7_privacy.privacy") is None
    import src.m7_privacy as m7
    assert not hasattr(m7, "rdp_accountant")


def test_epsilon_round_trips_through_opacus():
    sigma = noise_for(target_epsilon=2.0, sample_rate=0.01, steps=1000, delta=1e-5)
    eps = epsilon_for(sigma, sample_rate=0.01, steps=1000, delta=1e-5)
    assert eps == pytest.approx(2.0, abs=0.05)


def test_more_noise_means_smaller_epsilon():
    assert epsilon_for(2.0, 0.01, 500, 1e-5) < epsilon_for(1.0, 0.01, 500, 1e-5)


def test_non_private_models_report_no_epsilon():
    report = PrivacyReport.not_private("kernel SVM is not gradient-trained")
    assert report.epsilon is None and not report.dp_trained
