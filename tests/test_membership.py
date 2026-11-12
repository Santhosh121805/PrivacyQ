import numpy as np
import pytest

from src.m9_audit.membership import loss_threshold_attack, margin_loss, per_example_loss


def test_attack_auc_is_half_when_losses_are_identically_distributed():
    rng = np.random.default_rng(0)
    res = loss_threshold_attack(rng.exponential(1, 20000), rng.exponential(1, 20000))
    assert res["attack_auc"] == pytest.approx(0.5, abs=0.01)
    assert res["advantage"] < 0.03


def test_attack_detects_memorised_members():
    rng = np.random.default_rng(1)
    res = loss_threshold_attack(rng.exponential(0.1, 2000), rng.exponential(1.0, 2000))
    assert res["attack_auc"] > 0.85 and res["tpr_at_1pct_fpr"] > 0.05


def test_per_example_loss_and_margin_loss():
    assert per_example_loss(np.array([1, 0]), np.array([0.9, 0.1])) == pytest.approx([-np.log(0.9)] * 2)
    np.testing.assert_allclose(margin_loss(np.array([1, 0]), np.array([2.0, 2.0])), [-2.0, 2.0])
    with pytest.raises(ValueError):
        loss_threshold_attack(np.array([]), np.array([1.0]))
