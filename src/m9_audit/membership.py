"""Membership inference: does a trained model reveal who was in its training set?

Loss-threshold attack (Yeom et al., 2018): an attacker who knows a record and its
label computes the model's loss on it and guesses "member" when the loss is low.
Measured on an equal number of members (rows the model was trained on) and
non-members (rows from the same population it never saw).

Reported:
- attack ROC-AUC (0.5 = the attacker cannot tell members from non-members),
- TPR at 1 % FPR (how many members are confidently identified),
- membership advantage = max(TPR − FPR) (Yeom et al.), which differential
  privacy bounds by e^ε − 1 for an ε-DP training algorithm.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def per_example_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Binary cross-entropy of each example given its predicted positive probability."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def margin_loss(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Loss-like score for margin classifiers without probabilities (e.g. an SVM decision value)."""
    sign = np.where(np.asarray(y) == 1, 1.0, -1.0)
    return -sign * np.asarray(score, dtype=float)


def loss_threshold_attack(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> dict:
    m, n = np.asarray(member_losses, float), np.asarray(nonmember_losses, float)
    if len(m) == 0 or len(n) == 0:
        raise ValueError("need members and non-members")
    truth = np.r_[np.ones(len(m)), np.zeros(len(n))]
    attack_score = -np.r_[m, n]                    # lower loss -> more likely a member
    fpr, tpr, _ = roc_curve(truth, attack_score)
    return {
        "attack_auc": float(roc_auc_score(truth, attack_score)),
        "tpr_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
        "advantage": float(np.max(tpr - fpr)),
        "n_members": int(len(m)), "n_nonmembers": int(len(n)),
        "mean_member_loss": float(m.mean()), "mean_nonmember_loss": float(n.mean()),
    }
