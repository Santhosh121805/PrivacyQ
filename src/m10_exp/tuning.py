"""Shared model selection, applied identically to every model.

Inside each outer CV fold, the fold's training rows are split (stratified)
into inner-train and inner-val. For each hyperparameter setting the model is
fitted on inner-train (torch models may early-stop on inner-val), and the
setting with the best inner-val ROC-AUC is kept. The decision threshold is the
one maximising balanced accuracy (Youden's J) on inner-val scores. The selected
inner-train model is then scored once on the outer validation fold. Outer
validation folds and the locked hold-out are never used for any choice.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


def scores_of(model, X) -> np.ndarray:
    if hasattr(model, "decision_scores"):
        return np.asarray(model.decision_scores(X), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def youden_threshold(y, scores) -> float:
    fpr, tpr, thresholds = roc_curve(y, scores)
    best = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best])
    return threshold if np.isfinite(threshold) else float(np.max(scores))


def grid_points(grid: dict | None) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def inner_split(positions: np.ndarray, labels: np.ndarray, fraction: float, seed: int):
    inner_train, inner_val = train_test_split(
        positions, test_size=fraction, stratify=labels[positions], random_state=seed
    )
    return np.sort(inner_train), np.sort(inner_val)


def stratified_subsample(positions: np.ndarray, labels: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    if n is None or n >= len(positions):
        return positions
    keep, _ = train_test_split(positions, train_size=n, stratify=labels[positions], random_state=seed)
    return np.sort(keep)


def select_and_fit(factory, grid: dict | None, X_tr, y_tr, X_iv, y_iv) -> dict:
    """Fit every grid point on inner-train, pick by inner-val ROC-AUC."""
    best = None
    log = []
    total_seconds = 0.0
    for params in grid_points(grid):
        model = factory(**params)
        t0 = time.perf_counter()
        if getattr(model, "supports_eval_set", False):
            model.fit(X_tr, y_tr, eval_set=(X_iv, y_iv))
        else:
            model.fit(X_tr, y_tr)
        seconds = time.perf_counter() - t0
        total_seconds += seconds
        val_scores = scores_of(model, X_iv)
        auc = float(roc_auc_score(y_iv, val_scores))
        log.append({**params, "inner_val_auc": auc, "fit_seconds": seconds})
        if best is None or auc > best["inner_val_auc"]:
            best = {"model": model, "params": params, "inner_val_auc": auc,
                    "fit_seconds": seconds, "val_scores": val_scores}
    best["threshold"] = youden_threshold(y_iv, best["val_scores"])
    best["grid_log"] = log
    best["tuning_seconds"] = total_seconds
    return best
