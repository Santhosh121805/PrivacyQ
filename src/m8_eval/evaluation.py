from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(np.asarray(y_true), np.asarray(y_pred)))


def utility_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(labels, np.asarray(y_score)))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def compute_communication_cost(
    n_parameters: int,
    rounds: int,
    n_clients: int,
    communication_fraction: float = 1.0,
) -> float:
    if n_parameters < 0 or rounds < 0 or n_clients < 0:
        raise ValueError("n_parameters, rounds, and n_clients must be non-negative")
    return float(n_parameters * rounds * n_clients * communication_fraction)


def compute_compute_cost(n_samples: int, n_features: int, rounds: int, n_clients: int) -> float:
    if n_samples < 0 or n_features < 0 or rounds < 0 or n_clients < 0:
        raise ValueError("all inputs must be non-negative")
    return float(n_samples * n_features * rounds * n_clients)


HEADLINE_METRICS = ("balanced_accuracy", "recall", "roc_auc")


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Imbalance-aware metrics. Headline: balanced accuracy, recall, ROC-AUC.

    ``y_score`` is the positive-class probability. Models are trained with class
    weights, so the fixed 0.5 threshold approximates a balanced operating point;
    the threshold is never tuned on evaluation data.
    """
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix

    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(labels, scores)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "predicted_positive_rate": float(predictions.mean()),
        "prevalence": float(labels.mean()),
    }
