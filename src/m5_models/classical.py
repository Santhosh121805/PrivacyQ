"""Classical baselines with class-imbalance handling via class weights.

Imbalance is handled with class weights / positive-class weights, not SMOTE:
oversampling duplicates or interpolates individual records, which breaks the
per-example sensitivity assumption DP-SGD accounting relies on.

Hyperparameters are fixed, reasonable defaults chosen before looking at any
results; they are not tuned on validation or test folds.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

CLASSICAL_MODELS = ("logreg", "random_forest", "xgboost", "mlp", "svm")


def positive_weight(y: np.ndarray) -> float:
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("training labels contain no positive examples")
    return float((y == 0).sum() / n_pos)


def build_mlp(n_features: int, hidden: tuple[int, ...] = (64, 32)) -> torch.nn.Module:
    """Plain MLP returning one logit. Uses only Opacus-compatible layers."""
    layers: list[torch.nn.Module] = []
    width = n_features
    for h in hidden:
        layers += [torch.nn.Linear(width, h), torch.nn.ReLU()]
        width = h
    layers.append(torch.nn.Linear(width, 1))
    return torch.nn.Sequential(*layers)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """MLP trained with Adam on positive-class-weighted binary cross-entropy.

    With ``eval_set`` the weights from the epoch with the best eval ROC-AUC are
    kept (early stopping on the inner validation split).
    """

    supports_eval_set = True

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        epochs: int = 40,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        balance: bool = True,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.balance = balance
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None) -> "TorchMLPClassifier":
        import copy

        from sklearn.metrics import roc_auc_score

        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        self.model_ = build_mlp(X_t.shape[1], tuple(self.hidden))
        pos_weight = torch.tensor(positive_weight(y) if self.balance else 1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        best_auc, best_state, self.best_epoch_ = -np.inf, None, self.epochs
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(self.model_(xb).squeeze(-1), yb).backward()
                optimizer.step()
            if eval_set is not None:
                auc = roc_auc_score(eval_set[1], self.decision_function(eval_set[0]))
                if auc > best_auc:
                    best_auc, best_state, self.best_epoch_ = auc, copy.deepcopy(self.model_.state_dict()), epoch
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X_t.shape[1]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.as_tensor(np.asarray(X), dtype=torch.float32)).squeeze(-1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-np.clip(self.decision_function(X), -50, 50)))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SVMClassifier(BaseEstimator, ClassifierMixin):
    """Classical RBF-kernel SVM with balanced class weights.

    The classical counterpart of the quantum-kernel SVM: same SVC machinery, the
    kernel is exp(-gamma * ||x - x'||^2) instead of a quantum state fidelity.
    Scores are SVM decision values (not probabilities).
    """

    def __init__(self, C: float = 1.0, gamma: str | float = "scale", seed: int = 0):
        self.C = C
        self.gamma = gamma
        self.seed = seed

    def fit(self, X, y) -> "SVMClassifier":
        from sklearn.svm import SVC

        self.svc_ = SVC(kernel="rbf", C=self.C, gamma=self.gamma, class_weight="balanced", random_state=self.seed)
        self.svc_.fit(np.asarray(X, dtype=float), np.asarray(y))
        self.classes_ = np.array([0, 1])
        return self

    def decision_scores(self, X) -> np.ndarray:
        return self.svc_.decision_function(np.asarray(X, dtype=float))

    def n_parameters(self) -> int:
        return int(self.svc_.dual_coef_.size + 1)


def make_classical(
    name: str, y_train: np.ndarray, seed: int = 0, n_jobs: int = 4, **overrides
):
    """Return an unfitted, class-weighted classical baseline."""
    if name == "logreg":
        params = dict(C=1.0, max_iter=5000, class_weight="balanced", random_state=seed)
        return LogisticRegression(**{**params, **overrides})
    if name == "random_forest":
        params = dict(
            n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=n_jobs, random_state=seed,
        )
        return RandomForestClassifier(**{**params, **overrides})
    if name == "xgboost":
        params = dict(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
            scale_pos_weight=positive_weight(y_train), eval_metric="logloss",
            n_jobs=n_jobs, random_state=seed,
        )
        return XGBClassifier(**{**params, **overrides})
    if name == "mlp":
        return TorchMLPClassifier(seed=seed, **overrides)
    if name == "svm":
        return SVMClassifier(seed=seed, **overrides)
    raise ValueError(f"unknown classical model {name!r}; choose from {CLASSICAL_MODELS}")


def count_parameters(model) -> int | None:
    """Trainable parameter count where it is well defined (None for trees)."""
    if isinstance(model, LogisticRegression):
        return int(model.coef_.size + model.intercept_.size)
    if isinstance(model, TorchMLPClassifier):
        return int(sum(p.numel() for p in model.model_.parameters()))
    return None
