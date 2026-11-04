from __future__ import annotations

import autograd.numpy as anp
import numpy as np
import pennylane as qml
import pennylane.numpy as qnp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class LogisticRegressionBaseline(BaseEstimator, ClassifierMixin):
    """Small classical baseline that maintains a familiar scikit-learn interface."""

    def __init__(
        self,
        n_features: int | None = None,
        penalty: str = "l2",
        C: float = 1.0,
        max_iter: int = 200,
        random_state: int = 0,
    ) -> None:
        self.n_features = n_features
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        values = np.asarray(X, dtype=float)
        labels = np.asarray(y).reshape(-1)
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        if values.shape[0] != labels.shape[0]:
            raise ValueError("X and y must contain the same number of rows")

        self.model_ = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model_.fit(values, labels)
        self.n_features_in_ = values.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit must be called before predict_proba")
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit must be called before predict")
        return self.model_.predict(np.asarray(X, dtype=float))


class VariationalClassifier(BaseEstimator, ClassifierMixin):
    """Tiny variational quantum classifier implemented with a statevector simulator."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 1,
        learning_rate: float = 0.05,
        n_steps: int = 200,
        seed: int = 0,
    ) -> None:
        if not 1 <= n_qubits <= 12:
            raise ValueError("n_qubits must be between 1 and 12")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.learning_rate = float(learning_rate)
        self.n_steps = int(n_steps)
        self.seed = int(seed)
        self.weights_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.device = qml.device("default.qubit", wires=self.n_qubits)

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        if values.shape[1] != self.n_qubits:
            raise ValueError(f"variational model expects {self.n_qubits} features")
        return values

    def _circuit(self, features: np.ndarray, weights: np.ndarray):
        qml.AngleEmbedding(features, wires=range(self.n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        return qml.expval(qml.PauliZ(0))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VariationalClassifier":
        features = self._prepare_features(X)
        labels = np.asarray(y).reshape(-1)
        if features.shape[0] != labels.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        self.n_features_in_ = features.shape[1]

        rng = np.random.default_rng(self.seed)
        self.weights_ = qnp.array(rng.normal(0.0, 0.1, size=(self.n_layers, self.n_qubits, 3)), requires_grad=True)

        def loss_fn(weights, feature_vector, target):
            prediction = qml.QNode(self._circuit, self.device)(feature_vector, weights)
            probability = 1.0 / (1.0 + anp.exp(-prediction))
            return -(
                target * anp.log(probability + 1e-8)
                + (1.0 - target) * anp.log(1.0 - probability + 1e-8)
            )

        for _ in range(self.n_steps):
            gradients = [
                qml.grad(lambda w, x=x_i, target=y_i: loss_fn(w, x, target))(self.weights_)
                for x_i, y_i in zip(features, labels)
            ]
            averaged = np.mean(np.stack(gradients), axis=0)
            self.weights_ -= self.learning_rate * averaged
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("fit must be called before predict_proba")
        features = self._prepare_features(X)
        raw_scores = np.array([qml.QNode(self._circuit, self.device)(x, self.weights_) for x in features])
        probabilities = _sigmoid(raw_scores)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)


classical_baselines = {
    "logistic_regression": LogisticRegressionBaseline,
    "logistic": LogisticRegressionBaseline,
    "baseline": LogisticRegressionBaseline,
    "random_forest": RandomForestClassifier,
    "rf": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def make_model(name: str, **kwargs):
    model_name = name.lower().replace("-", "_")
    if model_name in {"logistic_regression", "logistic", "baseline", "classical"}:
        return LogisticRegressionBaseline(**kwargs)
    if model_name in {"variational", "vqc", "quantum"}:
        return VariationalClassifier(**kwargs)
    if model_name in {"random_forest", "rf"}:
        return RandomForestClassifier(**kwargs)
    if model_name in {"hist_gradient_boosting", "gradient_boosting"}:
        return HistGradientBoostingClassifier(**kwargs)
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**kwargs)
    raise ValueError(f"unknown model {name!r}")
