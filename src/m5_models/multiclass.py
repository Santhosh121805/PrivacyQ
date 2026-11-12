"""Three-class models (none / anxiety only / depression with or without anxiety).

Every model is class-weighted ("balanced": weight inversely proportional to class frequency) and
exposes ``class_scores(X)`` -> (n, 3) per-class scores (probabilities or one-vs-rest decision
values) used for one-vs-rest ROC-AUC, and ``predict(X)`` used for macro-F1 / balanced accuracy.

The QSVM uses the same exact statevector fidelity kernel as the binary ``QuantumKernelSVC``;
sklearn's SVC handles three classes natively (one-vs-one internally, one-vs-rest decision values
reported). The circuit gives no privacy.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.m5_models.quantum import QuantumKernelSVC, VQCClassifier

N_CLASSES = 3
CLASS_NAMES = ("none", "anxiety_only", "depression")


class _Base(BaseEstimator, ClassifierMixin):
    def predict(self, X) -> np.ndarray:
        return np.argmax(self.class_scores(X), axis=1)


class MulticlassLogReg(_Base):
    def __init__(self, C: float = 1.0, seed: int = 0):
        self.C = C
        self.seed = seed

    def fit(self, X, y):
        self.model_ = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000,
                                         random_state=self.seed).fit(X, y)
        return self

    def class_scores(self, X):
        return self.model_.predict_proba(X)

    def n_parameters(self) -> int:
        return int(self.model_.coef_.size + self.model_.intercept_.size)


class MulticlassXGB(_Base):
    def __init__(self, max_depth: int = 4, learning_rate: float = 0.05, seed: int = 0):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y):
        self.model_ = XGBClassifier(n_estimators=400, max_depth=self.max_depth, learning_rate=self.learning_rate,
                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                                    objective="multi:softprob", eval_metric="mlogloss", n_jobs=1,
                                    random_state=self.seed)
        self.model_.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        return self

    def class_scores(self, X):
        return self.model_.predict_proba(X)

    def n_parameters(self):
        return None


class MulticlassSVM(_Base):
    def __init__(self, C: float = 1.0, seed: int = 0):
        self.C = C
        self.seed = seed

    def fit(self, X, y):
        self.model_ = SVC(C=self.C, kernel="rbf", gamma="scale", class_weight="balanced",
                          decision_function_shape="ovr", random_state=self.seed).fit(X, y)
        return self

    def class_scores(self, X):
        return self.model_.decision_function(X)

    def predict(self, X):
        return self.model_.predict(X)

    def n_parameters(self) -> int:
        return int(self.model_.dual_coef_.size + self.model_.intercept_.size)


class MulticlassQSVM(_Base):
    def __init__(self, encoding: str = "angle", n_qubits: int = 8, C: float = 1.0, seed: int = 0):
        self.encoding = encoding
        self.n_qubits = n_qubits
        self.C = C
        self.seed = seed

    def _states(self, X):
        return QuantumKernelSVC(encoding=self.encoding, n_qubits=self.n_qubits)._states(X)

    def fit(self, X, y):
        self.train_states_ = self._states(X)
        K = QuantumKernelSVC._kernel(self.train_states_, self.train_states_)
        self.model_ = SVC(kernel="precomputed", C=self.C, class_weight="balanced",
                          decision_function_shape="ovr", random_state=self.seed).fit(K, y)
        return self

    def _k(self, X):
        return QuantumKernelSVC._kernel(self._states(X), self.train_states_)

    def class_scores(self, X):
        return self.model_.decision_function(self._k(X))

    def predict(self, X):
        return self.model_.predict(self._k(X))

    def n_parameters(self) -> int:
        return int(self.model_.dual_coef_.size + self.model_.intercept_.size)


class MulticlassVQC(_Base):
    """One-vs-rest VQC: one binary ``VQCClassifier`` per class (class k vs the rest), each with its own
    positive-class weight and early stopping on its own one-vs-rest inner-val AUC (same protocol as
    the binary VQC). Per-class scores are the three binary logits; prediction is their argmax."""

    supports_eval_set = True

    def __init__(self, encoding: str = "reupload", n_qubits: int = 4, n_layers: int = 1,
                 learning_rate: float = 0.05, epochs: int = 30, batch_size: int = 64, init_scale: float = 0.1,
                 seed: int = 0):
        self.encoding = encoding
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_scale = init_scale
        self.seed = seed

    def fit(self, X, y, eval_set=None):
        y = np.asarray(y)
        self.models_ = []
        for k in range(N_CLASSES):
            ev = None if eval_set is None else (eval_set[0], (np.asarray(eval_set[1]) == k).astype(int))
            m = VQCClassifier(encoding=self.encoding, n_qubits=self.n_qubits, n_layers=self.n_layers,
                              learning_rate=self.learning_rate, epochs=self.epochs, batch_size=self.batch_size,
                              init_scale=self.init_scale, seed=self.seed + k)
            self.models_.append(m.fit(X, (y == k).astype(int), eval_set=ev))
        return self

    def class_scores(self, X):
        return np.column_stack([m.decision_scores(X) for m in self.models_])

    def n_parameters(self) -> int:
        return int(sum(m.n_parameters() for m in self.models_))


MULTICLASS_MODELS = {"logreg": MulticlassLogReg, "xgboost": MulticlassXGB, "svm": MulticlassSVM,
                     "qsvm": MulticlassQSVM, "vqc": MulticlassVQC}
