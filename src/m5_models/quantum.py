"""Quantum models on a CPU statevector simulator (PennyLane ``default.qubit``).

The quantum circuit is a representation/feature layer. It provides NO privacy;
privacy in this project comes only from DP-SGD and federated aggregation.

Simulator choice: ``default.qubit`` with torch backprop and parameter
broadcasting evaluates a whole batch in one vectorised statevector pass. On this
workload (<= 8 qubits, ~1,200-row batches) it was ~70x faster than
``lightning.qubit`` with adjoint differentiation, which processes samples one
at a time (benchmark: 0.33 s vs 23 s per full-batch forward+backward, 8 qubits).

Models
------
- ``QuantumKernelSVC``: fidelity kernel k(x, x') = |<phi(x)|phi(x')>|^2 passed
  to ``SVC(kernel="precomputed")``. On a simulator the kernel is computed
  exactly from statevectors (one circuit per row, not per pair); on hardware it
  would be estimated with overlap circuits and shot noise. It is not
  gradient-trained, so DP-SGD does not apply to it.
- ``VQCClassifier``: encoding + StronglyEntanglingLayers, readout <Z_0>,
  logit = a*<Z_0> + b (two classical scalars). Kept shallow (1-2 layers) with
  small-angle initialisation because deep random circuits suffer from barren
  plateaus (vanishing gradients).
- ``HybridQNNClassifier``: Linear compression -> quantum layer (readout <Z_i>
  on every qubit) -> Linear head.

All torch models are trained with Adam on positive-class-weighted BCE and use
early stopping on a validation split carved from the training rows.
"""

from __future__ import annotations

import copy

import numpy as np
import pennylane as qml
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from src.m4_encoding import Encoding, make_encoding
from src.m5_models.classical import positive_weight

DTYPE = torch.float64


def _device(n_qubits: int):
    return qml.device("default.qubit", wires=n_qubits)


def _variational_layers(encoding: Encoding, x, weights) -> None:
    n = encoding.n_qubits
    n_layers = weights.shape[0]
    if encoding.reupload:
        for layer in range(n_layers):
            encoding.block(x)
            qml.StronglyEntanglingLayers(
                weights[layer : layer + 1], wires=range(n), ranges=[layer % (n - 1) + 1]
            )
    else:
        encoding.feature_map(x)
        qml.StronglyEntanglingLayers(weights, wires=range(n))


# ---------------------------------------------------------------- QSVM ----
class QuantumKernelSVC(BaseEstimator, ClassifierMixin):
    supports_eval_set = False

    def __init__(self, encoding: str = "zz", n_qubits: int = 4, C: float = 1.0, seed: int = 0):
        self.encoding = encoding
        self.n_qubits = n_qubits
        self.C = C
        self.seed = seed

    def _encoding(self) -> Encoding:
        return make_encoding(self.encoding, self.n_qubits)

    def _states(self, X: np.ndarray) -> np.ndarray:
        enc = self._encoding()

        @qml.qnode(_device(enc.n_qubits), interface="torch")
        def state(x):
            enc.feature_map(x)
            return qml.state()

        x = torch.as_tensor(enc.prepare(X), dtype=DTYPE)
        with torch.no_grad():
            out = state(x)
        return out.numpy().reshape(len(X), -1)

    @staticmethod
    def _kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs(a @ b.conj().T) ** 2

    def fit(self, X, y, eval_set=None) -> "QuantumKernelSVC":
        self.train_states_ = self._states(X)
        self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight="balanced", random_state=self.seed)
        self.svc_.fit(self._kernel(self.train_states_, self.train_states_), np.asarray(y))
        self.classes_ = np.array([0, 1])
        return self

    def decision_scores(self, X) -> np.ndarray:
        return self.svc_.decision_function(self._kernel(self._states(X), self.train_states_))

    def n_parameters(self) -> int:
        return int(self.svc_.dual_coef_.size + 1)  # dual coefficients + intercept


# ------------------------------------------------------- torch models -----
class _TorchQuantumBase(BaseEstimator, ClassifierMixin):
    supports_eval_set = True

    def _build(self, n_inputs: int) -> torch.nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    def _prepare(self, X) -> torch.Tensor:
        return torch.as_tensor(np.asarray(X, dtype=float), dtype=DTYPE)

    def fit(self, X, y, eval_set=None):
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        X_t, y_t = self._prepare(X), torch.as_tensor(np.asarray(y), dtype=DTYPE)
        self.module_ = self._build(X_t.shape[1])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight(y), dtype=DTYPE))
        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.learning_rate)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t), batch_size=self.batch_size,
            shuffle=True, generator=generator,
        )
        X_v = y_v = None
        if eval_set is not None:
            X_v, y_v = self._prepare(eval_set[0]), np.asarray(eval_set[1])
        best_auc, best_state, best_epoch = -np.inf, None, self.epochs
        self.history_ = []
        for epoch in range(1, self.epochs + 1):
            self.module_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss_fn(self.module_(xb), yb).backward()
                optimizer.step()
            if X_v is not None:
                auc = float(roc_auc_score(y_v, self._scores_tensor(X_v)))
                self.history_.append(auc)
                if auc > best_auc:
                    best_auc, best_state, best_epoch = auc, copy.deepcopy(self.module_.state_dict()), epoch
        if best_state is not None:
            self.module_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        self.classes_ = np.array([0, 1])
        return self

    def _scores_tensor(self, X_t: torch.Tensor) -> np.ndarray:
        self.module_.eval()
        with torch.no_grad():
            return self.module_(X_t).numpy()

    def decision_scores(self, X) -> np.ndarray:
        return self._scores_tensor(self._prepare(X))

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.module_.parameters()))


class _VQCModule(torch.nn.Module):
    def __init__(self, encoding: Encoding, n_layers: int, init_scale: float):
        super().__init__()
        n = encoding.n_qubits
        self.encoding = encoding
        self.weights = torch.nn.Parameter(init_scale * torch.randn(n_layers, n, 3, dtype=DTYPE))
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=DTYPE))
        self.bias = torch.nn.Parameter(torch.tensor(0.0, dtype=DTYPE))

        @qml.qnode(_device(n), interface="torch", diff_method="backprop")
        def circuit(x, weights):
            _variational_layers(encoding, x, weights)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x):
        return self.scale * self.circuit(x, self.weights) + self.bias


class VQCClassifier(_TorchQuantumBase):
    def __init__(self, encoding: str = "angle", n_qubits: int = 4, n_layers: int = 2,
                 learning_rate: float = 0.05, epochs: int = 30, batch_size: int = 64,
                 init_scale: float = 0.1, seed: int = 0):
        self.encoding = encoding
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_scale = init_scale
        self.seed = seed

    def _prepare(self, X) -> torch.Tensor:
        enc = make_encoding(self.encoding, self.n_qubits)
        return torch.as_tensor(enc.prepare(X), dtype=DTYPE)

    def _build(self, n_inputs: int) -> torch.nn.Module:
        return _VQCModule(make_encoding(self.encoding, self.n_qubits), self.n_layers, self.init_scale)


class _HybridModule(torch.nn.Module):
    def __init__(self, n_inputs: int, encoding: Encoding, n_layers: int, init_scale: float):
        super().__init__()
        n = encoding.n_qubits
        self.encoding = encoding
        self.compress = torch.nn.Linear(n_inputs, encoding.n_features(n_inputs), dtype=DTYPE)
        self.weights = torch.nn.Parameter(init_scale * torch.randn(n_layers, n, 3, dtype=DTYPE))
        self.head = torch.nn.Linear(n, 1, dtype=DTYPE)

        @qml.qnode(_device(n), interface="torch", diff_method="backprop")
        def circuit(x, weights):
            _variational_layers(encoding, x, weights)
            return [qml.expval(qml.PauliZ(w)) for w in range(n)]

        self.circuit = circuit

    def forward(self, x):
        z = self.compress(x)
        if self.encoding.name == "amplitude":
            z = torch.cat([z, torch.ones(z.shape[0], 1, dtype=DTYPE)], dim=1)
        else:
            z = np.pi * torch.sigmoid(z)  # angles in [0, pi], like the angle scaler
        q = torch.stack(self.circuit(z, self.weights), dim=-1)
        return self.head(q).squeeze(-1)


class HybridQNNClassifier(_TorchQuantumBase):
    def __init__(self, encoding: str = "angle", n_qubits: int = 4, n_layers: int = 2,
                 learning_rate: float = 0.01, epochs: int = 30, batch_size: int = 64,
                 init_scale: float = 0.1, seed: int = 0):
        self.encoding = encoding
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_scale = init_scale
        self.seed = seed

    def _build(self, n_inputs: int) -> torch.nn.Module:
        return _HybridModule(n_inputs, make_encoding(self.encoding, self.n_qubits), self.n_layers, self.init_scale)


QUANTUM_MODELS = {"qsvm": QuantumKernelSVC, "vqc": VQCClassifier, "hybrid": HybridQNNClassifier}


def vqc_resources(encoding: str, n_qubits: int, n_layers: int) -> dict:
    """Depth/gate counts of the full VQC circuit after native-gate decomposition."""
    from src.m4_encoding import circuit_resources

    enc = make_encoding(encoding, n_qubits)

    @qml.qnode(_device(n_qubits))
    def circuit(x, weights):
        _variational_layers(enc, x, weights)
        return qml.expval(qml.PauliZ(0))

    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 3.0, size=enc.n_features() + (1 if enc.name == "amplitude" else 0))
    return circuit_resources(circuit, x, rng.normal(size=(n_layers, n_qubits, 3)))
