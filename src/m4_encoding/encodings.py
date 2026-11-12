"""Quantum feature encodings with one shared interface.

Encodings are a *representation* layer only. They provide no privacy: any
privacy guarantee in this project comes from DP-SGD and federated aggregation.

Every ``Encoding`` exposes:
- ``n_features``: input dimension it consumes (n for angle/ZZ/re-uploading,
  2**n - 1 for amplitude; one amplitude slot is reserved for a constant).
- ``input_scale``: "angle" (features min-max scaled to [0, pi] on training
  rows) or "standard" (standardized features, used by amplitude encoding).
- ``prepare(X)``: final numpy-side preparation (amplitude appends the constant;
  basis binarises each angle-scaled feature at the fixed midpoint pi/2 of its
  [0, pi] range — no data-dependent threshold).
- ``feature_map(x)``: the data-only circuit (used by the quantum kernel).
- ``block(x)``: one upload of the data, interleaved with trainable layers by
  variational models when ``reupload`` is True.

All circuits accept a batch dimension (``x`` of shape (batch, n_features)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml

MIN_QUBITS, MAX_QUBITS = 4, 12
ENCODING_NAMES = ("angle", "amplitude", "zz", "reupload", "basis")
NATIVE_GATES = {"RX", "RY", "RZ", "CNOT", "Hadamard", "GlobalPhase"}


def _ring(n: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


@dataclass(frozen=True)
class Encoding:
    name: str
    n_qubits: int
    reps: int = 1

    def __post_init__(self) -> None:
        if self.name not in ENCODING_NAMES:
            raise ValueError(f"unknown encoding {self.name!r}; choose from {ENCODING_NAMES}")
        if not MIN_QUBITS <= self.n_qubits <= MAX_QUBITS:
            raise ValueError(f"n_qubits must be between {MIN_QUBITS} and {MAX_QUBITS}")
        if self.reps < 1:
            raise ValueError("reps must be at least 1")

    # ---- interface -------------------------------------------------------
    @property
    def n_wires(self) -> int:
        return self.n_qubits

    @property
    def reupload(self) -> bool:
        return self.name == "reupload"

    @property
    def input_scale(self) -> str:
        return "standard" if self.name == "amplitude" else "angle"

    def n_features(self, available: int | None = None) -> int:
        wanted = 2**self.n_qubits - 1 if self.name == "amplitude" else self.n_qubits
        return wanted if available is None else min(wanted, available)

    def prepare(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.name == "amplitude":
            if X.shape[1] > 2**self.n_qubits - 1:
                raise ValueError("too many features for amplitude encoding")
            # Constant slot keeps magnitude information that normalisation removes.
            return np.hstack([X, np.ones((X.shape[0], 1))])
        if X.shape[1] != self.n_qubits:
            raise ValueError(f"{self.name} encoding expects {self.n_qubits} features")
        if self.name == "basis":
            return (X >= np.pi / 2).astype(float)
        return X

    def feature_map(self, x) -> None:
        self._check(x)
        wires = range(self.n_qubits)
        if self.name == "angle":
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
        elif self.name == "amplitude":
            qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)
        elif self.name == "basis":
            # |b> = X^{b_1} ... X^{b_n} |0...0>; RX(pi*b) equals X^b up to a global phase and broadcasts.
            for w in wires:
                qml.RX(np.pi * x[..., w], wires=w)
        elif self.name == "zz":
            for _ in range(self.reps):
                for w in wires:
                    qml.Hadamard(wires=w)
                    qml.RZ(x[..., w], wires=w)
                for a, b in _ring(self.n_qubits):
                    qml.IsingZZ((np.pi - x[..., a]) * (np.pi - x[..., b]), wires=[a, b])
        else:  # reupload: data block repeated, separated by a fixed CNOT ring
            for _ in range(self.reps):
                qml.AngleEmbedding(x, wires=wires, rotation="Y")
                for a, b in _ring(self.n_qubits):
                    qml.CNOT(wires=[a, b])

    def block(self, x) -> None:
        """One data upload used between trainable layers (re-uploading only)."""
        self._check(x)
        qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")

    __call__ = feature_map

    def _check(self, x) -> None:
        expected = 2**self.n_qubits if self.name == "amplitude" else self.n_qubits
        if self.name == "amplitude":
            if x.shape[-1] > expected:
                raise ValueError("feature vector is too large for the configured qubits")
        elif x.shape[-1] != expected:
            raise ValueError(f"{self.name} encoding expects {expected} features")


def make_encoding(name: str, n_qubits: int, repetitions: int | None = None) -> Encoding:
    """ZZ and re-uploading default to 2 repetitions; angle/amplitude use 1."""
    if repetitions is None:
        repetitions = 2 if name in ("zz", "reupload") else 1
    return Encoding(name=name, n_qubits=n_qubits, reps=repetitions)


def circuit_resources(qnode, *args) -> dict:
    """Depth and gate counts after decomposing to {RX, RY, RZ, CNOT, H}."""
    decomposed = qml.transforms.decompose(qnode, gate_set=NATIVE_GATES)
    res = qml.specs(decomposed)(*args)["resources"]
    gates = {k: int(v) for k, v in res.gate_types.items() if k != "GlobalPhase"}
    return {
        "depth": int(res.depth),
        "n_gates": int(sum(gates.values())),
        "n_cnot": int(gates.get("CNOT", 0)),
        "gate_types": gates,
    }


def encoding_resources(encoding: Encoding) -> dict:
    dev = qml.device("default.qubit", wires=encoding.n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        encoding.feature_map(x)
        return qml.state()

    x = np.random.default_rng(0).uniform(0.1, 3.0, size=encoding.n_features() + (1 if encoding.name == "amplitude" else 0))
    return circuit_resources(circuit, x)
