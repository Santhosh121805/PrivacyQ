"""Quantum state visualisation: single-qubit Bloch vectors of encoded records.

For every record and qubit, the reduced density matrix rho_q of the encoded statevector gives the
Bloch vector r = (Tr rho_q X, Tr rho_q Y, Tr rho_q Z). Its length |r| is 1 when the qubit is in a
pure state (no entanglement with the other qubits) and below 1 when it is entangled. This is a
visual and descriptive tool only; it makes no claim about predictive value and gives no privacy.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from src.m4_encoding import make_encoding

_PAULI = {"x": np.array([[0, 1], [1, 0]]), "y": np.array([[0, -1j], [1j, 0]]), "z": np.diag([1, -1])}


def bloch_vectors(encoding: str, n_qubits: int, X: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Exact reduced-state Bloch vectors, shape (rows, n_qubits, 3), from ``default.qubit``."""
    enc = make_encoding(encoding, n_qubits)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def reduced(x, wire):
        enc.feature_map(x)
        return qml.density_matrix(wires=[wire])

    x = np.asarray(enc.prepare(X), dtype=float)
    out = np.empty((len(x), n_qubits, 3))
    for q in range(n_qubits):
        rho = np.concatenate([np.asarray(reduced(x[i:i + chunk], q)).reshape(-1, 2, 2)
                              for i in range(0, len(x), chunk)])
        for k, P in enumerate(_PAULI.values()):
            out[:, q, k] = np.einsum("nij,ji->n", rho, P).real
    return out


def statevector(encoding: str, n_qubits: int, x: np.ndarray) -> np.ndarray:
    """Exact encoded statevector of one record (2**n_qubits complex amplitudes), from ``default.qubit``."""
    enc = make_encoding(encoding, n_qubits)

    @qml.qnode(qml.device("default.qubit", wires=n_qubits))
    def state(v):
        enc.feature_map(v)
        return qml.state()

    return np.asarray(state(np.asarray(enc.prepare(np.atleast_2d(x)), dtype=float)[0]))


def class_separation(vectors: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per qubit: distance between the two classes' mean Bloch vectors (0 = no separation, max 2)."""
    y = np.asarray(y)
    return np.linalg.norm(vectors[y == 1].mean(0) - vectors[y == 0].mean(0), axis=-1)
