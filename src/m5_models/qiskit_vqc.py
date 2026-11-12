"""Independent Qiskit implementation of the VQC forward pass, used to cross-check PennyLane.

The circuit is rebuilt gate for gate from the PennyLane definition in ``src/m5_models/quantum.py``
(``_variational_layers``) without calling PennyLane:

- encoding "angle": RY(x_i) on wire i, then all variational layers;
  encoding "reupload": for each layer l, RY(x_i) on wire i, then variational layer l;
- variational layer l (PennyLane ``StronglyEntanglingLayers``): on every wire i,
  Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi) with (phi, theta, omega) = weights[l, i];
  then CNOT(i, (i + r) mod n) for i = 0..n-1, with range r = (l mod (n - 1)) + 1;
- readout <Z_0>; logit = scale * <Z_0> + bias (the two classical scalars of the PennyLane module).

Qiskit's ``Statevector`` simulator computes the exact expectation. Qiskit numbers qubits
little-endian, but wire i is qubit i in both frameworks and Z is measured on qubit 0, so the
expectation value is the same physical quantity. Training stays in PennyLane (torch backprop);
this module evaluates trained parameters. The circuit gives no privacy.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from src.m4_encoding import make_encoding

SUPPORTED = ("angle", "reupload")


def _layer(qc: QuantumCircuit, w: np.ndarray, r: int) -> None:
    n = qc.num_qubits
    for i in range(n):
        phi, theta, omega = (float(v) for v in w[i])
        qc.rz(phi, i)
        qc.ry(theta, i)
        qc.rz(omega, i)
    for i in range(n):
        qc.cx(i, (i + r) % n)


def vqc_circuit(encoding: str, x: np.ndarray, weights: np.ndarray) -> QuantumCircuit:
    """Qiskit circuit for one prepared input row ``x`` and weights of shape (n_layers, n_qubits, 3)."""
    if encoding not in SUPPORTED:
        raise ValueError(f"Qiskit VQC supports {SUPPORTED}, not {encoding!r}")
    n_layers, n, _ = weights.shape
    qc = QuantumCircuit(n)
    for layer in range(n_layers):
        if encoding == "reupload" or layer == 0:
            for i in range(n):
                qc.ry(float(x[i]), i)
        _layer(qc, weights[layer], layer % (n - 1) + 1)
    return qc


def expval_z0(encoding: str, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = weights.shape[1]
    z0 = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=n)
    return np.array([Statevector(vqc_circuit(encoding, x, weights)).expectation_value(z0).real for x in X])


def qiskit_decision_scores(model, X) -> np.ndarray:
    """Logits of a fitted PennyLane ``VQCClassifier`` recomputed entirely in Qiskit."""
    enc = make_encoding(model.encoding, model.n_qubits)
    module = model.module_
    weights = module.weights.detach().numpy()
    z = expval_z0(model.encoding, enc.prepare(X), weights)
    return float(module.scale.detach()) * z + float(module.bias.detach())
