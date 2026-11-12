import numpy as np
import pennylane as qml
import pytest

from src.m4_encoding import make_encoding
from src.m5_models.qiskit_vqc import expval_z0, qiskit_decision_scores, vqc_circuit
from src.m5_models.quantum import VQCClassifier, _variational_layers


def _pennylane_z0(encoding, X, weights):
    enc = make_encoding(encoding, weights.shape[1])

    @qml.qnode(qml.device("default.qubit", wires=weights.shape[1]))
    def circuit(x):
        _variational_layers(enc, x, weights)
        return qml.expval(qml.PauliZ(0))

    return np.array([float(circuit(x)) for x in X])


@pytest.mark.parametrize("encoding,n_qubits,n_layers", [("reupload", 4, 1), ("reupload", 4, 2), ("reupload", 6, 3),
                                                        ("angle", 4, 2), ("angle", 5, 3)])
def test_qiskit_forward_pass_equals_pennylane(encoding, n_qubits, n_layers):
    rng = np.random.default_rng(n_qubits * 10 + n_layers)
    X = rng.uniform(0, np.pi, (8, n_qubits))
    w = rng.normal(scale=1.0, size=(n_layers, n_qubits, 3))
    assert np.allclose(expval_z0(encoding, X, w), _pennylane_z0(encoding, X, w), atol=1e-12)


def test_qiskit_scores_a_trained_pennylane_vqc_identically():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, (120, 4))
    y = (X[:, 0] + X[:, 1] > np.pi).astype(int)
    model = VQCClassifier(encoding="reupload", n_qubits=4, n_layers=2, epochs=3, batch_size=32, seed=0).fit(X, y)
    assert np.allclose(qiskit_decision_scores(model, X), model.decision_scores(X), atol=1e-12)


def test_qiskit_circuit_structure_and_unsupported_encoding():
    w = np.zeros((2, 4, 3))
    ops = vqc_circuit("reupload", np.zeros(4), w).count_ops()
    assert ops["ry"] == 2 * 4 + 2 * 4 and ops["rz"] == 2 * 2 * 4 and ops["cx"] == 2 * 4
    with pytest.raises(ValueError):
        vqc_circuit("zz", np.zeros(4), w)
