import numpy as np
import pennylane as qml
import pytest

from src.m4_encoding import make_encoding


@pytest.mark.parametrize("name", ["angle", "amplitude", "zz", "reupload"])
def test_encodings_build_valid_circuits_on_expected_wires(name):
    circuit = make_encoding(name, n_qubits=4)
    device = qml.device("default.qubit", wires=4)

    @qml.qnode(device)
    def encoded_state(features):
        circuit(features)
        return qml.probs(wires=range(4))

    probabilities = encoded_state(np.array([0.2, 0.5, 1.0, 2.0]))

    assert circuit.n_wires == 4
    assert probabilities.shape == (16,)
    assert np.isclose(np.sum(probabilities), 1.0)


def test_angle_encoding_rejects_mismatched_feature_dimension():
    circuit = make_encoding("angle", n_qubits=4)
    device = qml.device("default.qubit", wires=4)

    @qml.qnode(device)
    def encoded_state(features):
        circuit(features)
        return qml.expval(qml.PauliZ(0))

    with pytest.raises(ValueError, match="expects 4 features"):
        encoded_state(np.array([0.1, 0.2]))
