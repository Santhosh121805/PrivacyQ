import pennylane as qml


def amplitude_encoding(n_qubits: int, repetitions: int = 1):
    if not 4 <= n_qubits <= 12:
        raise ValueError("n_qubits must be between 4 and 12")
    if repetitions != 1:
        raise ValueError("amplitude encoding is a single state-preparation operation")

    def circuit(features):
        if features.shape[-1] > 2**n_qubits:
            raise ValueError("feature vector is too large for the configured qubits")
        qml.AmplitudeEmbedding(
            features,
            wires=range(n_qubits),
            pad_with=0.0,
            normalize=True,
        )

    circuit.n_wires = n_qubits
    circuit.name = "amplitude"
    return circuit
