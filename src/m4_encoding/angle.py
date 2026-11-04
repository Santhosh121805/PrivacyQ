import pennylane as qml


def angle_encoding(n_qubits: int, repetitions: int = 1):
    if not 4 <= n_qubits <= 12:
        raise ValueError("n_qubits must be between 4 and 12")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")

    def circuit(features):
        if features.shape[-1] != n_qubits:
            raise ValueError(f"angle encoding expects {n_qubits} features")
        for _ in range(repetitions):
            qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")

    circuit.n_wires = n_qubits
    circuit.name = "angle"
    return circuit
