import math

import pennylane as qml


def zz_featuremap(n_qubits: int, repetitions: int = 1):
    if not 4 <= n_qubits <= 12:
        raise ValueError("n_qubits must be between 4 and 12")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")

    def circuit(features):
        if features.shape[-1] != n_qubits:
            raise ValueError(f"ZZ feature map expects {n_qubits} features")
        for _ in range(repetitions):
            for wire in range(n_qubits):
                qml.Hadamard(wires=wire)
                qml.RZ(features[wire], wires=wire)
            for wire in range(n_qubits):
                next_wire = (wire + 1) % n_qubits
                phase = (math.pi - features[wire]) * (math.pi - features[next_wire])
                qml.IsingZZ(phase, wires=[wire, next_wire])

    circuit.n_wires = n_qubits
    circuit.name = "zz"
    return circuit
