import numpy as np
import pennylane as qml
import pytest
from sklearn.metrics import roc_auc_score

from src.m5_models.noisy_kernel import NoiseModel, NoisyQuantumKernelSVC, noisy_fidelity_kernel
from src.m5_models.quantum import QuantumKernelSVC


def _full_circuit_p0(x, xp, noise: NoiseModel) -> float:
    """Full multi-qubit compute-uncompute circuit on default.mixed, noise after every pulse."""
    n = len(x)
    dev = qml.device("default.mixed", wires=n)
    p01, _ = noise.readout()

    @qml.qnode(dev)
    def circuit():
        for angles in (x, -xp):
            for w in range(n):
                qml.RY(angles[w], wires=w)
                for _ in range(noise.sx_per_ry):
                    noise.gate_channels(wire=w)   # creating the ops inside the QNode queues them
        for w in range(n):
            qml.BitFlip(p01, wires=w)          # symmetric readout error = bit flip before measurement
        return qml.probs(wires=range(n))

    return float(circuit()[0])


def test_factorised_kernel_matches_full_mixed_state_circuit():
    rng = np.random.default_rng(0)
    noise = NoiseModel(readout_p01=0.02, readout_p10=0.02, scale=5.0)
    xa, xb = rng.uniform(0, np.pi, (3, 4)), rng.uniform(0, np.pi, (2, 4))
    K = noisy_fidelity_kernel(xa, xb, noise)
    for i in range(3):
        for j in range(2):
            assert K[i, j] == pytest.approx(_full_circuit_p0(xa[i], xb[j], noise), abs=1e-10)


def test_zero_noise_reproduces_the_statevector_kernel():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, np.pi, (5, 4))
    ideal = QuantumKernelSVC(encoding="angle", n_qubits=4)._states(X)
    assert np.allclose(noisy_fidelity_kernel(X, X, NoiseModel(scale=0.0)), QuantumKernelSVC._kernel(ideal, ideal))


def test_noise_lowers_the_kernel_diagonal_and_rejects_entangling_encodings():
    X = np.random.default_rng(2).uniform(0, np.pi, (4, 4))
    d1 = np.diag(noisy_fidelity_kernel(X, X, NoiseModel(scale=1.0)))
    d10 = np.diag(noisy_fidelity_kernel(X, X, NoiseModel(scale=10.0)))
    assert (d10 < d1).all() and (d1 < 1).all()
    with pytest.raises(ValueError):
        NoisyQuantumKernelSVC(encoding="zz", n_qubits=4).fit(X, [0, 1, 0, 1])


def test_noisy_qsvm_with_shots_still_learns_a_simple_signal():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, np.pi, (160, 4))
    y = (X[:, 0] > np.pi / 2).astype(int)
    m = NoisyQuantumKernelSVC(n_qubits=4, C=1.0, noise_scale=1.0, shots=1024, seed=0).fit(X[:120], y[:120])
    assert roc_auc_score(y[120:], m.decision_scores(X[120:])) > 0.8
