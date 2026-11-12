import numpy as np
import pennylane as qml
import pytest
from sklearn.metrics import roc_auc_score

from src.m4_encoding import make_encoding
from src.m5_models.noisy_kernel import (
    NoiseModel, NoisyQuantumKernelSVC, _noise_ops, _pl_gate, _steps, full_noisy_fidelity_kernel,
    noisy_fidelity_kernel,
)
from src.m5_models.quantum import QuantumKernelSVC


def _end_to_end_p0(x, xp, encoding, noise: NoiseModel) -> float:
    """Whole compute-uncompute circuit run on default.mixed: noisy U(x), noisy U(x')^dagger,
    symmetric readout error (= bit flip before measurement), P(all 0)."""
    n = encoding.n_qubits
    p01, _ = noise.readout()

    @qml.qnode(qml.device("default.mixed", wires=n))
    def circuit():
        for name, wires, theta in _steps(encoding, x[None]):
            _pl_gate(name, wires, theta)
            _noise_ops(name, wires, noise)
        for name, wires, theta in reversed(_steps(encoding, xp[None])):
            _pl_gate(name, wires, theta, dagger=True)
            _noise_ops(name, wires, noise)
        for w in range(n):
            qml.BitFlip(p01, wires=w)
        return qml.probs(wires=range(n))

    return float(np.asarray(circuit()).reshape(-1)[0])


NOISY = NoiseModel(readout_p01=0.02, readout_p10=0.02, scale=5.0)


def test_factorised_angle_kernel_matches_end_to_end_circuit():
    rng = np.random.default_rng(0)
    enc = make_encoding("angle", 4)
    xa, xb = rng.uniform(0, np.pi, (3, 4)), rng.uniform(0, np.pi, (2, 4))
    K = noisy_fidelity_kernel(xa, xb, NOISY)
    for i in range(3):
        for j in range(2):
            assert K[i, j] == pytest.approx(_end_to_end_p0(xa[i], xb[j], enc, NOISY), abs=1e-10)


def test_full_zz_kernel_matches_end_to_end_circuit():
    rng = np.random.default_rng(4)
    enc = make_encoding("zz", 4)
    xa, xb = rng.uniform(0, np.pi, (2, 4)), rng.uniform(0, np.pi, (2, 4))
    K = full_noisy_fidelity_kernel(xa, xb, enc, NOISY)
    for i in range(2):
        for j in range(2):
            assert K[i, j] == pytest.approx(_end_to_end_p0(xa[i], xb[j], enc, NOISY), abs=1e-10)


def test_full_path_equals_factorised_path_for_angle():
    X = np.random.default_rng(5).uniform(0, np.pi, (4, 4))
    full = full_noisy_fidelity_kernel(X, X, make_encoding("angle", 4), NOISY)
    assert np.allclose(full, noisy_fidelity_kernel(X, X, NOISY), atol=1e-12)


@pytest.mark.parametrize("name", ["angle", "zz"])
def test_zero_noise_reproduces_the_statevector_kernel(name):
    X = np.random.default_rng(1).uniform(0, np.pi, (5, 4))
    ideal = QuantumKernelSVC(encoding=name, n_qubits=4)._states(X)
    K = full_noisy_fidelity_kernel(X, X, make_encoding(name, 4), NoiseModel(scale=0.0))
    assert np.allclose(K, QuantumKernelSVC._kernel(ideal, ideal), atol=1e-10)


def test_noise_lowers_the_diagonal_and_two_qubit_error_hits_zz_only():
    X = np.random.default_rng(2).uniform(0, np.pi, (4, 4))
    d1 = np.diag(noisy_fidelity_kernel(X, X, NoiseModel(scale=1.0)))
    d10 = np.diag(noisy_fidelity_kernel(X, X, NoiseModel(scale=10.0)))
    assert (d10 < d1).all() and (d1 < 1).all()
    for name, affected in (("angle", False), ("zz", True)):
        enc = make_encoding(name, 4)
        low = full_noisy_fidelity_kernel(X, X, enc, NoiseModel(cx_error=1e-4))
        high = full_noisy_fidelity_kernel(X, X, enc, NoiseModel(cx_error=5e-2))
        assert (np.diag(high) < np.diag(low) - 1e-3).all() == affected
    with pytest.raises(ValueError):
        NoisyQuantumKernelSVC(encoding="amplitude", n_qubits=4).fit(X, [0, 1, 0, 1])


def test_noisy_qsvm_with_shots_still_learns_a_simple_signal():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, np.pi, (160, 4))
    y = (X[:, 0] > np.pi / 2).astype(int)
    m = NoisyQuantumKernelSVC(n_qubits=4, C=1.0, noise_scale=1.0, shots=1024, seed=0).fit(X[:120], y[:120])
    assert roc_auc_score(y[120:], m.decision_scores(X[120:])) > 0.8
