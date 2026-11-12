"""Quantum-kernel SVM under a simulated hardware noise model (PennyLane ``default.mixed``).

This answers one question: how much of the noiseless QSVM result would survive on a
noisy device? It is a simulation with an assumed noise model, not a hardware run.
The circuit gives no privacy.

How a fidelity kernel is estimated on hardware (compute-uncompute):
    prepare |0..0>, apply U(x), apply U(x')^dagger, measure all qubits,
    k(x, x') = P(all qubits read 0), estimated from a finite number of shots.

Noise model (per physical gate, applied after the gate):
    depolarising channel (from the average gate error), amplitude damping (T1) and
    pure dephasing (T2) over the gate duration. Each RY counts as ``sx_per_ry``
    physical SX pulses (an RY compiles to two SX pulses plus virtual RZ on
    superconducting devices). Readout errors are asymmetric bit flips, P(1|0) and P(0|1).
    Every rate is multiplied by ``scale``, so 0 is the ideal simulator and 10 is ten
    times noisier than the default. The defaults are representative of published median
    figures for current superconducting processors, not a calibration snapshot of any
    one device. Randomised-benchmarking gate errors already include some decoherence, so
    adding T1/T2 on top errs on the noisy side.

Why this is exact and still cheap:
    the supported encoding (angle: one RY per qubit, no two-qubit gates) and the noise
    channels act on one qubit at a time, and readout errors are independent. So the
    noisy compute-uncompute circuit factorises over qubits and
        P(all 0) = prod_q Tr( rho_q(x) E_q(x') ),
    where rho_q(x) is qubit q's noisy encoded state (computed on ``default.mixed``) and
    E_q(x') is the Heisenberg-picture readout operator of the uncompute half (built from
    the same PennyLane channels' Kraus matrices). ``tests/test_noisy_kernel.py`` checks
    this against the full multi-qubit ``default.mixed`` circuit. Entangling encodings
    (ZZ, re-uploading) do not factorise and would need full 2^n x 2^n density matrices for
    every pair, so they are rejected here. Two-qubit gate errors and crosstalk therefore
    never enter this model: it cannot show the noise cost of entangling circuits.

Shot noise: with ``shots`` set, every kernel entry is replaced by Binomial(shots, k)/shots
(training Gram: upper triangle incl. the diagonal sampled once, then mirrored).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from src.m4_encoding import make_encoding


@dataclass(frozen=True)
class NoiseModel:
    t1_us: float = 200.0
    t2_us: float = 150.0
    sx_ns: float = 50.0
    sx_error: float = 3e-4
    readout_p01: float = 0.01   # P(read 1 | state 0)
    readout_p10: float = 0.02   # P(read 0 | state 1)
    sx_per_ry: int = 2
    scale: float = 1.0

    def gate_channels(self, wire: int = 0) -> list:
        """PennyLane channel ops applied after every physical SX pulse. Called inside a QNode,
        creating them queues them (each call adds one set of channels)."""
        if self.scale == 0:
            return []
        t = self.sx_ns * 1e-3                                    # microseconds
        gamma = 1 - np.exp(-self.scale * t / self.t1_us)
        t_phi_rate = max(1 / self.t2_us - 1 / (2 * self.t1_us), 0.0)  # pure dephasing 1/T_phi
        lam = 1 - np.exp(-2 * self.scale * t * t_phi_rate)
        p_dep = min(1.5 * self.scale * self.sx_error, 0.75)     # average infidelity r = 2p/3
        return [qml.DepolarizingChannel(p_dep, wires=wire), qml.AmplitudeDamping(gamma, wires=wire),
                qml.PhaseDamping(lam, wires=wire)]

    def readout(self) -> tuple[float, float]:
        return min(self.scale * self.readout_p01, 0.5), min(self.scale * self.readout_p10, 0.5)

    def as_dict(self) -> dict:
        return {"t1_us": self.t1_us, "t2_us": self.t2_us, "sx_ns": self.sx_ns, "sx_error": self.sx_error,
                "readout_p01": self.readout_p01, "readout_p10": self.readout_p10,
                "sx_per_ry": self.sx_per_ry, "scale": self.scale}


def _ry(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (..., 2, 2)


def _adjoint_noise(op: np.ndarray, noise: NoiseModel) -> np.ndarray:
    """Heisenberg picture of all noise after one RY: applies N^dagger(E) = sum K^dagger E K."""
    for _ in range(noise.sx_per_ry):
        for ch in reversed(noise.gate_channels()):
            op = sum(K.conj().T @ op @ K for K in ch.kraus_matrices())
    return op


def noisy_encoded_states(x: np.ndarray, noise: NoiseModel) -> np.ndarray:
    """rho_q(x) for every row and qubit, from ``default.mixed``. Shape (n, n_qubits, 2, 2)."""
    dev = qml.device("default.mixed", wires=1)

    @qml.qnode(dev)
    def state(theta):
        qml.RY(theta, wires=0)
        for _ in range(noise.sx_per_ry):
            noise.gate_channels()          # creating an op inside a QNode queues it
        return qml.density_matrix(wires=0)

    x = np.asarray(x, dtype=float)
    out = np.empty((x.shape[0], x.shape[1], 2, 2), dtype=complex)
    for q in range(x.shape[1]):
        out[:, q] = np.asarray(state(x[:, q])).reshape(x.shape[0], 2, 2)
    return out


def readout_operators(x: np.ndarray, noise: NoiseModel) -> np.ndarray:
    """E_q(x') = R^dagger N^dagger(Pi0) R with R = RY(-x'_q). Shape (n, n_qubits, 2, 2)."""
    p01, p10 = noise.readout()
    F = _adjoint_noise(np.diag([1 - p01, p10]).astype(complex), noise)
    R = _ry(-np.asarray(x, dtype=float))
    return np.conj(np.swapaxes(R, -1, -2)) @ F @ R


def noisy_fidelity_kernel(xa: np.ndarray, xb: np.ndarray, noise: NoiseModel) -> np.ndarray:
    """Exact (infinite-shot) P(all 0) of the noisy compute-uncompute circuit, rows of xa vs rows of xb."""
    rho, E = noisy_encoded_states(xa, noise), readout_operators(xb, noise)
    K = np.ones((len(xa), len(xb)))
    for q in range(rho.shape[1]):
        K *= np.einsum("aij,bji->ab", rho[:, q], E[:, q]).real
    return np.clip(K, 0.0, 1.0)


class NoisyQuantumKernelSVC(BaseEstimator, ClassifierMixin):
    """Fidelity-kernel SVM whose kernel is estimated under ``NoiseModel`` (angle encoding only)."""

    supports_eval_set = False

    def __init__(self, encoding: str = "angle", n_qubits: int = 8, C: float = 1.0, noise_scale: float = 1.0,
                 shots: int | None = None, noise: dict | None = None, seed: int = 0):
        self.encoding = encoding
        self.n_qubits = n_qubits
        self.C = C
        self.noise_scale = noise_scale
        self.shots = shots
        self.noise = noise
        self.seed = seed

    def _noise(self) -> NoiseModel:
        return NoiseModel(**{**(self.noise or {}), "scale": float(self.noise_scale)})

    def _prepare(self, X) -> np.ndarray:
        if self.encoding != "angle":
            raise ValueError("noisy kernel supports the product-state angle encoding only; entangling encodings "
                             "need full density matrices per pair (not implemented)")
        return make_encoding(self.encoding, self.n_qubits).prepare(X)

    def _sample(self, K: np.ndarray, rng, symmetric: bool) -> np.ndarray:
        if self.shots is None:
            return K
        est = rng.binomial(int(self.shots), K) / float(self.shots)
        if symmetric:
            upper = np.triu(est)
            est = upper + np.triu(est, 1).T
        return est

    def fit(self, X, y, eval_set=None) -> "NoisyQuantumKernelSVC":
        self.rng_ = np.random.default_rng(self.seed)
        self.train_x_ = self._prepare(X)
        K = noisy_fidelity_kernel(self.train_x_, self.train_x_, self._noise())
        K = 0.5 * (K + K.T)                    # one estimate per unordered pair (exact limit)
        self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight="balanced", random_state=self.seed)
        self.svc_.fit(self._sample(K, self.rng_, symmetric=True), np.asarray(y))
        self.classes_ = np.array([0, 1])
        return self

    def decision_scores(self, X) -> np.ndarray:
        K = noisy_fidelity_kernel(self._prepare(X), self.train_x_, self._noise())
        return self.svc_.decision_function(self._sample(K, self.rng_, symmetric=False))

    def n_parameters(self) -> int:
        return int(self.svc_.dual_coef_.size + 1)
