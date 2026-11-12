"""Quantum-kernel SVM under a simulated hardware noise model (PennyLane ``default.mixed``).

This answers one question: how much of the noiseless QSVM result would survive on a
noisy device? It is a simulation with an assumed noise model, not a hardware run.
The circuit gives no privacy.

How a fidelity kernel is estimated on hardware (compute-uncompute):
    prepare |0..0>, apply U(x), apply U(x')^dagger, measure all qubits,
    k(x, x') = P(all qubits read 0), estimated from a finite number of shots.

Noise model (applied after every physical gate):
    single-qubit pulses (SX): depolarising channel (from the average gate error), amplitude
    damping (T1) and pure dephasing (T2) over the pulse duration;
    two-qubit gates (CNOT/ECR): two-qubit depolarising channel (from the two-qubit gate error),
    then T1/T2 damping of both qubits over the gate duration;
    readout: asymmetric bit flips, P(1|0) and P(0|1).
    Physical compilation: RY = ``sx_per_ry`` SX pulses; H = one SX pulse; RZ is virtual (no
    pulse, no noise); IsingZZ(phi) = CNOT . RZ_target(phi) . CNOT (two two-qubit gates).
    Every rate is multiplied by ``scale`` (0 = ideal, 10 = ten times noisier). The defaults are
    representative of published median figures for current superconducting processors, not a
    calibration snapshot of any one device. Randomised-benchmarking gate errors already include
    some decoherence, so adding T1/T2 on top errs on the noisy side. Not modelled: idle
    decoherence of qubits waiting while others are gated, crosstalk, leakage, coherent errors.

Two exact evaluation paths:
    angle (one RY per qubit, no two-qubit gates): the noisy circuit factorises over qubits, so
        P(all 0) = prod_q Tr( rho_q(x) E_q(x') )
    with rho_q(x) from ``default.mixed`` and E_q(x') the Heisenberg-picture readout operator of the
    uncompute half. Cheap at any qubit count.
    zz (entangling): no factorisation. rho(x) is the full 2^n x 2^n noisy state from
    ``default.mixed``; E(x') is the full 2^n x 2^n Heisenberg-picture operator of the noisy
    uncompute circuit; k = Tr(rho(x) E(x')). Cost grows as 4^n per row.
    ``tests/test_noisy_kernel.py`` checks both against the full compute-uncompute circuit run end to
    end on ``default.mixed``.

Shot noise: with ``shots`` set, every kernel entry is replaced by Binomial(shots, k)/shots
(training Gram: upper triangle incl. the diagonal sampled once, then mirrored).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pennylane as qml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from src.m4_encoding import make_encoding

FULL_ENCODINGS = ("angle", "zz")
_PAULI = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_KERNEL_CACHE: OrderedDict = OrderedDict()
_READOUT_CACHE: OrderedDict = OrderedDict()


@dataclass(frozen=True)
class NoiseModel:
    t1_us: float = 200.0
    t2_us: float = 150.0
    sx_ns: float = 50.0
    sx_error: float = 3e-4
    readout_p01: float = 0.01   # P(read 1 | state 0)
    readout_p10: float = 0.02   # P(read 0 | state 1)
    sx_per_ry: int = 2
    cx_error: float = 5e-3      # two-qubit gate average error
    cx_ns: float = 300.0        # two-qubit gate duration
    scale: float = 1.0

    def _damping(self, ns: float) -> tuple[float, float]:
        t = ns * 1e-3                                                  # microseconds
        gamma = 1 - np.exp(-self.scale * t / self.t1_us)
        t_phi_rate = max(1 / self.t2_us - 1 / (2 * self.t1_us), 0.0)   # pure dephasing 1/T_phi
        return gamma, 1 - np.exp(-2 * self.scale * t * t_phi_rate)

    def gate_channels(self, wire: int = 0) -> list:
        """PennyLane channel ops after one SX pulse. Called inside a QNode, creating them queues them."""
        if self.scale == 0:
            return []
        gamma, lam = self._damping(self.sx_ns)
        p_dep = min(1.5 * self.scale * self.sx_error, 0.75)           # average infidelity r = 2p/3
        return [qml.DepolarizingChannel(p_dep, wires=wire), qml.AmplitudeDamping(gamma, wires=wire),
                qml.PhaseDamping(lam, wires=wire)]

    def cx_channels(self, a: int, b: int) -> list:
        """Ops after one two-qubit gate: two-qubit depolarising on (a, b) (average infidelity
        r = 4p/5, so p = 1.25 r), then T1/T2 damping of both qubits over the gate duration."""
        if self.scale == 0:
            return []
        p = min(1.25 * self.scale * self.cx_error, 15 / 16)
        pairs = [(A, B) for A in _PAULI for B in _PAULI]
        kraus = [np.sqrt(1 - p if i == 0 else p / 15) * np.kron(A, B) for i, (A, B) in enumerate(pairs)]
        gamma, lam = self._damping(self.cx_ns)
        return [qml.QubitChannel(kraus, wires=[a, b]),
                qml.AmplitudeDamping(gamma, wires=a), qml.PhaseDamping(lam, wires=a),
                qml.AmplitudeDamping(gamma, wires=b), qml.PhaseDamping(lam, wires=b)]

    def readout(self) -> tuple[float, float]:
        return min(self.scale * self.readout_p01, 0.5), min(self.scale * self.readout_p10, 0.5)

    def as_dict(self) -> dict:
        return {"t1_us": self.t1_us, "t2_us": self.t2_us, "sx_ns": self.sx_ns, "sx_error": self.sx_error,
                "readout_p01": self.readout_p01, "readout_p10": self.readout_p10, "sx_per_ry": self.sx_per_ry,
                "cx_error": self.cx_error, "cx_ns": self.cx_ns, "scale": self.scale}


def _ry(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (..., 2, 2)


# ------------------------------------------------ angle: factorised path ----
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
    """Angle encoding: exact (infinite-shot) P(all 0) of the noisy compute-uncompute circuit."""
    rho, E = noisy_encoded_states(xa, noise), readout_operators(xb, noise)
    K = np.ones((len(xa), len(xb)))
    for q in range(rho.shape[1]):
        K *= np.einsum("aij,bji->ab", rho[:, q], E[:, q]).real
    return np.clip(K, 0.0, 1.0)


# ------------------------------------------- full density-matrix path ----
def _steps(encoding, x: np.ndarray) -> list[tuple]:
    """Physical gate sequence of U(x): (name, wires, angle per row or None)."""
    n, steps = encoding.n_qubits, []
    if encoding.name == "angle":
        steps = [("RY", [w], x[:, w]) for w in range(n)]
    elif encoding.name == "zz":
        for _ in range(encoding.reps):
            for w in range(n):
                steps += [("H", [w], None), ("RZ", [w], x[:, w])]
            for a in range(n):
                b = (a + 1) % n
                steps += [("CNOT", [a, b], None), ("RZ", [b], (np.pi - x[:, a]) * (np.pi - x[:, b])),
                          ("CNOT", [a, b], None)]
    else:
        raise ValueError(f"noisy simulation supports {FULL_ENCODINGS}, not {encoding.name!r}")
    return steps


def _noise_ops(name: str, wires: list, noise: NoiseModel) -> list:
    if name == "RY":
        return [op for _ in range(noise.sx_per_ry) for op in noise.gate_channels(wires[0])]
    if name == "H":
        return noise.gate_channels(wires[0])
    if name == "CNOT":
        return noise.cx_channels(*wires)
    return []                                                          # virtual RZ


def _pl_gate(name: str, wires: list, theta, dagger: bool = False) -> None:
    sign = -1.0 if dagger else 1.0
    if name == "RY":
        qml.RY(sign * theta, wires=wires[0])
    elif name == "RZ":
        qml.RZ(sign * theta, wires=wires[0])
    elif name == "H":
        qml.Hadamard(wires=wires[0])
    else:
        qml.CNOT(wires=wires)


def _np_gate(name: str, theta) -> np.ndarray:
    """Gate matrix, shape (rows or 1, d, d)."""
    if name == "RY":
        return _ry(theta)
    if name == "RZ":
        z = np.exp(-0.5j * theta)
        zero = np.zeros_like(z)
        return np.stack([np.stack([z, zero], -1), np.stack([zero, np.conj(z)], -1)], -2)
    return (_H if name == "H" else _CNOT)[None]


def full_noisy_states(x: np.ndarray, encoding, noise: NoiseModel, chunk: int = 128) -> np.ndarray:
    """rho(x): noisy U(x) applied to |0..0> on ``default.mixed`` (all qubits). Shape (rows, 2^n, 2^n)."""
    n = encoding.n_qubits
    dev = qml.device("default.mixed", wires=n)

    @qml.qnode(dev)
    def state(xb):
        for name, wires, theta in _steps(encoding, xb):
            _pl_gate(name, wires, theta)
            _noise_ops(name, wires, noise)                             # creating ops inside a QNode queues them
        return qml.density_matrix(wires=range(n))

    x = np.asarray(x, dtype=float)
    out = [np.asarray(state(x[i:i + chunk])).reshape(-1, 2**n, 2**n) for i in range(0, len(x), chunk)]
    return np.concatenate(out)


def _conj(E: np.ndarray, A: np.ndarray, wires: list, n: int) -> np.ndarray:
    """A E A^dagger with A acting on ``wires`` (A: (rows or 1, d, d); E: (rows, 2^n, 2^n))."""
    k, B = len(wires), E.shape[0]
    T = E.reshape((B,) + (2,) * (2 * n))
    rows, front = [1 + w for w in wires], list(range(1, 1 + k))
    T = np.moveaxis(T, rows, front)
    shape = T.shape
    T = np.moveaxis((A @ T.reshape(B, 2**k, -1)).reshape(shape), front, rows)
    cols, back = [1 + n + w for w in wires], list(range(1 + 2 * n - k, 1 + 2 * n))
    T = np.moveaxis(T, cols, back)
    shape = T.shape
    T = (T.reshape(B, -1, 2**k) @ np.conj(np.swapaxes(A, -1, -2))).reshape(shape)
    return np.moveaxis(T, back, cols).reshape(B, 2**n, 2**n)


def full_readout_operators(x: np.ndarray, encoding, noise: NoiseModel) -> np.ndarray:
    """E(x') with P(all 0 after the noisy uncompute circuit U(x')^dagger on rho) = Tr(rho E(x')).

    The uncompute circuit runs the gates of U(x') reversed and inverted, noise after each.
    Peeling it from the measurement backwards visits the gates of U(x') in forward order: for
    each, apply the adjoint of its noise (sum K^dagger E K, channels in reverse), then
    E -> g E g^dagger. Shape (rows, 2^n, 2^n)."""
    n = encoding.n_qubits
    x = np.asarray(x, dtype=float)
    p01, p10 = noise.readout()
    pi0 = np.array([[1.0]])
    for _ in range(n):
        pi0 = np.kron(pi0, np.diag([1 - p01, p10]))
    E = np.repeat(pi0.astype(complex)[None], len(x), axis=0)
    for name, wires, theta in _steps(encoding, x):
        for ch in reversed(_noise_ops(name, wires, noise)):
            idx = [int(w) for w in ch.wires]
            E = sum(_conj(E, np.conj(np.asarray(K).T)[None], idx, n) for K in ch.kraus_matrices())
        E = _conj(E, _np_gate(name, theta), wires, n)
    return E


def full_noisy_fidelity_kernel(xa: np.ndarray, xb: np.ndarray, encoding, noise: NoiseModel,
                               readout: np.ndarray | None = None) -> np.ndarray:
    """Exact (infinite-shot) noisy compute-uncompute kernel, full density matrices."""
    rho = full_noisy_states(xa, encoding, noise)
    E = full_readout_operators(xb, encoding, noise) if readout is None else readout
    K = rho.reshape(len(rho), -1) @ np.swapaxes(E, -1, -2).reshape(len(E), -1).T
    return np.clip(K.real, 0.0, 1.0)


# --------------------------------------------------------------- model ----
def _digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a, dtype=float)
    return hashlib.sha1(a.tobytes() + str(a.shape).encode()).hexdigest()


def _cache_put(cache: OrderedDict, key, value, limit: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


class NoisyQuantumKernelSVC(BaseEstimator, ClassifierMixin):
    """Fidelity-kernel SVM whose kernel is estimated under ``NoiseModel``.

    angle: exact factorised path; zz: full density-matrix path. Exact kernels are cached per
    process (keyed by rows, encoding and noise), so the C grid and several shot settings on the
    same rows reuse one simulation; shots are sampled afterwards, per model."""

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
        if self.encoding not in FULL_ENCODINGS:
            raise ValueError(f"noisy kernel supports {FULL_ENCODINGS}; {self.encoding!r} is not implemented")
        return make_encoding(self.encoding, self.n_qubits).prepare(X)

    def _exact_kernel(self, xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
        noise = self._noise()
        base = (self.encoding, self.n_qubits, tuple(sorted(noise.as_dict().items())))
        key = base + (_digest(xa), _digest(xb))
        if key not in _KERNEL_CACHE:
            if self.encoding == "angle":
                K = noisy_fidelity_kernel(xa, xb, noise)
            else:
                enc = make_encoding(self.encoding, self.n_qubits)
                rkey = base + (_digest(xb),)
                if rkey not in _READOUT_CACHE:
                    _cache_put(_READOUT_CACHE, rkey, full_readout_operators(xb, enc, noise), 2)
                K = full_noisy_fidelity_kernel(xa, xb, enc, noise, readout=_READOUT_CACHE[rkey])
            _cache_put(_KERNEL_CACHE, key, K, 12)
        return _KERNEL_CACHE[key]

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
        K = self._exact_kernel(self.train_x_, self.train_x_)
        K = 0.5 * (K + K.T)                    # one estimate per unordered pair (exact limit)
        self.svc_ = SVC(kernel="precomputed", C=self.C, class_weight="balanced", random_state=self.seed)
        self.svc_.fit(self._sample(K, self.rng_, symmetric=True), np.asarray(y))
        self.classes_ = np.array([0, 1])
        return self

    def decision_scores(self, X) -> np.ndarray:
        K = self._exact_kernel(self._prepare(X), self.train_x_)
        return self.svc_.decision_function(self._sample(K, self.rng_, symmetric=False))

    def n_parameters(self) -> int:
        return int(self.svc_.dual_coef_.size + 1)
