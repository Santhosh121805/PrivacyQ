import numpy as np
import pytest

from src.m4_encoding import ENCODING_NAMES, encoding_resources, make_encoding
from src.m5_models.quantum import HybridQNNClassifier, QuantumKernelSVC, VQCClassifier
from src.m8_eval.stats import corrected_resampled_ttest, holm
from src.m10_exp.tuning import inner_split, select_and_fit, stratified_subsample, youden_threshold

rng = np.random.default_rng(0)


def _toy(d, n=80):
    X = rng.uniform(0, np.pi, size=(n, d))
    y = (np.sin(X[:, 0]) > 0.7).astype(int)
    y[:4] = [0, 1, 0, 1]
    return X, y


@pytest.mark.parametrize("name", ENCODING_NAMES)
def test_encodings_share_interface(name):
    enc = make_encoding(name, 4)
    X = rng.uniform(0, np.pi, size=(5, enc.n_features()))
    prepared = enc.prepare(X)
    assert prepared.shape[0] == 5
    assert encoding_resources(enc)["n_gates"] > 0


def test_amplitude_uses_2n_minus_1_features_plus_constant():
    enc = make_encoding("amplitude", 4)
    assert enc.n_features() == 15 and enc.n_features(available=10) == 10
    assert enc.prepare(np.zeros((2, 15))).shape == (2, 16)


def test_qubit_budget_is_enforced():
    with pytest.raises(ValueError):
        make_encoding("angle", 3)
    with pytest.raises(ValueError):
        make_encoding("angle", 13)


def test_quantum_kernel_is_a_valid_fidelity_kernel():
    X, y = _toy(4)
    model = QuantumKernelSVC(encoding="zz", n_qubits=4).fit(X, y)
    states = model.train_states_
    K = model._kernel(states, states)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)
    assert np.all(K >= -1e-12) and np.all(K <= 1 + 1e-12)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert model.decision_scores(X[:3]).shape == (3,)


@pytest.mark.parametrize("encoding", ["angle", "reupload", "amplitude"])
def test_vqc_trains_and_early_stops(encoding):
    d = 15 if encoding == "amplitude" else 4
    X, y = _toy(d)
    model = VQCClassifier(encoding=encoding, n_qubits=4, n_layers=1, epochs=2, batch_size=40)
    model.fit(X[:60], y[:60], eval_set=(X[60:], y[60:]))
    assert model.best_epoch_ in (1, 2) and len(model.history_) == 2
    assert model.n_parameters() == 4 * 3 + 2
    assert np.isfinite(model.decision_scores(X)).all()


def test_hybrid_qnn_compresses_then_measures_all_qubits():
    X = rng.normal(size=(60, 12))
    y = (X[:, 0] > 0).astype(int)
    model = HybridQNNClassifier(encoding="angle", n_qubits=4, n_layers=1, epochs=1).fit(X, y)
    assert model.n_parameters() == (12 * 4 + 4) + 4 * 3 + (4 + 1)


def test_tuning_selects_on_inner_val_and_sets_threshold():
    from sklearn.linear_model import LogisticRegression

    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=200) > 1).astype(int)
    pos = np.arange(200)
    sub = stratified_subsample(pos, y, 150, seed=0)
    tr, iv = inner_split(sub, y, 0.2, seed=0)
    assert not set(tr) & set(iv) and set(tr) | set(iv) == set(sub)
    best = select_and_fit(lambda C: LogisticRegression(C=C), {"C": [0.001, 1.0]}, X[tr], y[tr], X[iv], y[iv])
    assert best["params"]["C"] in (0.001, 1.0) and len(best["grid_log"]) == 2
    assert np.isfinite(best["threshold"])


def test_youden_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    t = youden_threshold(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]))
    assert 0.3 < t <= 0.8


def test_corrected_ttest_is_more_conservative_than_plain():
    a = np.array([0.80, 0.82, 0.81, 0.79, 0.80])
    b = a - np.array([0.01, 0.02, 0.0, 0.015, 0.01])
    plain = corrected_resampled_ttest(a, b, 0.0)["p"]
    corrected = corrected_resampled_ttest(a, b, 0.25)["p"]
    assert corrected > plain
    assert list(holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


def test_basis_encoding_prepares_the_computational_basis_state():
    import pennylane as qml

    enc = make_encoding("basis", 4)
    x = np.array([[0.2, 2.0, 3.1, 1.0]])            # angle-scaled features in [0, pi]
    bits = enc.prepare(x)[0]
    np.testing.assert_array_equal(bits, [0, 1, 1, 0])  # thresholded at pi/2

    @qml.qnode(qml.device("default.qubit", wires=4))
    def state(b):
        enc.feature_map(b)
        return qml.probs(wires=range(4))

    probs = state(bits)
    assert probs[int("0110", 2)] == pytest.approx(1.0)
    assert encoding_resources(enc)["n_gates"] == 4


def test_basis_kernel_is_one_for_equal_bitstrings_and_zero_otherwise():
    from src.m5_models.quantum import QuantumKernelSVC

    X = np.array([[0.1, 0.1, 3.0, 3.0], [0.2, 0.3, 2.9, 2.0], [3.0, 0.1, 0.1, 3.0]] * 4)
    y = np.array([0, 1, 0] * 4)
    m = QuantumKernelSVC(encoding="basis", n_qubits=4).fit(X, y)
    K = m._kernel(m.train_states_, m.train_states_)
    np.testing.assert_allclose(K[0, 1], 1.0, atol=1e-9)   # same bits
    np.testing.assert_allclose(K[0, 2], 0.0, atol=1e-9)   # different bits


def test_vqc_cobyla_optimiser_reduces_loss():
    X, y = _toy(4)
    model = VQCClassifier(encoding="angle", n_qubits=4, n_layers=1, optimizer="cobyla", maxiter=40, seed=0).fit(X, y)
    assert model.n_evaluations_ <= 41 and len(model.loss_history_) == model.n_evaluations_
    assert min(model.loss_history_) < model.loss_history_[0]
    assert np.isfinite(model.decision_scores(X)).all()
    with pytest.raises(ValueError):
        VQCClassifier(optimizer="sgd").fit(X, y)
