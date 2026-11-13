import numpy as np

from src.m8_eval.state_viz import bloch_vectors, class_separation


def test_angle_encoding_gives_pure_states_on_the_xz_great_circle():
    X = np.array([[0.0, np.pi / 2, np.pi, 1.0]])
    r = bloch_vectors("angle", 4, X)[0]
    # RY(theta)|0> has Bloch vector (sin theta, 0, cos theta)
    assert np.allclose(r, np.column_stack([np.sin(X[0]), np.zeros(4), np.cos(X[0])]), atol=1e-12)
    assert np.allclose(np.linalg.norm(r, axis=1), 1.0)


def test_zz_encoding_entangles_so_reduced_states_are_mixed():
    X = np.random.default_rng(0).uniform(0, np.pi, (20, 4))
    lengths = np.linalg.norm(bloch_vectors("zz", 4, X), axis=-1)
    assert (lengths <= 1 + 1e-9).all() and lengths.mean() < 0.95


def test_class_separation_is_zero_for_identical_classes_and_two_for_opposite_poles():
    v = np.zeros((4, 1, 3)); v[:2, 0, 2] = 1; v[2:, 0, 2] = -1
    assert np.allclose(class_separation(v, np.array([0, 0, 1, 1])), 2.0)
    assert np.allclose(class_separation(np.ones((4, 1, 3)), np.array([0, 0, 1, 1])), 0.0)


def test_statevector_is_normalised_and_matches_angle_product_state():
    from src.m8_eval.state_viz import statevector
    x = np.array([0.3, 1.1, 2.0, 0.7])
    psi = statevector("angle", 4, x)
    assert psi.shape == (16,) and np.isclose(np.sum(np.abs(psi) ** 2), 1.0)
    # angle encoding = RY(x_q) on each qubit: P(|0000>) = prod cos^2(x_q / 2)
    assert np.isclose(np.abs(psi[0]) ** 2, np.prod(np.cos(x / 2) ** 2))
    assert np.isclose(np.sum(np.abs(statevector("zz", 4, x)) ** 2), 1.0)
