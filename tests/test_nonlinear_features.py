import numpy as np
import pytest

from src.m3_features.features import FeatureTransformer


@pytest.mark.parametrize("method", ["umap", "autoencoder"])
def test_nonlinear_reducers_fit_on_train_only_and_angle_scale(method):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12))
    X[:, :3] += np.repeat(rng.integers(0, 2, 300)[:, None], 3, axis=1) * 3.0     # two clusters
    y = rng.integers(0, 2, 300)
    ft = FeatureTransformer(d_prime=4, method=method, output="angle", seed=0).fit(X[:200], y[:200])
    tr, te = ft.transform(X[:200]), ft.transform(X[200:])
    assert tr.shape == (200, 4) and te.shape == (100, 4)
    assert tr.min() >= 0 and tr.max() <= np.pi + 1e-9                     # [0, pi] fitted on train rows
    assert te.min() >= 0 and te.max() <= np.pi + 1e-9                     # test rows clipped, never refit
    again = FeatureTransformer(d_prime=4, method=method, output="angle", seed=0).fit(X[:200], y[:200])
    assert np.allclose(again.transform(X[200:]), te)                       # seeded, reproducible


def test_autoencoder_reconstruction_loss_decreases():
    X = np.random.default_rng(1).normal(size=(256, 10))
    ft = FeatureTransformer(d_prime=3, method="autoencoder", seed=0).fit(X, np.zeros(256))
    h = ft.reducer.loss_history_
    assert h[-1] < 0.8 * h[0]
