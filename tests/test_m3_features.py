import numpy as np

from src.m3_features.features import FeatureTransformer


def test_scaler_and_feature_selection_are_fitted_on_training_rows_only():
    train = np.array(
        [[0.0, 0.0, 5.0], [1.0, 0.1, 4.0], [2.0, 0.0, 3.0], [3.0, 0.1, 2.0]]
    )
    labels = np.array([0, 0, 1, 1])
    transformer = FeatureTransformer(d_prime=2, seed=11).fit(train, labels)
    learned_mean = transformer.scaler.mean_.copy()

    test_values = np.array([[1000.0, -1000.0, 100.0]])
    transformed = transformer.transform(test_values)

    np.testing.assert_allclose(transformer.scaler.mean_, learned_mean)
    assert transformed.shape == (1, 2)
    assert np.all((transformed >= 0.0) & (transformed <= np.pi))


def test_optional_pca_returns_requested_dimension():
    rng = np.random.default_rng(5)
    train = rng.normal(size=(40, 7))
    labels = np.tile([0, 1], 20)
    transformer = FeatureTransformer(d_prime=3, use_pca=True, seed=5)

    transformed = transformer.fit_transform(train, labels)

    assert transformed.shape == (40, 3)
    assert np.all((transformed >= 0.0) & (transformed <= np.pi))
