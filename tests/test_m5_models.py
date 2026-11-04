import numpy as np

from src.m5_models import make_model


def test_make_model_supports_classical_and_variational_variants():
    logistic = make_model("logistic_regression", n_features=4)
    variational = make_model("variational", n_qubits=4, n_layers=1)

    X = np.array([[0.1, 0.2, 0.3, 0.4], [0.8, 0.7, 0.6, 0.5]])
    y = np.array([0, 1])

    logistic.fit(X, y)
    variational.fit(X, y)

    assert logistic.predict(X).shape == (2,)
    assert variational.predict(X).shape == (2,)
