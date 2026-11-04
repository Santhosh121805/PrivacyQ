"""Model definitions for the hybrid quantum-classical research scaffold."""

from .classical import CLASSICAL_MODELS, TorchMLPClassifier, build_mlp, count_parameters, make_classical
from .models import (
    LogisticRegressionBaseline,
    VariationalClassifier,
    classical_baselines,
    make_model,
)

__all__ = [
    "CLASSICAL_MODELS",
    "TorchMLPClassifier",
    "build_mlp",
    "count_parameters",
    "make_classical",
    "LogisticRegressionBaseline",
    "VariationalClassifier",
    "classical_baselines",
    "make_model",
]
