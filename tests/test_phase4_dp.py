import math

import numpy as np
import pytest
import torch
from opacus import GradSampleModule

from src.m5_models.quantum_dp import build_hybrid, build_vqc
from src.m7_privacy.dp_training import DPClassifier

rng = np.random.default_rng(0)


@pytest.mark.parametrize("encoding", ["reupload", "angle"])
def test_opacus_per_sample_grads_match_manual_for_quantum_layer(encoding):
    torch.manual_seed(0)
    model = build_vqc(encoding, 4, 2)
    x = torch.rand(6, 4, dtype=torch.float64)
    y = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.float64)
    gsm = GradSampleModule(model, loss_reduction="sum")
    torch.nn.functional.binary_cross_entropy_with_logits(gsm(x).reshape(-1), y, reduction="sum").backward()
    qlayer = model[0]
    opacus_grads = qlayer.weights.grad_sample
    manual = []
    for i in range(6):
        m = build_vqc(encoding, 4, 2)
        m.load_state_dict({k.replace("_module.", ""): v for k, v in gsm.state_dict().items()})
        torch.nn.functional.binary_cross_entropy_with_logits(m(x[i:i + 1]).reshape(-1), y[i:i + 1], reduction="sum").backward()
        manual.append(m[0].weights.grad)
    torch.testing.assert_close(opacus_grads, torch.stack(manual))


def _data(d, n=400):
    X = rng.uniform(0, np.pi, size=(n, d))
    y = (X[:, 0] > 2.2).astype(int)
    return X, y


@pytest.mark.parametrize("kind,d,params", [
    ("logreg", 4, {}),
    ("vqc", 4, {"encoding": "reupload", "n_qubits": 4, "n_layers": 1}),
    ("hybrid", 6, {"encoding": "angle", "n_qubits": 4, "n_layers": 1}),
])
def test_dp_training_hits_target_epsilon_via_opacus(kind, d, params):
    X, y = _data(d)
    model = DPClassifier(kind=kind, epsilon=2.0, delta=1e-5, epochs=2, batch_size=64, seed=1, model_params=params).fit(X, y)
    report = model.privacy_
    assert report.dp_trained and report.accountant == "opacus.rdp"
    assert report.epsilon == pytest.approx(2.0, rel=0.05)
    assert report.noise_multiplier > 0 and report.steps > 0
    assert np.isfinite(model.decision_scores(X)).all()


def test_infinite_epsilon_reports_no_epsilon():
    X, y = _data(4)
    model = DPClassifier(kind="logreg", epsilon=None, epochs=1).fit(X, y)
    assert model.privacy_.epsilon is None and not model.privacy_.dp_trained


def test_hybrid_builds_with_amplitude():
    model = build_hybrid(10, "amplitude", 4, 1)
    assert model(torch.rand(3, 10, dtype=torch.float64)).shape == (3, 1)
    assert not math.isnan(float(model(torch.rand(2, 10, dtype=torch.float64)).sum()))
