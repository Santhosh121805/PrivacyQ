import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score

from src.m4_encoding import make_encoding
from src.m6_federated.fedavg import fedavg_train, group_partition, stratified_iid_partition
from src.m6_federated.federated import dirichlet_partition, partition_frame


def _disjoint_cover(parts, n):
    rows = np.concatenate(parts)
    return len(rows) == n and len(set(rows.tolist())) == n


def test_dirichlet_partition_is_disjoint_and_complete():
    y = np.r_[np.ones(30), np.zeros(270)].astype(int)
    assert _disjoint_cover(dirichlet_partition(y, 5, 0.5, 0), 300)


def test_partition_frame_returns_client_subsets():
    frame = pd.DataFrame({"x": range(90), "label": [0, 1, 0] * 30})
    parts = partition_frame(frame, n_clients=3, iid=True, seed=0)
    assert sum(len(p) for p in parts) == 90 and all(len(p) == 30 for p in parts)


def test_stratified_iid_keeps_label_ratio():
    y = np.r_[np.ones(30), np.zeros(270)].astype(int)
    parts = stratified_iid_partition(y, 3, 0)
    assert _disjoint_cover(parts, 300)
    assert all(y[p].sum() == 10 for p in parts)


def test_group_partition_never_splits_a_group():
    groups = np.random.default_rng(0).integers(0, 12, 300)
    parts = group_partition(groups, 3, 0)
    assert _disjoint_cover(parts, 300)
    owners = [set(groups[p]) for p in parts]
    assert all(not (owners[i] & owners[j]) for i in range(3) for j in range(i + 1, 3))


@pytest.mark.parametrize("epsilon", [2.0, None])
def test_dp_fedavg_trains_and_accounts_per_client(epsilon):
    rng = np.random.default_rng(1)
    X = rng.uniform(0, np.pi, (450, 4))
    y = (X[:, 0] > 2.2).astype(int)
    parts = stratified_iid_partition(y, 3, 0)
    enc = make_encoding("reupload", 4)

    def evaluate(module):
        with torch.no_grad():
            s = module(torch.as_tensor(enc.prepare(X), dtype=torch.float64)).reshape(-1).numpy()
        return {"auc": roc_auc_score(y, s)}

    out = fedavg_train("vqc", {"encoding": "reupload", "n_qubits": 4, "n_layers": 1},
                       [(X[p], y[p]) for p in parts], rounds=2, local_epochs=1, learning_rate=0.05,
                       sample_rate=0.1, epsilon=epsilon, delta=1e-5, max_grad_norm=1.0, seed=0,
                       prepare=enc.prepare, evaluate=evaluate)
    hist = out["history"]
    assert list(hist["round"]) == [1, 2] and "auc" in hist
    assert hist["bytes"].iloc[0] == 2 * 3 * out["n_params"] * 8
    if epsilon is None:
        assert out["epsilon_spent_max"] is None
    else:
        assert all(e == pytest.approx(2.0, rel=0.05) for e in out["client_epsilons"])
