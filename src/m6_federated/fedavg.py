"""In-process FedAvg with optional client-side DP-SGD (Opacus).

Each client ("school") trains locally on its own rows; the server averages the
returned weights, weighted by client size (FedAvg). With a target epsilon, every
client trains with Opacus DP-SGD (Poisson sampling, per-sample clipping,
Gaussian noise). Its noise multiplier is calibrated by Opacus for the client's
total number of local steps over all rounds, and its own Opacus RDP accountant
tracks every step. Each record belongs to exactly one client, so the
record-level guarantee is the client's (epsilon, delta) (parallel composition);
the run reports the maximum spent epsilon over clients.

Federated averaging alone is not a formal privacy guarantee; the formal
guarantee comes from DP-SGD. Secure aggregation is not implemented (stub), so
the server sees individual (already DP-noised) client updates.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

from src.m7_privacy.dp_training import build_module

BYTES_PER_PARAM = 8  # float64 weights


def stratified_iid_partition(labels: np.ndarray, n_clients: int, seed: int) -> list[np.ndarray]:
    """Random split preserving the label ratio in every client."""
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_clients)]
    for cls in np.unique(labels):
        members = rng.permutation(np.where(labels == cls)[0])
        for client, part in enumerate(np.array_split(members, n_clients)):
            buckets[client].extend(part.tolist())
    return [np.array(sorted(b), dtype=int) for b in buckets]


def group_partition(groups: np.ndarray, n_clients: int, seed: int) -> list[np.ndarray]:
    """Non-IID split by site: whole groups (e.g. U.S. states) are assigned to
    clients, largest first, each to the client with the fewest rows so far."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    ids, sizes = np.unique(groups, return_counts=True)
    order = sorted(range(len(ids)), key=lambda i: (-sizes[i], rng.random()))
    load = np.zeros(n_clients)
    members: list[list[int]] = [[] for _ in range(n_clients)]
    for i in order:
        client = int(np.argmin(load))
        members[client].extend(np.where(groups == ids[i])[0].tolist())
        load[client] += sizes[i]
    return [np.array(sorted(m), dtype=int) for m in members]


def _client_pos_weight(y: np.ndarray) -> float:
    pos = int((y == 1).sum())
    return float((y == 0).sum() / pos) if pos else 1.0


class _Client:
    def __init__(self, module, X, y, *, lr, batch_size, epsilon, delta, max_grad_norm, total_steps, seed):
        self.n = len(y)
        generator = torch.Generator().manual_seed(seed)
        X_t = torch.as_tensor(X, dtype=torch.float64)
        y_t = torch.as_tensor(y, dtype=torch.float64)
        self.module = module
        self.optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        self.loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_t, y_t),
                                                  batch_size=batch_size, shuffle=True, generator=generator)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(_client_pos_weight(y), dtype=torch.float64))
        self.engine = None
        self.noise_multiplier = None
        if epsilon is not None:
            steps_per_epoch = len(self.loader)
            self.noise_multiplier = float(get_noise_multiplier(
                target_epsilon=float(epsilon), target_delta=float(delta), sample_rate=1.0 / steps_per_epoch,
                steps=int(total_steps(steps_per_epoch)), accountant="rdp"))
            self.engine = PrivacyEngine(accountant="rdp", secure_mode=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.module, self.optimizer, self.loader = self.engine.make_private(
                    module=self.module, optimizer=self.optimizer, data_loader=self.loader,
                    noise_multiplier=self.noise_multiplier, max_grad_norm=float(max_grad_norm),
                    poisson_sampling=True)

    @property
    def inner(self) -> torch.nn.Module:
        return getattr(self.module, "_module", self.module)

    def train(self, global_state: dict, epochs: int) -> dict:
        self.inner.load_state_dict(global_state)
        self.module.train()
        for _ in range(epochs):
            for xb, yb in self.loader:
                if len(xb) == 0:
                    continue
                self.optimizer.zero_grad()
                self.loss_fn(self.module(xb).reshape(-1), yb).backward()
                self.optimizer.step()
        return {k: v.detach().clone() for k, v in self.inner.state_dict().items()}

    def epsilon(self, delta: float) -> float | None:
        return None if self.engine is None else float(self.engine.accountant.get_epsilon(delta=delta))


def fedavg_train(kind: str, model_params: dict, clients: list[tuple[np.ndarray, np.ndarray]], *,
                 rounds: int, local_epochs: int, learning_rate: float, sample_rate: float,
                 epsilon: float | None, delta: float, max_grad_norm: float, seed: int,
                 prepare=lambda X: X, evaluate=None) -> dict:
    """Run FedAvg. Returns the final global module, per-round history and privacy summary."""
    torch.manual_seed(seed)
    prepared = [(np.asarray(prepare(X), dtype=float), np.asarray(y)) for X, y in clients]
    global_module = build_module(kind, prepared[0][0].shape[1], model_params)
    n_params = int(sum(p.numel() for p in global_module.parameters()))
    workers = []
    for i, (X, y) in enumerate(prepared):
        workers.append(_Client(copy.deepcopy(global_module), X, y, lr=learning_rate,
                               batch_size=max(1, int(round(sample_rate * len(y)))),
                               epsilon=epsilon, delta=delta, max_grad_norm=max_grad_norm,
                               total_steps=lambda per_epoch: rounds * local_epochs * per_epoch,
                               seed=seed * 100 + i))
    sizes = np.array([w.n for w in workers], dtype=float)
    history = []
    for r in range(1, rounds + 1):
        state = {k: v.clone() for k, v in global_module.state_dict().items()}
        updates = [w.train(state, local_epochs) for w in workers]
        averaged = {k: sum(u[k] * (s / sizes.sum()) for u, s in zip(updates, sizes)) for k in state}
        global_module.load_state_dict(averaged)
        bytes_round = 2 * len(workers) * n_params * BYTES_PER_PARAM  # global down + update up, every client
        entry = {"round": r, "bytes": bytes_round, "cumulative_bytes": bytes_round * r}
        if evaluate is not None:
            entry.update(evaluate(global_module))
        history.append(entry)
    eps = [w.epsilon(delta) for w in workers]
    return {
        "module": global_module,
        "history": pd.DataFrame(history),
        "n_params": n_params,
        "client_sizes": sizes.astype(int).tolist(),
        "client_epsilons": eps,
        "epsilon_spent_max": None if epsilon is None else max(eps),
        "noise_multipliers": [w.noise_multiplier for w in workers],
    }
