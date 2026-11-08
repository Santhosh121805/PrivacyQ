from __future__ import annotations

import numpy as np
import pandas as pd


def iid_partition(indices: np.ndarray | list[int], n_clients: int, seed: int = 0) -> list[np.ndarray]:
    if n_clients < 1:
        raise ValueError("n_clients must be at least 1")
    values = np.asarray(list(indices), dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(values)
    return np.array_split(shuffled, n_clients)


def dirichlet_partition(
    labels: np.ndarray | list[int],
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 0,
) -> list[np.ndarray]:
    """Label-skewed split: each class's rows are divided across clients in
    Dirichlet(alpha) proportions. Every row goes to exactly one client."""
    if n_clients < 1:
        raise ValueError("n_clients must be at least 1")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    values = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_clients)]
    for cls in np.unique(values):
        members = rng.permutation(np.where(values == cls)[0])
        shares = rng.dirichlet(np.full(n_clients, alpha))
        cuts = (np.cumsum(shares)[:-1] * len(members)).astype(int)
        for client, part in enumerate(np.split(members, cuts)):
            buckets[client].extend(part.tolist())
    return [np.array(sorted(b), dtype=int) for b in buckets]


def weighted_fedavg(client_updates: list[np.ndarray | dict[str, np.ndarray]], weights: list[float] | None = None):
    if not client_updates:
        raise ValueError("at least one client update is required")
    if weights is None:
        weights = [1.0] * len(client_updates)
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != len(client_updates):
        raise ValueError("weights must match the number of client updates")
    if np.allclose(weights, 0.0):
        raise ValueError("weights cannot all be zero")

    first = client_updates[0]
    if isinstance(first, dict):
        averaged: dict[str, np.ndarray] = {}
        for key in first:
            stacked = np.stack([np.asarray(client[key], dtype=float) for client in client_updates])
            averaged[key] = np.average(stacked, axis=0, weights=weights)
        return averaged

    stacked = np.stack([np.asarray(client, dtype=float) for client in client_updates])
    return np.average(stacked, axis=0, weights=weights)


def fedavg(*args, **kwargs):
    return weighted_fedavg(*args, **kwargs)


def federated_average(*args, **kwargs):
    return weighted_fedavg(*args, **kwargs)


def partition_frame(
    frame: pd.DataFrame,
    label_column: str = "label",
    n_clients: int = 3,
    iid: bool = True,
    alpha: float = 0.5,
    seed: int = 0,
) -> list[pd.DataFrame]:
    assert label_column in frame.columns
    indices = np.arange(len(frame))
    if iid:
        client_index_sets = iid_partition(indices, n_clients=n_clients, seed=seed)
    else:
        client_index_sets = dirichlet_partition(frame[label_column].to_numpy(), n_clients=n_clients, alpha=alpha, seed=seed)
    return [frame.iloc[part].copy() for part in client_index_sets]
