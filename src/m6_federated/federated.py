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
    if n_clients < 1:
        raise ValueError("n_clients must be at least 1")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    values = np.asarray(labels, dtype=int)
    unique = np.unique(values)
    rng = np.random.default_rng(seed)
    allocations = rng.dirichlet(np.full(len(unique), alpha), size=n_clients)
    client_indices: list[np.ndarray] = []
    for client in range(n_clients):
        counts = np.round(allocations[client] * len(values)).astype(int)
        counts[-1] = len(values) - counts[:-1].sum()
        counts = np.clip(counts, 0, None)
        selected = []
        for class_id, class_index in enumerate(unique):
            members = np.where(values == class_index)[0]
            if len(members) == 0:
                continue
            take = min(counts[class_id], len(members))
            selected.extend(members[:take])
        if len(selected) < len(values) / max(1, n_clients):
            remaining = np.setdiff1d(np.arange(len(values)), np.array(selected, dtype=int))
            selected.extend(remaining[: max(0, len(values) - len(selected))])
        client_indices.append(np.array(sorted(selected), dtype=int))
    return client_indices


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
    return [frame.iloc[indices].copy() for indices in client_index_sets]
