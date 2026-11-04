"""Federated learning helpers for in-process client simulation."""

from .federated import (
    dirichlet_partition,
    fedavg,
    federated_average,
    iid_partition,
    partition_frame,
    weighted_fedavg,
)

__all__ = [
    "iid_partition",
    "dirichlet_partition",
    "weighted_fedavg",
    "fedavg",
    "federated_average",
    "partition_frame",
]
