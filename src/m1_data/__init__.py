"""Dataset generation and loading interfaces."""

from __future__ import annotations

from src.m1_data.adodas import AdoDASLoader
from src.m1_data.loader import CSVDatasetLoader, DatasetLoader
from src.m1_data.mhtext import MHTextLoader
from src.m1_data.nsch import NSCHLoader
from src.m1_data.synthetic import SyntheticAdolescentLoader, SyntheticDatasetLoader


def make_loader(data_cfg: dict, seed: int = 0) -> DatasetLoader:
    """Build a loader from the ``data`` section of a YAML config."""
    cfg = dict(data_cfg)
    kind = cfg.pop("loader")
    if kind == "nsch":
        return NSCHLoader(**cfg)
    if kind == "adodas":
        return AdoDASLoader(**cfg)
    if kind == "mhtext":
        return MHTextLoader(**cfg)
    if kind == "csv":
        return CSVDatasetLoader(**cfg)
    if kind == "synthetic_adolescent":
        return SyntheticAdolescentLoader(seed=cfg.pop("seed", seed), **cfg)
    if kind == "synthetic":
        return SyntheticDatasetLoader(seed=cfg.pop("seed", seed), **cfg)
    raise ValueError(f"unknown loader {kind!r}")


__all__ = [
    "AdoDASLoader",
    "CSVDatasetLoader",
    "DatasetLoader",
    "MHTextLoader",
    "NSCHLoader",
    "SyntheticAdolescentLoader",
    "SyntheticDatasetLoader",
    "make_loader",
]
