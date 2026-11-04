"""Pluggable tabular dataset loaders.

Every loader returns a pandas DataFrame holding feature columns, a binary
``label_column`` (0/1) and optional metadata columns (listed in
``metadata_columns``) that must never be used as model features, e.g. a
group/site column used later for federated partitioning or survey weights.

``is_synthetic`` must be True for any generated data so that every downstream
output can be labelled as synthetic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd


class DatasetLoader(ABC):
    """Interface for tabular datasets consumed by the experiment runners."""

    label_column: str = "label"
    name: str = "unnamed"
    is_synthetic: bool = False
    metadata_columns: tuple[str, ...] = ()
    group_column: str | None = None

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a fresh table containing features and the target label."""

    def source_tag(self) -> str:
        """Short provenance string written next to every reported number."""
        prefix = "SYNTHETIC" if self.is_synthetic else "REAL"
        return f"{prefix}:{self.name}"

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "is_synthetic": self.is_synthetic,
            "label_column": self.label_column,
            "metadata_columns": list(self.metadata_columns),
            "group_column": self.group_column,
        }


@dataclass
class CSVDatasetLoader(DatasetLoader):
    """Load any CSV with a binary target.

    ``positive_values`` maps the raw target to 1 when the raw value is in the
    list and 0 otherwise; if omitted the target must already be 0/1. Rows whose
    raw target is missing are dropped. ``categorical_columns`` are cast to
    strings so the preprocessor one-hot encodes them.
    """

    path: str = ""
    target_column: str = "label"
    positive_values: Sequence[object] | None = None
    feature_columns: Sequence[str] | None = None
    drop_columns: Sequence[str] = ()
    categorical_columns: Sequence[str] = ()
    group_column: str | None = None
    weight_column: str | None = None
    name: str = "csv"
    is_synthetic: bool = False
    label_column: str = "label"
    metadata_columns: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("CSVDatasetLoader requires a path")
        self.metadata_columns = tuple(
            column for column in (self.group_column, self.weight_column) if column
        )

    def load(self) -> pd.DataFrame:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"dataset file not found: {path}")
        frame = pd.read_csv(path)
        if self.target_column not in frame:
            raise ValueError(f"target column {self.target_column!r} not in {path}")
        frame = frame.dropna(subset=[self.target_column]).reset_index(drop=True)

        raw_target = frame.pop(self.target_column)
        if self.positive_values is not None:
            labels = raw_target.isin(list(self.positive_values)).astype(int)
        else:
            if not set(pd.unique(raw_target)).issubset({0, 1}):
                raise ValueError("target must be 0/1 unless positive_values is given")
            labels = raw_target.astype(int)

        frame = frame.drop(columns=[c for c in self.drop_columns if c in frame])
        keep = list(self.feature_columns) if self.feature_columns else [
            c for c in frame.columns if c not in self.metadata_columns
        ]
        missing = set(keep) - set(frame.columns)
        if missing:
            raise ValueError(f"feature columns not found: {sorted(missing)}")
        result = frame[keep + [c for c in self.metadata_columns if c in frame]].copy()
        for column in self.categorical_columns:
            if column in result:
                result[column] = result[column].map(lambda v: v if pd.isna(v) else str(v))
        result[self.label_column] = labels.to_numpy()
        if labels.nunique() != 2:
            raise ValueError("target must contain both classes")
        return result
