from abc import ABC, abstractmethod

import pandas as pd


class DatasetLoader(ABC):
    """Interface for tabular datasets consumed by the experiment runner."""

    label_column: str

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a fresh table containing features and the target label."""
