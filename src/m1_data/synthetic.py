from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.m1_data.loader import DatasetLoader


@dataclass
class SyntheticDatasetLoader(DatasetLoader):
    n_samples: int = 2000
    n_numeric: int = 16
    n_categorical: int = 4
    positive_fraction: float = 0.25
    missing_fraction: float = 0.03
    seed: int = 0
    label_column: str = "label"

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        if self.n_numeric < 1 or self.n_categorical < 0:
            raise ValueError("at least one numeric feature is required")
        if not 0 < self.positive_fraction < 1:
            raise ValueError("positive_fraction must be between 0 and 1")
        if not 0 <= self.missing_fraction < 1:
            raise ValueError("missing_fraction must be in [0, 1)")

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n_positive = int(round(self.n_samples * self.positive_fraction))
        labels = np.zeros(self.n_samples, dtype=np.int64)
        labels[:n_positive] = 1
        rng.shuffle(labels)

        values: dict[str, object] = {}
        for index in range(self.n_numeric):
            signal = labels * (1.1 if index < 4 else 0.25)
            values[f"numeric_{index:02d}"] = rng.normal(signal, 1.0)

        for index in range(self.n_categorical):
            categories = np.array(["low", "medium", "high"])
            probabilities = np.where(
                labels[:, None] == 1,
                np.array([[0.22, 0.38, 0.40]]),
                np.array([[0.40, 0.38, 0.22]]),
            )
            values[f"category_{index:02d}"] = [
                rng.choice(categories, p=row) for row in probabilities
            ]

        frame = pd.DataFrame(values)
        if self.missing_fraction:
            missing = rng.random(frame.shape) < self.missing_fraction
            frame = frame.mask(missing)

        frame[self.label_column] = labels
        frame["subject_id"] = [f"S{index:06d}" for index in range(self.n_samples)]
        frame["name"] = [f"Synthetic Participant {index:06d}" for index in range(self.n_samples)]
        return frame
