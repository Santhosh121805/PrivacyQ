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
    name: str = "generic-generator-v0"
    is_synthetic: bool = True

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


@dataclass
class SyntheticAdolescentLoader(DatasetLoader):
    """SYNTHETIC adolescent-survey-like data for pipeline validation only.

    Mimics the structure of a school/household survey: demographics, Likert
    items, binned sleep and screen-time hours, binary adverse-experience and
    family-history flags, a nominal school/site id, and a ~8% positive class.
    The label depends on non-linear interactions (bullying x short sleep,
    screen time x sex, a threshold on adverse-experience count, poverty x
    caregiver mental health) so the task is not trivially linear.

    Numbers produced from this loader say nothing about real adolescents.
    """

    n_samples: int = 3000
    positive_fraction: float = 0.08
    missing_fraction: float = 0.03
    n_sites: int = 10
    seed: int = 0
    name: str = "adolescent-generator-v1"
    is_synthetic: bool = True
    label_column: str = "label"
    group_column: str | None = "site"
    metadata_columns: tuple[str, ...] = ("site",)

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.n_samples
        site = rng.integers(0, self.n_sites, size=n)
        site_effect = rng.normal(0.0, 0.4, size=self.n_sites)[site]
        age = rng.integers(12, 18, size=n).astype(float)
        female = rng.random(n) < 0.5
        poverty = np.clip(rng.normal(250, 110, size=n), 50, 400).round()
        parent_mh = np.clip(rng.poisson(1.2, size=n) + 1, 1, 5).astype(float)
        family_history = (rng.random(n) < 0.12 + 0.05 * (parent_mh >= 3)).astype(float)
        aces = (rng.random((n, 8)) < np.linspace(0.05, 0.25, 8)).astype(float)
        sleep = np.clip(np.round(rng.normal(4.2 - 0.15 * (age - 12), 1.1)), 1, 7)
        screen = np.clip(np.round(rng.normal(3.4 + 0.2 * (age - 12) / 2, 1.0)), 1, 5)
        activity = np.clip(np.round(rng.normal(2.5, 0.9, size=n)), 1, 4)
        bullied = np.clip(rng.poisson(0.6, size=n) + 1, 1, 5).astype(float)
        latent_mood = rng.normal(0.0, 1.0, size=n)
        likert = np.clip(
            np.round(2.0 + 0.6 * latent_mood[:, None] + rng.normal(0, 0.8, size=(n, 6))), 1, 4
        )

        ace_count = aces.sum(axis=1)
        logit = (
            0.9 * latent_mood
            + 0.35 * (bullied - 1) * (sleep <= 3)
            + 0.45 * (screen - 3) * female
            + 1.1 * (ace_count >= 3)
            + 0.5 * family_history
            + 0.3 * (parent_mh - 2) * (poverty < 150)
            - 0.25 * (activity - 2.5)
            + 0.15 * (age - 14.5)
            + site_effect
        )
        lo, hi = -15.0, 5.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if (1 / (1 + np.exp(-(logit + mid)))).mean() > self.positive_fraction:
                hi = mid
            else:
                lo = mid
        label = (rng.random(n) < 1 / (1 + np.exp(-(logit + lo)))).astype(int)

        frame = pd.DataFrame(
            {
                "age": age,
                "female": female.astype(float),
                "race": rng.choice(["a", "b", "c"], size=n, p=[0.7, 0.15, 0.15]),
                "poverty_ratio": poverty,
                "parent_mental_health": parent_mh,
                "family_history_depression": family_history,
                **{f"ace_{i + 1}": aces[:, i] for i in range(8)},
                "sleep_hours_bin": sleep,
                "screen_time_bin": screen,
                "physical_activity": activity,
                "bullied": bullied,
                **{f"likert_{i + 1}": likert[:, i] for i in range(6)},
            }
        )
        if self.missing_fraction:
            frame = frame.mask(rng.random(frame.shape) < self.missing_fraction)
        frame["site"] = site
        frame[self.label_column] = label
        return frame
