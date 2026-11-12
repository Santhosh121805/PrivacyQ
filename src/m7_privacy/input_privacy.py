"""Input-stage privacy: applied to the data BEFORE quantum encoding.

Two mechanisms, each with its own accounting, separate from training-time DP-SGD:

1. ``KAnonymizer`` (k-anonymity, no ε): quasi-identifiers are generalised
   (age into bands, poverty ratio into bands), then every record whose
   generalised quasi-identifier combination occurs fewer than k times in the
   *training* rows has all its quasi-identifier cells suppressed (set to
   missing; local suppression). Records are kept, so no one is silently
   dropped. The fitted set of "safe" combinations is reused for other rows:
   a record whose combination is not safe is suppressed the same way.
   The k-anonymity of the transformed training rows is checked with
   ``k_anonymity_report`` and returned in ``report_``.

2. ``LaplaceLDP`` (local differential privacy, pure ε-LDP, δ = 0): each
   record's reduced feature vector (the encoder input, values clipped to
   [low, high]) is perturbed with Laplace noise of scale b = d · (high − low) / ε,
   where d is the number of features (L1 sensitivity of a whole record).
   Clipping back to [low, high] afterwards is post-processing. The guarantee
   covers the record's features only — labels are not perturbed — and holds
   given the feature projection, which is fitted on training rows.

Neither mechanism is quantum; quantum encoding provides no privacy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.m2_preprocess.preprocessing import k_anonymity_report

# Demographic quasi-identifiers in the NSCH feature whitelist, with generalisation rules.
NSCH_QUASI_IDENTIFIERS: dict[str, list | None] = {
    "sc_age_years": [11, 13, 15, 17],          # bands 12-13, 14-15, 16-17
    "sc_sex": None,
    "sc_racer": None,
    "sc_hispanic_r": None,
    "fpl_i1": [0, 99, 199, 399, 400],          # poverty ratio <100, 100-199, 200-399, 400+
    "family_r": None,
    "hhlanguage": None,
}


@dataclass
class KAnonymizer:
    k: int = 5
    quasi_identifiers: dict[str, list | None] = field(default_factory=lambda: dict(NSCH_QUASI_IDENTIFIERS))

    def _generalise(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col, edges in self.quasi_identifiers.items():
            if col not in out:
                raise ValueError(f"quasi-identifier {col!r} missing from frame")
            if edges is not None:
                band = pd.cut(pd.to_numeric(out[col], errors="coerce"), edges, labels=False, include_lowest=True)
                out[col] = band.astype(float) + 1.0     # ordinal band index 1..n, NaN stays NaN
        return out

    def _keys(self, frame: pd.DataFrame) -> pd.Series:
        cols = list(self.quasi_identifiers)
        return frame[cols].astype(str).agg("|".join, axis=1)

    def fit(self, frame: pd.DataFrame) -> "KAnonymizer":
        if self.k < 2:
            raise ValueError("k must be at least 2")
        gen = self._generalise(frame)
        counts = self._keys(gen).value_counts()
        self.safe_keys_ = set(counts[counts >= self.k].index)
        transformed = self.transform(frame)
        rep = k_anonymity_report(transformed, list(self.quasi_identifiers), self.k)
        self.report_ = {"k": self.k, "n_rows": int(len(frame)), "n_suppressed": int(self.last_suppressed_),
                        "suppression_rate": float(self.last_suppressed_ / max(1, len(frame))),
                        "min_class_size": rep["minimum_group_size"], "k_anonymous": bool(rep["compliant"]),
                        "n_classes_kept": int(len(self.safe_keys_))}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "safe_keys_"):
            raise RuntimeError("fit must be called before transform")
        gen = self._generalise(frame)
        unsafe = ~self._keys(gen).isin(self.safe_keys_).to_numpy()
        cols = list(self.quasi_identifiers)
        gen.loc[unsafe, cols] = np.nan
        self.last_suppressed_ = int(unsafe.sum())
        return gen


@dataclass
class LaplaceLDP:
    epsilon: float | None = None          # None = no local DP (ε = ∞)
    low: float = 0.0
    high: float = float(np.pi)
    seed: int = 0

    def noise_scale(self, n_features: int) -> float:
        if self.epsilon is None:
            return 0.0
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return n_features * (self.high - self.low) / float(self.epsilon)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(np.asarray(X, dtype=float), self.low, self.high)
        if self.epsilon is None:
            return X
        rng = np.random.default_rng(self.seed)
        noisy = X + rng.laplace(0.0, self.noise_scale(X.shape[1]), size=X.shape)
        return np.clip(noisy, self.low, self.high)

    def report(self, n_features: int) -> dict:
        return {"epsilon_input": None if self.epsilon is None else float(self.epsilon), "delta_input": 0.0,
                "mechanism": "Laplace, record-level L1 sensitivity d·(high−low)", "n_features": n_features,
                "noise_scale": self.noise_scale(n_features), "covers": "features only (labels not perturbed)"}
