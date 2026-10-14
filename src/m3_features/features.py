from functools import partial

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeatureTransformer:
    """Fit feature transformations on training data and reuse them unchanged."""

    def __init__(
        self,
        d_prime: int,
        use_pca: bool = False,
        n_selected: int | None = None,
        seed: int = 0,
    ) -> None:
        if not 1 <= d_prime <= 12:
            raise ValueError("d_prime must be between 1 and 12")
        self.d_prime = d_prime
        self.use_pca = use_pca
        self.n_selected = n_selected
        self.seed = seed
        self.scaler = StandardScaler()
        self.selector = SelectKBest(score_func=partial(mutual_info_classif, random_state=seed))
        self.pca: PCA | None = PCA(n_components=d_prime, random_state=seed) if use_pca else None
        self.angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi), clip=True)
        self.n_input_features_: int | None = None
        self.is_fitted_ = False

    def fit(self, features: pd.DataFrame | np.ndarray, labels: np.ndarray) -> "FeatureTransformer":
        values = self._as_array(features)
        labels = np.asarray(labels).reshape(-1)
        if values.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have the same number of rows")
        if values.shape[1] < self.d_prime:
            raise ValueError("d_prime cannot exceed the number of input features")
        self.n_input_features_ = values.shape[1]

        selected_count = self.n_selected
        if selected_count is None:
            selected_count = min(values.shape[1], self.d_prime * (2 if self.use_pca else 1))
        if selected_count < self.d_prime or selected_count > values.shape[1]:
            raise ValueError("n_selected must be between d_prime and the input feature count")

        standardized = self.scaler.fit_transform(values)
        self.selector.set_params(k=selected_count)
        selected = self.selector.fit_transform(standardized, labels)
        reduced = self.pca.fit_transform(selected) if self.pca is not None else selected
        self.angle_scaler.fit(reduced)
        self.is_fitted_ = True
        return self

    def transform(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("fit must be called before transform")
        values = self._as_array(features)
        if values.shape[1] != self.n_input_features_:
            raise ValueError("input feature count differs from fitted training data")
        standardized = self.scaler.transform(values)
        selected = self.selector.transform(standardized)
        reduced = self.pca.transform(selected) if self.pca is not None else selected
        return self.angle_scaler.transform(reduced)

    def fit_transform(
        self,
        features: pd.DataFrame | np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        return self.fit(features, labels).transform(features)

    @staticmethod
    def _as_array(features: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = features.to_numpy(dtype=float) if isinstance(features, pd.DataFrame) else np.asarray(features, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("features must be a finite two-dimensional matrix")
        return values
