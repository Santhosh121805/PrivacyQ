from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_IDENTIFIER_COLUMNS = {
    "id",
    "subject_id",
    "participant_id",
    "name",
    "identifier",
}


class TabularPreprocessor:
    """Impute and encode using statistics learned from training rows only."""

    def __init__(self, identifier_columns: Iterable[str] | None = None) -> None:
        self.identifier_columns = set(identifier_columns or ()) | DEFAULT_IDENTIFIER_COLUMNS
        self.transformer: ColumnTransformer | None = None
        self.feature_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []

    def fit(self, frame: pd.DataFrame, label_column: str) -> "TabularPreprocessor":
        self._validate_frame(frame, label_column)
        features = frame.drop(columns=[label_column])
        self.feature_columns_ = [
            column for column in features.columns
            if column.lower() not in self.identifier_columns
        ]
        features = features[self.feature_columns_]
        self.numeric_columns_ = features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns_ = [
            column for column in self.feature_columns_ if column not in self.numeric_columns_
        ]
        if not self.feature_columns_:
            raise ValueError("no usable feature columns remain after identifier removal")

        transformers = []
        if self.numeric_columns_:
            transformers.append(
                ("numeric", SimpleImputer(strategy="median"), self.numeric_columns_)
            )
        if self.categorical_columns_:
            categorical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, self.categorical_columns_))

        self.transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.transformer.fit(features)
        return self

    def transform(self, frame: pd.DataFrame, label_column: str | None = None) -> pd.DataFrame:
        if self.transformer is None:
            raise RuntimeError("fit must be called before transform")
        if label_column is not None and label_column in frame:
            frame = frame.drop(columns=[label_column])
        missing = set(self.feature_columns_) - set(frame.columns)
        if missing:
            raise ValueError(f"input is missing fitted feature columns: {sorted(missing)}")
        transformed = self.transformer.transform(frame[self.feature_columns_])
        names = self.transformer.get_feature_names_out()
        result = pd.DataFrame(transformed, columns=names, index=frame.index, dtype=float)
        if result.isna().any().any():
            raise ValueError("preprocessing produced missing feature values")
        return result

    def fit_transform(self, frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
        return self.fit(frame, label_column).transform(frame, label_column)

    @staticmethod
    def _validate_frame(frame: pd.DataFrame, label_column: str) -> None:
        if label_column not in frame.columns:
            raise ValueError(f"label column {label_column!r} is not present")
        if frame.empty:
            raise ValueError("cannot fit preprocessing on an empty frame")


def k_anonymity_report(
    frame: pd.DataFrame,
    quasi_identifier_columns: Iterable[str],
    k: int = 5,
) -> dict[str, int | bool]:
    if k < 1:
        raise ValueError("k must be at least 1")
    columns = list(quasi_identifier_columns)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"unknown quasi-identifier columns: {sorted(missing)}")
    if not columns:
        raise ValueError("at least one quasi-identifier column is required")
    if frame.empty:
        minimum_group_size = 0
    else:
        group_sizes = frame.groupby(columns, dropna=False, observed=True).size()
        minimum_group_size = int(group_sizes.min())
    return {
        "k": k,
        "minimum_group_size": minimum_group_size,
        "compliant": minimum_group_size >= k,
    }


def check_k_anonymity(
    frame: pd.DataFrame,
    quasi_identifier_columns: Iterable[str],
    k: int = 5,
) -> bool:
    return bool(k_anonymity_report(frame, quasi_identifier_columns, k)["compliant"])
