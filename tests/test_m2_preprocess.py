import numpy as np
import pandas as pd

from src.m1_data.synthetic import SyntheticDatasetLoader
from src.m2_preprocess.preprocessing import (
    TabularPreprocessor,
    check_k_anonymity,
    k_anonymity_report,
)


def test_preprocessing_drops_identifiers_and_removes_missing_values():
    frame = SyntheticDatasetLoader(n_samples=100, seed=3).load()
    result = TabularPreprocessor().fit_transform(frame, "label")

    assert not result.isna().any().any()
    assert not any("subject_id" in column or "name" in column for column in result.columns)
    assert np.isfinite(result.to_numpy()).all()


def test_k_anonymity_helper_reports_smallest_group():
    frame = pd.DataFrame({"band": ["a", "a", "b", "b", "b"]})

    report = k_anonymity_report(frame, ["band"], k=3)

    assert report == {"k": 3, "minimum_group_size": 2, "compliant": False}
    assert check_k_anonymity(frame, ["band"], k=2)
