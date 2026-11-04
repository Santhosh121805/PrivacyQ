import numpy as np
import pandas as pd
import pytest

from src.m1_data import CSVDatasetLoader, SyntheticAdolescentLoader, SyntheticDatasetLoader, make_loader
from src.m1_data.nsch import FEATURES, assert_no_leakage
from src.m3_features.features import FeatureTransformer
from src.m5_models.classical import make_classical
from src.m8_eval.evaluation import binary_metrics
from src.m10_exp.common import cv_folds, holdout_split


def test_synthetic_loaders_are_flagged_synthetic():
    assert SyntheticAdolescentLoader(n_samples=50).is_synthetic
    assert SyntheticDatasetLoader(n_samples=50).is_synthetic
    assert SyntheticAdolescentLoader(n_samples=50).source_tag().startswith("SYNTHETIC:")


def test_synthetic_adolescent_is_seeded_and_imbalanced():
    a = SyntheticAdolescentLoader(n_samples=4000, seed=3).load()
    b = SyntheticAdolescentLoader(n_samples=4000, seed=3).load()
    assert a.equals(b)
    assert 0.05 <= a["label"].mean() <= 0.11


def test_csv_loader_maps_target_and_keeps_metadata(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"x": [1, 2, 3, 4], "school": [1, 1, 2, 2], "dep": ["yes", "no", None, "no"]}).to_csv(path, index=False)
    loader = CSVDatasetLoader(path=str(path), target_column="dep", positive_values=["yes"], group_column="school")
    frame = loader.load()
    assert frame["label"].tolist() == [1, 0, 0]
    assert "school" in loader.metadata_columns and not loader.is_synthetic


def test_nsch_whitelist_has_no_diagnosis_or_treatment_columns():
    assert_no_leakage(list(FEATURES))
    with pytest.raises(ValueError):
        assert_no_leakage(["k2q33a"])


def test_make_loader_dispatch():
    assert isinstance(make_loader({"loader": "synthetic_adolescent", "n_samples": 20}), SyntheticAdolescentLoader)


def test_mi_and_pca_paths_return_k_angle_features():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    for method in ("mi", "pca"):
        out = FeatureTransformer(d_prime=4, method=method, seed=0).fit_transform(X, y)
        assert out.shape == (200, 4)
        assert np.all((out >= 0) & (out <= np.pi))
    mi = FeatureTransformer(d_prime=1, method="mi", seed=0).fit(X, y)
    assert mi.selector.get_support()[0]


def test_holdout_and_folds_are_disjoint_and_stratified():
    y = np.array([1] * 20 + [0] * 180)
    dev, hold = holdout_split(y, 0.2, seed=1)
    assert not set(dev) & set(hold)
    assert abs(y[hold].mean() - 0.1) < 0.03
    folds = cv_folds(dev, y, n_splits=4, n_repeats=2, seed=1)
    assert len(folds) == 8
    for f in folds:
        assert not set(f["train"]) & set(f["val"]) and not set(f["val"]) & set(hold)


def test_metrics_penalise_always_negative_classifier():
    y = np.array([0] * 90 + [1] * 10)
    m = binary_metrics(y, np.zeros(100))
    assert m["accuracy"] == 0.9 and m["balanced_accuracy"] == 0.5 and m["recall"] == 0.0


@pytest.mark.parametrize("name", ["logreg", "random_forest", "xgboost", "mlp"])
def test_classical_baselines_fit_and_score(name):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 1.2).astype(int)
    overrides = {"random_forest": {"n_estimators": 20}, "mlp": {"epochs": 3}}.get(name, {})
    model = make_classical(name, y, seed=0, n_jobs=1, **overrides).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (300,) and np.all((p >= 0) & (p <= 1))
