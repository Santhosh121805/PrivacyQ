from src.m1_data.synthetic import SyntheticDatasetLoader


def test_synthetic_loader_is_seeded_and_has_imbalanced_binary_label():
    first = SyntheticDatasetLoader(n_samples=200, seed=9).load()
    second = SyntheticDatasetLoader(n_samples=200, seed=9).load()

    assert first.equals(second)
    assert first["label"].mean() == 0.25
    assert set(first["label"].unique()) == {0, 1}
    assert {"subject_id", "name"}.issubset(first.columns)


def test_synthetic_loader_rejects_invalid_parameters():
    try:
        SyntheticDatasetLoader(n_samples=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid n_samples to fail")
