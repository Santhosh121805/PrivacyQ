from src.m1_data.synthetic import SyntheticDatasetLoader
from src.m2_preprocess.preprocessing import TabularPreprocessor
from src.m10_exp import runner


def test_runner_fits_preprocessing_on_training_rows_only(monkeypatch):
    seen = []
    original_fit = TabularPreprocessor.fit

    def recording_fit(self, frame, label_column):
        seen.append(set(frame.index))
        return original_fit(self, frame, label_column)

    monkeypatch.setattr(TabularPreprocessor, "fit", recording_fit)
    frame = SyntheticDatasetLoader(n_samples=200, seed=1).load()
    config = {"project": {"seed": 1}, "split": {"test_size": 0.2, "validation_size": 0.2},
              "features": {"d_prime": 4}}
    runner._prepare_training_data(frame, config)

    assert len(seen) == 1
    assert len(seen[0]) == 120  # 60% of 200 rows; val and test rows never seen
