from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.m1_data.synthetic import SyntheticDatasetLoader
from src.m2_preprocess.preprocessing import TabularPreprocessor
from src.m3_features.features import FeatureTransformer
from src.m4_encoding import make_encoding
from src.m5_models import make_model
from src.m6_federated import partition_frame
from src.m7_privacy import PrivacyReport, secure_aggregation_status
from src.m8_eval import utility_metrics
from src.m9_audit import loss_threshold_membership_inference


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _build_dataset(config: dict) -> pd.DataFrame:
    dataset_cfg = config.get("dataset", {})
    loader = SyntheticDatasetLoader(
        n_samples=int(dataset_cfg.get("n_samples", 200)),
        n_numeric=int(dataset_cfg.get("n_numeric", 8)),
        n_categorical=int(dataset_cfg.get("n_categorical", 2)),
        positive_fraction=float(dataset_cfg.get("positive_fraction", 0.25)),
        missing_fraction=float(dataset_cfg.get("missing_fraction", 0.0)),
        seed=int(config.get("project", {}).get("seed", 0)),
    )
    return loader.load()


def _prepare_training_data(frame: pd.DataFrame, config: dict):
    """Split rows FIRST, then fit every transformation on the training rows only."""
    split_cfg = config.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    validation_size = float(split_cfg.get("validation_size", 0.2))
    seed = int(config.get("project", {}).get("seed", 0))
    labels = frame["label"].to_numpy(dtype=int)
    positions = np.arange(len(frame))

    train_pos, test_pos = train_test_split(
        positions, test_size=test_size, random_state=seed, stratify=labels
    )
    train_pos, val_pos = train_test_split(
        train_pos,
        test_size=validation_size / (1.0 - test_size),
        random_state=seed,
        stratify=labels[train_pos],
    )

    preprocessor = TabularPreprocessor()
    X_train = preprocessor.fit_transform(frame.iloc[train_pos], label_column="label").to_numpy(dtype=float)
    X_val = preprocessor.transform(frame.iloc[val_pos], label_column="label").to_numpy(dtype=float)
    X_test = preprocessor.transform(frame.iloc[test_pos], label_column="label").to_numpy(dtype=float)
    y_train, y_val, y_test = labels[train_pos], labels[val_pos], labels[test_pos]

    feature_cfg = config.get("features", {})
    transformer = FeatureTransformer(
        d_prime=int(feature_cfg.get("d_prime", 4)),
        use_pca=bool(feature_cfg.get("use_pca", False)),
        seed=seed,
    )
    X_train_reduced = transformer.fit_transform(X_train, y_train)
    X_val_reduced = transformer.transform(X_val)
    X_test_reduced = transformer.transform(X_test)

    return {
        "X_train": X_train_reduced,
        "X_val": X_val_reduced,
        "X_test": X_test_reduced,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_transformer": transformer,
    }


def run_experiment(config_path: str | Path) -> dict:
    config = load_config(config_path)
    project_cfg = config.get("project", {})
    output_dir = Path(project_cfg.get("output_dir", "results"))
    run_name = project_cfg.get("name", "dp-fedq-run")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"SYNTHETIC-{run_name}-{run_id}"  # legacy runner: synthetic data only
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = _build_dataset(config)
    prepared = _prepare_training_data(frame, config)
    X_train = prepared["X_train"]
    X_val = prepared["X_val"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_val = prepared["y_val"]
    y_test = prepared["y_test"]

    encoding_cfg = config.get("encoding", {})
    encoding_name = encoding_cfg.get("name", "angle")
    encoding_repetitions = int(encoding_cfg.get("repetitions", 1))
    n_qubits = int(config.get("model", {}).get("n_qubits", min(X_train.shape[1], 4)))
    circuit = make_encoding(encoding_name, n_qubits=n_qubits, repetitions=encoding_repetitions)
    model_cfg = config.get("model", {})
    model = make_model(
        "variational",
        n_qubits=n_qubits,
        n_layers=int(model_cfg.get("n_layers", 1)),
        learning_rate=float(config.get("federated", {}).get("learning_rate", 0.05)),
        n_steps=200,
        seed=int(project_cfg.get("seed", 0)),
    )
    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    test_predictions = model.predict(X_test)

    metrics = utility_metrics(y_test, test_predictions, test_probs)
    election_loss = loss_threshold_membership_inference(np.abs(y_test - test_probs), threshold=0.5)

    federated_cfg = config.get("federated", {})
    partitions = partition_frame(
        frame,
        label_column="label",
        n_clients=int(federated_cfg.get("n_clients", 3)),
        iid=bool(federated_cfg.get("iid", True)),
        alpha=float(federated_cfg.get("dirichlet_alpha", 0.5)),
        seed=int(project_cfg.get("seed", 0)),
    )
    privacy = PrivacyReport.not_private(
        "this smoke runner trains the VQC without DP-SGD; no epsilon is reported"
    )

    results = {
        "data_source": "SYNTHETIC:generic-generator-v0",
        "warning": "SYNTHETIC DATA - not a finding",
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "dataset_shape": list(frame.shape),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "encoding": {"name": encoding_name, "n_qubits": n_qubits, "repetitions": encoding_repetitions},
        "circuit": {"n_wires": int(circuit.n_wires), "name": getattr(circuit, "name", encoding_name)},
        "metrics": metrics,
        "validation_metrics": utility_metrics(y_val, model.predict(X_val), val_probs),
        "federated_partitions": len(partitions),
        "privacy": privacy.as_dict(),
        "audit": {"loss_threshold_membership_inference": election_loss.tolist()},
        "secure_aggregation": secure_aggregation_status(used=False),
    }

    with (run_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    pd.DataFrame({"true": y_test, "pred": test_predictions, "score": test_probs}).to_csv(
        run_dir / "predictions.csv",
        index=False,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DP-FedQ synthetic experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration")
    args = parser.parse_args()
    results = run_experiment(args.config)
    print(json.dumps({"run_id": results["run_id"], "metrics": results["metrics"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
