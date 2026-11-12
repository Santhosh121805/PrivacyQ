# DP-FedQ Project Rules

## Design rules

1. Privacy comes only from federated learning plus DP-SGD (and optional secure aggregation later). Quantum computation is not a privacy mechanism.
2. Compare every quantum model against classical baselines on identical features and splits. Do not claim quantum advantage.
3. Split the test data before fitting preprocessing, scalers, feature selectors, or PCA. Fit transformations on training data only; never tune on test data.
4. Define each run with one YAML config, fixed seeds, and logged software versions. Do not hard-code data or output paths.
5. Use circuits with 4 to 12 qubits, a statevector simulator, and CPU-only execution.
6. Never commit raw data. Keep `data/README.md`; ignore raw and processed data.

## Module map

- `m1_data`: seeded synthetic generator and loader interface.
- `m2_preprocess`: cleaning, imputation, categorical encoding, identifier removal, and k-anonymity helper.
- `m3_features`: train-only scaling, mutual-information selection, PCA, UMAP and autoencoder reduction, and angle scaling.
- `m4_encoding`: angle, amplitude, and ZZ feature maps.
- `m5_models`: variational circuit, quantum kernel (statevector and noisy `default.mixed`), three-class models, Qiskit cross-check, and classical baselines.
- `m6_federated`: IID/Dirichlet partitions and in-process weighted FedAvg.
- `m7_privacy`: per-example DP-SGD, RDP accounting, input-stage k-anonymity and local DP, and pairwise-masked secure aggregation.
- `m8_eval`: utility metrics and compute/communication cost.
- `m9_audit`: loss-threshold membership inference, re-identification risk, and a gradient-inversion stub (reports "not implemented").
- `m10_exp`: YAML-driven runner and result logging.

## Commands

- `make setup`: create `.venv` and install pinned dependencies.
- `make test`: run pytest.
- `python -m src.m10_exp.runner --config configs/smoke.yaml`: run the small synthetic end-to-end experiment.
