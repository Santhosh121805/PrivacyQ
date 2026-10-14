# DP-FedQ

DP-FedQ is a research scaffold for a privacy-preserving, hybrid quantum-classical classifier. Phase 1 uses synthetic tabular data only, a CPU statevector simulator, and an in-process federated averaging loop.

Privacy claims in this project come from federated training with differentially private SGD and its stated accountant assumptions. Quantum computation itself is not a privacy mechanism. Quantum models are compared with classical models using identical training/test splits and feature transforms; this project makes no quantum-advantage claim.

## Setup and run

Python 3.10 or newer is required. `make setup` creates `.venv` and installs the pinned dependencies. `make test` runs the tests. Run the end-to-end smoke configuration with:

```sh
python -m src.m10_exp.runner --config configs/smoke.yaml
```

Each run is configured by one YAML file and writes JSON/CSV artifacts below `results/<run_id>/`. `configs/default.yaml` is the larger synthetic experiment configuration.

## Structure

- `src/m1_data` through `src/m10_exp`: data, preprocessing, features, encodings, models, federated training, privacy, evaluation, audits, and experiment runner.
- `configs`: reproducible smoke and default run definitions.
- `tests`: focused tests for module behavior and privacy/federation invariants.
- `data/README.md`: data handling notes; raw and processed data are ignored by Git.

## Phase 1 limitations

- The dataset is synthetic and is not representative of real adolescent mental-health populations. No real data is included.
- Quantum circuits run on a CPU statevector simulator; no quantum hardware or shot-based execution is included.
- The quantum-kernel and gradient-inversion modules are explicit stubs, not implemented methods.
- The federated loop is an in-process FedAvg baseline, not a secure communication system. Secure aggregation is a stub.
- Differential privacy guarantees depend on the implemented sampling, accountant, parameters, and neighboring-dataset definition; reported epsilon is not a blanket guarantee for arbitrary deployment.
