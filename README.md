# DP-FedQ — Privacy-Preserving Adolescent Mental Health Using Quantum AI

A hybrid classical–quantum pipeline that identifies adolescents currently at risk
of (diagnosed) depression from tabular survey data. Utility, privacy cost and
computational cost are all measured under the same protocol.

- **Privacy** comes only from differential privacy (DP-SGD via Opacus, with
  Opacus's RDP accountant as the single source of ε) and federated aggregation.
  Quantum circuits are a representation layer and provide **no** privacy.
- Every quantum model is compared against classical baselines on identical rows,
  features and folds. No quantum advantage is assumed or claimed.

## Data

Primary dataset: **NSCH 2024 public-use file**, adolescents aged 12–17
(21,789 labelled rows, 66 whitelisted predictors, 9.96% positive). See
[data/README.md](data/README.md) for the source, checksum, target definition,
leakage guard and the label limitation: the target is caregiver-reported,
provider-*diagnosed* depression, so results identify adolescents resembling
those already diagnosed. They identify who is at risk now. They do not forecast
who will become depressed.

```sh
python scripts/download_nsch.py 2024      # -> data/raw/nsch_2024/ (git-ignored)
```

Any tabular CSV with a binary target can be used via `loader: csv` in a config
(`src/m1_data/loader.py`). Synthetic data (`loader: synthetic_adolescent`) is for
tests and smoke runs only; its outputs go to `results/SYNTHETIC-*`.

## Setup and run

Python 3.10+. `make setup` creates `.venv` with pinned dependencies; `make test`
runs the tests. Each run is defined by one YAML config and writes to
`results/<name>-<timestamp>/` together with the config copy, package versions,
git commit and data checksum.

```sh
python -m src.m10_exp.baselines      --config configs/phase1_nsch2024.yaml   # Phase 1
python -m src.m10_exp.exp1_encodings --config configs/exp1_nsch2024.yaml     # Experiment 1
```

## Evaluation protocol

- A stratified 20% hold-out is locked away and is not used for any choice.
- Outer evaluation: 3 × 5-fold repeated stratified CV on the remaining rows.
  The same seed gives the same folds in every experiment, so results pair up
  fold by fold.
- All preprocessing (median/mode imputation, one-hot encoding, scaling, mutual
  information selection, PCA) is fitted inside each fold on training rows only.
- **Model selection is identical for every model:** the fold's training rows are
  split into inner-train / inner-val. Hyperparameters, early-stopping epoch and
  the decision threshold (Youden's J, maximising balanced accuracy) are chosen
  on inner-val only.
- Headline metrics: **balanced accuracy, recall, ROC-AUC** (plus PR-AUC). Plain
  accuracy is reported but never used as a headline.
- Significance: paired over folds, using the Nadeau–Bengio corrected resampled
  t-test (fold scores are correlated because training sets overlap), with
  Wilcoxon signed-rank as a check and Holm correction across comparisons.

## Method notes

- **Class imbalance** is handled with class weights / positive-class weights, not
  SMOTE. Oversampling copies or interpolates individual records, so one person
  can influence several training examples. That breaks the per-example
  sensitivity DP-SGD accounting assumes, and the reported ε would no longer hold.
- **Quantum subsampling.** Quantum models train on a stratified subsample of
  1,500 rows per fold (~150 positives; inner-train 1,200, inner-val 300). They
  are scored on the full outer validation fold. Classical models are run on both
  the same subsample and the full data.
- **Simulator.** PennyLane `default.qubit` with torch backprop and parameter
  broadcasting. `lightning.qubit` is installed, but its adjoint
  differentiation processes samples one at a time. Here it was ~70× slower
  (8 qubits, 1,200-row batch: 23 s vs 0.33 s per forward+backward).
- **Quantum kernel.** The fidelity kernel |⟨φ(x)|φ(x′)⟩|² is computed exactly from
  simulated statevectors. On hardware it would be estimated with overlap
  circuits and shot noise, at roughly twice the feature-map depth.
- **Barren plateaus.** Gradients of deep, randomly initialised variational
  circuits vanish exponentially with qubit count. VQCs are therefore kept to 1–2
  StronglyEntanglingLayers, initialised with small angles (N(0, 0.1²)).
- **DP and the QSVM.** DP-SGD applies only to gradient-trained models (VQC,
  hybrid QNN, MLP, logistic regression). The kernel SVM is not gradient-trained,
  so it is evaluated without DP and reports no ε.
- **Encoding input sizes differ by design.** Angle, ZZ and re-uploading use one
  feature per qubit. Amplitude encoding uses up to 2ⁿ − 1 features, with one
  amplitude reserved for a constant that keeps magnitude information. At
  8 qubits that is all 95 encoded features.

## Structure

`src/m1_data` … `src/m10_exp` follow the module map in [CLAUDE.md](CLAUDE.md):
data loaders, preprocessing, feature reduction, encodings, classical and quantum
models, federated simulation, privacy accounting, evaluation and statistics,
audits, and experiment runners. `src/m10_exp/runner.py` is a legacy
synthetic-only smoke runner.

## Known stubs

Secure aggregation and gradient-inversion auditing are explicit stubs. Federated
training and DP-SGD experiments are Phases 4–5.
