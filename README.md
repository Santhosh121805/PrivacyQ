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
python -m src.m10_exp.baselines      --config configs/phase1_nsch2024.yaml        # Phase 1 (untuned baselines)
python -m src.m10_exp.exp1_encodings --config configs/exp1_nsch2024_final.yaml    # Exp 1: encodings
python -m src.m10_exp.exp2_compare   --config configs/exp2_nsch2024.yaml          # Exp 2: quantum vs classical
python -m src.m10_exp.exp3_privacy   --config configs/exp3_nsch2024.yaml          # Exp 3: privacy-utility curve
python -m src.m10_exp.exp3_privacy   --config configs/exp3_sensitivity_clipping.yaml
python -m src.m10_exp.exp4_federated --config configs/exp4_nsch2024.yaml          # Exp 4: federated + DP
python -m src.m10_exp.final_report   --config configs/final_report.yaml           # results/RESULTS.md
```

The `resume:` blocks in the Exp 1/Exp 2 configs reuse rows from interrupted runs; delete
them to recompute everything from scratch. Update the run directories in
`configs/exp4_nsch2024.yaml` (`reference_run`) and `configs/final_report.yaml` after re-running.

## What was found (plain English)

All numbers are from `results/RESULTS.md` (real NSCH 2024 data, mean over folds).

1. **Classical models do well and agree.** Tuned logistic regression, XGBoost and an MLP all
   reach ROC-AUC ≈ 0.89 on all 95 features. Logistic regression is as good as the
   non-linear models, so the signal is mostly additive.
2. **Encoding choice matters a lot for the VQC, less for the kernel.** Swapping encodings
   moved VQC ROC-AUC by 7–17 points (10–26%), in line with the survey's 10–30% claim; for the
   quantum kernel SVM the swing was 4–7 points. Simple angle / re-uploading encodings won;
   the entangling ZZ map and amplitude encoding were worst.
3. **No quantum advantage.** On identical rows and features, the best quantum kernel
   (0.847) was statistically tied with logistic regression (0.851, Holm p = 0.70) and was not
   significantly different from XGBoost or the MLP on those small reduced inputs; the VQC (0.814)
   and hybrid QNN (0.784) were significantly worse than logistic regression (Holm p = 0.012 and 0.041). All quantum models were far below the
   full-data classical models (0.888), mostly because of the 1,500-row, 4–8-feature limit.
4. **Privacy costs accuracy, and quantum does not degrade differently once the classical
   baseline is set up fairly.** Under DP-SGD the VQC kept 0.805 ROC-AUC at ε = 4 (0.653 at
   ε = 0.5). With the shared clipping norm of 1, logistic regression on the same rows looked
   much worse under DP, but a sensitivity check with clipping norm 10 showed that was an
   artefact: logistic regression then matched the VQC at ε = 4 and was higher at ε = 0.5
   (0.758 vs 0.653; not significant: Holm p = 0.64; Wilcoxon sensitivity check Holm p = 0.10). Logistic regression on the full data stayed best at every ε (0.802 at
   ε = 0.5, 0.856 at ε = 4).
5. **Federated + DP works for the classical model, not for the small quantum one.** With
   FedAvg and client-side DP at ε = 4, full-data logistic regression matched centralised DP
   (≈0.86) for 3, 5 and 10 clients, IID or split by state. The VQC, trained on 1,200 rows split
   across clients, fell from 0.805 (centralised) to ≈0.66 with 10 clients: each client has
   too few records for DP noise to average out.

6. **The locked hold-out confirms the cross-validation numbers.** Scored once at the end
   (`python -m src.m10_exp.holdout_eval --config configs/holdout_final.yaml`, which refuses to
   run twice): LogReg 0.892, QSVM 0.854, DP LogReg ε = 4 0.861 — each CV mean lies inside the
   hold-out 95% interval.

Bottom line: on this dataset, quantum encodings are a workable representation layer but
give no accuracy or privacy benefit over a well-set-up classical baseline; the privacy
guarantee always comes from DP-SGD, never from the quantum circuit.

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
  t-test (fold scores are correlated because training sets overlap), Holm-adjusted
  within each family of comparisons; Wilcoxon signed-rank (Holm-adjusted) is reported
  as a sensitivity check. Unless stated otherwise every p-value in this repository is
  the Holm-adjusted corrected t. Rationale and audit: top of `results/RESULTS.md` and
  `results/significance/changed.md`.

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
  hybrid QNN, MLP, logistic regression). Opacus has no per-sample gradient rule for a
  PennyLane circuit, so `src/m5_models/quantum_dp.py` registers one (verified exact in tests);
  Opacus then clips, adds noise and accounts as usual. The kernel SVM is not gradient-trained,
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

## Known stubs and limits

Gradient-inversion auditing is an explicit stub that reports "not implemented". Secure aggregation is
implemented (pairwise-masked, `src/m7_privacy/secure_aggregation.py`) but was not used in Experiments 3–4.
Input-stage privacy (k-anonymity, local DP) is evaluated in Experiment 5 only. RandomForest was dropped from
all results on supervisor instruction. Status of every architecture-diagram item: `CONFORMANCE.md`. See the Limitations section of
`results/RESULTS.md`.
