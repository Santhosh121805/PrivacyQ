# Conformance with the architecture diagram

This document states, for every box of the architecture diagram ("Quantum AI for Mental Health",
boxes 1–15), what this repository actually implements. It is kept exactly true: a sub-item is
**BUILT** only if code exists, runs on the real data, and has a test or a results file behind it.

Status values:

- **BUILT**: implemented, exercised, evidence listed.
- **PARTIAL**: implemented in a narrower form than the diagram names; the note says exactly how.
- **PHASE-2**: not implemented. Either scheduled in the current conformance sprint (noted as
  "scheduled") or out of scope for a stated reason.

Paths are relative to the repository root. `results/` is git-ignored and produced by the commands in
`README.md`; the run directories referenced are those listed in `configs/final_report.yaml`.

_Last updated: 2026-11-12, after Priority 1, Priority 2 and Priority 3 item 7._

## Pipeline flow as built

NSCH 2024 survey → cleaning, identifier removal, train/test split →
**[input-stage privacy: k-anonymity (k = 5) on quasi-identifiers]** → imputation and one-hot encoding →
feature reduction (PCA / mutual information) → **[input-stage privacy: local DP on the encoder input, ε_input]** →
quantum encoding → model (QSVM / VQC / hybrid QNN, or classical baselines) →
**[training-stage privacy: DP-SGD (ε_training, Opacus) and FedAvg]** → evaluation (cross-validation plus a
locked hold-out) → report.

Privacy therefore operates at **two points**, both belonging to box 4:

| Stage | Mechanism | Budget recorded | Where |
|---|---|---|---|
| Input (before encoding) | Anonymisation (identifier removal) | — | `src/m2_preprocess/preprocessing.py` |
| Input (before encoding) | k-anonymity, k = 5: generalisation + local suppression of 7 demographic quasi-identifiers, rule fitted on training rows | k, suppression rate, k-anonymous flag per fold | `src/m7_privacy/input_privacy.py` (`KAnonymizer`) |
| Input (before encoding) | Local DP: Laplace noise on each training record's 8 encoder-input features | ε_input (pure ε-LDP, δ = 0), noise scale | `src/m7_privacy/input_privacy.py` (`LaplaceLDP`) |
| Training | DP-SGD | ε_training, δ = 10⁻⁵ (Opacus RDP accountant) | `src/m7_privacy/dp_training.py` |
| Training | Federated averaging | — (data stays on clients) | `src/m6_federated/fedavg.py` |
| Training | Secure aggregation | — | PHASE-2 (stub); scheduled Priority 3, item 9 |

Both budgets are stored as separate columns (`epsilon_input`, `epsilon_training`) in
`results/exp5-nsch2024-input-privacy-20261112-120542/fold_metrics.csv`. They are not combined into one number:
local DP covers the features of each training record (not its label); DP-SGD covers the trained model.

**Scope:** input-stage privacy is a pipeline stage used in Experiment 5. Experiments 1–4 (encoding sweep,
quantum vs classical, privacy–utility curve, federated) were run before it existed and use training-stage
privacy only.

---

## Box 1: Data sources

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Survey responses | BUILT | `src/m1_data/nsch.py`, `data/README.md`, `tests/test_phase1_pipeline.py` | NSCH 2024 topical file, adolescents 12–17, 21,789 records, 66 whitelisted questions. |
| Public datasets | BUILT | `scripts/download_nsch.py`, `data/README.md` | NSCH public-use file from the U.S. Census Bureau (checksum recorded). |
| Clinical notes (de-identified) | PHASE-2 | — | No such dataset is accessible to this project. |
| Counselling records | PHASE-2 | — | No such dataset is accessible to this project. |
| Social media text | PHASE-2 | — | No such dataset is accessible; the project uses survey responses only. |
| Wearable / IoT data | PHASE-2 | — | No such dataset is accessible to this project. |
| Ethical use / consent / de-identification | PARTIAL | `data/README.md`, `src/m2_preprocess/preprocessing.py` | Data is the Census Bureau's de-identified public-use file (consent handled by the survey); the project drops identifier columns. No consent process of our own. |

## Box 2: Data preprocessing and anonymisation

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Data cleaning | BUILT | `src/m1_data/nsch.py` (`FEATURES`, `assert_no_leakage`) | Fixed whitelist; diagnosis/treatment columns excluded; unknown labels dropped. |
| Noise removal | PHASE-2 | — | No explicit noise-removal step; survey answers are categorical codes. |
| Tokenisation / encoding | PARTIAL | `src/m2_preprocess/preprocessing.py` | Categorical one-hot encoding (95 columns). No tokenisation: there is no text. |
| Anonymisation (PII removal) | BUILT | `src/m2_preprocess/preprocessing.py` (`DEFAULT_IDENTIFIER_COLUMNS`), `tests/test_m2_preprocess.py` | Identifier and metadata columns are never used as features. |
| Anonymisation (k-anonymity) | BUILT | `src/m7_privacy/input_privacy.py` (`KAnonymizer`), `tests/test_input_privacy.py`, Exp 5 | k = 5 on 7 demographic quasi-identifiers; generalisation then local suppression; fitted on training rows; used in Experiment 5. |
| Normalisation | BUILT | `src/m3_features/features.py`, `src/m10_exp/exp2_compare.py` | Standardisation and [0, π] angle scaling, fitted on training rows only. |
| Train-test split | BUILT | `src/m10_exp/common.py` (`holdout_split`, `cv_folds`), `tests/test_phase1_pipeline.py` | Locked 20 % hold-out (scored once) plus repeated stratified CV. |

## Box 3: Feature representation

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Text features (BERT / RoBERTa, sentiment, contextual) | PHASE-2 | — | Requires text data the project does not have. |
| Structured features: behavioural, demographic | BUILT | `src/m1_data/nsch.py` | School, social, sleep, screen time, ACEs, family, demographics. |
| Structured features: physiological signals | PHASE-2 | — | No physiological data in NSCH. |
| Dimensionality reduction: PCA | BUILT | `src/m3_features/features.py`, `tests/test_phase1_pipeline.py` | Fitted inside each fold. |
| Dimensionality reduction: mutual information | BUILT | `src/m3_features/features.py` | Not named in the diagram; used in the encoding sweep. |
| Dimensionality reduction: UMAP, autoencoder | PHASE-2 | — | Scheduled: Priority 3, item 11. |

## Box 4: Privacy-preserving mechanisms

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Differential privacy (training, DP-SGD) | BUILT | `src/m7_privacy/dp_training.py`, `src/m7_privacy/accounting.py`, `tests/test_phase4_dp.py`, Exp 3 results | Opacus DP-SGD; ε from Opacus's RDP accountant only. |
| Federated simulation | BUILT | `src/m6_federated/fedavg.py`, `tests/test_phase5_federated.py`, Exp 4 results | FedAvg, 3/5/10 clients, IID and by-state, client-side DP. |
| Secure aggregation | PHASE-2 | `src/m7_privacy/secure_aggregation.py` (stub that reports "not implemented") | Scheduled: Priority 3, item 9. |
| Homomorphic encryption | PHASE-2 | — | Computationally impractical at this scale; a toy version would misrepresent its cost. |
| Privacy risk analysis (membership inference test) | BUILT | `src/m9_audit/membership.py`, `tests/test_membership.py`, `src/m10_exp/exp6_membership.py`, `results/exp6-nsch2024-membership-inference-20261112-120934/` | Loss-threshold attack, 5 folds. See box 12. (`src/m9_audit/audit.py` is an older unused helper, not part of this test.) |
| Input-stage privacy: k-anonymity | BUILT | `src/m7_privacy/input_privacy.py`, `tests/test_input_privacy.py`, `results/exp5-nsch2024-input-privacy-20261112-120542/` | k = 5 achieved on every fold's training rows; suppresses 13.0 % of rows on full data, 43.5 % on 1.2k-row subsamples; no measurable AUC cost (e.g. LogReg full 0.853 → 0.858). |
| Input-stage privacy: local differential privacy | BUILT | same | Laplace ε-LDP on encoder inputs, ε_input ∈ {32, 16, 8, 4}. Large cost: QSVM 0.848 → 0.821 / 0.709 / 0.570 / 0.500; LogReg full 0.853 → 0.764 at ε_input = 4. Labels are not perturbed. |

## Box 5: Quantum encoding

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Amplitude encoding | BUILT | `src/m4_encoding/encodings.py`, `tests/test_phase2_quantum.py`, Exp 1 | |
| Angle encoding | BUILT | same | Final QSVM uses it. |
| Basis encoding | BUILT | `src/m4_encoding/encodings.py` (`basis`), `tests/test_phase2_quantum.py::test_basis_encoding_prepares_the_computational_basis_state`, `results/exp1-nsch2024-encodings-with-basis-20261112/summary.md` | Each angle-scaled feature binarised at π/2, prepared as \|b⟩ with RX(π·b). Same 10 folds as Exp 1. Worst encoding in most settings: QSVM 0.658–0.779, VQC 0.517–0.723 (near chance with PCA). With basis included the encoding swing reaches 21–29 % (QSVM, PCA) and 43–57 % (VQC, PCA). |
| Quantum feature maps (ZZ) | BUILT | same | Plus data re-uploading (not in the diagram). |

## Box 6: Quantum model (simulation)

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Variational quantum circuit (VQC) | BUILT | `src/m5_models/quantum.py`, Exp 1–4 | StronglyEntanglingLayers, 1–2 layers. |
| Quantum neural network (hybrid QNN) | BUILT | `src/m5_models/quantum.py`, Exp 2 | Linear compression → quantum layer → linear head. |
| Parameterised gates, entanglement layers | BUILT | `src/m5_models/quantum.py` | Rot gates plus CNOT entanglers. |
| Simulator: PennyLane | BUILT | `requirements.txt`, all quantum code | `default.qubit`, exact statevector. |
| Simulator: Qiskit | PHASE-2 | — | Scheduled: Priority 4, item 12. |

## Box 7: Hybrid quantum-classical learning

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Classical optimiser: Adam | BUILT | `src/m5_models/quantum.py`, `src/m7_privacy/dp_training.py` | |
| Classical optimiser: COBYLA | BUILT | `src/m5_models/quantum.py` (`VQCClassifier(optimizer="cobyla")`), `tests/test_phase2_quantum.py::test_vqc_cobyla_optimiser_reduces_loss`, `results/exp2-nsch2024-vqc-cobyla-20261112-121718/comparison.md` | SciPy COBYLA, 300 loss evaluations, full batch. VQC re-upload PCA-4, 15 folds: 0.799 ± 0.041 vs Adam 0.814 ± 0.012 (Wilcoxon p = 0.11, not significant); less stable; 12.7 s vs 44.8 s per fit. |
| Loss function: classification | BUILT | `src/m5_models/quantum.py` | Class-weighted binary cross-entropy. |
| Loss function: regression | PHASE-2 | — | No continuous target in the task. |
| Update / evaluate loop | BUILT | `src/m5_models/quantum.py` (`_TorchQuantumBase.fit`) | Early stopping on inner validation. |

## Box 8: Simulation environment

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| PennyLane | BUILT | all quantum code | |
| Qiskit | PHASE-2 | — | Scheduled: Priority 4, item 12. |
| Cirq, TensorFlow Quantum | PHASE-2 | — | Redundant with PennyLane and Qiskit; would add no finding. |
| Noise simulation (realistic hardware) | PARTIAL | `src/m5_models/noisy_kernel.py`, `tests/test_noisy_kernel.py` (4 tests, incl. exact match to the full multi-qubit `default.mixed` circuit), `configs/exp8_noise_nsch2024.yaml`, `results/exp8-nsch2024-hardware-noise-20261112-125925/noise_report.md` | Best configuration (QSVM, angle, 8 qubits, PCA-8) re-run on the 15 Exp 2 folds on `default.mixed` with T1/T2, depolarising gate error, readout error and finite shots: ROC-AUC 0.847 ideal → 0.847 (noise x1, exact) / 0.850 (x1, 4,096 shots) / 0.846 (x1, 1,024 shots) / 0.845 (x10, 4,096 shots); none of the noisy settings differs significantly from the noise-free baseline (paired Wilcoxon, Holm-adjusted). Noise model is assumed (representative published values), not a device calibration. **Not covered:** entangling circuits (ZZ, VQC, hybrid) and two-qubit gate errors — this circuit has none; no real hardware. |
| Different numbers of qubits | BUILT | Exp 1 (4, 6, 8 qubits) | |
| Multiple quantum circuit designs | BUILT | Exp 1, Exp 2 | 4 encodings × 3 models. |
| Scalability experiments (number of clients) | BUILT | Exp 4 (3, 5, 10 clients) | |
| Heterogeneous (non-IID) data | BUILT | `src/m6_federated/fedavg.py` (`group_partition`), Exp 4 | Clients split by U.S. state. |
| Communication constraints | PARTIAL | Exp 4 `fold_metrics.csv`; `results/exp7-nsch2024-federated-cost-20261112-123743/communication_cost.md` | Payload, per-round bytes and server time measured; no bandwidth limit imposed and no network latency (in-process simulation). |

## Box 9: Mental health analysis tasks

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Mental health state classification | PARTIAL | `src/m1_data/nsch.py` (`TARGETS`) | Binary: currently diagnosed depression vs not. Multi-class scheduled: Priority 3, item 8. |
| Risk level prediction | PARTIAL | all experiments | Binary risk score with a threshold; no graded levels. |
| Topic and trend analysis | PHASE-2 | — | Requires text data. |
| Emotion and sentiment analysis | PHASE-2 | — | Requires text data. |
| Personalised insights | PHASE-2 | — | Requires text data; also out of scope for a research tool. |

## Box 10: Evaluation and comparison

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Logistic regression | BUILT | `src/m5_models/classical.py`, Exp 2 | |
| SVM (classical) | BUILT | `src/m5_models/classical.py` (`SVMClassifier`), `tests/test_phase1_pipeline.py`, `results/exp2-nsch2024-svm-baseline-20261112-121554/comparison.md` | RBF kernel, C tuned on inner-val, γ fixed at sklearn "scale"; same 15 folds as Exp 2. PCA-8 / 1.2k rows: 0.838 ± 0.016, below QSVM 0.847 (13/15 folds, Wilcoxon p = 0.026, Nadeau–Bengio p = 0.36, not significant) and below LogReg 0.851 (0/15 folds). Full data, all 95 features: 0.880 vs LogReg 0.888. |
| Random forest | PARTIAL | `results/phase1-nsch2024-depression-*` | Run in Phase 1, then excluded from all results on supervisor instruction. |
| BERT + ML | PHASE-2 | — | Requires text data. |
| Deep learning (LSTM, CNN) | PHASE-2 | — | Require sequence data; the data is tabular. An MLP baseline is built instead. |
| XGBoost, MLP | BUILT | `src/m5_models/classical.py`, Exp 2 | Not named in the diagram. |
| Quantum: VQC, QNN, hybrid | BUILT | Exp 1–2 | Plus QSVM (not in the diagram). |

## Box 11: Performance metrics

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Accuracy, precision, recall, F1 | BUILT | `src/m8_eval/evaluation.py` (`binary_metrics`), `tests/test_phase1_pipeline.py` | Headline metrics are balanced accuracy, recall and ROC-AUC. |
| Training time | BUILT | `fit_seconds` / `tuning_seconds` in every `fold_metrics.csv` | |
| Resource usage (qubits, gates) | BUILT | `src/m4_encoding/encodings.py` (`circuit_resources`), Exp 1 `circuit_resources.csv` | Depth and gate counts after decomposition. |

## Box 12: Privacy evaluation

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Membership inference risk | BUILT | `src/m9_audit/membership.py`, `tests/test_membership.py`, `results/exp6-nsch2024-membership-inference-20261112-120934/summary.md` | Attack AUC 0.496–0.503 for DP LogReg and DP VQC at every ε ∈ {0.5…8, ∞} and 0.502 for the QSVM: no measurable leakage even without DP. Positive control (overfit MLP) 0.558, so the attack works. Loss-threshold attack only; stronger attacks (e.g. LiRA) not run. |
| Re-identification risk | BUILT | `src/m9_audit/reidentification.py`, `tests/test_reidentification.py`, `scripts/reidentification_report.py`, `results/reidentification/risk.md` | 17,431 dev rows: 36.8 % unique on 7 raw demographic QIs (62.7 % with state); after k = 5 anonymisation 0.0 % unique, max risk 0.20 — but 22.9 % unique again if the attacker also knows the U.S. state. |
| Effect of privacy parameters (ε, noise) | BUILT | Exp 3 (`results/exp3-*`), `results/RESULTS.md` Tables 5–7; Exp 5 | Training ε ∈ {0.5, 1, 2, 4, 8, ∞}; input ε ∈ {4, 8, 16, 32, ∞}. |
| Data utility trade-off | BUILT | same | Privacy–utility curve. |

## Box 13: Scalability analysis

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Varying number of clients | BUILT | Exp 4 | |
| Non-IID data distribution | BUILT | Exp 4 | |
| Communication latency | PARTIAL | `src/m6_federated/fedavg.py`, `tests/test_phase5_federated.py::test_fedavg_records_measured_payload_and_timings`, `scripts/federated_cost_report.py`, `results/exp7-nsch2024-federated-cost-20261112-123743/communication_cost.md` | Measured: serialised update size (LogReg 2,136 B, VQC 1,812 B per update), bytes per round (10.9–42.7 kB), slowest-client time per round (0.05–0.42 s), rounds to converge (LogReg 2–11 median; VQC 24–29 of 30 — not converged). **Network transfer time is not measured**: training is in-process with no network, and bytes ÷ assumed bandwidth would be a simulation. |
| Server aggregation load | BUILT | same | Wall-clock deserialise + weighted average per round: 1.1–3.1 ms (LogReg, 3–10 clients), 1.4–5.9 ms (VQC); grows roughly linearly with clients. 5 folds, by-state clients, ε = 4; re-run reproduced Exp 4 AUC exactly. |

## Box 14: Output and visualisation

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Performance comparison graphs | BUILT | `results/figures/*.png`, `src/m10_exp/final_report.py` | |
| Feature visualisation | PARTIAL | Phase 1 `mi_selected_features.csv` | MI-selected features listed; no feature-importance plot. |
| Attention visualisation | PHASE-2 | — | No attention model (no text data). |
| Quantum state visualisation: statevector | BUILT | `api.py` (`/api/inspect`), PrivacyQ Inspect page | Probabilities of all 2ⁿ basis states for a real record. |
| Quantum state visualisation: Bloch sphere | BUILT | `src/m8_eval/state_viz.py`, `tests/test_state_viz.py` (3 tests), `scripts/state_visualisation.py`, `results/state_visualisation/bloch_angle_vs_zz.png`, `bloch_summary.md` | Exact single-qubit Bloch vectors of 1,200 real training records (Exp 2 fold 1, PCA-8), coloured by label, angle vs ZZ. Angle: all states pure (|r| = 1.000). ZZ: entangled, mean |r| 0.47–0.71 per qubit. Class-mean distance up to 0.51 (angle, qubit 1). Descriptive only. |
| Research report and insights | BUILT | `results/RESULTS.md`, `README.md`, `results/VIVA.md` | |

## Box 15: Key outcomes

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Feasibility of quantum AI for mental-health analysis | BUILT | Exp 1–2 | Feasible on a simulator; no accuracy advantage over classical. |
| Privacy-preserving insights | BUILT | Exp 3–4 | |
| Comparative performance results | BUILT | `results/RESULTS.md` Tables 1–4, 11 | |
| Guidelines for real-world deployment | PARTIAL | `results/RESULTS.md` (Limitations) | Limitations stated; no deployment guideline document. |
| Research report | BUILT | `results/RESULTS.md` | |

---

## Change log

- 2026-11-12: baseline written before the conformance sprint.
- 2026-11-12: Priority 1 — input-stage privacy (k-anonymity, local DP) built and measured (Experiment 5); flow section rewritten.
- 2026-11-12: Priority 2 item 2 — re-identification risk built and measured.
- 2026-11-12: Priority 2 item 1 — membership inference test built and measured (Experiment 6).
- 2026-11-12: Priority 2 items 4 and 5 — classical SVM baseline and COBYLA optimiser built and measured on the Exp 2 folds.
- 2026-11-12: Priority 2 item 3 — basis encoding built and swept on the Exp 1 folds; merged encoding report written.
- 2026-11-12: Priority 2 item 6 — measured payload, client time, server aggregation time and rounds to converge (Experiment 7).
- 2026-11-12: Priority 3 item 7 — best configuration re-run under a simulated hardware noise model (Experiment 8).
- 2026-11-12: Priority 3 item 10 — Bloch-sphere state visualisation of real encoded records (angle vs ZZ).
