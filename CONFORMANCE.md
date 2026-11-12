# Conformance with the architecture diagram

**Project:** Privacy-Preserving Adolescent Mental Health Using Quantum AI (DP-FedQ)
**Status as of:** 12 November 2026

This document states, for every box of the architecture diagram ("Quantum AI for Mental Health", boxes 1–15),
what the repository implements. A sub-item is **BUILT** only if code exists, has been run on the real data, and
has a test or a results file behind it.

| Status | Meaning |
|---|---|
| **BUILT** | Implemented and exercised; evidence listed. |
| **PARTIAL** | Implemented in a narrower form than the diagram names; the note says exactly how. |
| **PHASE-2** | Not implemented; the note gives the reason. |

Paths are relative to the repository root. `results/` is produced by the commands in `README.md`; the run
directories are listed in `configs/final_report.yaml`. The quantum circuits are a representation layer only:
privacy comes from DP-SGD, federated learning, secure aggregation and the input-stage mechanisms, never from the
quantum computation.

**Significance policy (all experiments).** Comparisons are paired over the same cross-validation folds. Primary
test: Nadeau–Bengio corrected resampled t-test, Holm-adjusted within each pre-declared family; "significant" means
Holm-adjusted p < 0.05. The Wilcoxon signed-rank test, Holm-adjusted in the same families, is reported as a
sensitivity check. Rationale: top of `results/RESULTS.md`. Every comparison with both p-values:
`results/significance/audit.csv` and `results/significance/changed.md`.

## Pipeline flow as built

NSCH 2024 survey → cleaning, identifier removal, train/test split →
**[input-stage privacy: k-anonymity (k = 5) on quasi-identifiers]** → imputation and one-hot encoding →
feature reduction (PCA, mutual information, UMAP or autoencoder) → **[input-stage privacy: local DP on the encoder
input, ε_input]** → quantum encoding → model (QSVM, VQC, hybrid QNN, or classical baselines) →
**[training-stage privacy: DP-SGD (ε_training, Opacus), FedAvg, optional secure aggregation]** → evaluation
(cross-validation and a locked hold-out) → report.

Privacy operates at **two stages**, both belonging to box 4:

| Stage | Mechanism | Budget recorded | Where |
|---|---|---|---|
| Input (before encoding) | Anonymisation: identifier columns never used as features | — | `src/m2_preprocess/preprocessing.py` |
| Input (before encoding) | k-anonymity, k = 5: generalisation and local suppression of 7 demographic quasi-identifiers; rule fitted on training rows | k, suppression rate, k-anonymous flag per fold | `src/m7_privacy/input_privacy.py` (`KAnonymizer`) |
| Input (before encoding) | Local differential privacy: Laplace noise on each training record's encoder-input features | ε_input (pure ε-LDP, δ = 0) and noise scale | `src/m7_privacy/input_privacy.py` (`LaplaceLDP`) |
| Training | DP-SGD | ε_training, δ = 10⁻⁵ (Opacus RDP accountant) | `src/m7_privacy/dp_training.py` |
| Training | Federated averaging | — (data stays on clients) | `src/m6_federated/fedavg.py` |
| Training | Secure aggregation: pairwise Diffie–Hellman masks; the server receives only masked vectors and recovers only the weighted sum | — (hides individual updates; not a DP budget) | `src/m7_privacy/secure_aggregation.py`, `src/m6_federated/fedavg.py` |

The two budgets are recorded separately (`epsilon_input`, `epsilon_training` in
`results/exp5-nsch2024-input-privacy-20261112-120542/fold_metrics.csv`) and are not combined into one number:
local DP covers the features of each training record (not its label); DP-SGD covers the trained model.

**Where each stage was used.** Input-stage privacy (k-anonymity and local DP) exists in **Experiment 5 only**.
Experiments 1–4 (encoding sweep, quantum vs classical, privacy–utility curve, federated learning) use
training-stage privacy only, and secure aggregation was used only in Experiments 7b and 7c.

---

## Box 1: Data sources

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Survey responses | BUILT | `src/m1_data/nsch.py`, `data/README.md`, `tests/test_phase1_pipeline.py` | NSCH 2024 topical file: 21,789 adolescents aged 12–17 with a determinable depression label; 66 whitelisted questions. |
| Public datasets | BUILT | `scripts/download_nsch.py`, `data/README.md` | NSCH public-use file from the U.S. Census Bureau; checksum recorded. |
| Clinical notes (de-identified) | PHASE-2 | — | Requires free-text clinical records; no accessible dataset provides them. |
| Counselling records | PHASE-2 | — | Requires free-text counselling records; no accessible dataset provides them. |
| Social media text | PHASE-2 | — | Requires social-media text linked to mental-health outcomes; no accessible dataset provides it. |
| Wearable / IoT data | PHASE-2 | — | Requires physiological sensor data; no accessible dataset provides it. |
| Ethical use / consent / de-identification | PARTIAL | `data/README.md`, `src/m2_preprocess/preprocessing.py` | The data is the Census Bureau's de-identified public-use file (consent handled by the survey); the project drops identifier columns. There is no consent process of the project's own. |

## Box 2: Data preprocessing and anonymisation

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Data cleaning | BUILT | `src/m1_data/nsch.py` (`FEATURES`, `assert_no_leakage`) | Fixed whitelist; diagnosis and treatment columns excluded; rows with an unknown label dropped. |
| Noise removal | PHASE-2 | — | Not applicable: survey answers are categorical codes, not noisy signals. |
| Tokenisation / encoding | PARTIAL | `src/m2_preprocess/preprocessing.py` | Categorical one-hot encoding (95 columns). Tokenisation requires text data, which no accessible dataset provides. |
| Anonymisation (PII removal) | BUILT | `src/m2_preprocess/preprocessing.py` (`DEFAULT_IDENTIFIER_COLUMNS`), `tests/test_m2_preprocess.py` | Identifier and metadata columns are never used as features. |
| Anonymisation (k-anonymity) | BUILT | `src/m7_privacy/input_privacy.py` (`KAnonymizer`), `tests/test_input_privacy.py`, Experiment 5 | k = 5 on 7 demographic quasi-identifiers; generalisation then local suppression; fitted on training rows. Used in Experiment 5. |
| Normalisation | BUILT | `src/m3_features/features.py`, `src/m10_exp/exp2_compare.py` | Standardisation and [0, π] angle scaling, fitted on training rows only. |
| Train-test split | BUILT | `src/m10_exp/common.py` (`holdout_split`, `cv_folds`), `tests/test_phase1_pipeline.py` | Locked 20 % hold-out (4,358 rows) plus repeated stratified cross-validation. |

## Box 3: Feature representation

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Text features (BERT / RoBERTa, sentiment, contextual) | PHASE-2 | — | Requires text data; no accessible dataset provides it. |
| Structured features: behavioural, demographic | BUILT | `src/m1_data/nsch.py` | School, social, sleep, screen time, adverse childhood experiences, family, demographics. |
| Structured features: physiological signals | PHASE-2 | — | Requires physiological sensor data; no accessible dataset provides it (NSCH has none). |
| Dimensionality reduction: PCA | BUILT | `src/m3_features/features.py`, `tests/test_phase1_pipeline.py` | Fitted inside each fold on training rows. |
| Dimensionality reduction: mutual information | BUILT | `src/m3_features/features.py` | Not named in the diagram; used in the encoding sweep. |
| Dimensionality reduction: UMAP, autoencoder | BUILT | `src/m3_features/nonlinear.py`, `src/m3_features/features.py`, `tests/test_nonlinear_features.py` (5 tests), `configs/exp10_umap_autoencoder_nsch2024.yaml`, `results/exp10-nsch2024-umap-autoencoder-20261112-143551/reduction_report.md` | Both fitted inside each fold on that fold's inner-train rows only (`feature_matrices` in `src/m10_exp/exp2_compare.py`; tested by `tests/test_nonlinear_features.py::test_feature_matrices_fit_reducer_on_inner_train_rows_only`). Experiment 10, same 15 folds as Experiment 2. ROC-AUC on the 1,500-row subsample, PCA / autoencoder / UMAP: QSVM 0.847 / 0.837 / 0.774; LogReg 0.851 / 0.836 / 0.768; RBF-SVM 0.838 / 0.833 / 0.777. UMAP is below PCA for every model (significant, Holm p ≤ 0.007). The autoencoder is not significantly different from PCA for any model. QSVM vs the classical models: not significant on either reduction. |

## Box 4: Privacy-preserving mechanisms

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Differential privacy (training, DP-SGD) | BUILT | `src/m7_privacy/dp_training.py`, `src/m7_privacy/accounting.py`, `tests/test_phase4_dp.py`, Experiment 3 | Opacus DP-SGD; ε read only from Opacus's RDP accountant. |
| Federated simulation | BUILT | `src/m6_federated/fedavg.py`, `tests/test_phase5_federated.py`, Experiment 4 | FedAvg with 3, 5 and 10 clients, IID and split by U.S. state, client-side DP-SGD. |
| Secure aggregation | BUILT | `src/m7_privacy/secure_aggregation.py`, `src/m6_federated/fedavg.py` (`secure_aggregation=True`), `tests/test_secure_aggregation.py` (13 tests), `configs/exp7b_secure_aggregation_nsch2024.yaml`, `results/exp7b-nsch2024-federated-secure-aggregation-20261112-153004/`, `scripts/secagg_overhead.py`, `results/secagg_overhead/overhead.md` | Pairwise additive masking (the masking core of Bonawitz et al., 2017): 2048-bit MODP Diffie–Hellman (RFC 3526 group 14, checked against the RFC's closed form) and SHA-256 / SHAKE-256 per-round masks. Updates are encoded as fixed-point integers (24 fractional bits) and masked and summed modulo 2⁶⁴, so the masks cancel exactly (tested bit for bit); client sizes are masked too. Each masked vector is computationally indistinguishable from random to the server (checked statistically by tests); a server colluding with 3 of 5 clients recovers only the 2 honest clients' sum. Overflow: negative values round-trip exactly in two's complement; each client refuses entries above a declared public bound, and setup refuses any configuration whose worst-case sum comes within 8 bits of 2⁶³. Worst case measured in Experiment 7c (10 clients, 3,786-row client, largest weight 8.94): 20.6 bits of headroom. Experiment 7b (Experiment 7 with secure aggregation): every stored metric identical to Experiment 7 in 30 of 30 runs. Cost: one-off key setup 0.5 / 1.5 / 6.1 s for 3 / 5 / 10 clients; masking 0.10–0.16 ms per client per round. Not implemented: client-dropout recovery, public-key authentication (honest-but-curious server only), defence against malicious clients. Secure aggregation is not differential privacy; the formal guarantee remains DP-SGD. |
| Homomorphic encryption | PHASE-2 | — | Computationally impractical at this scale; a toy version would misrepresent its cost. |
| Privacy risk analysis (membership inference test) | BUILT | `src/m9_audit/membership.py`, `tests/test_membership.py`, `src/m10_exp/exp6_membership.py`, `results/exp6-nsch2024-membership-inference-20261112-120934/` | Loss-threshold attack, 5 folds; results in box 12. |
| Input-stage privacy: k-anonymity | BUILT | `src/m7_privacy/input_privacy.py`, `tests/test_input_privacy.py`, `results/exp5-nsch2024-input-privacy-20261112-120542/` | Experiment 5 only. k = 5 achieved on every fold's training rows; suppresses 13.0 % of rows on full data and 43.5 % on 1,200-row subsamples. ROC-AUC unchanged within fold-to-fold variation (e.g. LogReg full data 0.853 → 0.858); this difference was not significance-tested. |
| Input-stage privacy: local differential privacy | BUILT | same | Experiment 5 only. Laplace ε-LDP on the encoder inputs, ε_input ∈ {32, 16, 8, 4}; labels are not perturbed. Large utility cost: QSVM 0.848 → 0.821 / 0.709 / 0.570 / 0.500; LogReg full data 0.853 → 0.764 at ε_input = 4. |

## Box 5: Quantum encoding

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Amplitude encoding | BUILT | `src/m4_encoding/encodings.py`, `tests/test_phase2_quantum.py`, Experiment 1 | |
| Angle encoding | BUILT | same | Used by the best quantum model (QSVM). |
| Basis encoding | BUILT | `src/m4_encoding/encodings.py`, `tests/test_phase2_quantum.py::test_basis_encoding_prepares_the_computational_basis_state`, `results/exp1-nsch2024-encodings-with-basis-20261112/summary.md` | Each angle-scaled feature binarised at π/2 and prepared as \|b⟩. Same 10 folds as Experiment 1. Worst encoding in most settings: QSVM 0.658–0.779, VQC 0.517–0.723. |
| Quantum feature maps (ZZ) | BUILT | same | Data re-uploading is also implemented (not in the diagram). |

## Box 6: Quantum model (simulation)

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Variational quantum circuit (VQC) | BUILT | `src/m5_models/quantum.py`, Experiments 1–4 | Strongly entangling layers, 1–2 layers. |
| Quantum neural network (hybrid QNN) | BUILT | `src/m5_models/quantum.py`, Experiment 2 | Linear compression → quantum layer → linear head. |
| Parameterised gates, entanglement layers | BUILT | `src/m5_models/quantum.py` | Rotation gates with CNOT entanglers. |
| Simulator: PennyLane | BUILT | `requirements.txt`, all quantum code | `default.qubit` (exact statevector); `default.mixed` for the noise simulation. |
| Simulator: Qiskit | BUILT (cross-check) | `src/m5_models/qiskit_vqc.py`, `results/qiskit_crosscheck/crosscheck.md` | Qiskit evaluates the trained VQC; identical to PennyLane on all 15 Experiment 2 folds (box 8). Training and the other models use PennyLane. |

## Box 7: Hybrid quantum-classical learning

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Classical optimiser: Adam | BUILT | `src/m5_models/quantum.py`, `src/m7_privacy/dp_training.py` | |
| Classical optimiser: COBYLA | BUILT | `src/m5_models/quantum.py` (`VQCClassifier(optimizer="cobyla")`), `tests/test_phase2_quantum.py::test_vqc_cobyla_optimiser_reduces_loss`, `results/exp2-nsch2024-vqc-cobyla-20261112-121718/comparison.md` | SciPy COBYLA, 300 loss evaluations, full batch. VQC (re-uploading, PCA-4), 15 folds: 0.799 ± 0.041 vs Adam 0.814 ± 0.012 (not significant, Holm p 0.55); 12.7 s vs 44.8 s per fit. |
| Loss function: classification | BUILT | `src/m5_models/quantum.py` | Class-weighted binary cross-entropy. |
| Loss function: regression | PHASE-2 | — | Not applicable: the task has no continuous target. |
| Update / evaluate loop | BUILT | `src/m5_models/quantum.py` (`_TorchQuantumBase.fit`) | Early stopping on an inner validation split. |

## Box 8: Simulation environment

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| PennyLane | BUILT | all quantum code | |
| Qiskit | BUILT (cross-check) | `src/m5_models/qiskit_vqc.py`, `tests/test_qiskit_vqc.py` (7 tests), `scripts/qiskit_crosscheck.py`, `results/qiskit_crosscheck/crosscheck.md` | Independent Qiskit 1.4.4 implementation of the VQC forward pass, written gate for gate without PennyLane; equal to PennyLane to 10⁻¹² in tests. On all 15 Experiment 2 folds: largest score difference 9.8 × 10⁻¹⁵ over 52,293 validation rows; ROC-AUC, balanced accuracy and predictions identical. Training stays in PennyLane; the QSVM, hybrid QNN and noisy circuits are not reimplemented in Qiskit. |
| Cirq, TensorFlow Quantum | PHASE-2 | — | Redundant with PennyLane and Qiskit; would add no finding. |
| Noise simulation (realistic hardware): QSVM | BUILT | `src/m5_models/noisy_kernel.py`, `tests/test_noisy_kernel.py` (7 tests), `configs/exp8_noise_nsch2024.yaml`, `configs/exp8b_noise_zz_nsch2024.yaml`, `results/exp8-nsch2024-hardware-noise-20261112-125925/noise_report.md`, `results/exp8b-nsch2024-hardware-noise-zz-vs-angle-4q-20261112-144158/noise_report.md` | Kernel estimated as on hardware (compute-uncompute, finite shots) on `default.mixed`, with T1/T2 decay, one- and two-qubit gate error and readout error; tested against the end-to-end mixed-state circuit. **Angle kernel:** none of the noisy settings differs significantly from the noise-free baseline, at 8 qubits (0.847 → 0.845–0.850, Experiment 8) or at 4 qubits (0.835 → 0.832–0.836, Experiment 8b). **ZZ kernel (4 qubits):** lower at realistic noise with 1,024 shots (0.793 → 0.786) and at 10× noise (→ 0.769), but not significant under the primary test (Holm p 1.0 and 0.19); unchanged at 4,096 shots. ZZ is significantly below angle at every setting. The Experiment 8b results are at 4 qubits (reduced from 8 because the full density-matrix simulation of ZZ was too slow) and are not directly comparable with the 8-qubit results. The noise model uses representative published values, not a device calibration; no real hardware was used. |
| Noise simulation: VQC | PHASE-2 | — | Training a variational circuit on a mixed-state simulator needs a density-matrix forward and backward pass for every optimiser step, far more costly than the kernel; Experiment 8b already shows how two-qubit error affects entangling circuits. |
| Different numbers of qubits | BUILT | Experiment 1 | 4, 6 and 8 qubits. |
| Multiple quantum circuit designs | BUILT | Experiments 1 and 2 | 5 encodings × 3 models. |
| Scalability experiments (number of clients) | BUILT | Experiment 4 | 3, 5 and 10 clients. |
| Heterogeneous (non-IID) data | BUILT | `src/m6_federated/fedavg.py` (`group_partition`), Experiment 4 | Clients split by U.S. state. |
| Communication constraints | PARTIAL | `results/exp7-nsch2024-federated-cost-20261112-123743/communication_cost.md` | Payload, bytes per round and server time measured; no bandwidth limit imposed and no network latency (in-process simulation). |

## Box 9: Mental health analysis tasks

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Mental health state classification: binary | BUILT | `src/m1_data/nsch.py` (`TARGETS`), Experiments 1–8 | Currently diagnosed depression (K2Q32B) vs not; 21,789 adolescents. |
| Mental health state classification: three classes | BUILT | `src/m1_data/nsch.py` (`three_class_labels`), `src/m5_models/multiclass.py`, `src/m10_exp/exp9_multiclass.py`, `src/m10_exp/holdout_multiclass.py`, `tests/test_multiclass.py` (11 tests), `results/exp9-nsch2024-multiclass-20261112-140242/multiclass_report.md`, `results/exp9b-nsch2024-multiclass-vqc-20261112-140937/multiclass_report.md`, `results/holdout-multiclass-20261112-171606/` | None / anxiety only / depression (with or without anxiety), from the "currently has" items K2Q32B and K2Q33B; merging the first two classes gives the binary target exactly. No stress variable exists in NSCH 2024. Classes 78.0 / 12.0 / 10.0 %. Same population, hold-out and fold assignments as Experiment 2; 165 rows with undetermined anxiety are excluded (141 from training and validation; 24 from three-class hold-out scoring under a rule committed before scoring). Class-weighted models; headline metrics macro-F1 and macro one-vs-rest ROC-AUC. Cross-validation (macro-F1 / macro AUC): XGBoost full data 0.548 / 0.822; LogReg full data 0.542 / 0.820; QSVM 0.497 / 0.762 and LogReg 0.503 / 0.774 on the same PCA-8 subsample; VQC 0.450 / 0.744 and LogReg 0.483 / 0.753 on the same PCA-4 subsample. No quantum-vs-classical difference is significant. Locked hold-out (scored once; macro AUC [95 % CI]): XGBoost 0.816 [0.804, 0.828], LogReg 0.819 [0.807, 0.832], QSVM 0.760 [0.746, 0.774], VQC 0.740 [0.725, 0.755]; every cross-validation mean lies inside its hold-out interval. |
| Risk level prediction | PARTIAL | all experiments | A binary risk score with a decision threshold; no graded risk levels. |
| Topic and trend analysis | PHASE-2 | — | Requires text data; no accessible dataset provides it. |
| Emotion and sentiment analysis | PHASE-2 | — | Requires text data; no accessible dataset provides it. |
| Personalised insights | PHASE-2 | — | Requires text data linked to individuals; no accessible dataset provides it. |

## Box 10: Evaluation and comparison

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Logistic regression | BUILT | `src/m5_models/classical.py`, Experiment 2 | |
| SVM (classical) | BUILT | `src/m5_models/classical.py` (`SVMClassifier`), `tests/test_phase1_pipeline.py`, `results/exp2-nsch2024-svm-baseline-20261112-121554/comparison.md` | RBF kernel, same 15 folds as Experiment 2. PCA-8 subsample: 0.838 vs QSVM 0.847 and LogReg 0.851 (neither difference significant). Full data, all 95 features: 0.880 vs LogReg 0.888 (significant, Holm p 0.017). |
| Random forest | PARTIAL | `results/phase1-nsch2024-depression-20261104-183743/` | Run in Phase 1, then excluded from all results on supervisor instruction. |
| BERT + ML | PHASE-2 | — | Requires text data; no accessible dataset provides it. |
| Deep learning (LSTM, CNN) | PHASE-2 | — | Require sequence data (text or physiological time series); no accessible dataset provides it, and the survey data is tabular. An MLP baseline is built instead. |
| XGBoost, MLP | BUILT | `src/m5_models/classical.py`, Experiment 2 | Not named in the diagram. |
| Quantum: VQC, QNN, hybrid | BUILT | Experiments 1 and 2 | Plus a quantum-kernel SVM (QSVM), not named in the diagram. |

## Box 11: Performance metrics

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Accuracy, precision, recall, F1 | BUILT | `src/m8_eval/evaluation.py` (`binary_metrics`, `multiclass_metrics`), `tests/test_phase1_pipeline.py`, `tests/test_multiclass.py` | Headline metrics: balanced accuracy, recall and ROC-AUC (binary); macro-F1 and macro one-vs-rest ROC-AUC (three classes). Plain accuracy is never a headline. |
| Training time | BUILT | `fit_seconds` and `tuning_seconds` in every `fold_metrics.csv` | |
| Statistical testing | BUILT | `src/m8_eval/stats.py`, `tests/test_stats.py`, `scripts/significance_audit.py`, `results/significance/` | Nadeau–Bengio corrected t (primary) and Wilcoxon (sensitivity), Holm within each family. Of 542 comparisons on 10–15 folds, 154 are significant under the primary test. The comparisons behind the main conclusion hold under both tests: QSVM ties LogReg (Holm p 0.70); VQC is below LogReg (0.012); the hybrid QNN is below LogReg in ROC-AUC (0.041) and balanced accuracy (0.010). |
| Resource usage (qubits, gates) | BUILT | `src/m4_encoding/encodings.py` (`circuit_resources`), Experiment 1 | Circuit depth and gate counts after decomposition to native gates. |

## Box 12: Privacy evaluation

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Membership inference risk | BUILT | `src/m9_audit/membership.py`, `tests/test_membership.py`, `results/exp6-nsch2024-membership-inference-20261112-120934/summary.md` | Attack AUC 0.496–0.503 for DP LogReg and DP VQC at every ε from 0.5 to 8 and without DP, and 0.502 for the QSVM: no measurable leakage. A deliberately overfitted MLP (positive control) gives 0.558, so the attack can detect leakage. Loss-threshold attack only; stronger attacks (e.g. LiRA) not run. |
| Re-identification risk | BUILT | `src/m9_audit/reidentification.py`, `tests/test_reidentification.py`, `scripts/reidentification_report.py`, `results/reidentification/risk.md` | 17,431 development rows: 36.8 % unique on 7 demographic quasi-identifiers; after k = 5 anonymisation 0.0 % unique (maximum re-identification risk 0.20), but 22.9 % unique again if the attacker also knows the U.S. state. |
| Gradient inversion attack | PHASE-2 | `src/m9_audit/audit.py` (returns "not implemented") | Not implemented within the project's time; membership inference is the attack evaluated. |
| Effect of privacy parameters (ε, noise) | BUILT | Experiment 3, `results/RESULTS.md` Tables 5–7; Experiment 5 | Training ε ∈ {0.5, 1, 2, 4, 8, ∞}; input ε ∈ {4, 8, 16, 32, ∞}. On the hold-out, the gap from the best non-private model to DP LogReg at ε = 4 is 0.031 ROC-AUC; in cross-validation about half of that gap is DP noise (0.016, not significant) and half the change of training method. |
| Data utility trade-off | BUILT | same | Privacy–utility curve. |

## Box 13: Scalability analysis

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Varying number of clients | BUILT | Experiment 4 | 3, 5 and 10 clients. |
| Non-IID data distribution | BUILT | Experiment 4 | Clients split by U.S. state. |
| Communication latency | PARTIAL | `src/m6_federated/fedavg.py`, `tests/test_phase5_federated.py::test_fedavg_records_measured_payload_and_timings`, `scripts/federated_cost_report.py`, `results/exp7-nsch2024-federated-cost-20261112-123743/communication_cost.md` | Measured: update size (LogReg 2,136 B, VQC 1,812 B), bytes per round (10.9–42.7 kB), slowest-client time per round (0.05–0.42 s), rounds to converge (LogReg 2–11; the VQC had not converged within 30 rounds). Network transfer time is not measured: training is in-process, with no network. |
| Server aggregation load | BUILT | same | 1.1–3.1 ms per round (LogReg, 3–10 clients) and 1.4–5.9 ms (VQC), growing roughly linearly with the number of clients. |

## Box 14: Output and visualisation

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Performance comparison graphs | BUILT | `results/figures/`, `src/m10_exp/final_report.py` | |
| Feature visualisation | PARTIAL | Phase 1 `mi_selected_features.csv` | Selected features are listed; there is no feature-importance plot. |
| Attention visualisation | PHASE-2 | — | Requires an attention model over text data; no accessible dataset provides such data. |
| Quantum state visualisation: statevector | BUILT | `api.py` (`POST /api/inspect`), PrivacyQ Inspect page | Probabilities of all 2ⁿ basis states for a real record. |
| Quantum state visualisation: Bloch sphere | BUILT | `src/m8_eval/state_viz.py`, `tests/test_state_viz.py` (3 tests), `scripts/state_visualisation.py`, `results/state_visualisation/bloch_angle_vs_zz.png` | Exact single-qubit Bloch vectors of 1,200 real training records, coloured by label, angle vs ZZ encoding. Angle: all states pure. ZZ: entangled (mean Bloch-vector length 0.47–0.71 per qubit). Descriptive only. |
| Research report and insights | BUILT | `results/RESULTS.md`, `README.md` | |

## Box 15: Key outcomes

| Sub-item | Status | Evidence | Note |
|---|---|---|---|
| Feasibility of quantum AI for mental-health analysis | BUILT | Experiments 1, 2 and 9 | Feasible on a simulator; no accuracy advantage over classical models. |
| Privacy-preserving insights | BUILT | Experiments 3–7 | |
| Comparative performance results | BUILT | `results/RESULTS.md` Tables 1–4, 11 and 12 | |
| Guidelines for real-world deployment | PARTIAL | `results/RESULTS.md` (Limitations) | Limitations are stated; there is no separate deployment-guideline document. |
| Research report | BUILT | `results/RESULTS.md` | |
