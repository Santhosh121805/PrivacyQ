# Results report — Mental Health Text Classification dataset (second data source), beside NSCH 2024

One section per architecture box. Every number is read from a results/ file at build time; the traceability appendix lists each one with its file. NSCH numbers are read from the NSCH results/ files and were not recomputed. Inputs are the frozen MiniLM sentence encoder (BERT family: sentence-transformers/all-MiniLM-L6-v2), never fine-tuned. Quantum circuits are simulated. Federated clients are simulated. The locked hold-out of this dataset has not been scored.

## Box 1 — Data Sources (Adolescent Mental Health)

**What this step does:** Adds a second, unrelated data source: public social-media posts labelled Normal / Anxiety / Depression / Suicidal.

| Item | Value |
|---|---|
| Dataset | Mukherjee (2025), Mental Health Text Classification Dataset (4-class), Hugging Face, CC BY 4.0 (data/README.md) |
| Raw rows | 49,612 |
| After deduplication (exact + near ≥ 0.9) | 48,444 |
| Processed rows (subsample + quarantined block) | 24,572 |
| Who wrote it | Adults and adolescents, unknown ages; NOT an adolescent sample (see Box 9) |

*Table 1.2. The six sources in the diagram — coverage now*

| Source in the diagram | NSCH | Mental Health Text |
|---|---|---|
| Clinical notes (de-identified) | No | No: not available, no clinical-notes dataset is part of this project |
| Counselling records | No | No: not available, no counselling-records dataset is part of this project |
| Survey responses | Yes | No |
| Social media text | No | **Yes: BUILT** |
| Wearable/IoT data (optional) | No | No: not available, no sensor dataset is part of this project |
| Public datasets | Yes | Yes |

Source: results/mhtext/phase1/manifest.json · data/README.md

## Box 2 — Data Preprocessing & Anonymization

**What this step does:** Deduplicates before any split, drops conflicting labels, quarantines the block without IDs, locks a hold-out, blocks keyword-leakage columns.

| Step | Rows |
|---|---|
| Exact duplicates dropped | 664 |
| Exact conflicting-label rows dropped | 33 |
| Near duplicates dropped (TF-IDF cosine ≥ 0.90) | 381 |
| Near conflicting-label rows dropped | 90 |
| Confirmation pass on subsample + quarantine, rows dropped | 13 |
| Locked hold-out fraction | 20% (not scored) |
| Blocked columns (label leakage) | has_suicidal_keyword, has_stress_keyword, has_help_keyword |

Anonymisation: there are no quasi-identifier columns to generalise, so k-anonymity does not apply. Direct identifiers can be scrubbed from the text (URLs, e-mails, handles, communities, numbers); Box 12 shows this does not reduce re-identification.

Source: results/mhtext/phase1/manifest.json

## Box 3 — Feature Representation

**What this step does:** Turns each post into numbers with frozen models only (nothing fine-tuned).

*Table 3.1. Text features — coverage now*

| Diagram item | NSCH | Mental Health Text |
|---|---|---|
| Transformer embeddings (e.g., BERT, RoBERTa) | Not available | **BUILT**: frozen MiniLM sentence encoder (BERT family), 384 dims, whole post and per-sentence mean + max |
| Sentiment & emotion features | Not available | **BUILT**: frozen emotion classifier (j-hartmann/emotion-english-distilroberta-base), 7 emotion probabilities |
| Contextual representations | Not available | **BUILT**: the sentence-encoder embedding is contextual |
| Physiological signals (optional) | Not available | Not available: no sensor data |

*Table 3.2. Mutual information with the 4-class label, by feature family*

| Family | Columns | Max MI | Mean MI | Best rank |
|---|---|---|---|---|
| MiniLM pooled embedding | 384 | 0.233 | 0.040 | 6 |
| MiniLM sentence mean | 384 | 0.181 | 0.052 | 20 |
| MiniLM sentence max | 384 | 0.250 | 0.068 | 4 |
| emotion | 7 | 0.129 | 0.079 | 65 |
| shipped: length/style | 13 | 0.303 | 0.160 | 1 |
| shipped: sentiment | 2 | 0.194 | 0.191 | 14 |

PCA vs mutual-information selection to reach 8 quantum inputs: see Box 15 (P4) — it does not replicate across the two datasets.

Source: results/mhtext/phase2/manifest.json · results/mhtext/phase2/mi_by_family.csv

## Box 4 — Privacy-Preserving Mechanisms

**What this step does:** Applies the NSCH mechanisms with their NSCH settings: DP-SGD during training, Laplace local DP on the encoder inputs, federated training (Box 13); identifier scrubbing for text.

| Mechanism | NSCH | Mental Health Text |
|---|---|---|
| k-anonymity on quasi-identifiers | Used (k = 5) | Not applicable: free text has no quasi-identifier columns |
| Identifier scrubbing (URLs, e-mails, handles, communities, numbers) | n/a | BUILT (Box 12: no effect on re-identification) |
| DP-SGD (Opacus, ε 0.5-8, δ 1e-5) | Used | BUILT (same settings) |
| Laplace local DP on the 8 encoder inputs (ε_input 4-32) | Used | BUILT (same settings) |
| Federated averaging with client DP (ε = 4) | Used | BUILT, simulated clients (Box 13) |

Source: results/mhtext/phase5/dp/run.json · results/mhtext/phase5/ldp/run.json

## Box 5 — Quantum Encoding

**What this step does:** Maps 4-8 reduced inputs onto qubits with five encodings (exp1 method, 1,200 rows, 10 folds).

| Model | Features | Encoding | 4 qubits | 6 qubits | 8 qubits |
|---|---|---|---|---|---|
| QSVM | mi | amplitude | 0.757 | 0.764 | 0.750 |
| QSVM | mi | angle | 0.702 | 0.721 | 0.737 |
| QSVM | mi | basis | 0.639 | 0.666 | 0.665 |
| QSVM | mi | reupload | 0.697 | 0.715 | 0.732 |
| QSVM | mi | zz | 0.670 | 0.688 | 0.685 |
| QSVM | pca | amplitude | 0.777 | 0.742 | 0.753 |
| QSVM | pca | angle | 0.803 | 0.804 | 0.804 |
| QSVM | pca | basis | 0.735 | 0.732 | 0.702 |
| QSVM | pca | reupload | 0.796 | 0.797 | 0.797 |
| QSVM | pca | zz | 0.758 | 0.750 | 0.730 |
| VQC | mi | amplitude | 0.635 | 0.600 | 0.560 |
| VQC | mi | angle | 0.684 | 0.699 | 0.718 |
| VQC | mi | basis | 0.575 | 0.532 | 0.516 |
| VQC | mi | reupload | 0.683 | 0.701 | 0.707 |
| VQC | mi | zz | 0.631 | 0.626 | 0.615 |
| VQC | pca | amplitude | 0.604 | 0.595 | 0.581 |
| VQC | pca | angle | 0.784 | 0.784 | 0.782 |
| VQC | pca | basis | 0.553 | 0.513 | 0.509 |
| VQC | pca | reupload | 0.779 | 0.778 | 0.775 |
| VQC | pca | zz | 0.678 | 0.670 | 0.617 |

Best encodings on both datasets: angle and re-upload on PCA inputs; basis is worst.

Source: results/mhtext/phase4/binary/sweep_summary.csv

## Box 6 — Quantum Model (Simulation)

**What this step does:** Trains QSVM, VQC and hybrid QNN on the binary Depression vs Suicidal view, 1,200 rows per fold, 15 folds (exp2 method).

| Model | Input | ROC AUC | Balanced acc. | Recall | Accuracy | Quarantine AUC |
|---|---|---|---|---|---|---|
| QSVM, angle, 8 qubits | pca8 | 0.803 | 0.729 | 0.778 | 0.721 | 0.760 |
| VQC, re-upload, 4 qubits | pca4 | 0.781 | 0.713 | 0.770 | 0.705 | 0.745 |
| Hybrid QNN, angle, 8 qubits | all384 | 0.658 | 0.610 | 0.679 | 0.600 | 0.616 |

Four-class and three-class quantum views: deliberately not run (reason in results/mhtext/phase4/binary/phase4.md).

Source: results/mhtext/phase4/binary/summary.csv

## Box 7 — Hybrid Quantum-Classical Learning

**What this step does:** Trains a classical layer (384 → 8) and an 8-qubit circuit end to end.

Hybrid QNN ROC AUC 0.658 ± 0.045, against MLP on the same rows and all 384 dims 0.815. It is the weakest quantum model here, as on NSCH (Box 15, P6).

Source: results/mhtext/phase4/binary/summary.csv

## Box 8 — Simulation Environment

**What this step does:** Records the software, the simulator and the wall time of each run.

| Run | Wall time (min) |
|---|---|
| Phase 4 quantum sweep + comparison, 642 tasks of the second launch (whole run 11:15-13:37 incl. restart) | 116 |
| Phase 4 all-rows classical ceiling | 15 |
| Phase 5 DP-SGD | 7 |
| Phase 5 membership inference | 4 |
| Phase 5 input local DP | 2 |
| Phase 5 re-identification | 22 |
| Phase 6 federated (120 tasks) | 23 |

Python 3.10.10, statevector simulation, 6 CPU workers. Packages: PennyLane 0.40.0, PennyLane_Lightning 0.40.0, PyYAML 6.0.2, matplotlib 3.9.3, numpy 1.26.4, opacus 1.5.2, pandas 2.2.3, scikit-learn 1.5.2, scipy 1.15.3, torch 2.2.2, xgboost 2.1.3.

*Table 8.3. Qubit count 4 → 8 (paired NB-t, Holm over each dataset's table)*: see Box 15, P3.

Source: results/mhtext/phase4/binary/run.json · results/mhtext/phase4/binary_ceiling/run.json · results/mhtext/phase5/dp/run.json · results/mhtext/phase5/mia/run.json · results/mhtext/phase5/ldp/run.json · results/mhtext/phase5/reid/run.json · results/mhtext/phase6/run.json

## Box 9 — Mental Health Analysis Tasks

**What this step does:** Defines what is predicted on whom.

| Task | NSCH | Mental Health Text |
|---|---|---|
| Binary | provider-diagnosed depression, adolescents 12-17 | Depression vs Suicidal (the hard, length-free boundary) |
| Multi-class | none / anxiety only / depression | 4-class (Normal, Anxiety, Depression, Suicidal) and 3-class (Normal dropped) |
| Stress | Not available | Not available: no stress label |
| Topic and trend analysis | Not available | Not available: posts carry no timestamps |
| Emotion and sentiment analysis | Not available | **BUILT** as features (Box 3); not a prediction target |
| Personalized insights (simulated) | Not available | Not available: no per-person history |

Population limit: authors' ages are unknown; regex counts find an age of 12-17 stated in 0.9% of development posts. This dataset is not an adolescent sample.

Source: results/mhtext/phase5/reid/self_disclosures.csv

## Box 10 — Evaluation & Comparison

**What this step does:** Compares every model against the length-only reference (multi-class) and quantum against classical on the same rows and inputs (binary). NB-t corrected resampled t-test, Holm.

*4-class, cross-validation (15 folds)*

| Model | Macro AUC | Macro-F1 | Balanced acc. | Accuracy | Quarantine macro AUC |
|---|---|---|---|---|---|
| All pooled features, XGBoost | 0.952 | 0.802 | 0.796 | 0.816 | 0.856 |
| All pooled features, MLP | 0.951 | 0.793 | 0.807 | 0.810 | 0.864 |
| CNN over tokens (embedding trained from scratch) | 0.950 | 0.785 | 0.791 | 0.802 | 0.750 |
| LSTM over tokens (embedding trained from scratch) | 0.949 | 0.777 | 0.786 | 0.795 | 0.760 |
| All pooled features, LogReg | 0.945 | 0.780 | 0.792 | 0.797 | 0.859 |
| All pooled features, SVM (RBF) | 0.941 | 0.808 | 0.807 | 0.820 | 0.872 |
| Frozen MiniLM sentence encoder (BERT family), whole post, LogReg | 0.941 | 0.764 | 0.785 | 0.787 | 0.865 |
| Frozen MiniLM sentence encoder (BERT family), per-sentence mean + max, LogReg | 0.936 | 0.756 | 0.775 | 0.777 | 0.843 |
| Length only (log words + log characters), LogReg | 0.776 | 0.460 | 0.482 | 0.548 | 0.641 |

*3-class, cross-validation (15 folds)*

| Model | Macro AUC | Macro-F1 | Balanced acc. | Accuracy | Quarantine macro AUC |
|---|---|---|---|---|---|
| All pooled features, XGBoost | 0.901 | 0.787 | 0.781 | 0.758 | 0.828 |
| All pooled features, MLP | 0.899 | 0.774 | 0.788 | 0.751 | 0.845 |
| CNN over tokens (embedding trained from scratch) | 0.896 | 0.756 | 0.778 | 0.727 | 0.700 |
| LSTM over tokens (embedding trained from scratch) | 0.894 | 0.757 | 0.781 | 0.730 | 0.678 |
| All pooled features, SVM (RBF) | 0.893 | 0.790 | 0.790 | 0.764 | 0.852 |
| All pooled features, LogReg | 0.888 | 0.763 | 0.775 | 0.739 | 0.838 |
| Frozen MiniLM sentence encoder (BERT family), whole post, LogReg | 0.886 | 0.750 | 0.772 | 0.732 | 0.835 |
| Frozen MiniLM sentence encoder (BERT family), per-sentence mean + max, LogReg | 0.873 | 0.741 | 0.759 | 0.720 | 0.806 |
| Length only (log words + log characters), LogReg | 0.617 | 0.403 | 0.450 | 0.421 | 0.515 |

*Diagram items — coverage now*

| Diagram item | NSCH | Mental Health Text |
|---|---|---|
| BERT + ML | Not available | **BUILT**: frozen MiniLM sentence encoder (BERT family) + LogReg / XGBoost / MLP / SVM |
| Deep Learning (LSTM, CNN) | Not available (MLP instead) | **BUILT**: LSTM and CNN over token sequences |
| Quantum Neural Network | hybrid QNN | hybrid QNN (Box 7) |

*Binary view: quantum vs classical on the same rows and inputs (ROC AUC)*

| Quantum | Classical | Quantum | Classical | Diff | Holm p |
|---|---|---|---|---|---|
| qsvm/subsample/pca8 | logreg/subsample/pca8 | 0.803 | 0.804 | -0.000 | 1.000 |
| qsvm/subsample/pca8 | xgboost/subsample/pca8 | 0.803 | 0.802 | +0.002 | 1.000 |
| qsvm/subsample/pca8 | mlp/subsample/pca8 | 0.803 | 0.802 | +0.001 | 1.000 |
| qsvm/subsample/pca8 | svm/subsample/pca8 | 0.803 | 0.804 | -0.001 | 1.000 |
| vqc/subsample/pca4 | logreg/subsample/pca4 | 0.781 | 0.803 | -0.022 | 0.014 |
| vqc/subsample/pca4 | xgboost/subsample/pca4 | 0.781 | 0.798 | -0.017 | 0.103 |
| vqc/subsample/pca4 | mlp/subsample/pca4 | 0.781 | 0.800 | -0.019 | 0.029 |
| vqc/subsample/pca4 | svm/subsample/pca4 | 0.781 | 0.801 | -0.020 | 0.029 |
| hybrid/subsample/all384 | logreg/subsample/all384 | 0.658 | 0.803 | -0.145 | 0.001 |
| hybrid/subsample/all384 | xgboost/subsample/all384 | 0.658 | 0.802 | -0.144 | 0.001 |
| hybrid/subsample/all384 | mlp/subsample/all384 | 0.658 | 0.815 | -0.157 | 0.000 |
| hybrid/subsample/all384 | svm/subsample/all384 | 0.658 | 0.812 | -0.154 | 0.001 |

Length-only floor 0.558; classical ceiling, 5,044 rows 0.828; classical ceiling, all rows (≈ 17,600 per fold) 0.832.

Source: results/mhtext/phase3/summary.csv · results/mhtext/phase4/binary/tests_same_rows_features.csv · results/mhtext/phase4/binary/summary.csv

## Box 11 — Performance Metrics

**What this step does:** Reports accuracy only beside class-balanced metrics.

Every table in Boxes 6 and 10 reports accuracy beside balanced accuracy, recall, macro-F1 and ROC AUC; per-class precision, recall, F1 and AUC are in results/mhtext/phase3/summary.csv.

Source: results/mhtext/phase3/summary.csv

## Box 12 — Privacy Evaluation

**What this step does:** Measures re-identification, membership inference and the utility cost of DP, beside NSCH.

*Summary carried from results/mhtext/phase5/phase5.md (every figure there is read from the files below)*

- **The 8 inputs fed to the quantum circuit (step 5) still identify the post 32.6 % of the time** from a 12-word fragment, when the PCA projection is known. 8 PCA components are not anonymous. Keeping the projection secret would lower linkage, but secrecy of a projection is not a privacy guarantee and is not claimed.
- **A post is far more identifying than a survey row:** a 5-word fragment singles out 95.0 % of posts, against 62.7 % of NSCH adolescents on 7 demographics + state (raw) and 0.0 % on the 7 demographics after k-anonymity (k = 5). Scrubbing identifiers leaves it at 95.2 %; k-anonymity has no columns to act on.
- **The small quantum models leak little, and that comes from their low capacity, not from anything quantum.** Membership-attack AUC: QSVM 0.508, VQC 0.502; the overfitted MLP control reaches 0.732 (NSCH 0.558).
- **Local DP at ε = 32 defeats the linkage attack we implemented, at a QSVM cost of 0.007 AUC. The formal guarantee at this budget is weak and must not be read as a general privacy claim.**
- DP-SGD at ε = 0.5 costs the same-rows LogReg 0.019 AUC on text against 0.195 on NSCH, about tenfold less; the cause is not established.

| Measure | NSCH | Mental Health Text |
|---|---|---|
| Unique rows, most identifying table (NSCH: 7 demographics + state; text: 5-word fragment) | 62.7% | 95.0% |
| Unique rows after the pipeline's protection | 0.0% (k-anonymity) | 95.2% (scrubbing) |
| Linkage through the 8 quantum inputs (12-word fragment, projection known) | n/a | 32.6% |
| Overfit MLP control, membership attack AUC | 0.558 | 0.732 |
| QSVM, membership attack AUC | 0.502 | 0.508 |
| VQC (no DP), membership attack AUC | 0.498 | 0.502 |

Source: results/mhtext/phase5/phase5.md · results/mhtext/phase5/reid/fragment_risk.csv · results/mhtext/phase5/reid/linkage.csv · results/mhtext/phase5/mia/summary.csv · results/reidentification/risk.csv · results/exp6-nsch2024-membership-inference-20261112-120934/summary.csv

## Box 13 — Scalability Analysis (Simulation)

**What this step does:** FedAvg with client DP-SGD (ε = 4) over 3, 5 and 10 SIMULATED clients (exp4 method).

| Arm | Partition | 3 clients | 5 clients | 10 clients | NSCH (iid) 3 / 5 / 10 |
|---|---|---|---|---|---|
| vqc_sub | iid | 0.755 | 0.747 | 0.724 | 0.748 / 0.709 / 0.657 |
| vqc_sub | by_length | 0.761 | 0.747 | 0.725 | 0.748 / 0.709 / 0.657 |
| logreg_full | iid | 0.781 | 0.780 | 0.786 | 0.861 / 0.859 / 0.860 |
| logreg_full | by_length | 0.780 | 0.779 | 0.784 | 0.861 / 0.859 / 0.860 |

Communication per FedAvg round: VQC 2.44 kB, LogReg 36.09 kB (mean over clients and folds). by_length is the non-IID substitute for NSCH's by_state (posts carry no site).

Source: results/mhtext/phase6/fold_metrics.csv · results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv · results/mhtext/phase6/phase6.md

## Box 14 — Output & Visualization

**What this step does:** Lists what this dataset adds to the visual outputs.

| Diagram item | NSCH | Mental Health Text |
|---|---|---|
| Feature and attention visualization: attention | Not available | Not produced: the encoder is used frozen and attention maps were not extracted; available in principle, not built |
| Quantum state visualization | Built (Box 5 of the NSCH report) | Not repeated: same circuits, different inputs |
| Results tables | Built | Built (this report and the phase reports) |

Source: results/mhtext/phase4/binary/phase4.md

## Box 15 — Key Outcomes

**What this step does:** States which findings replicate across two unrelated datasets, and the limits.

*Table 15.1. Patterns tested on both datasets (same method, same statistics)*

| # | Pattern | NSCH 2024 | Mental Health Text | Status |
|---|---|---|---|---|
| P1 | QSVM ties the best classical model on matched rows | 0.847 vs 0.851, p 0.704 | 0.803 vs 0.804, p 1.000 | Replicates |
| P2 | VQC loses by about 0.02, significantly | 0.814 vs 0.837, p 0.012 | 0.781 vs 0.803, p 0.014 | Replicates |
| P3 | 4 → 8 qubits gives no significant gain for any quantum model | 0 of 20 cells significant either way; headline configs QSVM +0.012, VQC re-upload -0.013 | 0 of 20 cells gain significantly; 2 drop (VQC, weak encodings); headline QSVM +0.001, VQC re-upload -0.004 | Replicates, in this form |
| P5 | Federated VQC falls from 3 to 10 clients; LogReg stays flat | VQC -0.091, p 0.046; LogReg -0.001, p 0.842 | VQC -0.031, p 0.003; LogReg +0.005, p 0.467 | Replicates |
| P6 | Hybrid QNN is the lowest quantum model on ROC AUC | 0.784 (VQC 0.814, VQC-COBYLA 0.799, QSVM 0.847) | 0.658 (VQC 0.781, QSVM 0.803) | Replicates on ROC AUC; on NSCH the hybrid edges the VQC on accuracy, precision and F1 |

P3 is stated in its measured form. 'Changes almost nothing' holds for the headline encodings (angle and re-upload on PCA). For weak encodings the means move more (up to 0.084 on NSCH, 0.075 on text across 4, 6, 8 qubits), not significantly on NSCH and as two significant drops on text. On neither dataset do more qubits help.

Table 15.1 holds four replications from the list of candidates (P1, P2, P3, P6) plus the federated one (P5): five in total. P4 (PCA vs mutual-information selection) was a candidate and is not in the table; it is reported below as a disagreement.

### Where the two datasets disagree

**P4 — the better way to reduce inputs to 4-8 qubits depends on the kind of data.** Same comparison on both datasets (exp1 sweep: PCA vs mutual-information column selection, every model x encoding x qubit cell, paired NB-t, Holm with NSCH's families: non-basis encodings in one family, basis in another).

| | NSCH 2024 (95 survey codes) | Mental Health Text (384-dim sentence embedding) |
|---|---|---|
| Cells compared | 33 | 33 |
| PCA significantly better | 0 | 20 |
| MI significantly better | 6 (all basis-encoding cells) | 0 |
| PCA higher mean, any significance | 22 | 27 |

Finding: the better reduction depends on the kind of data. On sentence embeddings PCA is clearly better; on survey codes PCA never wins significantly and MI selection wins outright for the basis encoding. Anyone doing quantum ML should choose the reduction for the data type, not carry one choice from tabular to text inputs or back. Why the two differ is not tested here (one candidate explanation: information spread across many correlated embedding dimensions versus concentrated in a few survey columns).

**Cell-level caveat (a correction to an earlier summary of ours).** An earlier summary said "PCA beats MI for every model" on text. That was a per-model average. At cell level PCA has the higher mean in 27 of 33 text cells, not all: MI has the higher mean in 6, for example QSVM amplitude at 6 qubits (0.764 vs 0.742), VQC amplitude at 4 qubits (0.635 vs 0.604), VQC amplitude at 6 qubits (0.600 vs 0.595); none of these is significant.

NSCH's separate comparison of PCA against UMAP and an autoencoder (NSCH report, Box 3, Table 3.3) is a different comparison and is not part of P4.

**On P5, plainly:** NSCH's federated VQC drop has Holm p 0.046; it only just clears the threshold, and on its own that would be weak evidence. The text drop is a third the size (-0.031 vs -0.091) but far tighter (95 % CI -0.045 to -0.016) and clearly significant (p 0.003). Neither dataset alone would settle it; the replication is what makes it solid.

**What is new on this dataset, and does not replicate because NSCH cannot show it:** a free-text post is far more identifying than a row of survey codes, and the 8 numbers fed to the quantum circuit still identify the post about a third of the time (Box 12). The small quantum models leak little, and that comes from their low capacity, not from anything quantum.

**Limits.** Authors' ages are unknown, so this is not an adolescent sample. Labels come from the source corpora, not from clinicians. Clients are simulated. Circuits are simulated, not run on hardware. Quantum results use the binary view only. The locked hold-out has not been scored.

Source: results/mhtext/phase7/pattern_checks/p3_qubits_8_vs_4.csv · results/mhtext/phase7/pattern_checks/p4_mi_vs_pca_text.csv · results/mhtext/phase7/pattern_checks/p4_mi_vs_pca_nsch_stored.csv · results/mhtext/phase6/tests_replication_clients.csv · results/exp2-nsch2024-quantum-vs-classical-20261108-165426/summary.csv · results/exp2-nsch2024-quantum-vs-classical-20261108-165426/tests_same_rows_features.csv · results/mhtext/phase4/binary/summary.csv · results/mhtext/phase4/binary/tests_same_rows_features.csv

## Coverage update: what this dataset added to the architecture

| Box | Diagram item | Before (NSCH only) | Now |
|---|---|---|---|
| 1 | Social media text | Not available on this dataset | **BUILT** |
| 3 | Transformer embeddings (e.g., BERT, RoBERTa) | Not available on this dataset | **BUILT** (frozen MiniLM, BERT family) |
| 3 | Sentiment & emotion features | Not available on this dataset | **BUILT** (frozen emotion classifier) |
| 3 | Contextual representations | Not available on this dataset | **BUILT** (contextual sentence embeddings) |
| 9 | Emotion and sentiment analysis | Not available on this dataset | **BUILT** as features (not a target) |
| 10 | BERT + ML | Not available on this dataset | **BUILT** |
| 10 | Deep Learning (LSTM, CNN) | Not available on this dataset | **BUILT** |
| 12 | Re-identification of free text | not applicable (no text) | **BUILT** (fragment risk, linkage) |

Still unavailable, with the reason:

| Box | Diagram item | Reason |
|---|---|---|
| 1 | Clinical notes (de-identified) | No clinical-notes dataset is part of this project; access needs a data-use agreement with a health provider. |
| 1 | Counselling records | No counselling-records dataset is part of this project; such records are not public. |
| 1 | Wearable/IoT data (optional) | No sensor dataset is part of this project. |
| 3 | Physiological signals (optional) | Needs sensor data (above). |
| 9 | Stress | Neither dataset has a stress label. |
| 9 | Topic and trend analysis | Needs text over time; the posts carry no timestamps. |
| 9 | Personalized insights (simulated) | Needs a per-person history; posts are not linked to authors. |
| 14 | Attention visualization | Available in principle (frozen transformer); not extracted. |

## Traceability appendix

| Box | Item | Value | File |
|---|---|---|---|
| Box 1 | raw rows | 49,612 | results/mhtext/phase1/manifest.json |
| Box 1 | rows after dedup | 48,444 | results/mhtext/phase1/manifest.json |
| Box 1 | processed rows 4-class | 24,572 | results/mhtext/phase1/manifest.json |
| Box 2 | exact duplicates dropped | 664 | results/mhtext/phase1/manifest.json |
| Box 2 | exact conflict rows | 33 | results/mhtext/phase1/manifest.json |
| Box 2 | near duplicates dropped | 381 | results/mhtext/phase1/manifest.json |
| Box 2 | near conflict rows | 90 | results/mhtext/phase1/manifest.json |
| Box 2 | confirmation pass dropped | 13 | results/mhtext/phase1/manifest.json |
| Box 2 | hold-out fraction | 20% | results/mhtext/phase1/manifest.json |
| Box 3 | embedding dims | 384 | results/mhtext/phase2/manifest.json |
| Box 3 | emotion labels | 7 | results/mhtext/phase2/manifest.json |
| Box 3 | max MI pooled embedding | 0.233 | results/mhtext/phase2/mi_by_family.csv |
| Box 3 | mean MI pooled embedding | 0.040 | results/mhtext/phase2/mi_by_family.csv |
| Box 3 | max MI sentence mean | 0.181 | results/mhtext/phase2/mi_by_family.csv |
| Box 3 | mean MI sentence mean | 0.052 | results/mhtext/phase2/mi_by_family.csv |
| Box 3 | max MI sentence max | 0.250 | results/mhtext/phase2/mi_by_family.csv |
| Box 3 | mean MI sentence max | 0.068 | results/mhtext/phase2/mi_by_family.csv |
| Box 3 | max MI emotion | 0.129 | results/mhtext/phase2/mi_by_family.csv |
| Box 3 | mean MI emotion | 0.079 | results/mhtext/phase2/mi_by_family.csv |
| Box 3 | max MI shipped: length/style | 0.303 | results/mhtext/phase2/mi_by_family.csv |
| Box 3 | mean MI shipped: length/style | 0.160 | results/mhtext/phase2/mi_by_family.csv |
| Box 3 | max MI shipped: sentiment | 0.194 | results/mhtext/phase2/mi_by_family.csv |
| Box 3 | mean MI shipped: sentiment | 0.191 | results/mhtext/phase2/mi_by_family.csv |
| Box 5 | sweep qsvm mi amplitude 4q | 0.757 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi amplitude 6q | 0.764 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi amplitude 8q | 0.750 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi angle 4q | 0.702 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi angle 6q | 0.721 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi angle 8q | 0.737 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi basis 4q | 0.639 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi basis 6q | 0.666 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi basis 8q | 0.665 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi reupload 4q | 0.697 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi reupload 6q | 0.715 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi reupload 8q | 0.732 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi zz 4q | 0.670 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi zz 6q | 0.688 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm mi zz 8q | 0.685 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca amplitude 4q | 0.777 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca amplitude 6q | 0.742 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca amplitude 8q | 0.753 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca angle 4q | 0.803 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca angle 6q | 0.804 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca angle 8q | 0.804 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca basis 4q | 0.735 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca basis 6q | 0.732 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca basis 8q | 0.702 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca reupload 4q | 0.796 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca reupload 6q | 0.797 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca reupload 8q | 0.797 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca zz 4q | 0.758 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca zz 6q | 0.750 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep qsvm pca zz 8q | 0.730 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi amplitude 4q | 0.635 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi amplitude 6q | 0.600 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi amplitude 8q | 0.560 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi angle 4q | 0.684 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi angle 6q | 0.699 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi angle 8q | 0.718 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi basis 4q | 0.575 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi basis 6q | 0.532 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi basis 8q | 0.516 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi reupload 4q | 0.683 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi reupload 6q | 0.701 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi reupload 8q | 0.707 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi zz 4q | 0.631 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi zz 6q | 0.626 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc mi zz 8q | 0.615 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca amplitude 4q | 0.604 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca amplitude 6q | 0.595 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca amplitude 8q | 0.581 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca angle 4q | 0.784 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca angle 6q | 0.784 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca angle 8q | 0.782 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca basis 4q | 0.553 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca basis 6q | 0.513 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca basis 8q | 0.509 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca reupload 4q | 0.779 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca reupload 6q | 0.778 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca reupload 8q | 0.775 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca zz 4q | 0.678 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca zz 6q | 0.670 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 5 | sweep vqc pca zz 8q | 0.617 | results/mhtext/phase4/binary/sweep_summary.csv |
| Box 6 | qsvm ROC AUC | 0.803 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | qsvm bal acc | 0.729 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | qsvm recall | 0.778 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | qsvm accuracy | 0.721 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | qsvm quarantine AUC | 0.760 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | vqc ROC AUC | 0.781 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | vqc bal acc | 0.713 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | vqc recall | 0.770 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | vqc accuracy | 0.705 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | vqc quarantine AUC | 0.745 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | hybrid ROC AUC | 0.658 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | hybrid bal acc | 0.610 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | hybrid recall | 0.679 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | hybrid accuracy | 0.600 | results/mhtext/phase4/binary/summary.csv |
| Box 6 | hybrid quarantine AUC | 0.616 | results/mhtext/phase4/binary/summary.csv |
| Box 7 | hybrid ROC AUC | 0.658 | results/mhtext/phase4/binary/summary.csv |
| Box 7 | hybrid ROC AUC sd | 0.045 | results/mhtext/phase4/binary/summary.csv |
| Box 7 | mlp all384 sub AUC | 0.815 | results/mhtext/phase4/binary/summary.csv |
| Box 8 | wall time Phase 4 quantum sweep + comparison, 642 tasks of the second launch (whole run 11:15-13:37 incl. restart) | 116 | results/mhtext/phase4/binary/run.json |
| Box 8 | wall time Phase 4 all-rows classical ceiling | 15 | results/mhtext/phase4/binary_ceiling/run.json |
| Box 8 | wall time Phase 5 DP-SGD | 7 | results/mhtext/phase5/dp/run.json |
| Box 8 | wall time Phase 5 membership inference | 4 | results/mhtext/phase5/mia/run.json |
| Box 8 | wall time Phase 5 input local DP | 2 | results/mhtext/phase5/ldp/run.json |
| Box 8 | wall time Phase 5 re-identification | 22 | results/mhtext/phase5/reid/run.json |
| Box 8 | wall time Phase 6 federated (120 tasks) | 23 | results/mhtext/phase6/run.json |
| Box 9 | share stating age 12-17 | 0.9% | results/mhtext/phase5/reid/self_disclosures.csv |
| Box 10 | 4-class xgboost macro AUC | 0.952 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class xgboost macro F1 | 0.802 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class xgboost bal acc | 0.796 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class xgboost accuracy | 0.816 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class xgboost quarantine macro AUC | 0.856 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class mlp macro AUC | 0.951 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class mlp macro F1 | 0.793 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class mlp bal acc | 0.807 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class mlp accuracy | 0.810 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class mlp quarantine macro AUC | 0.864 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class cnn macro AUC | 0.950 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class cnn macro F1 | 0.785 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class cnn bal acc | 0.791 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class cnn accuracy | 0.802 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class cnn quarantine macro AUC | 0.750 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class lstm macro AUC | 0.949 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class lstm macro F1 | 0.777 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class lstm bal acc | 0.786 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class lstm accuracy | 0.795 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class lstm quarantine macro AUC | 0.760 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class logreg macro AUC | 0.945 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class logreg macro F1 | 0.780 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class logreg bal acc | 0.792 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class logreg accuracy | 0.797 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class logreg quarantine macro AUC | 0.859 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class svm macro AUC | 0.941 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class svm macro F1 | 0.808 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class svm bal acc | 0.807 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class svm accuracy | 0.820 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class svm quarantine macro AUC | 0.872 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class bert_pooled_logreg macro AUC | 0.941 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class bert_pooled_logreg macro F1 | 0.764 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class bert_pooled_logreg bal acc | 0.785 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class bert_pooled_logreg accuracy | 0.787 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class bert_pooled_logreg quarantine macro AUC | 0.865 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class bert_sentence_logreg macro AUC | 0.936 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class bert_sentence_logreg macro F1 | 0.756 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class bert_sentence_logreg bal acc | 0.775 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class bert_sentence_logreg accuracy | 0.777 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class bert_sentence_logreg quarantine macro AUC | 0.843 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class length_only_logreg macro AUC | 0.776 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class length_only_logreg macro F1 | 0.460 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class length_only_logreg bal acc | 0.482 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class length_only_logreg accuracy | 0.548 | results/mhtext/phase3/summary.csv |
| Box 10 | 4-class length_only_logreg quarantine macro AUC | 0.641 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class xgboost macro AUC | 0.901 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class xgboost macro F1 | 0.787 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class xgboost bal acc | 0.781 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class xgboost accuracy | 0.758 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class xgboost quarantine macro AUC | 0.828 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class mlp macro AUC | 0.899 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class mlp macro F1 | 0.774 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class mlp bal acc | 0.788 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class mlp accuracy | 0.751 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class mlp quarantine macro AUC | 0.845 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class cnn macro AUC | 0.896 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class cnn macro F1 | 0.756 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class cnn bal acc | 0.778 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class cnn accuracy | 0.727 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class cnn quarantine macro AUC | 0.700 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class lstm macro AUC | 0.894 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class lstm macro F1 | 0.757 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class lstm bal acc | 0.781 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class lstm accuracy | 0.730 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class lstm quarantine macro AUC | 0.678 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class svm macro AUC | 0.893 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class svm macro F1 | 0.790 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class svm bal acc | 0.790 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class svm accuracy | 0.764 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class svm quarantine macro AUC | 0.852 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class logreg macro AUC | 0.888 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class logreg macro F1 | 0.763 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class logreg bal acc | 0.775 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class logreg accuracy | 0.739 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class logreg quarantine macro AUC | 0.838 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class bert_pooled_logreg macro AUC | 0.886 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class bert_pooled_logreg macro F1 | 0.750 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class bert_pooled_logreg bal acc | 0.772 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class bert_pooled_logreg accuracy | 0.732 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class bert_pooled_logreg quarantine macro AUC | 0.835 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class bert_sentence_logreg macro AUC | 0.873 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class bert_sentence_logreg macro F1 | 0.741 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class bert_sentence_logreg bal acc | 0.759 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class bert_sentence_logreg accuracy | 0.720 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class bert_sentence_logreg quarantine macro AUC | 0.806 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class length_only_logreg macro AUC | 0.617 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class length_only_logreg macro F1 | 0.403 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class length_only_logreg bal acc | 0.450 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class length_only_logreg accuracy | 0.421 | results/mhtext/phase3/summary.csv |
| Box 10 | 3-class length_only_logreg quarantine macro AUC | 0.515 | results/mhtext/phase3/summary.csv |
| Box 10 | qsvm/subsample/pca8 vs logreg/subsample/pca8 q | 0.803 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | qsvm/subsample/pca8 vs logreg/subsample/pca8 c | 0.804 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | qsvm/subsample/pca8 vs logreg/subsample/pca8 p | 1.000 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | qsvm/subsample/pca8 vs xgboost/subsample/pca8 q | 0.803 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | qsvm/subsample/pca8 vs xgboost/subsample/pca8 c | 0.802 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | qsvm/subsample/pca8 vs xgboost/subsample/pca8 p | 1.000 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | qsvm/subsample/pca8 vs mlp/subsample/pca8 q | 0.803 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | qsvm/subsample/pca8 vs mlp/subsample/pca8 c | 0.802 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | qsvm/subsample/pca8 vs mlp/subsample/pca8 p | 1.000 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | qsvm/subsample/pca8 vs svm/subsample/pca8 q | 0.803 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | qsvm/subsample/pca8 vs svm/subsample/pca8 c | 0.804 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | qsvm/subsample/pca8 vs svm/subsample/pca8 p | 1.000 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs logreg/subsample/pca4 q | 0.781 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs logreg/subsample/pca4 c | 0.803 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs logreg/subsample/pca4 p | 0.014 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs xgboost/subsample/pca4 q | 0.781 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs xgboost/subsample/pca4 c | 0.798 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs xgboost/subsample/pca4 p | 0.103 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs mlp/subsample/pca4 q | 0.781 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs mlp/subsample/pca4 c | 0.800 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs mlp/subsample/pca4 p | 0.029 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs svm/subsample/pca4 q | 0.781 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs svm/subsample/pca4 c | 0.801 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | vqc/subsample/pca4 vs svm/subsample/pca4 p | 0.029 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs logreg/subsample/all384 q | 0.658 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs logreg/subsample/all384 c | 0.803 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs logreg/subsample/all384 p | 0.001 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs xgboost/subsample/all384 q | 0.658 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs xgboost/subsample/all384 c | 0.802 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs xgboost/subsample/all384 p | 0.001 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs mlp/subsample/all384 q | 0.658 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs mlp/subsample/all384 c | 0.815 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs mlp/subsample/all384 p | 0.000 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs svm/subsample/all384 q | 0.658 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs svm/subsample/all384 c | 0.812 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | hybrid/subsample/all384 vs svm/subsample/all384 p | 0.001 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 10 | length floor AUC | 0.558 | results/mhtext/phase4/binary/summary.csv |
| Box 10 | dev-rows ceiling AUC | 0.828 | results/mhtext/phase4/binary/summary.csv |
| Box 10 | all-rows ceiling AUC | 0.832 | results/mhtext/phase4/binary/summary.csv |
| Box 12 | NSCH unique 7QI+state | 62.7% | results/reidentification/risk.csv |
| Box 12 | text unique 5-word | 95.0% | results/mhtext/phase5/reid/fragment_risk.csv |
| Box 12 | NSCH unique after k-anon | 0.0% | results/reidentification/risk.csv |
| Box 12 | text unique 5-word scrubbed | 95.2% | results/mhtext/phase5/reid/fragment_risk.csv |
| Box 12 | linkage PCA-8 12 words | 32.6% | results/mhtext/phase5/reid/linkage.csv |
| Box 12 | NSCH mlp_overfit MIA | 0.558 | results/exp6-nsch2024-membership-inference-20261112-120934/summary.csv |
| Box 12 | text mlp_overfit MIA | 0.732 | results/mhtext/phase5/mia/summary.csv |
| Box 12 | NSCH qsvm MIA | 0.502 | results/exp6-nsch2024-membership-inference-20261112-120934/summary.csv |
| Box 12 | text qsvm MIA | 0.508 | results/mhtext/phase5/mia/summary.csv |
| Box 12 | NSCH vqc MIA | 0.498 | results/exp6-nsch2024-membership-inference-20261112-120934/summary.csv |
| Box 12 | text vqc MIA | 0.502 | results/mhtext/phase5/mia/summary.csv |
| Box 13 | fed vqc_sub iid 3 | 0.755 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | fed vqc_sub iid 5 | 0.747 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | fed vqc_sub iid 10 | 0.724 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | NSCH fed vqc_sub iid 3 | 0.748 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | NSCH fed vqc_sub iid 5 | 0.709 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | NSCH fed vqc_sub iid 10 | 0.657 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | fed vqc_sub by_length 3 | 0.761 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | fed vqc_sub by_length 5 | 0.747 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | fed vqc_sub by_length 10 | 0.725 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | NSCH fed vqc_sub iid 3 | 0.748 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | NSCH fed vqc_sub iid 5 | 0.709 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | NSCH fed vqc_sub iid 10 | 0.657 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | fed logreg_full iid 3 | 0.781 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | fed logreg_full iid 5 | 0.780 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | fed logreg_full iid 10 | 0.786 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | NSCH fed logreg_full iid 3 | 0.861 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | NSCH fed logreg_full iid 5 | 0.859 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | NSCH fed logreg_full iid 10 | 0.860 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | fed logreg_full by_length 3 | 0.780 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | fed logreg_full by_length 5 | 0.779 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | fed logreg_full by_length 10 | 0.784 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | NSCH fed logreg_full iid 3 | 0.861 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | NSCH fed logreg_full iid 5 | 0.859 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | NSCH fed logreg_full iid 10 | 0.860 | results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv |
| Box 13 | vqc kB per round | 2.44 | results/mhtext/phase6/fold_metrics.csv |
| Box 13 | logreg kB per round | 36.09 | results/mhtext/phase6/fold_metrics.csv |
| Box 15 | NSCH qsvm AUC | 0.847 | results/exp2-nsch2024-quantum-vs-classical-20261108-165426/summary.csv |
| Box 15 | NSCH logreg pca8 sub AUC | 0.851 | results/exp2-nsch2024-quantum-vs-classical-20261108-165426/summary.csv |
| Box 15 | NSCH qsvm vs logreg p | 0.704 | results/exp2-nsch2024-quantum-vs-classical-20261108-165426/tests_same_rows_features.csv |
| Box 15 | text qsvm AUC | 0.803 | results/mhtext/phase4/binary/summary.csv |
| Box 15 | text logreg pca8 AUC | 0.804 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 15 | text qsvm vs logreg p | 1.000 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 15 | NSCH vqc AUC | 0.814 | results/exp2-nsch2024-quantum-vs-classical-20261108-165426/summary.csv |
| Box 15 | NSCH logreg pca4 AUC | 0.837 | results/exp2-nsch2024-quantum-vs-classical-20261108-165426/summary.csv |
| Box 15 | NSCH vqc vs logreg p | 0.012 | results/exp2-nsch2024-quantum-vs-classical-20261108-165426/tests_same_rows_features.csv |
| Box 15 | text vqc AUC | 0.781 | results/mhtext/phase4/binary/summary.csv |
| Box 15 | text logreg pca4 AUC | 0.803 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 15 | text vqc vs logreg p | 0.014 | results/mhtext/phase4/binary/tests_same_rows_features.csv |
| Box 15 | NSCH significant qubit cells | 0 | results/mhtext/phase7/pattern_checks/p3_qubits_8_vs_4.csv |
| Box 15 | text significant qubit drops | 2 | results/mhtext/phase7/pattern_checks/p3_qubits_8_vs_4.csv |
| Box 15 | NSCH fed VQC drop p | 0.046 | results/mhtext/phase6/tests_replication_clients.csv |
| Box 15 | NSCH fed LogReg p | 0.842 | results/mhtext/phase6/tests_replication_clients.csv |
| Box 15 | text fed VQC drop p | 0.003 | results/mhtext/phase6/tests_replication_clients.csv |
| Box 15 | text fed LogReg p | 0.467 | results/mhtext/phase6/tests_replication_clients.csv |
| Box 15 | NSCH hybrid AUC | 0.784 | results/exp2-nsch2024-quantum-vs-classical-20261108-165426/summary.csv |
| Box 15 | NSCH VQC COBYLA AUC | 0.799 | results/exp2-nsch2024-vqc-cobyla-20261112-121718/comparison.csv |
| Box 15 | text hybrid AUC | 0.658 | results/mhtext/phase4/binary/summary.csv |
| Box 15 | NSCH max qubit range | 0.084 | results/mhtext/phase7/pattern_checks/p3_qubits_8_vs_4.csv |
| Box 15 | text max qubit range | 0.075 | results/mhtext/phase7/pattern_checks/p3_qubits_8_vs_4.csv |
| Box 15 | NSCH PCA sig wins | 0 | results/mhtext/phase7/pattern_checks/p4_mi_vs_pca_nsch_stored.csv |
| Box 15 | text PCA sig wins | 20 | results/mhtext/phase7/pattern_checks/p4_mi_vs_pca_text.csv |
| Box 15 | NSCH MI sig wins | 6 | results/mhtext/phase7/pattern_checks/p4_mi_vs_pca_nsch_stored.csv |
| Box 15 | text MI sig wins | 0 | results/mhtext/phase7/pattern_checks/p4_mi_vs_pca_text.csv |
| Box 15 | NSCH PCA higher mean | 22 | results/mhtext/phase7/pattern_checks/p4_mi_vs_pca_nsch_stored.csv |
| Box 15 | text PCA higher mean | 27 | results/mhtext/phase7/pattern_checks/p4_mi_vs_pca_text.csv |
