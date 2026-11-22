# Phase 4 — quantum models on Mental Health Text (binary view)

Source: REAL:MHText. Positive class Suicidal, negative Depression. Development rows 6,306 (2,685 positive); quarantined block 4,793 rows (2,396 positive), scored by every fold's model and never used for fitting or tuning. Locked hold-out not used.

Method held fixed from NSCH: 5x3 repeated stratified K-fold; quantum and the same-rows classical arms train on the same 1,500-row subsample per fold (1,200 inner-train / 300 inner-validation), same PCA inputs, same per-fold seed; NB-t corrected resampled t-test, Holm within metric (primary). Input: frozen MiniLM sentence encoder (BERT family; sentence-transformers/all-MiniLM-L6-v2), 384 dims, reduced inside each fold.

Wall time: 1.94 h for the last launch (whole run 11:15-13:37 = 2.37 h including a restart; 655 tasks, 6 workers).

## Headline: quantum vs classical on the same rows and inputs

| Model | Input | ROC AUC | Bal. acc. | Recall | Accuracy | Quarantine AUC | best classical, same input | NB-t Holm p (AUC) |
|---|---|---|---|---|---|---|---|---|
| QSVM (angle, 8 qubits) | PCA-8 of MiniLM | 0.803 ± 0.012 | 0.729 | 0.778 | 0.721 | 0.760 | SVM (RBF) 0.804 | 1.000 |
| VQC (re-upload, 4 qubits) | PCA-4 of MiniLM | 0.781 ± 0.015 | 0.713 | 0.770 | 0.705 | 0.745 | Logistic regression 0.803 | 0.014 |
| Hybrid QNN (angle, 8 qubits) | all 384 MiniLM dims | 0.658 ± 0.045 | 0.610 | 0.679 | 0.600 | 0.616 | MLP 0.815 | 0.000 |

Every same-rows pair (ROC AUC; Holm over all 12 pairs within the metric):

| Quantum | Classical | Quantum AUC | Classical AUC | Diff [95% CI] | Quantum wins /15 | NB-t Holm p | Significant |
|---|---|---|---|---|---|---|---|
| qsvm/subsample/pca8 | logreg/subsample/pca8 | 0.803 | 0.804 | -0.000 [-0.005, +0.004] | 7 | 1.000 | no |
| qsvm/subsample/pca8 | xgboost/subsample/pca8 | 0.803 | 0.802 | +0.002 [-0.007, +0.010] | 10 | 1.000 | no |
| qsvm/subsample/pca8 | mlp/subsample/pca8 | 0.803 | 0.802 | +0.001 [-0.007, +0.009] | 9 | 1.000 | no |
| qsvm/subsample/pca8 | svm/subsample/pca8 | 0.803 | 0.804 | -0.001 [-0.004, +0.003] | 8 | 1.000 | no |
| vqc/subsample/pca4 | logreg/subsample/pca4 | 0.781 | 0.803 | -0.022 [-0.034, -0.010] | 0 | 0.014 | yes |
| vqc/subsample/pca4 | xgboost/subsample/pca4 | 0.781 | 0.798 | -0.017 [-0.031, -0.003] | 1 | 0.103 | no |
| vqc/subsample/pca4 | mlp/subsample/pca4 | 0.781 | 0.800 | -0.019 [-0.031, -0.007] | 0 | 0.029 | yes |
| vqc/subsample/pca4 | svm/subsample/pca4 | 0.781 | 0.801 | -0.020 [-0.033, -0.008] | 0 | 0.029 | yes |
| hybrid/subsample/all384 | logreg/subsample/all384 | 0.658 | 0.803 | -0.145 [-0.205, -0.085] | 0 | 0.001 | yes |
| hybrid/subsample/all384 | xgboost/subsample/all384 | 0.658 | 0.802 | -0.144 [-0.200, -0.087] | 0 | 0.001 | yes |
| hybrid/subsample/all384 | mlp/subsample/all384 | 0.658 | 0.815 | -0.157 [-0.213, -0.101] | 0 | 0.000 | yes |
| hybrid/subsample/all384 | svm/subsample/all384 | 0.658 | 0.812 | -0.154 [-0.211, -0.097] | 0 | 0.001 | yes |

## Floor and ceiling (same folds)

- **Length-only floor** (logistic regression on log words + log characters, same 1,200 rows): ROC AUC 0.558 ± 0.012, quarantine 0.463.
- **Classical ceiling, 5,044 rows** (all 384 dims, every development row of the fold's training part): best SVM (RBF) ROC AUC 0.828. This is NOT a full-data ceiling. The cap is not the SVM's n² cost: Phase 1 drew a 15,000-row stratified subsample of the 38,859 deduplicated ID-bearing rows (so that the frozen-encoder feature extraction and the 5x3 grids stay tractable on CPU) and locked 20 % of it as the hold-out, which leaves 6,306 Depression/Suicidal development rows in this view.
- **Classical ceiling, all rows** (Logistic regression, all 384 dims; the fold's development training rows plus the 12,544 deduplicated Depression/Suicidal ID rows outside the Phase 1 subsample, ≈ 17,588 training rows per fold; same validation folds and quarantine): ROC AUC 0.832 ± 0.008, quarantine 0.784. Extra rows were checked against every processed row (exact and TF-IDF cosine ≥ 0.90): 0 exact and 2 near duplicates dropped. This is the true binary ceiling (SVM not re-run at this size).

| Quantum | AUC | vs length floor: diff, Holm p | vs ceiling (dev rows): diff, Holm p | vs ceiling (all rows): diff, Holm p |
|---|---|---|---|---|
| QSVM (angle, 8 qubits) | 0.803 | +0.246, p 0.000 | -0.024, p 0.000 | -0.029, p 0.000 |
| VQC (re-upload, 4 qubits) | 0.781 | +0.223, p 0.000 | -0.047, p 0.000 | -0.051, p 0.000 |
| Hybrid QNN (angle, 8 qubits) | 0.658 | +0.100, p 0.003 | -0.169, p 0.000 | -0.174, p 0.000 |

## All arms

| Model | Regime | Input | ROC AUC | Bal. acc. | Recall | PR AUC | Accuracy | F1 | Quarantine AUC | Tuning s |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP | subsample | all 384 MiniLM dims | 0.815 ± 0.009 | 0.737 | 0.813 | 0.725 | 0.726 | 0.716 | 0.772 | 7 |
| SVM (RBF) | subsample | all 384 MiniLM dims | 0.812 ± 0.010 | 0.736 | 0.801 | 0.719 | 0.727 | 0.713 | 0.768 | 3 |
| Logistic regression | subsample | all 384 MiniLM dims | 0.803 ± 0.013 | 0.728 | 0.799 | 0.716 | 0.718 | 0.706 | 0.759 | 0 |
| XGBoost | subsample | all 384 MiniLM dims | 0.802 ± 0.013 | 0.729 | 0.772 | 0.713 | 0.722 | 0.703 | 0.756 | 91 |
| Hybrid QNN (angle, 8 qubits) | subsample | all 384 MiniLM dims | 0.658 ± 0.045 | 0.610 | 0.679 | 0.575 | 0.600 | 0.588 | 0.616 | 262 |
| Logistic regression | subsample | log words + log characters | 0.558 ± 0.012 | 0.537 | 0.564 | 0.472 | 0.533 | 0.495 | 0.463 | 0 |
| Logistic regression | subsample | PCA-4 of MiniLM | 0.803 ± 0.011 | 0.732 | 0.798 | 0.699 | 0.723 | 0.710 | 0.756 | 0 |
| SVM (RBF) | subsample | PCA-4 of MiniLM | 0.801 ± 0.011 | 0.729 | 0.779 | 0.703 | 0.721 | 0.703 | 0.749 | 0 |
| MLP | subsample | PCA-4 of MiniLM | 0.800 ± 0.014 | 0.728 | 0.796 | 0.700 | 0.718 | 0.706 | 0.761 | 5 |
| XGBoost | subsample | PCA-4 of MiniLM | 0.798 ± 0.011 | 0.726 | 0.758 | 0.708 | 0.722 | 0.698 | 0.752 | 2 |
| VQC (re-upload, 4 qubits) | subsample | PCA-4 of MiniLM | 0.781 ± 0.015 | 0.713 | 0.770 | 0.681 | 0.705 | 0.689 | 0.745 | 113 |
| SVM (RBF) | subsample | PCA-8 of MiniLM | 0.804 ± 0.013 | 0.727 | 0.768 | 0.711 | 0.721 | 0.699 | 0.758 | 0 |
| Logistic regression | subsample | PCA-8 of MiniLM | 0.804 ± 0.010 | 0.731 | 0.808 | 0.703 | 0.719 | 0.710 | 0.768 | 0 |
| QSVM (angle, 8 qubits) | subsample | PCA-8 of MiniLM | 0.803 ± 0.012 | 0.729 | 0.778 | 0.710 | 0.721 | 0.703 | 0.760 | 1 |
| MLP | subsample | PCA-8 of MiniLM | 0.802 ± 0.013 | 0.730 | 0.811 | 0.708 | 0.717 | 0.709 | 0.776 | 5 |
| XGBoost | subsample | PCA-8 of MiniLM | 0.802 ± 0.014 | 0.726 | 0.783 | 0.716 | 0.717 | 0.702 | 0.760 | 3 |
| SVM (RBF) | full | all 384 MiniLM dims | 0.828 ± 0.009 | 0.756 | 0.791 | 0.745 | 0.751 | 0.729 | 0.784 | 33 |
| MLP | full | all 384 MiniLM dims | 0.823 ± 0.009 | 0.750 | 0.791 | 0.743 | 0.744 | 0.724 | 0.775 | 21 |
| Logistic regression | full | all 384 MiniLM dims | 0.820 ± 0.008 | 0.744 | 0.773 | 0.734 | 0.740 | 0.716 | 0.769 | 1 |
| XGBoost | full | all 384 MiniLM dims | 0.818 ± 0.009 | 0.743 | 0.791 | 0.735 | 0.736 | 0.718 | 0.771 | 148 |
| Logistic regression | full | log words + log characters | 0.561 ± 0.013 | 0.544 | 0.491 | 0.476 | 0.552 | 0.477 | 0.464 | 0 |
| XGBoost | full | PCA-4 of MiniLM | 0.806 ± 0.011 | 0.733 | 0.751 | 0.720 | 0.730 | 0.703 | 0.759 | 3 |
| MLP | full | PCA-4 of MiniLM | 0.806 ± 0.010 | 0.733 | 0.764 | 0.712 | 0.728 | 0.705 | 0.764 | 17 |
| Logistic regression | full | PCA-4 of MiniLM | 0.803 ± 0.010 | 0.735 | 0.779 | 0.699 | 0.729 | 0.709 | 0.760 | 0 |
| SVM (RBF) | full | PCA-4 of MiniLM | 0.803 ± 0.010 | 0.735 | 0.761 | 0.702 | 0.732 | 0.707 | 0.756 | 3 |
| XGBoost | full | PCA-8 of MiniLM | 0.809 ± 0.011 | 0.736 | 0.770 | 0.723 | 0.731 | 0.709 | 0.767 | 4 |
| MLP | full | PCA-8 of MiniLM | 0.809 ± 0.012 | 0.733 | 0.758 | 0.718 | 0.729 | 0.704 | 0.774 | 17 |
| SVM (RBF) | full | PCA-8 of MiniLM | 0.808 ± 0.011 | 0.738 | 0.756 | 0.714 | 0.735 | 0.708 | 0.764 | 4 |
| Logistic regression | full | PCA-8 of MiniLM | 0.805 ± 0.010 | 0.730 | 0.757 | 0.703 | 0.727 | 0.701 | 0.771 | 0 |
| Logistic regression | all_rows | all 384 MiniLM dims | 0.832 ± 0.008 | 0.751 | 0.774 | 0.749 | 0.748 | 0.723 | 0.784 | 5 |
| XGBoost | all_rows | all 384 MiniLM dims | 0.829 ± 0.008 | 0.749 | 0.791 | 0.749 | 0.743 | 0.723 | 0.786 | 312 |

## Beside NSCH (numbers read from results/exp2-nsch2024-quantum-vs-classical-20261108-165426, not recomputed)

| | NSCH 2024 (survey, 95 columns) | Mental Health Text binary (MiniLM) |
|---|---|---|
| QSVM (angle, 8 qubits) AUC | 0.847 | 0.803 |
| Logistic regression, same rows + input | 0.851 | 0.804 |
| QSVM vs logistic regression, NB-t Holm p | 0.704 | 1.000 |
| VQC (re-upload, 4 qubits) AUC | 0.814 | 0.781 |
| Logistic regression, same rows + input | 0.837 | 0.803 |
| VQC vs logistic regression, NB-t Holm p | 0.012 | 0.014 |
| Hybrid QNN AUC (all inputs) | 0.784 | 0.658 |
| Classical ceiling, development rows of the fold | 0.888 (every NSCH development row) | 0.828 (5,044 rows) |
| Classical ceiling, all available rows | 0.888 (same: NSCH had no subsample cap) | 0.832 (≈ 17,588 rows) |
| Length-only floor | n/a | 0.558 |

NSCH Holm families had 9 same-rows pairs (3 classical models); this view has 12 (SVM added), so its Holm correction is slightly stricter.

## Deliberately not run: four-class and three-class quantum views

The multiclass quantum views were deliberately not run (decision recorded after the binary result). The binary Depression vs Suicidal view was chosen as the hard, length-free, tangled boundary - the fairest test quantum models could get on this dataset - and the QSVM tied classical there. A multiclass quantum run would be one-vs-rest, repeating that binary result four times and averaging it, and it would require extending the VQC and hybrid QNN beyond the NSCH implementations (three-class VQC, binary-only hybrid), which breaks the 'method held fixed' design that makes the NSCH comparison valid. Multiclass results for this dataset are the Phase 3 classical and sequence baselines (results/mhtext/phase3/).

## Encoding x qubit sweep (exp1 method; 1,200 rows; 2 repeats x 5 folds; ROC AUC, quarantine in brackets)

| Model | Features | Encoding | 4 qubits | 6 qubits | 8 qubits |
|---|---|---|---|---|---|
| Logistic regression (no encoding) | mi | none | 0.701 (0.656) | 0.722 (0.663) | 0.738 (0.685) |
| Logistic regression (no encoding) | pca | none | 0.803 (0.759) | 0.804 (0.768) | 0.804 (0.769) |
| QSVM | mi | amplitude | 0.757 (0.697) | 0.764 (0.712) | 0.750 (0.703) |
| QSVM | mi | angle | 0.702 (0.656) | 0.721 (0.663) | 0.737 (0.688) |
| QSVM | mi | basis | 0.639 (0.601) | 0.666 (0.627) | 0.665 (0.629) |
| QSVM | mi | reupload | 0.697 (0.652) | 0.715 (0.659) | 0.732 (0.682) |
| QSVM | mi | zz | 0.670 (0.627) | 0.688 (0.635) | 0.685 (0.644) |
| QSVM | pca | amplitude | 0.777 (0.746) | 0.742 (0.690) | 0.753 (0.701) |
| QSVM | pca | angle | 0.803 (0.756) | 0.804 (0.761) | 0.804 (0.764) |
| QSVM | pca | basis | 0.735 (0.668) | 0.732 (0.675) | 0.702 (0.656) |
| QSVM | pca | reupload | 0.796 (0.754) | 0.797 (0.751) | 0.797 (0.751) |
| QSVM | pca | zz | 0.758 (0.697) | 0.750 (0.686) | 0.730 (0.683) |
| VQC | mi | amplitude | 0.635 (0.595) | 0.600 (0.579) | 0.560 (0.544) |
| VQC | mi | angle | 0.684 (0.635) | 0.699 (0.649) | 0.718 (0.664) |
| VQC | mi | basis | 0.575 (0.557) | 0.532 (0.527) | 0.516 (0.512) |
| VQC | mi | reupload | 0.683 (0.636) | 0.701 (0.652) | 0.707 (0.652) |
| VQC | mi | zz | 0.631 (0.598) | 0.626 (0.605) | 0.615 (0.597) |
| VQC | pca | amplitude | 0.604 (0.574) | 0.595 (0.566) | 0.581 (0.555) |
| VQC | pca | angle | 0.784 (0.754) | 0.784 (0.769) | 0.782 (0.764) |
| VQC | pca | basis | 0.553 (0.538) | 0.513 (0.517) | 0.509 (0.508) |
| VQC | pca | reupload | 0.779 (0.741) | 0.778 (0.737) | 0.775 (0.744) |
| VQC | pca | zz | 0.678 (0.626) | 0.670 (0.609) | 0.617 (0.585) |

Every row reports ROC AUC beside balanced accuracy and recall in the CSVs; accuracy is never reported alone.
Quantum results are simulated (statevector), not hardware; quantum models are not a privacy mechanism.
