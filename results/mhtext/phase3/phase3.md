# Mental Health Text — Phase 3 (box 10): baselines

Same repeated stratified folds for every model within a view (5 x 3 on development rows); hyperparameters and early stopping on an inner split of each fold's training rows; scalers fitted on inner-train rows only. The locked hold-out is not used. Mean ± SD over folds.
Views: 4-class: 11,992 development rows, 15 folds; 3-class: 7,203 development rows, 15 folds.
Accuracy is shown only beside recall, F1 and ROC-AUC. Labels whose source posts were filed under two classes were dropped in Phase 1; that soft Depression/Suicidal boundary caps the achievable score between those two classes, so a middling per-class score there is not by itself a modelling failure.

## 4-class: macro metrics (cross-validation), sorted by macro ROC-AUC

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Macro ROC-AUC |
|---|---|---|---|---|---|
| XGBoost, all pooled features | 0.816 ± 0.007 | 0.809 ± 0.007 | 0.796 ± 0.011 | 0.802 ± 0.008 | 0.952 ± 0.002 |
| MLP, all pooled features | 0.810 ± 0.006 | 0.787 ± 0.010 | 0.807 ± 0.010 | 0.793 ± 0.008 | 0.951 ± 0.002 |
| 1-D CNN on token sequences | 0.802 ± 0.009 | 0.788 ± 0.014 | 0.791 ± 0.018 | 0.785 ± 0.010 | 0.950 ± 0.003 |
| BiLSTM on token sequences | 0.795 ± 0.012 | 0.791 ± 0.013 | 0.786 ± 0.014 | 0.777 ± 0.014 | 0.949 ± 0.003 |
| LogReg, all pooled features | 0.797 ± 0.008 | 0.771 ± 0.007 | 0.792 ± 0.009 | 0.780 ± 0.007 | 0.945 ± 0.002 |
| SVM (RBF), all pooled features | 0.820 ± 0.007 | 0.811 ± 0.008 | 0.807 ± 0.011 | 0.808 ± 0.008 | 0.941 ± 0.003 |
| Frozen MiniLM sentence encoder (BERT family), whole text + LogReg | 0.787 ± 0.006 | 0.751 ± 0.006 | 0.785 ± 0.005 | 0.764 ± 0.005 | 0.941 ± 0.002 |
| Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg | 0.777 ± 0.009 | 0.743 ± 0.011 | 0.775 ± 0.010 | 0.756 ± 0.010 | 0.936 ± 0.002 |
| Length only (LogReg on log words, log characters of the text) | 0.548 ± 0.012 | 0.473 ± 0.009 | 0.482 ± 0.012 | 0.460 ± 0.010 | 0.776 ± 0.007 |

### 4-class: per class (recall / F1 / one-vs-rest ROC-AUC, means)

| Model | Normal | Anxiety | Depression | Suicidal |
|---|---|---|---|---|
| XGBoost, all pooled features | 0.957 / 0.936 / 0.988 | 0.819 / 0.847 / 0.987 | 0.737 / 0.747 / 0.921 | 0.672 / 0.679 / 0.913 |
| MLP, all pooled features | 0.935 / 0.936 / 0.987 | 0.872 / 0.816 / 0.982 | 0.675 / 0.726 / 0.921 | 0.748 / 0.695 / 0.914 |
| 1-D CNN on token sequences | 0.946 / 0.931 / 0.986 | 0.848 / 0.824 / 0.984 | 0.693 / 0.718 / 0.915 | 0.677 / 0.666 / 0.914 |
| BiLSTM on token sequences | 0.959 / 0.930 / 0.986 | 0.818 / 0.813 / 0.981 | 0.577 / 0.671 / 0.914 | 0.788 / 0.693 / 0.913 |
| LogReg, all pooled features | 0.928 / 0.931 / 0.986 | 0.864 / 0.809 / 0.981 | 0.682 / 0.713 / 0.908 | 0.695 / 0.666 / 0.905 |
| SVM (RBF), all pooled features | 0.958 / 0.938 / 0.984 | 0.844 / 0.853 / 0.975 | 0.705 / 0.742 / 0.905 | 0.721 / 0.698 / 0.899 |
| Frozen MiniLM sentence encoder (BERT family), whole text + LogReg | 0.920 / 0.921 / 0.981 | 0.858 / 0.762 / 0.976 | 0.657 / 0.701 / 0.902 | 0.704 / 0.674 / 0.904 |
| Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg | 0.911 / 0.919 / 0.981 | 0.852 / 0.764 / 0.973 | 0.641 / 0.686 / 0.896 | 0.697 / 0.654 / 0.894 |
| Length only (LogReg on log words, log characters of the text) | 0.807 / 0.801 / 0.903 | 0.401 / 0.228 / 0.715 | 0.441 / 0.478 / 0.765 | 0.280 / 0.334 / 0.724 |

## 3-class: macro metrics (cross-validation), sorted by macro ROC-AUC

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Macro ROC-AUC |
|---|---|---|---|---|---|
| XGBoost, all pooled features | 0.758 ± 0.013 | 0.794 ± 0.015 | 0.781 ± 0.013 | 0.787 ± 0.013 | 0.901 ± 0.007 |
| MLP, all pooled features | 0.751 ± 0.014 | 0.766 ± 0.017 | 0.788 ± 0.012 | 0.774 ± 0.014 | 0.899 ± 0.008 |
| 1-D CNN on token sequences | 0.727 ± 0.021 | 0.764 ± 0.018 | 0.778 ± 0.018 | 0.756 ± 0.021 | 0.896 ± 0.008 |
| BiLSTM on token sequences | 0.730 ± 0.018 | 0.758 ± 0.015 | 0.781 ± 0.017 | 0.757 ± 0.018 | 0.894 ± 0.008 |
| SVM (RBF), all pooled features | 0.764 ± 0.011 | 0.795 ± 0.012 | 0.790 ± 0.012 | 0.790 ± 0.011 | 0.893 ± 0.007 |
| LogReg, all pooled features | 0.739 ± 0.011 | 0.754 ± 0.014 | 0.775 ± 0.010 | 0.763 ± 0.011 | 0.888 ± 0.007 |
| Frozen MiniLM sentence encoder (BERT family), whole text + LogReg | 0.732 ± 0.011 | 0.736 ± 0.014 | 0.772 ± 0.009 | 0.750 ± 0.011 | 0.886 ± 0.006 |
| Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg | 0.720 ± 0.014 | 0.729 ± 0.015 | 0.759 ± 0.012 | 0.741 ± 0.014 | 0.873 ± 0.007 |
| Length only (LogReg on log words, log characters of the text) | 0.421 ± 0.015 | 0.409 ± 0.016 | 0.450 ± 0.018 | 0.403 ± 0.014 | 0.617 ± 0.010 |

### 3-class: per class (recall / F1 / one-vs-rest ROC-AUC, means)

| Model | Anxiety | Depression | Suicidal |
|---|---|---|---|
| XGBoost, all pooled features | 0.861 / 0.887 / 0.989 | 0.757 / 0.764 / 0.854 | 0.726 / 0.710 / 0.860 |
| MLP, all pooled features | 0.894 / 0.854 / 0.984 | 0.708 / 0.749 / 0.851 | 0.762 / 0.719 / 0.861 |
| 1-D CNN on token sequences | 0.903 / 0.860 / 0.990 | 0.625 / 0.700 / 0.842 | 0.807 / 0.707 / 0.857 |
| BiLSTM on token sequences | 0.897 / 0.846 / 0.987 | 0.611 / 0.701 / 0.837 | 0.835 / 0.723 / 0.857 |
| SVM (RBF), all pooled features | 0.863 / 0.879 / 0.984 | 0.730 / 0.762 / 0.840 | 0.776 / 0.730 / 0.856 |
| LogReg, all pooled features | 0.887 / 0.849 / 0.982 | 0.707 / 0.740 / 0.835 | 0.732 / 0.700 / 0.848 |
| Frozen MiniLM sentence encoder (BERT family), whole text + LogReg | 0.887 / 0.814 / 0.980 | 0.686 / 0.730 / 0.829 | 0.743 / 0.707 / 0.848 |
| Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg | 0.872 / 0.815 / 0.973 | 0.673 / 0.717 / 0.814 | 0.732 / 0.690 / 0.834 |
| Length only (LogReg on log words, log characters of the text) | 0.548 / 0.322 / 0.716 | 0.409 / 0.466 / 0.556 | 0.393 / 0.421 / 0.579 |

## Each model against the length-only baseline (NB-t, Holm within view x metric)

| View | Metric | Model | Model | Length only | Difference [95% CI] | Holm p | Significant |
|---|---|---|---|---|---|---|---|
| 3-class | macro_auc | XGBoost, all pooled features | 0.901 | 0.617 | +0.284 [+0.271, +0.297] | 5.21e-16 | yes |
| 3-class | macro_auc | MLP, all pooled features | 0.899 | 0.617 | +0.282 [+0.268, +0.296] | 1.33e-15 | yes |
| 3-class | macro_auc | 1-D CNN on token sequences | 0.896 | 0.617 | +0.279 [+0.263, +0.295] | 3.35e-15 | yes |
| 3-class | macro_auc | BiLSTM on token sequences | 0.894 | 0.617 | +0.277 [+0.259, +0.294] | 7.95e-15 | yes |
| 3-class | macro_auc | SVM (RBF), all pooled features | 0.893 | 0.617 | +0.276 [+0.263, +0.290] | 1.39e-15 | yes |
| 3-class | macro_auc | LogReg, all pooled features | 0.888 | 0.617 | +0.271 [+0.258, +0.285] | 1.69e-15 | yes |
| 3-class | macro_auc | Frozen MiniLM sentence encoder (BERT family), whole text + LogReg | 0.886 | 0.617 | +0.269 [+0.255, +0.283] | 2.17e-15 | yes |
| 3-class | macro_auc | Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg | 0.873 | 0.617 | +0.256 [+0.242, +0.271] | 3.28e-15 | yes |
| 3-class | macro_f1 | SVM (RBF), all pooled features | 0.790 | 0.403 | +0.387 [+0.366, +0.408] | 4.96e-15 | yes |
| 3-class | macro_f1 | XGBoost, all pooled features | 0.787 | 0.403 | +0.383 [+0.364, +0.403] | 2.62e-15 | yes |
| 3-class | macro_f1 | MLP, all pooled features | 0.774 | 0.403 | +0.370 [+0.349, +0.391] | 9.29e-15 | yes |
| 3-class | macro_f1 | LogReg, all pooled features | 0.763 | 0.403 | +0.359 [+0.340, +0.379] | 4.96e-15 | yes |
| 3-class | macro_f1 | BiLSTM on token sequences | 0.757 | 0.403 | +0.353 [+0.319, +0.387] | 2.42e-12 | yes |
| 3-class | macro_f1 | 1-D CNN on token sequences | 0.756 | 0.403 | +0.352 [+0.320, +0.384] | 2.05e-12 | yes |
| 3-class | macro_f1 | Frozen MiniLM sentence encoder (BERT family), whole text + LogReg | 0.750 | 0.403 | +0.347 [+0.326, +0.368] | 1.81e-14 | yes |
| 3-class | macro_f1 | Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg | 0.741 | 0.403 | +0.337 [+0.316, +0.359] | 3.18e-14 | yes |
| 4-class | macro_auc | XGBoost, all pooled features | 0.952 | 0.776 | +0.176 [+0.168, +0.184] | 3.55e-16 | yes |
| 4-class | macro_auc | MLP, all pooled features | 0.951 | 0.776 | +0.175 [+0.166, +0.183] | 6.59e-16 | yes |
| 4-class | macro_auc | 1-D CNN on token sequences | 0.950 | 0.776 | +0.174 [+0.167, +0.180] | 3.34e-17 | yes |
| 4-class | macro_auc | BiLSTM on token sequences | 0.949 | 0.776 | +0.172 [+0.164, +0.180] | 4.72e-16 | yes |
| 4-class | macro_auc | LogReg, all pooled features | 0.945 | 0.776 | +0.169 [+0.161, +0.177] | 6.59e-16 | yes |
| 4-class | macro_auc | SVM (RBF), all pooled features | 0.941 | 0.776 | +0.164 [+0.155, +0.173] | 9.4e-16 | yes |
| 4-class | macro_auc | Frozen MiniLM sentence encoder (BERT family), whole text + LogReg | 0.941 | 0.776 | +0.164 [+0.156, +0.172] | 6.59e-16 | yes |
| 4-class | macro_auc | Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg | 0.936 | 0.776 | +0.160 [+0.152, +0.167] | 6.59e-16 | yes |
| 4-class | macro_f1 | SVM (RBF), all pooled features | 0.808 | 0.460 | +0.347 [+0.336, +0.359] | 7.02e-18 | yes |
| 4-class | macro_f1 | XGBoost, all pooled features | 0.802 | 0.460 | +0.342 [+0.330, +0.353] | 7.02e-18 | yes |
| 4-class | macro_f1 | MLP, all pooled features | 0.793 | 0.460 | +0.333 [+0.317, +0.349] | 5.68e-16 | yes |
| 4-class | macro_f1 | 1-D CNN on token sequences | 0.785 | 0.460 | +0.324 [+0.307, +0.342] | 2.14e-15 | yes |
| 4-class | macro_f1 | LogReg, all pooled features | 0.780 | 0.460 | +0.320 [+0.309, +0.330] | 4.79e-18 | yes |
| 4-class | macro_f1 | BiLSTM on token sequences | 0.777 | 0.460 | +0.317 [+0.290, +0.343] | 3.19e-13 | yes |
| 4-class | macro_f1 | Frozen MiniLM sentence encoder (BERT family), whole text + LogReg | 0.764 | 0.460 | +0.304 [+0.292, +0.316] | 3.54e-17 | yes |
| 4-class | macro_f1 | Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg | 0.756 | 0.460 | +0.295 [+0.282, +0.309] | 3.24e-16 | yes |

## Four-class against three-class (the length-confound sensitivity run)

| Model | 4-class macro AUC | 3-class macro AUC | 4-class macro F1 | 3-class macro F1 |
|---|---|---|---|---|
| XGBoost, all pooled features | 0.952 | 0.901 | 0.802 | 0.787 |
| MLP, all pooled features | 0.951 | 0.899 | 0.793 | 0.774 |
| 1-D CNN on token sequences | 0.950 | 0.896 | 0.785 | 0.756 |
| BiLSTM on token sequences | 0.949 | 0.894 | 0.777 | 0.757 |
| LogReg, all pooled features | 0.945 | 0.888 | 0.780 | 0.763 |
| SVM (RBF), all pooled features | 0.941 | 0.893 | 0.808 | 0.790 |
| Frozen MiniLM sentence encoder (BERT family), whole text + LogReg | 0.941 | 0.886 | 0.764 | 0.750 |
| Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg | 0.936 | 0.873 | 0.756 | 0.741 |
| Length only (LogReg on log words, log characters of the text) | 0.776 | 0.617 | 0.460 | 0.403 |

Chance level for macro one-vs-rest ROC-AUC is 0.5 in both views. Macro F1 is not comparable across views at face value (four classes vs three, different class shares).

## Sensitivity: the quarantined 9,582-row block (different provenance), scored by every fold's model

| View | Model | Macro AUC, cross-validation | Macro AUC, quarantined block | Macro F1, quarantined block |
|---|---|---|---|---|
| 3-class | SVM (RBF), all pooled features | 0.893 | 0.852 | 0.666 |
| 3-class | MLP, all pooled features | 0.899 | 0.845 | 0.663 |
| 3-class | LogReg, all pooled features | 0.888 | 0.838 | 0.660 |
| 3-class | Frozen MiniLM sentence encoder (BERT family), whole text + LogReg | 0.886 | 0.835 | 0.656 |
| 3-class | XGBoost, all pooled features | 0.901 | 0.828 | 0.616 |
| 3-class | Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg | 0.873 | 0.806 | 0.623 |
| 3-class | 1-D CNN on token sequences | 0.896 | 0.700 | 0.398 |
| 3-class | BiLSTM on token sequences | 0.894 | 0.678 | 0.397 |
| 3-class | Length only (LogReg on log words, log characters of the text) | 0.617 | 0.515 | 0.305 |
| 4-class | SVM (RBF), all pooled features | 0.941 | 0.872 | 0.665 |
| 4-class | Frozen MiniLM sentence encoder (BERT family), whole text + LogReg | 0.941 | 0.865 | 0.644 |
| 4-class | MLP, all pooled features | 0.951 | 0.864 | 0.636 |
| 4-class | LogReg, all pooled features | 0.945 | 0.859 | 0.634 |
| 4-class | XGBoost, all pooled features | 0.952 | 0.856 | 0.612 |
| 4-class | Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg | 0.936 | 0.843 | 0.605 |
| 4-class | BiLSTM on token sequences | 0.949 | 0.760 | 0.468 |
| 4-class | 1-D CNN on token sequences | 0.950 | 0.750 | 0.443 |
| 4-class | Length only (LogReg on log words, log characters of the text) | 0.776 | 0.641 | 0.312 |

The block has no shipped numeric features (no Unique_ID rows); for models that use them they enter as the training mean after scaling. Its Normal posts are longer (median 31 words vs 10).

Run time 175 min. Files: fold_metrics.csv, summary.csv, tests_vs_length_only.csv, run.json.
