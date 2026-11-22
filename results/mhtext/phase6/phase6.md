# Phase 6 — SIMULATED federated learning on Mental Health Text (binary view), beside NSCH

Box 13. Clients are SIMULATED: partitions of one dataset on one machine, not separate institutions. exp4 method unchanged (FedAvg, 30 rounds x 1 local epoch, client-side DP-SGD at ε = 4, δ = 1e-5; arms VQC re-upload 4 qubits on 1,200 rows / PCA-4 and LogReg on every development training row / all 384 MiniLM dims), read from configs/exp4_nsch2024.yaml. 5x2 folds = the Phase 5 DP-SGD folds; centralised references are those Phase 5 runs (as exp4 used exp3). Partitions: iid (as NSCH) and by_length (non-IID substitute for NSCH's by_state: whole post-length deciles per client; the posts carry no site). NSCH numbers read from results/exp4-nsch2024-federated-dp-20261108-170320/summary.csv and its fold_metrics.csv; no NSCH model was run.

## ROC AUC by number of clients (ε = 4)

| Arm | Partition | Centralised | 3 clients | 5 clients | 10 clients | NSCH partition | NSCH centralised | NSCH 3 | NSCH 5 | NSCH 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| vqc_sub | iid | 0.771 | 0.755 | 0.747 | 0.724 | iid | 0.805 | 0.748 | 0.709 | 0.657 |
| vqc_sub | by_length | 0.771 | 0.761 | 0.747 | 0.725 | by_state | 0.805 | 0.759 | 0.706 | 0.658 |
| logreg_full | iid | 0.772 | 0.781 | 0.780 | 0.786 | iid | 0.856 | 0.861 | 0.859 | 0.860 |
| logreg_full | by_length | 0.772 | 0.780 | 0.779 | 0.784 | by_state | 0.856 | 0.860 | 0.861 | 0.858 |

## Federated vs centralised DP-SGD at the same ε (NB-t, Holm within arm)

| Arm | Partition | Clients | Federated | Centralised DP | Diff | Holm p |
|---|---|---|---|---|---|---|
| vqc_sub | iid | 3 | 0.755 | 0.771 | -0.016 | 0.322 |
| vqc_sub | iid | 5 | 0.747 | 0.771 | -0.024 | 0.251 |
| vqc_sub | iid | 10 | 0.724 | 0.771 | -0.047 | 0.002 |
| vqc_sub | by_length | 3 | 0.761 | 0.771 | -0.010 | 0.322 |
| vqc_sub | by_length | 5 | 0.747 | 0.771 | -0.023 | 0.322 |
| vqc_sub | by_length | 10 | 0.725 | 0.771 | -0.045 | 0.002 |
| logreg_full | iid | 3 | 0.781 | 0.772 | +0.008 | 1.000 |
| logreg_full | iid | 5 | 0.780 | 0.772 | +0.008 | 1.000 |
| logreg_full | iid | 10 | 0.786 | 0.772 | +0.013 | 0.139 |
| logreg_full | by_length | 3 | 0.780 | 0.772 | +0.007 | 1.000 |
| logreg_full | by_length | 5 | 0.779 | 0.772 | +0.007 | 1.000 |
| logreg_full | by_length | 10 | 0.784 | 0.772 | +0.012 | 0.119 |

## Replication test: does the federated VQC fall with more clients while LogReg holds flat?

Rule declared before the run (module docstring): primary partition iid. Replicates if (a) the VQC's AUC change from 3 to 10 clients is a significant drop, (b) LogReg's change is not significant or below 0.01, and (c) the VQC's drop is significantly larger than LogReg's (paired over folds). NB-t, Holm within each block. The NSCH rows apply the same statistics to its stored fold metrics.

| Dataset | Partition | Quantity | Mean change in AUC [95% CI] | Holm p |
|---|---|---|---|---|
| Mental Health Text | iid | VQC: 10 clients minus 3 | -0.031 [-0.045, -0.016] | 0.003 |
| Mental Health Text | iid | LogReg: 10 clients minus 3 | +0.005 [-0.010, +0.020] | 0.467 |
| Mental Health Text | iid | VQC change minus LogReg change | -0.036 [-0.054, -0.017] | 0.003 |
| Mental Health Text | by_length | VQC: 10 clients minus 3 | -0.036 [-0.054, -0.017] | 0.006 |
| Mental Health Text | by_length | LogReg: 10 clients minus 3 | +0.005 [-0.012, +0.021] | 0.520 |
| Mental Health Text | by_length | VQC change minus LogReg change | -0.040 [-0.065, -0.016] | 0.009 |
| NSCH 2024 | iid | VQC: 10 clients minus 3 | -0.091 [-0.166, -0.016] | 0.046 |
| NSCH 2024 | iid | LogReg: 10 clients minus 3 | -0.001 [-0.016, +0.013] | 0.842 |
| NSCH 2024 | iid | VQC change minus LogReg change | -0.090 [-0.158, -0.022] | 0.046 |
| NSCH 2024 | by_state | VQC: 10 clients minus 3 | -0.101 [-0.182, -0.020] | 0.040 |
| NSCH 2024 | by_state | LogReg: 10 clients minus 3 | -0.002 [-0.022, +0.019] | 0.870 |
| NSCH 2024 | by_state | VQC change minus LogReg change | -0.099 [-0.172, -0.027] | 0.037 |

- **Mental Health Text, iid: pattern present** — (a) VQC drop significant: yes; (b) LogReg flat: yes; (c) VQC drop larger than LogReg's: yes.
- **Mental Health Text, by_length: pattern present** — (a) VQC drop significant: yes; (b) LogReg flat: yes; (c) VQC drop larger than LogReg's: yes.
- **NSCH 2024, iid: pattern present** — (a) VQC drop significant: yes; (b) LogReg flat: yes; (c) VQC drop larger than LogReg's: yes.
- **NSCH 2024, by_state: pattern present** — (a) VQC drop significant: yes; (b) LogReg flat: yes; (c) VQC drop larger than LogReg's: yes.

**Verdict (primary, iid on both datasets): REPLICATES.**

Communication per task is in fold_metrics.csv (bytes_per_round, total_bytes); convergence per round in round_history.csv (logged for plots only, never used for a choice).
