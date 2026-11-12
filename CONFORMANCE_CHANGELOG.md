# CONFORMANCE.md change log

Revision history of `CONFORMANCE.md` (the printable document carries no internal notes; the full diff history is in git).

- 2026-11-12: baseline written before the conformance sprint.
- 2026-11-12: Priority 1 — input-stage privacy (k-anonymity, local DP) built and measured (Experiment 5); flow section rewritten.
- 2026-11-12: Priority 2 item 2 — re-identification risk built and measured.
- 2026-11-12: Priority 2 item 1 — membership inference test built and measured (Experiment 6).
- 2026-11-12: Priority 2 items 4 and 5 — classical SVM baseline and COBYLA optimiser built and measured on the Exp 2 folds.
- 2026-11-12: Priority 2 item 3 — basis encoding built and swept on the Exp 1 folds; merged encoding report written.
- 2026-11-12: Priority 2 item 6 — measured payload, client time, server aggregation time and rounds to converge (Experiment 7).
- 2026-11-12: Priority 3 item 7 — best configuration re-run under a simulated hardware noise model (Experiment 8).
- 2026-11-12: ZZ feature-map QSVM under the same noise model (Experiment 8b, 4 qubits): angle robust, ZZ degrades; VQC under noise marked PHASE-2.
- 2026-11-12: Priority 3 item 8 — three-class task (none / anxiety only / depression), classical, QSVM and one-vs-rest VQC (Experiments 9, 9b).
- 2026-11-12: Priority 3 item 9 — secure aggregation (pairwise DH masks) built, wired into FedAvg, verified identical results (Experiment 7b), overhead measured.
- 2026-11-12: Priority 4 item 12 — independent Qiskit VQC forward pass; identical to PennyLane on all 15 Exp 2 folds.
- 2026-11-12: Secure-aggregation overflow: two's-complement round-trip tests, declared per-entry bounds enforced client-side, capacity check with an 8-bit margin; Exp 7 worst case measured (Experiment 7c) at 20.6 bits of headroom.
- 2026-11-12: Multi-class hold-out scored once under the pre-specified rule (configurations committed before scoring); every CV mean inside its hold-out interval.
- 2026-11-12: Pre-specified multi-class hold-out rule committed (three-class metrics exclude the 24 undetermined hold-out rows; binary depression metrics use all rows). No hold-out evaluation run.
- 2026-11-12: Primary significance test switched to the Nadeau–Bengio corrected resampled t (Holm); Wilcoxon-Holm kept as a sensitivity check. All reports regenerated (originals kept as `*.superseded-sig-20261112.*`); 542 comparisons audited, 114 decisions differ between the tests (`results/significance/changed.md`); claims in this document updated; load-bearing conclusions unchanged.
- 2026-11-12: Corrections after review: noise statements scoped to angle vs ZZ; 4-qubit caveat; 6-qubit timing figures marked as estimates; multi-class hold-out handling and macro-F1 for all rows; secure-aggregation exactness and wording; UMAP/AE per-fold fitting documented and tested.
- 2026-11-12: Priority 3 item 10 — Bloch-sphere state visualisation of real encoded records (angle vs ZZ).
- 2026-11-12: Priority 3 item 11 — UMAP and autoencoder reduction through the Experiment 2 comparison (Experiment 10).
