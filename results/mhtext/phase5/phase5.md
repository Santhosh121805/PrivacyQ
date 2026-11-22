# Phase 5 — privacy on Mental Health Text (binary Depression vs Suicidal view), beside NSCH

Boxes 4 (privacy-preserving mechanisms) and 12 (privacy evaluation). Source REAL:MHText. Every model-based privacy parameter is read from the NSCH experiment configs (exp3, exp5, exp6) and the NSCH numbers are read from its results/ files; no NSCH experiment was run. Inputs: frozen MiniLM sentence encoder (BERT family) embeddings. The locked hold-out is not used. Quantum circuits are simulated; a quantum model is not a privacy mechanism.

## Summary

- **The 8 inputs fed to the quantum circuit (step 5) still identify the post 32.6 % of the time** from a 12-word fragment, when the PCA projection is known. 8 PCA components are not anonymous. Keeping the projection secret would lower linkage, but secrecy of a projection is not a privacy guarantee and is not claimed.
- **A post is far more identifying than a survey row:** a 5-word fragment singles out 95.0 % of posts, against 62.7 % of NSCH adolescents on 7 demographics + state (raw) and 0.0 % on the 7 demographics after k-anonymity (k = 5). Scrubbing identifiers leaves it at 95.2 %; k-anonymity has no columns to act on.
- **The small quantum models leak little, and that comes from their low capacity, not from anything quantum.** Membership-attack AUC: QSVM 0.508, VQC 0.502; the overfitted MLP control reaches 0.732 (NSCH 0.558).
- **Local DP at ε = 32 defeats the linkage attack we implemented, at a QSVM cost of 0.007 AUC. The formal guarantee at this budget is weak and must not be read as a general privacy claim.**
- DP-SGD at ε = 0.5 costs the same-rows LogReg 0.019 AUC on text against 0.195 on NSCH, about tenfold less; the cause is not established.

## 1. Re-identification risk (box 12) beside NSCH

Same prosecutor-risk scale as the NSCH table (results/reidentification/risk.csv): the attacker knows something about a person and that they are in the released table; risk = 1 / (rows that match what the attacker knows). NSCH: the attacker knows quasi-identifier values. Text: the attacker knows a fragment of the person's post (k consecutive words, one random fragment per post, seed 2026). Development rows only; the locked hold-out is not read (as for NSCH).

| Released table | What the attacker knows | Rows | Unique (risk = 1) | Rows violating k = 5 | Average risk | Maximum risk |
|---|---|---|---|---|---|---|
| NSCH 2024 survey | 10 demographic/household QIs, raw values | 17,431 | 49.3 % | 61.6 % | 0.561 | 1.000 |
| NSCH 2024 survey | 7 demographic QIs, raw values | 17,431 | 36.8 % | 51.6 % | 0.446 | 1.000 |
| NSCH 2024 survey | 7 demographic QIs, raw values + U.S. state | 17,431 | 62.7 % | 74.6 % | 0.703 | 1.000 |
| NSCH 2024 survey | 7 demographic QIs, generalised (age and poverty bands) | 17,431 | 3.2 % | 9.9 % | 0.083 | 1.000 |
| NSCH 2024 survey | 7 demographic QIs, generalised + state | 17,431 | 32.2 % | 57.7 % | 0.462 | 1.000 |
| NSCH 2024 survey | 7 demographic QIs after k-anonymity (k = 5, as used in the pipeline) | 17,431 | 0.0 % | 0.0 % | 0.026 | 0.200 |
| NSCH 2024 survey | 7 demographic QIs after k-anonymity + state | 17,431 | 22.9 % | 47.8 % | 0.369 | 1.000 |
| Mental Health Text posts, as released | a 3-word fragment | 11,992 | 66.3 % | 80.1 % | 0.734 | 1.000 |
| Mental Health Text posts, as released | a 5-word fragment | 11,992 | 95.0 % | 98.3 % | 0.966 | 1.000 |
| Mental Health Text posts, as released | a 8-word fragment | 11,992 | 99.3 % | 99.9 % | 0.996 | 1.000 |
| Mental Health Text posts, as released | a 12-word fragment | 11,992 | 99.5 % | 99.9 % | 0.997 | 1.000 |
| Mental Health Text posts, as released | a 20-word fragment | 11,992 | 99.5 % | 99.9 % | 0.997 | 1.000 |
| Mental Health Text posts, scrubbed (URLs, e-mails, handles, communities, numbers removed) | a 3-word fragment | 11,992 | 65.6 % | 79.5 % | 0.728 | 1.000 |
| Mental Health Text posts, scrubbed (URLs, e-mails, handles, communities, numbers removed) | a 5-word fragment | 11,992 | 95.2 % | 98.5 % | 0.968 | 1.000 |
| Mental Health Text posts, scrubbed (URLs, e-mails, handles, communities, numbers removed) | a 8-word fragment | 11,992 | 99.2 % | 99.8 % | 0.995 | 1.000 |
| Mental Health Text posts, scrubbed (URLs, e-mails, handles, communities, numbers removed) | a 12-word fragment | 11,992 | 99.4 % | 99.9 % | 0.996 | 1.000 |
| Mental Health Text posts, scrubbed (URLs, e-mails, handles, communities, numbers removed) | a 20-word fragment | 11,992 | 99.5 % | 99.9 % | 0.996 | 1.000 |

Share of posts that contain at least one span found in no other post (an attacker who knows any such span singles the post out with certainty):

| Corpus | Text | Span | Posts | Posts with a unique span |
|---|---|---|---|---|
| dev | as released | any 5-word span | 11,992 | 93.7 % |
| dev | as released | any 8-word span | 11,992 | 84.2 % |
| dev | scrubbed (URLs, e-mails, handles, communities, numbers removed) | any 5-word span | 11,992 | 93.8 % |
| dev | scrubbed (URLs, e-mails, handles, communities, numbers removed) | any 8-word span | 11,992 | 84.4 % |
| dev+quarantine | as released | any 5-word span | 21,574 | 96.4 % |
| dev+quarantine | as released | any 8-word span | 21,574 | 90.8 % |
| dev+quarantine | scrubbed (URLs, e-mails, handles, communities, numbers removed) | any 5-word span | 21,574 | 96.4 % |
| dev+quarantine | scrubbed (URLs, e-mails, handles, communities, numbers removed) | any 8-word span | 21,574 | 91.0 % |

Sensitivity, larger corpus (development rows + quarantined block, 21,574 posts, as released): uniqueness 3-word fragment 58.6 %, 5-word fragment 94.2 %, 8-word fragment 99.4 %, 12-word fragment 99.6 %, 20-word fragment 99.6 %.

### Quasi-identifiers written into the text

NSCH's quasi-identifiers (age, sex, state) are columns that can be generalised or suppressed. In free text the author writes them into the post. Regex counts on the development posts (a lower bound: patterns such as '(16F)', 'I am 15', '17 years old', a U.S. state name; counted, never read or printed):

| Disclosure | Posts |
|---|---|
| States an age | 3.2 % |
| States an age of 12-17 | 0.9 % |
| States sex/gender | 0.6 % |
| Names a U.S. state | 0.2 % |
| Mentions a school level | 0.9 % |
| Age and sex | 0.5 % |
| Age, sex and state | 0.0 % |
| Any of age, sex, state | 3.5 % |

### Linking a fragment to the released vectors

If only model inputs are released (not text), an attacker embeds the known fragment with the public MiniLM model and looks for the nearest released vector. Rank 1 = the post is re-identified. Released vectors: the 384-dim MiniLM embedding, or the 8 PCA encoder inputs the quantum circuit receives (worst case: the PCA projection is public), with Laplace local DP on the released encoder inputs at the exp5 budgets.

| Released | Fragment words | ε_input | Linked (top-1) | Top-10 | Median rank | Chance (top-1) |
|---|---|---|---|---|---|---|
| MiniLM embedding (384 dims) | 8 | ∞ | 49.1 % | 59.6 % | 2 | 0.00008 |
| PCA-8 encoder input | 8 | ∞ | 23.1 % | 28.6 % | 662 | 0.00008 |
| PCA-8 encoder input | 8 | 32 | 0.0 % | 0.3 % | 3,941 | 0.00008 |
| PCA-8 encoder input | 8 | 16 | 0.0 % | 0.1 % | 4,806 | 0.00008 |
| PCA-8 encoder input | 8 | 8 | 0.0 % | 0.1 % | 5,283 | 0.00008 |
| PCA-8 encoder input | 8 | 4 | 0.0 % | 0.1 % | 5,584 | 0.00008 |
| MiniLM embedding (384 dims) | 12 | ∞ | 59.8 % | 70.6 % | 1 | 0.00008 |
| PCA-8 encoder input | 12 | ∞ | 32.6 % | 38.3 % | 155 | 0.00008 |
| PCA-8 encoder input | 12 | 32 | 0.0 % | 0.4 % | 3,527 | 0.00008 |
| PCA-8 encoder input | 12 | 16 | 0.0 % | 0.1 % | 4,498 | 0.00008 |
| PCA-8 encoder input | 12 | 8 | 0.0 % | 0.1 % | 5,141 | 0.00008 |
| PCA-8 encoder input | 12 | 4 | 0.0 % | 0.1 % | 5,534 | 0.00008 |
| MiniLM embedding (384 dims) | 20 | ∞ | 72.3 % | 82.5 % | 1 | 0.00008 |
| PCA-8 encoder input | 20 | ∞ | 45.1 % | 50.9 % | 8 | 0.00008 |
| PCA-8 encoder input | 20 | 32 | 0.0 % | 0.5 % | 3,072 | 0.00008 |
| PCA-8 encoder input | 20 | 16 | 0.0 % | 0.1 % | 4,172 | 0.00008 |
| PCA-8 encoder input | 20 | 8 | 0.0 % | 0.1 % | 4,945 | 0.00008 |
| PCA-8 encoder input | 20 | 4 | 0.0 % | 0.1 % | 5,356 | 0.00008 |

## 2. Membership inference (box 12) beside NSCH

exp6 method unchanged: loss-threshold attack on equal numbers of members (the model's training rows) and non-members (unseen validation rows), 5 folds; attack ROC AUC 0.5 = no membership signal; mlp_overfit is the positive control (no early stopping, 200 epochs). Text inputs: all 384 MiniLM dims (LogReg, MLP), PCA-4 (VQC), PCA-8 (QSVM).

| Arm | ε | Text: attack AUC | Text: TPR @ 1 % FPR | Text: advantage | Text members | NSCH: attack AUC | NSCH: advantage |
|---|---|---|---|---|---|---|---|
| dp_logreg_full | 0.5 | 0.511 ± 0.007 | 0.010 | 0.034 | 4,035 | 0.498 | 0.010 |
| dp_logreg_full | 1 | 0.515 ± 0.007 | 0.010 | 0.043 | 4,035 | 0.498 | 0.010 |
| dp_logreg_full | 2 | 0.520 ± 0.006 | 0.012 | 0.056 | 4,035 | 0.500 | 0.011 |
| dp_logreg_full | 4 | 0.523 ± 0.005 | 0.011 | 0.063 | 4,035 | 0.500 | 0.010 |
| dp_logreg_full | 8 | 0.526 ± 0.006 | 0.011 | 0.067 | 4,035 | 0.501 | 0.011 |
| dp_logreg_full | ∞ | 0.526 ± 0.010 | 0.011 | 0.055 | 4,035 | 0.503 | 0.015 |
| dp_vqc | 0.5 | 0.505 ± 0.016 | 0.010 | 0.032 | 1,200 | 0.502 | 0.024 |
| dp_vqc | 1 | 0.506 ± 0.015 | 0.009 | 0.032 | 1,200 | 0.502 | 0.024 |
| dp_vqc | 2 | 0.504 ± 0.013 | 0.010 | 0.030 | 1,200 | 0.497 | 0.019 |
| dp_vqc | 4 | 0.504 ± 0.013 | 0.009 | 0.029 | 1,200 | 0.496 | 0.019 |
| dp_vqc | 8 | 0.503 ± 0.013 | 0.009 | 0.030 | 1,200 | 0.496 | 0.021 |
| dp_vqc | ∞ | 0.502 ± 0.014 | 0.010 | 0.032 | 1,200 | 0.498 | 0.018 |
| qsvm | ∞ | 0.508 ± 0.014 | 0.011 | 0.038 | 1,200 | 0.502 | 0.026 |
| mlp_overfit | ∞ | 0.732 ± 0.016 | 0.014 | 0.595 | 1,200 | 0.558 | 0.228 |

## 3. DP-SGD privacy-utility (box 4 mechanism, box 12 cost) beside NSCH

exp3 method unchanged (Opacus DP-SGD, δ = 1e-5, fixed hyperparameters, 5x2 folds). Arms: VQC re-upload 4 qubits and LogReg on the same 1,200 rows and PCA-4 inputs; LogReg on every development training row and all 384 dims. Cell: ROC AUC (change vs the same model without DP; NB-t Holm p within arm). NSCH: change, from its tests_private_vs_nonprivate.csv.

| Arm | ε = 0.5 | ε = 1 | ε = 2 | ε = 4 | ε = 8 | ε = ∞ |
|---|---|---|---|---|---|---|
| vqc_sub | 0.721 (-0.064, p 0.174; NSCH -0.165) | 0.753 (-0.032, p 0.237; NSCH -0.089) | 0.763 (-0.023, p 0.237; NSCH -0.049) | 0.771 (-0.015, p 0.237; NSCH -0.013) | 0.773 (-0.013, p 0.237; NSCH -0.009) | 0.785 |
| logreg_sub | 0.781 (-0.019, p 0.342; NSCH -0.195) | 0.786 (-0.014, p 0.342; NSCH -0.129) | 0.790 (-0.011, p 0.342; NSCH -0.101) | 0.792 (-0.008, p 0.342; NSCH -0.091) | 0.793 (-0.007, p 0.342; NSCH -0.088) | 0.800 |
| logreg_full | 0.731 (-0.059, p 0.018; NSCH -0.070) | 0.745 (-0.045, p 0.041; NSCH -0.043) | 0.761 (-0.030, p 0.085; NSCH -0.025) | 0.772 (-0.018, p 0.112; NSCH -0.016) | 0.780 (-0.010, p 0.185; NSCH -0.011) | 0.790 |

VQC vs LogReg on the same rows and inputs at each ε (NB-t, Holm within the pair):

| ε | VQC AUC | LogReg AUC | Diff | VQC wins | Holm p | AUC drop VQC / LogReg |
|---|---|---|---|---|---|---|
| 0.5 | 0.721 | 0.781 | -0.060 | 1/10 | 0.162 | 0.064 / 0.019 |
| 1 | 0.753 | 0.786 | -0.033 | 2/10 | 0.162 | 0.032 / 0.014 |
| 2 | 0.763 | 0.790 | -0.027 | 1/10 | 0.162 | 0.023 / 0.011 |
| 4 | 0.771 | 0.792 | -0.022 | 0/10 | 0.053 | 0.015 / 0.008 |
| 8 | 0.773 | 0.793 | -0.021 | 0/10 | 0.053 | 0.013 / 0.007 |
| ∞ | 0.785 | 0.800 | -0.015 | 0/10 | 0.010 |  |

## 4. Input-stage local DP before encoding (box 4) beside NSCH

exp5 method unchanged: Laplace pure ε-LDP on the 8 PCA encoder inputs of training rows, validation rows clean; dp_logreg adds DP-SGD at ε_training = 4. k-anonymity: NOT APPLICABLE to free text (there are no quasi-identifier columns to generalise; the identifying content is the text itself - see section 1). Cell: text ROC AUC (NSCH, k-anonymity off).

| Arm | Regime | ε_input = ∞ | ε_input = 32 | ε_input = 16 | ε_input = 8 | ε_input = 4 |
|---|---|---|---|---|---|---|
| qsvm | subsample | 0.804 (0.848) | 0.797 (0.821) | 0.759 (0.709) | 0.662 (0.570) | 0.558 (0.500) |
| logreg | subsample | 0.804 (0.852) | 0.799 (0.846) | 0.784 (0.826) | 0.742 (0.760) | 0.673 (0.660) |
| dp_logreg | subsample | 0.796 (0.801) | 0.792 (0.670) | 0.766 (0.543) | 0.678 (0.440) | 0.581 (0.388) |
| logreg | full | 0.805 (0.853) | 0.802 (0.849) | 0.796 (0.844) | 0.778 (0.825) | 0.719 (0.764) |

## What differs from NSCH (explicit comparison)

**1. A post is far more identifying than a row of survey codes.** On NSCH the most identifying table (7 demographic quasi-identifiers + U.S. state, raw values) leaves 62.7 % of adolescents unique, and the pipeline's k-anonymity (k = 5) brings that to 0.0 % with average prosecutor risk 0.026. On this dataset an attacker who knows just 5 consecutive words of a post singles it out in 95.0 % of cases (average risk 0.966); 8 words: 99.3 %. Even a 3-word fragment (66.3 % unique) is more identifying than NSCH's raw 10-column demographic table. 93.7 % of posts contain a 5-word span that appears in no other post.

**2. Removing direct identifiers does not help, and k-anonymity has nothing to act on.** Scrubbing URLs, e-mails, handles, community names and every number leaves 5-word uniqueness at 95.2 % (from 95.0 %). Explicit quasi-identifiers are rare in the text (an age in 3.2 % of posts, a U.S. state in 0.2 %); the identifying content is the wording itself, which no column-level generalisation or suppression can reach. On NSCH the same threat was answered by k-anonymity; for text there is no equivalent, so the only safe release is no release of text. The source corpora are public social-media posts (drawn largely from Reddit), so a fragment can also be searched for on the open web, where the original post may still be online with its author's account (not checked here: no web search was run on any post). This re-identification risk is a property of the data, not of any model trained on it.

**3. The 8 numbers fed to the quantum circuit still identify the post about a third of the time.** These are the PCA encoder inputs of architecture step 5 (quantum encoding). Most readers would assume 8 PCA components are anonymous; they are not. An attacker who knows a 12-word fragment, embeds it with the public MiniLM model and projects it with the same PCA finds the right post among 11,992 as the nearest released 8-number vector 32.6 % of the time (chance 0.00008); with a 20-word fragment, 45.1 %. Through the full 384-dim MiniLM embedding it is 59.8 %. Encoding into a quantum state does not change this: the circuit's input is these 8 numbers. This assumes the projection is known to the attacker. If the projection were kept secret, linkage would fall; but secrecy of a projection is not a privacy guarantee, and we do not claim it as one.

**3b. Local DP at ε = 32 defeats the linkage attack we implemented, at a QSVM cost of 0.007 AUC** (top-1 linkage at most 0.0 %; QSVM ROC AUC 0.797 from 0.804; on NSCH the same ε_input cost the QSVM 0.027). The formal guarantee at this budget is weak and must not be read as a general privacy claim: e^32 is astronomically large, so the bound is close to vacuous. It does not bound a stronger attack.

**4. Text models memorise individuals more readily.** The deliberately overfitted MLP (positive control) reaches membership-attack AUC 0.732 with advantage 0.595 on text, against 0.558 / 0.228 on NSCH with the same training-row count (1,200): unique, high-dimensional posts are easy to memorise. The properly trained models leak little on both datasets, but the non-private LogReg on all 384 dims is measurably above chance on text (0.526 vs NSCH 0.503), and DP-SGD lowers it to 0.511 at ε = 0.5. The VQC and QSVM stay at 0.502 and 0.508, as on NSCH. The small quantum models leak little, and that comes from their low capacity, not from anything quantum.

**5. DP-SGD utility cost at ε = 0.5.** Same-rows LogReg: 0.019 ROC AUC on text vs 0.195 on NSCH, about a tenfold difference. VQC: 0.064 vs 0.165. LogReg on all inputs and every development row: 0.059 vs 0.070. The cause of the difference is not established. The VQC stays below LogReg at every ε, as on NSCH. DP-SGD bounds what a trained model reveals about its training rows; it does nothing for points 1-3, which concern the released data itself.

**Bottom line.** On NSCH, privacy is mostly about the model, because the released table can be made k-anonymous. On free text, the data is the dominant risk: posts are effectively unique and stay unique after scrubbing, and even their embeddings link back. Text must stay where it was collected; only protected model inputs (local DP) or protected models (DP-SGD, federated - Phase 6) should leave it.
