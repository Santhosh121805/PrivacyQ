# HOWTO — run PrivacyQ

PrivacyQ is a research configuration tool for privacy-preserving mental-health models.
**Research configuration tool. Not for clinical or individual use.**

## Start it

From the project folder (`C:\Users\Santhosh S\Desktop\major`) in PowerShell:

```powershell
.venv\Scripts\activate
uvicorn api:app --reload
```

Open **http://127.0.0.1:8000/** in a browser. Stop with `Ctrl+C`.

## First time only (or if `models/` was deleted)

```powershell
python scripts/build_config_models.py     # ~15 s: caches the Inspect models in models/config_models.pkl
python scripts/record_test_status.py      # runs the test suite, writes results/test_status.json for the Overview card
```

Without the model cache, Overview, Configure and Evidence still work (they only read `results/`); Inspect
shows the command to run. Without `results/test_status.json` the "Tests passing" card says so instead of
showing a number.

## Measured on 8 Nov 2026 (development laptop)

| What | Measured |
|---|---|
| Page served after `uvicorn` starts | 1.7 s |
| Overview / Configure / Evidence usable | immediately (read only `results/`) |
| Inspect models loaded (background thread) | 6.8 s; up to ~15 s when the laptop is busy (torch + PennyLane imports) |
| `GET /api/configure` | 4 ms |
| `POST /api/inspect`, QSVM, full six-step trace | 22 ms (angle) – 52 ms (ZZ) |
| `POST /api/inspect`, logistic regression | 8 ms |
| `GET /api/circuit` (`qml.draw_mpl`), first draw / cached | 45–148 ms / 2 ms |

The QSVM does no training-side kernel work per request: the statevectors of its 1,200 training rows are
stored in the cached model, so a prediction is one circuit simulation plus one matrix product.
**On the day, start the server a minute before the demo.**

## Pages

| Page | What it does |
|---|---|
| Overview | Four headline numbers from `results/`, the privacy–utility curve (Observable Plot; classical blue, quantum teal), what the tool does and what was found. |
| Configure | Encoding, model, feature count, ε and number of sites. The panel shows the **measured** ROC-AUC for that configuration (mean ± std over folds), the gap to the unprivatised classical baseline, balanced accuracy, recall, training cost and a plain-English verdict. A combination that was never run is labelled "Not measured" and the nearest measured configuration is shown, with every difference listed. Nothing is interpolated. |
| Inspect | Traces a real NSCH record (a dev record none of the cached models was trained on) through the configured pipeline in six steps, then shows the circuit drawn by PennyLane `qml.draw_mpl()`. |
| Evidence | The experiment tables and report figures, read from `results/tables` and `results/figures`. |

Configurations are deep-linkable, e.g.
`http://127.0.0.1:8000/#configure?model=vqc&enc=reupload&f=4&eps=4&sites=10`
(the VQC at ε = 4 split across 10 sites: 0.658, "too few rows per site").

## What the numbers mean (say this if asked)

- **Configure numbers are cross-validation means** from Experiments 1–4; the baseline bar is logistic
  regression on all 95 features and the full data (0.888 cross-validated; 0.892 on the locked hold-out).
- **Only what was run can be shown.** DP was run for the VQC (re-uploading, 4 features) and logistic
  regression; federation only at ε = 4. QSVM has no private or federated version: it is not trained by
  gradient descent, so DP-SGD and FedAvg do not apply. The tool says this instead of inventing a number.
- **Logistic regression under DP** is shown with the pre-declared clipping norm of 1; the panel also states
  the post-hoc clipping-10 sensitivity result, which closes the apparent gap to the VQC.
- **Best-model-to-DP-model gap on the hold-out** is 0.8924 → 0.8609 = **0.0316**, displayed as 0.032 (three decimals).
  It is not all privacy cost: the two models are trained differently (tuned sklearn LogReg vs SGD-trained LogReg).
  On the 10 shared cross-validation folds the gap is 0.032 = 0.016 (SGD training, no noise) + 0.016 (DP noise at
  ε = 4; not significant, Nadeau–Bengio Holm p = 0.33). No confidence interval was computed for the hold-out difference
  itself, and the hold-out scores were not saved, so it cannot be added without re-scoring the hold-out.
- **The QSVM's output is a decision score, not a probability.** Inspect labels it that way.
- **VQC and Hybrid QNN weights are not cached**, so their Inspect trace uses the QSVM fitted with the same
  encoding and feature count on the same rows; the page says so.

## Assets

Tailwind (Play CDN 3.4.16) and Observable Plot (0.6.16, ESM) load from their CDNs as specified; if the
network is unavailable both fall back automatically to identical local copies in `web/vendor/`, so the tool
also works offline. Inter and JetBrains Mono are served locally. The Tailwind Play script prints a
"not for production" warning in the browser console; that is expected for a no-build tool.

Dependencies beyond the pipeline (pinned in `requirements.txt`): `fastapi==0.115.6`, `uvicorn==0.32.1`,
`watchfiles==0.24.0` (makes `--reload` use file notifications instead of polling `.venv`).

---

# HOWTO — the "three schools" story demo (for the project review)

A four-screen walkthrough for a non-technical audience. It **replays a real federated run** from
Experiment 4; nothing is simulated or trained at launch.

```powershell
.venv\Scripts\activate
uvicorn story:app --port 8001
```

Open **http://127.0.0.1:8001/**. Use **Next** to move between screens; on screen 2 press
**Start federated training** (Next unlocks when the replay finishes). Fully offline.

First time only (or if `results/story/story.json` is missing):

```powershell
python scripts/build_story_data.py     # ~2 min: "trained alone" runs + replay data
```

Without that file the page still opens and says what to run.

## What is being replayed

- Experiment 4, logistic regression, 3 clients split by U.S. state (non-IID), client-side DP-SGD at
  ε = 4, outer fold r0f0. The three "schools" are those three clients: real NSCH 2024 records, one
  third each (16–18 states per school). The school names are invented.
- Accuracy per round: the recorded validation ROC-AUC of that run (30 rounds), measured on 3,487 records
  never used for training. The curve is shown as recorded — it fluctuates under DP noise and ends at
  **0.855** (the final model; not the best round).
- Privacy meter: ε after each round, recomputed with Opacus's RDP accountant from the run's own noise
  multiplier (σ = 1.22), sampling rate and steps; the final value equals the logged ε = 3.994.
- Model update size: 96 weights × 8 bytes = **768 bytes (0.8 KB)** per school per round.

## "Trained alone" baseline (run 9 Nov 2026)

Each school trains logistic regression on only its own rows (own preprocessing), scored on the same
3,487 test records:

| | School 1 | School 2 | School 3 |
|---|---|---|---|
| Alone, same DP (ε = 4) — shown in the demo | 0.829 | 0.829 | 0.840 |
| Alone, no privacy (for reference) | 0.866 | 0.867 | 0.871 |
| Together (FedAvg + DP, ε = 4) | 0.855 | | |

Across all 10 Experiment 4 test splits: alone with DP 0.819 ± 0.024, together 0.860 ± 0.011; together
was better in 10 of 10 splits (Nadeau–Bengio corrected t p = 0.006; Wilcoxon sensitivity check p = 0.002). **The comparison only holds at equal
privacy:** a school alone without DP (0.860 on average) matches the private shared model. The demo
says "with the same privacy protection" for that reason. Raw runs: `results/story_alone_runs.csv`.

## Timing (measured)

Replay animation 20 s (30 rounds; the first three slow, the rest fast). Full walkthrough with
presenter-style pauses: ~86 s.
