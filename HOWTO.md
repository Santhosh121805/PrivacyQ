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
- **Privacy cost on the hold-out** is 0.8924 → 0.8609 = **0.0316**, displayed as 0.032 (three decimals).
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
