# HOWTO — run the web demo

## Start it

From the project folder (`C:\Users\Santhosh S\Desktop\major`) in PowerShell:

```powershell
.venv\Scripts\activate
uvicorn api:app --reload
```

Then open **http://127.0.0.1:8000/** in a browser. Stop with `Ctrl+C`.

Fully offline: Tailwind, Observable Plot, d3 and the Inter / JetBrains Mono fonts are local copies in
`web/vendor/` (see `web/vendor/README.md`). Nothing is trained at launch.

## Startup and latency (measured 8 Nov 2026 on the development laptop)

| What | Measured |
|---|---|
| Page served after `uvicorn` starts | ~3.5–4 s |
| Models ready (first prediction possible) | ~7–15 s, depending on machine load |
| QSVM prediction (final angle model), server time | 26–45 ms |
| Logistic regression prediction, server time | 12–38 ms |
| Full `/api/predict` round trip, circuit already drawn | 100–220 ms |
| Same, first time a circuit is drawn for a record | 250–870 ms (`qml.draw_mpl`; cached afterwards) |

Models load in a background thread so the page appears immediately; the page shows
"Loading the cached models…" until they are ready. Load time is almost entirely importing torch,
PennyLane and scikit-learn (required by the pipeline's own model classes), so it depends on how busy
the laptop is: 6.9 s on a quiet machine, ~14 s with many browser windows open. **On the day, start the
server a minute before the demo.** The QSVM does not recompute training-side kernel work per request: the
statevectors of its 1,200 training rows are cached in `models/demo_models.pkl`; each request simulates
one circuit for the new record and takes one matrix product against those cached states.

The page shows the server-side timings under the Predict button.

## First time only (or if `models/` was deleted)

```powershell
python scripts/build_demo_models.py
```

Fits the final configurations once (~15 s) and caches them in `models/`. It reproduces the models scored
on the locked hold-out, checks their C and thresholds match the hold-out run exactly, and never scores
the hold-out again. Without the cache the page still opens and each section explains what to run.

Dependencies beyond the pipeline (pinned in `requirements.txt`): `fastapi==0.115.6`, `uvicorn==0.32.1`,
`watchfiles==0.24.0` (makes `--reload` use file notifications instead of polling every file in `.venv`).

## Sections (one scrolling document)

| Section | What it shows |
|---|---|
| Try it | The 8 most informative survey questions as inputs; load a random real NSCH record (a dev record neither model was trained on), a diagnosed one, or typical answers. **Predict** shows logistic regression (probability) and the QSVM (decision score) side by side, each with a bar against its threshold, and says explicitly when they disagree. |
| Inside the pipeline | The same record through every stage as tables of real values: raw → imputed and scaled → reduced to 8 → quantum state (256 probabilities) → measured fidelities → score and decision. Switch encodings to watch the circuit (drawn by PennyLane `qml.draw_mpl`) and the prediction change. |
| Privacy explorer | Slider over the measured ε (0.5, 1, 2, 4, 8, ∞); AUC for classical and quantum at that ε; Observable Plot curve with the point marked; hold-out headline. |
| Results | Headline tables and the four report figures, read from `results/`. |

Classical models are blue (`#4269d0`), quantum models teal (`#6cc5b0`) in every chart.

## Deep links (rehearsal)

- `http://127.0.0.1:8000/#try?record=9517` — a diagnosed adolescent both models flag.
- `http://127.0.0.1:8000/#pipeline?record=9517&enc=zz` — same record with the ZZ feature map, which does
  **not** flag it, while the final angle model does.
- `http://127.0.0.1:8000/#privacy?eps=0` — privacy explorer at ε = 0.5.

## Things to say while demoing

- Research screening aid, not a diagnosis (the header says so throughout).
- Logistic regression gives a probability; the QSVM gives an SVM decision score, which is not a
  probability. Each is compared with its own threshold, chosen on inner-validation rows.
- Inputs not on the form come from the loaded record, or from training medians.
- Privacy headline: ε = 4 costs **0.0316** ROC-AUC on the hold-out (0.8924 → 0.8609) — say "about 0.03".
  (0.031 comes from subtracting already-rounded numbers.)
