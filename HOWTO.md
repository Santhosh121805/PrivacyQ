# HOWTO — run the web demo

## Start it

From the project folder (`C:\Users\Santhosh S\Desktop\major`) in PowerShell:

```powershell
.venv\Scripts\activate
uvicorn api:app --reload
```

Then open **http://127.0.0.1:8000/** in a browser. Stop the server with `Ctrl+C`.

The server starts in about 7 seconds and works fully offline. Nothing is trained at launch: the fitted
models are loaded from `models/demo_models.pkl`, and the Results and Privacy pages read the saved files
in `results/`.

## First time only (or if `models/` was deleted)

```powershell
python scripts/build_demo_models.py
```

This fits the final configurations once (about 15 seconds) and caches them in `models/`. It reproduces
the models evaluated on the locked hold-out, checks that their selected C and thresholds match the
hold-out run exactly, and never scores the hold-out again. If the cache is missing, the app still opens
and each section shows this instruction instead of a result.

Requirements beyond the pipeline: `fastapi==0.115.6` and `uvicorn==0.32.1` (in `requirements.txt`).

## Sections

| Section | What it shows |
|---|---|
| 1 · Try it | The 8 most informative survey questions as inputs. Load a random real NSCH record (a dev record neither model was trained on) or a diagnosed one, or use typical answers. **Predict** runs logistic regression and the QSVM side by side; a red banner appears when they disagree. |
| 2 · Inside the pipeline | The same record through every stage with real numbers: raw answers → imputed/one-hot (95) → standardized → PCA-8 → rotation angles → 256-amplitude quantum state → fidelities with training states → score → decision. Switch encodings (angle, amplitude, ZZ, re-uploading) to see the circuit and prediction change. Angle is the final evaluated model. |
| 3 · Privacy explorer | Slider over the measured ε values (0.5, 1, 2, 4, 8, ∞), the cross-validation privacy–utility curve, and the hold-out headline. |
| 4 · Results | Headline tables and the four figures, read from `results/tables` and `results/figures`. |

## Deep links (useful during the viva)

- `http://127.0.0.1:8000/#try?record=9517` — opens Try it with record 9517 and predicts
  (a diagnosed adolescent both models flag).
- `http://127.0.0.1:8000/#pipeline?record=9517&enc=zz` — the same record with the ZZ feature map
  (the ZZ model does **not** flag this record, while the final angle model does).
- `#privacy`, `#results` — jump to a section.

## Things to say while demoing

- The header line is deliberate: this is a research screening aid, not a diagnosis.
- Logistic regression shows a probability; the QSVM shows an SVM decision score, which is not a
  probability. Each is compared with its own threshold, chosen on inner-validation rows.
- The QSVM was trained on 1,200 rows and PCA-8; logistic regression on ~14,000 rows and all 95 features —
  the same final configurations as the hold-out table.
- Inputs not shown on the form come from the loaded record, or from training medians when no record is
  loaded.
- Privacy headline: ε = 4 costs 0.0316 ROC-AUC on the hold-out (0.8924 → 0.8609). Quote it as about
  0.03; 0.031 comes from subtracting already-rounded numbers.
