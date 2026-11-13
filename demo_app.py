"""Architecture walkthrough demo — one section per box of docs/architecture.jpg.

Start:  python demo_app.py        then open  http://127.0.0.1:8002/      (works offline)

The page reads only demo/manifest.json (built from real results by scripts/build_demo_manifest.py) and one live
endpoint. POST /api/live runs the real cached models (models/config_models.pkl, models/screening_models.pkl) on one
row of the locked hold-out: de-identify → preprocess → reduce to 8 angles → encode on 8 qubits (PennyLane statevector,
computed now) → QSVM decision and risk band. Only the row's raw survey answers are stored in the manifest. Nothing is
trained here, and no metric is computed on the hold-out.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "demo" / "manifest.json"
app = FastAPI(title="Architecture walkthrough", docs_url=None, redoc_url=None)
_STATE: dict = {}


@app.exception_handler(Exception)
async def any_error(request, exc):
    return JSONResponse(status_code=503, content={"error": str(exc) or type(exc).__name__})


def _load() -> dict:
    if not _STATE:
        for name, path in (("manifest", MANIFEST), ("cache", ROOT / "models/config_models.pkl"),
                           ("screen", ROOT / "models/screening_models.pkl")):
            if not path.exists():
                raise FileNotFoundError(f"{path.relative_to(ROOT)} is missing — run scripts/build_demo_manifest.py "
                                        "(and the model-cache scripts it names)")
        _STATE["manifest"] = json.loads(MANIFEST.read_text(encoding="utf-8"))
        _STATE["cache"] = pickle.load(open(ROOT / "models/config_models.pkl", "rb"))
        _STATE["screen"] = pickle.load(open(ROOT / "models/screening_models.pkl", "rb"))
    return _STATE


@app.post("/api/live")
def live():
    import pennylane as qml

    from src.m4_encoding import make_encoding
    from src.m5_models.multiclass import CLASS_NAMES
    from src.m8_eval.screening import risk_band

    t0 = time.perf_counter()
    s = _load()
    man, cache, screen = s["manifest"], s["cache"], s["screen"]
    rec, labels = man["live"], man["labels"]
    row = pd.DataFrame([rec["raw"]])
    row["label"] = rec["survey_label"]
    shown = list(rec["display"])[:4]
    steps = [{"title": "Real record", "detail": f"Dataset row {rec['row_index']} (hold-out). 4 of "
                                                f"{len(rec['raw']) - len(rec['identifier_columns'])} answers:",
              "rows": [[labels.get(k, k), rec["display"][k]] for k in shown]}]
    # 1. de-identify: drop identifier / metadata columns before anything else
    row_d = row.drop(columns=rec["identifier_columns"])
    steps.append({"title": "De-identify", "detail": "Identifier and metadata columns removed before modelling (values not shown).",
                  "rows": [["Removed", ", ".join(rec["identifier_columns"])]]})
    for c in rec["identifier_columns"]:
        row_d[c] = np.nan                                     # the preprocessor expects the columns; they carry no values
    # 2. preprocess with the transformations fitted on the training rows
    Xdf = cache["prep"].transform(row_d, "label")
    X = Xdf.to_numpy()
    steps.append({"title": "Preprocess", "detail": f"Imputed, one-hot encoded ({X.shape[1]} columns), standardised (fitted on 1,200 training rows). First 3:",
                  "rows": [[labels.get(c, c), f"{v:.3f}"] for c, v in zip(Xdf.columns[:3], X[0][:3])]})
    # 3. reduce to 8 rotation angles
    entry = cache["qsvm"][("angle", 8)]
    z = entry["transformer"].transform(X)[0]
    steps.append({"title": "Reduce to 8 angles", "detail": "PCA to 8 components, scaled to [0, π] — one angle per qubit (radians):",
                  "rows": [["Qubits 0–7", ", ".join(f"{v:.2f}" for v in z)]]})
    # 4. encode: statevector computed now
    enc = make_encoding("angle", 8)

    @qml.qnode(qml.device("default.qubit", wires=8))
    def state(v):
        enc.feature_map(v)
        return qml.state()
    probs = np.abs(np.asarray(state(z))) ** 2
    top = np.argsort(probs)[::-1][:3]
    steps.append({"title": "Encode (8 qubits)", "detail": f"Statevector of 256 amplitudes (norm {probs.sum():.6f}). Most probable states:",
                  "rows": [[f"|{i:08b}⟩", f"{probs[i]:.4f}"] for i in top]})
    # 5. predict: QSVM decision + screening risk band
    score = float(entry["model"].decision_scores(z.reshape(1, -1))[0])
    p = float(screen["risk"]["model"].predict_proba(X)[0, 1])
    band = risk_band(p, screen["risk"]["cuts"])
    p3 = screen["three_class"]["model"].class_scores(X)[0]
    steps.append({"title": "Predict", "detail": "QSVM decision vs its threshold; risk band from the 95-column logistic regression.",
                  "rows": [["QSVM score", f"{score:.3f} vs {entry['threshold']:.3f} → {'flagged' if score >= entry['threshold'] else 'not flagged'}"],
                           ["Depression probability", f"{p:.3f} → {band.upper()} (cuts {screen['risk']['cuts']['low_moderate']:.3f} / "
                                                      f"{screen['risk']['cuts']['moderate_high']:.3f})"],
                           ["Three classes", " · ".join(f"{c.replace('_', ' ')} {v:.3f}" for c, v in zip(CLASS_NAMES, p3))]]})
    return {"row_index": rec["row_index"], "split": rec["split"], "steps": steps, "band": band,
            "survey_label": "currently diagnosed depression" if rec["survey_label"] else "no current depression diagnosis",
            "ms": round(1000 * (time.perf_counter() - t0), 1)}


@app.get("/api/{rest:path}")
def unknown(rest: str):
    return JSONResponse(status_code=404, content={"error": f"unknown endpoint /api/{rest}"})


app.mount("/fonts", StaticFiles(directory=ROOT / "web" / "vendor" / "fonts"), name="fonts")
app.mount("/", StaticFiles(directory=ROOT / "demo", html=True), name="demo")

if __name__ == "__main__":
    import uvicorn

    print("Architecture walkthrough: http://127.0.0.1:8002/")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="warning")
