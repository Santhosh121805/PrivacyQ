"""PrivacyQ — research configuration tool for privacy-preserving mental-health models.

Run:  uvicorn api:app --reload      then open  http://127.0.0.1:8000/

Research configuration tool. Not for clinical or individual use.
Every reported number is read from experiment results in results/ (see
privacyq.py); nothing is trained at launch. The Inspect page traces a record
through models cached in models/config_models.pkl (build once with
`python scripts/build_config_models.py`). Every failure is a JSON message.
"""

from __future__ import annotations

import io
import json
import math
import pickle
import random
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import privacyq as pq

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models" / "config_models.pkl"
RESULTS = ROOT / "results"
BUILD_HINT = "Run `python scripts/build_config_models.py` once to fit and cache the Inspect models (~15 s)."
NON_FEATURE_COLUMNS = ["label", "state_fips", "survey_weight", "survey_year"]

app = FastAPI(title="PrivacyQ", docs_url=None, redoc_url=None)


class ToolError(Exception):
    pass


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", []) if p not in ("body", "query"))
    return JSONResponse(status_code=422, content={"error": f"Invalid request{(' (' + where + ')') if where else ''}: "
                                                           f"{first.get('msg', 'malformed')}"})


@app.exception_handler(Exception)
async def any_error(request: Request, exc: Exception):  # never a bare 500
    status = 422 if isinstance(exc, (ValueError, KeyError)) else 503
    return JSONResponse(status_code=status, content={"error": str(exc) or type(exc).__name__})


# ------------------------------------------------------------------ models
_MODELS: dict | None = None
_LOCK = threading.Lock()


def models() -> dict:
    global _MODELS
    if _MODELS is None:
        with _LOCK:
            if _MODELS is None:
                if not MODELS.exists():
                    raise ToolError(f"Cached models not found at models/config_models.pkl. {BUILD_HINT}")
                with MODELS.open("rb") as fh:
                    _MODELS = pickle.load(fh)
    return _MODELS


SCREENING = ROOT / "models" / "screening_models.pkl"
_SCREENING = None


def screening_models() -> dict | None:
    """Box 9 models (three classes, graded risk), built by scripts/build_screening_models.py."""
    global _SCREENING
    if _SCREENING is None and SCREENING.exists():
        with SCREENING.open("rb") as fh:
            _SCREENING = pickle.load(fh)
    return _SCREENING


def _screening(record_id: int, X: np.ndarray) -> dict:
    from src.m5_models.multiclass import CLASS_NAMES
    from src.m8_eval.screening import risk_band

    s = screening_models()
    if s is None:
        return {"error": "Screening models not built. Run `python scripts/build_screening_models.py` once (~20 s)."}
    probs = s["three_class"]["model"].class_scores(X)[0]
    p = float(s["risk"]["model"].predict_proba(X)[0, 1])
    r = s["risk"]
    return {"three_class": {"probabilities": {c: float(v) for c, v in zip(CLASS_NAMES, probs)},
                            "predicted": CLASS_NAMES[int(np.argmax(probs))],
                            "survey_answer": s["pool_three_class"].get(int(record_id)),
                            "C": s["three_class"]["C"], "inner_val_macro_auc": s["three_class"]["inner_val_macro_auc"],
                            "n_train": s["three_class"]["n_train"]},
            "risk": {"probability": p, "band": risk_band(p, r["cuts"]), "cuts": r["cuts"],
                     "inner_val_bands": r["inner_val_bands"], "inner_val_auc": r["inner_val_auc"],
                     "n_inner_val": r["n_inner_val"], "n_inner_val_pos": r["n_inner_val_pos"],
                     "rule_tried_first": r.get("rule_tried_first")}}


@app.on_event("startup")
def warm() -> None:
    """Load cached models in the background (torch/PennyLane imports dominate) so pages are served at once."""
    def load():
        try:
            pq.catalogue()
            models()
        except Exception:
            pass  # surfaced per request
    threading.Thread(target=load, name="warmup", daemon=True).start()


def _clean(v):
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pq.read_csv(path)
    except pq.CatalogueError as e:
        raise ToolError(str(e)) from e


def _guard(fn):
    try:
        return fn()
    except pq.CatalogueError as e:
        raise ToolError(str(e)) from e


# ------------------------------------------------------------------ overview
@app.get("/api/overview")
def overview():
    r = _guard(pq.runs)
    base = _guard(pq.baseline)
    tests = _csv(r["exp2"] / "tests_same_rows_features.csv")
    t = tests[(tests.metric == "roc_auc") & (tests.quantum == "qsvm/subsample/pca8") &
              (tests.classical == "logreg/subsample/pca8")].iloc[0]
    status_file = RESULTS / "test_status.json"
    test_status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else None
    e3 = pd.concat([_csv(r["exp3"] / "summary.csv"), _csv(r["exp3_sensitivity"] / "summary.csv")])
    e3["eps"] = [pq._eps(v) for v in e3.target_epsilon]
    series = []
    for arm, label, family, dash in [("logreg_full", "Logistic regression · full data, 95 features", "classical", False),
                                     ("logreg_sub_clip10", "Logistic regression · same 1.2k rows as VQC (clip 10)", "classical", True),
                                     ("vqc_sub", "VQC · re-uploading, 4 qubits, 1.2k rows", "quantum", False)]:
        d = e3[e3.arm == arm].set_index("eps")
        series.append({"arm": arm, "label": label, "family": family, "dashed": dash, "points": [
            {"eps": e, "auc": _clean(d.loc[e, "roc_auc_mean"]), "std": _clean(d.loc[e, "roc_auc_std"])}
            for e in pq.EPS_ORDER if e in d.index]})
    return {
        "records": 21789,
        "holdout_auc": base["holdout_auc"],
        "privacy_cost": base["holdout_auc"] - base["dp4_holdout_auc"],
        "dp4_holdout_auc": base["dp4_holdout_auc"],
        "qsvm_auc": float(t.quantum_mean), "logreg_matched_auc": float(t.classical_mean),
        "qvc_holm_p": float(t.p_nb_t_holm), "qvc_folds": int(t.k),  # Holm p of the primary test (Nadeau-Bengio corrected t)
        "tests": test_status,
        "curve": {"epsilons": pq.EPS_ORDER, "series": series, "folds": 10},
    }


# ------------------------------------------------------------------ configure
@app.get("/api/configure")
def configure(model: str, encoding: str, features: int, eps: str, sites: int):
    return _guard(lambda: pq.resolve(model, encoding, features, eps, sites))


# ------------------------------------------------------------------ inspect
@app.get("/api/sample")
def sample(record_id: int | None = None):
    pool = models()["pool"]
    if record_id is not None and record_id not in pool.index:
        raise ValueError(f"record {record_id} is not in the demo record pool")
    row = record_id if record_id is not None else int(random.choice(list(pool.index)))
    return {"record_id": row, "survey_label": int(pool.loc[row, "label"]),
            "note": "a dev record that none of the cached models was trained on"}


class InspectRequest(BaseModel):
    record_id: int
    model: str = "qsvm"
    encoding: str = "angle"
    features: int = 8


def _record(record_id: int) -> pd.DataFrame:
    pool = models()["pool"]
    if record_id not in pool.index:
        raise ValueError(f"record {record_id} is not in the demo record pool")
    rec = pool.loc[[record_id]].copy()
    for col in NON_FEATURE_COLUMNS[1:]:
        rec[col] = np.nan
    return rec.reset_index(drop=True)


def _qsvm_inputs(record_id: int, encoding: str, features: int):
    m = models()
    if (encoding, features) not in m["qsvm"]:
        raise ValueError("encoding/features combination not cached")
    entry = m["qsvm"][(encoding, features)]
    X = m["prep"].transform(_record(record_id), "label").to_numpy()
    return entry, X, entry["transformer"].transform(X)


@lru_cache(maxsize=128)
def _circuit_svg(encoding: str, features: int, record_id: int) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pennylane as qml

    from src.m4_encoding import make_encoding

    _, _, z = _qsvm_inputs(record_id, encoding, features)
    enc = make_encoding(encoding, features)

    @qml.qnode(qml.device("default.qubit", wires=features))
    def circuit(x):
        enc.feature_map(x)
        return qml.state()

    style = {"svg.fonttype": "none", "font.family": "monospace", "font.monospace": ["JetBrains Mono", "DejaVu Sans Mono"],
             "text.color": "#1b1e23", "patch.facecolor": "#ffffff", "patch.edgecolor": "#6b6f76",
             "patch.linewidth": 1.0, "lines.color": "#9498a0", "lines.linewidth": 1.0}
    with plt.rc_context(style):
        fig, _ = qml.draw_mpl(circuit, decimals=None if encoding == "amplitude" else 2, level="device",
                              style="rcParams")(enc.prepare(z)[0])
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.find("<svg"):]


@app.get("/api/circuit")
def circuit(encoding: str, features: int, record_id: int):
    if encoding not in pq.ENCODINGS or features not in (4, 6, 8):
        raise ValueError("unknown encoding or feature count")
    t0 = time.perf_counter()
    svg = _circuit_svg(encoding, features, record_id)
    return Response(svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "max-age=3600", "X-Render-Ms": f"{1000 * (time.perf_counter() - t0):.1f}"})


@app.post("/api/inspect")
def inspect(req: InspectRequest):
    if req.model not in pq.MODELS or req.encoding not in pq.ENCODINGS or req.features not in (4, 6, 8):
        raise ValueError("unknown model, encoding or feature count")
    t0 = time.perf_counter()
    m = models()
    rec = _record(req.record_id)
    raw = rec.drop(columns=NON_FEATURE_COLUMNS).iloc[0]
    imputed = m["prep"].transform(rec, "label")
    X = imputed.to_numpy()
    n = req.features
    steps = [{"title": "Raw survey answers", "detail": f"{int(raw.notna().sum())} of 66 whitelisted answers present.",
              "rows": [[k, v if isinstance(v, str) else _clean(v)] for k, v in raw.items()]}]
    note = None
    if req.model == "logreg":
        lr = m["logreg"][n]
        ft = lr["transformer"]
        std, red, z = ft.scaler.transform(X)[0], ft.pca.transform(ft.scaler.transform(X))[0], ft.transform(X)[0]
        p = float(lr["model"].predict_proba(z.reshape(1, -1))[0, 1])
        steps += [
            {"title": "Imputed and scaled", "detail": "Missing answers filled with training medians or modes, categories "
             "one-hot encoded (95 columns), then standardized on the 1,200 training rows.",
             "rows": [[c, f"{a:.2f} → {b:.3f}"] for c, a, b in zip(imputed.columns, X[0], std)]},
            {"title": f"Reduced to {n} features", "detail": f"PCA to {n} components, rescaled to [0, π] on training rows.",
             "rows": [[f"PC{i + 1}", f"{a:.3f} → {b:.3f}"] for i, (a, b) in enumerate(zip(red, z))]},
            {"title": "Encoded", "detail": "Not applicable: logistic regression uses the reduced values directly.", "rows": []},
            {"title": "Measured", "detail": "Not applicable: no quantum circuit in a classical model.", "rows": []},
            {"title": "Probability", "detail": "Logistic regression probability compared with the threshold chosen on "
             "inner-validation rows.", "rows": [["probability", round(p, 4)], ["threshold", round(lr["threshold"], 4)],
                                                ["decision", "above threshold" if p >= lr["threshold"] else "below threshold"]]},
        ]
        circuit_url = None
    else:
        entry = m["qsvm"][(req.encoding, n)]
        ft, model = entry["transformer"], entry["model"]
        std = ft.scaler.transform(X)[0]
        red = ft.pca.transform(ft.scaler.transform(X))[0]
        z = ft.transform(X)
        state = model._states(z)[0]
        probs = np.abs(state) ** 2
        top = np.argsort(-probs)[:8]
        k_row = model._kernel(model._states(z), model.train_states_)[0]
        svc = model.svc_
        sv_k = k_row[svc.support_]
        near = np.argsort(-sv_k)[:5]
        score = float(model.decision_scores(z)[0])
        reduced_rows = ([[f"PC{i + 1} → qubit {i}", f"{a:.3f} → {b:.3f} rad"] for i, (a, b) in enumerate(zip(red, z[0]))]
                        if req.encoding != "amplitude" else
                        [[f"PC{i + 1}", f"{a:.3f}"] for i, a in enumerate(red[:12])] +
                        [["…", f"{len(red)} components + 1 constant → {2 ** n} amplitudes"]])
        steps += [
            {"title": "Imputed and scaled", "detail": "Missing answers filled with training medians or modes, categories "
             "one-hot encoded (95 columns), then standardized on the 1,200 training rows.",
             "rows": [[c, f"{a:.2f} → {b:.3f}"] for c, a, b in zip(imputed.columns, X[0], std)]},
            {"title": f"Reduced to {n} features" if req.encoding != "amplitude" else f"Reduced for {n}-qubit amplitude encoding",
             "detail": (f"PCA to {n} components, rescaled to rotation angles in [0, π]." if req.encoding != "amplitude" else
                        f"PCA to {len(red)} components (2^{n} − 1 slots), standardized, then normalized into amplitudes."),
             "rows": reduced_rows},
            {"title": "Encoded", "detail": f"{pq.ENCODINGS[req.encoding]} encoding on {n} qubits → {len(state)} amplitudes "
             f"(norm {np.linalg.norm(state):.6f}). Eight most probable basis states:",
             "rows": [[f"|{format(int(i), f'0{n}b')}⟩", f"{probs[i]:.4f}"] for i in top],
             "probabilities": [float(v) for v in probs]},
            {"title": "Measured", "detail": "Fidelity |⟨φ(x)|φ(xᵢ)⟩|² with each training state — on hardware, the probability "
             "of reading all-zeros from an overlap circuit.",
             "rows": [["training states compared", int(len(k_row))], ["fidelity min / median / max",
                      f"{k_row.min():.3f} / {np.median(k_row):.3f} / {k_row.max():.3f}"],
                      ["support vectors", int(len(svc.support_))]] +
                     [[f"nearest support vector {j + 1} ({'diagnosed' if svc.dual_coef_[0][i] > 0 else 'not diagnosed'})",
                       f"{sv_k[i]:.4f}"] for j, i in enumerate(near)]},
            {"title": "Score (not a probability)", "detail": "The QSVM outputs an SVM decision score — weighted fidelities plus "
             "an intercept — compared with the threshold chosen on inner-validation rows.",
             "rows": [["decision score", round(score, 4)], ["threshold", round(entry["threshold"], 4)],
                      ["decision", "above threshold" if score >= entry["threshold"] else "below threshold"]]},
        ]
        circuit_url = f"/api/circuit?encoding={req.encoding}&features={n}&record_id={req.record_id}"
        if req.model in ("vqc", "hybrid"):
            note = (f"{pq.MODELS[req.model]} weights are not cached in this tool, so the trace uses the QSVM fitted with the "
                    f"same encoding and {n} features on the same 1,200 rows.")
    return {"record_id": req.record_id, "survey_label": int(models()["pool"].loc[req.record_id, "label"]),
            "model": req.model, "encoding": req.encoding, "features": n, "note": note, "steps": steps,
            "circuit_url": circuit_url, "screening": _screening(req.record_id, X),
            "latency_ms": 1000 * (time.perf_counter() - t0)}


# ------------------------------------------------------------------ evidence
@app.get("/api/evidence")
def evidence():
    tables = RESULTS / "tables"
    t4 = _csv(tables / "table4_quantum_vs_classical.csv")
    figs = [{"file": f, "url": f"/figures/{f}", "exists": (RESULTS / "figures" / f).exists()} for f in
            ("privacy_utility_curve.png", "encoding_comparison.png", "roc_curves.png", "federated_convergence.png")]
    return {"matched": _records(t4[t4.Metric == "ROC-AUC"].drop(columns=["Metric"])),
            "holdout": _records(_csv(tables / "table11_holdout.csv")),
            "encodings": _records(_csv(tables / "table2_encodings.csv")),
            "figures": figs}


# ------------------------------------------------------------------ federated demo (Overview "How it works")
@app.get("/api/federated")
def federated():
    """Replay of the recorded Experiment 4 run (LogReg, 3 clients split by U.S. state, ε = 4, fold r0f0) from
    results/story/story.json, plus the measured serialised update size for that same setup (Experiment 7)."""
    story_file = RESULTS / "story" / "story.json"
    if not story_file.exists():
        raise ToolError("Replay data not found (results/story/story.json). Run `python scripts/build_story_data.py` once.")
    story = json.loads(story_file.read_text(encoding="utf-8"))
    cost_files = sorted(RESULTS.glob("exp7-nsch2024-federated-cost-*/communication_cost.csv"))
    if not cost_files:
        raise ToolError("Measured payload not found (results/exp7-*/communication_cost.csv).")
    cost = _csv(cost_files[-1])
    row = cost[(cost.arm == "logreg_full") & (cost.clients == 3) & (cost.partition == "by_state")].iloc[0]
    return {"replay": story["replay"], "alone_auc_eps4": story["replay_alone_auc_eps4"],
            "update_bytes": int(row.payload_bytes_per_update), "update_bytes_source": str(cost_files[-1].parent.name)}


@app.get("/api/{rest:path}")
def unknown(rest: str):
    return JSONResponse(status_code=404, content={"error": f"unknown endpoint /api/{rest}"})


if (RESULTS / "figures").exists():
    app.mount("/figures", StaticFiles(directory=RESULTS / "figures"), name="figures")
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
