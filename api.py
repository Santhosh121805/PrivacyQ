"""Web demo for DP-FedQ: FastAPI JSON API + static frontend (web/).

Run:  uvicorn api:app --reload      then open  http://127.0.0.1:8000/

Research screening aid, not a diagnosis. Nothing is trained at launch: fitted
models are loaded from models/demo_models.pkl (build once with
`python scripts/build_demo_models.py`). Results and figures are read from
results/, never recomputed. Every failure is returned as a JSON message.
"""

from __future__ import annotations

import io
import math
import pickle
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models" / "demo_models.pkl"
RESULTS = ROOT / "results"
ENCODINGS = ("angle", "amplitude", "zz", "reupload")
BUILD_HINT = "Run `python scripts/build_demo_models.py` once to fit and cache the demo models."

# The 8 features most often chosen by mutual information (Phase 1, MI top-8, 15 folds).
# Answer labels are from the NSCH 2024 codebook (nsch_2024_topical.do).
INPUTS = [
    ("makefriend", "Difficulty making or keeping friends", {1: "No difficulty", 2: "A little difficulty", 3: "A lot of difficulty"}),
    ("k7q85_r", "Stays calm and in control when challenged", {1: "Always", 2: "Usually", 3: "Sometimes", 4: "Never"}),
    ("bullied_r", "Bullied, picked on or excluded by others", {1: "Never (past 12 months)", 2: "1–2 times (past 12 months)",
                                                               3: "1–2 times per month", 4: "1–2 times per week", 5: "Almost every day"}),
    ("k2q01", "General health", {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}),
    ("k7q02r_r", "Days of school missed (illness/injury, past 12 months)", {1: "None", 2: "1–3 days", 3: "4–6 days",
                                                                             4: "7–10 days", 5: "11 or more days", 6: "Not enrolled in school"}),
    ("k7q83_r", "Does all required homework", {1: "Always", 2: "Usually", 3: "Sometimes", 4: "Never"}),
    ("k7q04r_r", "Times the school contacted the family about problems", {1: "None", 2: "1 time", 3: "2 or more times"}),
    ("k7q84_r", "Works to finish tasks they start", {1: "Always", 2: "Usually", 3: "Sometimes", 4: "Never"}),
]
INPUT_KEYS = [k for k, _, _ in INPUTS]

app = FastAPI(title="DP-FedQ demo", docs_url=None, redoc_url=None)


class DemoError(Exception):
    pass


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    return JSONResponse(status_code=422, content={"error": f"Invalid request{(' at ' + where) if where else ''}: "
                                                           f"{first.get('msg', 'malformed body')}"})


@app.exception_handler(Exception)
async def any_error(request: Request, exc: Exception):  # never a bare 500
    status = 503 if isinstance(exc, DemoError) else 422 if isinstance(exc, (ValueError, KeyError)) else 503
    return JSONResponse(status_code=status, content={"error": str(exc) or type(exc).__name__})


@lru_cache(maxsize=1)
def bundle() -> dict:
    if not MODELS.exists():
        raise DemoError(f"Cached models not found at models/demo_models.pkl. {BUILD_HINT}")
    with MODELS.open("rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=1)
def runs() -> dict:
    cfg = ROOT / "configs" / "final_report.yaml"
    if not cfg.exists():
        raise DemoError("configs/final_report.yaml not found; results cannot be located.")
    return {k: ROOT / v for k, v in yaml.safe_load(cfg.read_text(encoding="utf-8"))["runs"].items()}


def _csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DemoError(f"Result file missing: {path.relative_to(ROOT)}. Re-run the experiment or final report.")
    return pd.read_csv(path)


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


@app.on_event("startup")
def warm() -> None:
    """Load cached models and import the quantum stack so the first click is fast (no training)."""
    try:
        bundle()
        import src.m5_models.quantum  # noqa: F401
    except Exception:  # surfaced per request instead
        pass


# --------------------------------------------------------------------- API ---
@app.get("/api/sample")
def sample(positive: bool | None = None, record_id: int | None = None):
    b = bundle()
    pool = b["pool"]
    if record_id is not None:
        if record_id not in pool.index:
            raise ValueError(f"record {record_id} is not in the demo record pool")
        row = record_id
    else:
        if positive is not None:
            pool = pool[pool.label == int(positive)]
        row = int(random.choice(list(pool.index)))
    rec = b["pool"].loc[row]
    from src.m1_data.nsch import FEATURES

    return {
        "record_id": row,
        "true_label": int(rec.label),
        "inputs": [{"key": k, "label": lab, "options": [{"value": v, "text": t} for v, t in opts.items()],
                    "value": _clean(rec[k])} for k, lab, opts in INPUTS],
        "answers": [{"key": k, "description": FEATURES[k][1], "value": _clean(rec[k])} for k in FEATURES],
        "source": f"{b['source']} — dev record not used to train either model",
    }


class PredictRequest(BaseModel):
    record_id: int | None = None
    edits: dict[str, float | None] = {}
    encoding: str = "angle"


def _record_frame(b: dict, req: PredictRequest) -> pd.DataFrame:
    from src.m1_data.nsch import FEATURES

    if req.record_id is not None:
        if req.record_id not in b["pool"].index:
            raise ValueError(f"record {req.record_id} is not in the demo record pool")
        rec = b["pool"].loc[[req.record_id], list(FEATURES)].copy()
    else:  # no record loaded: everything else imputed with the training medians / modes
        rec = pd.DataFrame([{k: np.nan for k in FEATURES}])
    for key, value in req.edits.items():
        if key not in INPUT_KEYS:
            raise ValueError(f"'{key}' is not an editable input")
        allowed = dict((k, o) for k, _, o in INPUTS)[key]
        if value is not None and int(value) not in allowed:
            raise ValueError(f"{key}={value} is not a valid answer")
        rec[key] = np.nan if value is None else float(value)
    rec["label"] = 0  # placeholder column expected by the preprocessor; never read
    for col in ("state_fips", "survey_weight", "survey_year"):
        rec[col] = np.nan
    return rec.reset_index(drop=True)


def _decision(value: float, threshold: float) -> dict:
    return {"threshold": threshold, "at_risk": bool(value >= threshold), "distance": value - threshold}


@lru_cache(maxsize=64)
def _circuit_svg(encoding: str, angles: tuple) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pennylane as qml

    from src.m4_encoding import make_encoding

    enc = make_encoding(encoding, 8)
    dev = qml.device("default.qubit", wires=8)

    @qml.qnode(dev)
    def circuit(x):
        enc.feature_map(x)
        return qml.state()

    x = enc.prepare(np.array([angles]))[0]
    fig, _ = qml.draw_mpl(circuit, decimals=None if encoding == "amplitude" else 2, level="device",
                          style="black_white")(x)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.find("<svg"):]


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}")
    b = bundle()
    rec = _record_frame(b, req)

    # LogReg (final classical model): imputation/one-hot -> standardize -> logistic regression
    lr = b["logreg"]
    x_lr = lr["scaler"].transform(lr["prep"].transform(rec, "label").to_numpy())
    p_lr = float(lr["model"].predict_proba(x_lr)[0, 1])

    # QSVM path for the selected encoding (angle = the final, evaluated model)
    q = b["qsvm"]
    imputed = q["prep"].transform(rec, "label")
    X = imputed.to_numpy()

    def qsvm_score(enc: str):
        m = q["by_encoding"][enc]
        z = m["transformer"].transform(X)
        return m, z, float(m["model"].decision_scores(z)[0])

    final_m, _, final_score = qsvm_score("angle")
    m, z, score = qsvm_score(req.encoding)
    ft, model = m["transformer"], m["model"]
    standardized = ft.scaler.transform(X)[0]
    reduced = ft.pca.transform(ft.scaler.transform(X))[0]
    state = model._states(z)[0]
    probs = np.abs(state) ** 2
    order = np.argsort(-probs)[:8]
    k_row = model._kernel(model._states(z), model.train_states_)[0]
    svc = model.svc_
    sv_k = k_row[svc.support_]
    contrib = svc.dual_coef_[0] * sv_k
    near = np.argsort(-sv_k)[:5]

    raw = rec.drop(columns=["label", "state_fips", "survey_weight", "survey_year"]).iloc[0]
    cv = b["qsvm_exp1_cv_auc_pca8_8q"][req.encoding]
    trace = [
        {"stage": "Raw answers", "detail": f"{int(raw.notna().sum())} of 66 whitelisted answers present, "
                                           f"{int(raw.isna().sum())} missing", "values": [
            {"name": k, "value": _clean(v) if not isinstance(v, str) else v} for k, v in raw.items()]},
        {"stage": "Imputed + one-hot", "detail": "missing answers filled with training medians/modes; categories "
                                                 f"expanded to 0/1 columns → {X.shape[1]} numbers",
         "values": [{"name": c, "value": float(v)} for c, v in zip(imputed.columns, X[0])]},
        {"stage": "Standardized", "detail": "each column centred and scaled with the 1,200 training rows",
         "values": [{"name": c, "value": float(v)} for c, v in zip(imputed.columns, standardized)]},
        {"stage": "Reduced", "detail": (f"PCA to {len(reduced)} components, then the constant slot; normalized "
                                        "into 256 amplitudes" if req.encoding == "amplitude" else
                                        "PCA to 8 components, then scaled to rotation angles in [0, π]"),
         "values": [{"name": f"PC{i + 1}", "value": float(v)} for i, v in enumerate(reduced[:8])],
         "angles": None if req.encoding == "amplitude" else [float(v) for v in z[0]]},
        {"stage": "Quantum state", "detail": f"8 qubits → {len(state)} amplitudes (norm {np.linalg.norm(state):.6f})",
         "probabilities": [float(p) for p in probs],
         "top": [{"basis": format(int(i), "08b"), "amplitude_re": float(state[i].real),
                  "amplitude_im": float(state[i].imag), "probability": float(probs[i])} for i in order]},
        {"stage": "Measured", "detail": "fidelity |⟨φ(x)|φ(xᵢ)⟩|² with each training state — on hardware, the "
                                        "probability of reading all-zeros from an overlap circuit",
         "kernel": {"min": float(k_row.min()), "median": float(np.median(k_row)), "max": float(k_row.max()),
                    "n_train": int(len(k_row)), "n_support": int(len(svc.support_))},
         "nearest_support_vectors": [{"fidelity": float(sv_k[i]), "label": "diagnosed" if svc.dual_coef_[0][i] > 0
                                      else "not diagnosed", "weight": float(svc.dual_coef_[0][i])} for i in near]},
        {"stage": "Score", "detail": "Σ wᵢ·fidelityᵢ over support vectors + intercept, compared with the threshold "
                                     "chosen on inner-validation rows",
         "positive_part": float(contrib[contrib > 0].sum()), "negative_part": float(contrib[contrib < 0].sum()),
         "intercept": float(svc.intercept_[0]), "score": score, **_decision(score, m["threshold"])},
    ]
    return {
        "record_id": req.record_id,
        "logreg": {"probability": p_lr, **_decision(p_lr, lr["threshold"]), "kind": "probability",
                   "trained_rows": lr["n_train"], "C": lr["params"].get("C")},
        "qsvm": {"score": final_score, **_decision(final_score, final_m["threshold"]), "kind": "decision score",
                 "encoding": "angle", "trained_rows": q["n_train"], "C": final_m["params"].get("C")},
        "selected_encoding": {"encoding": req.encoding, "score": score, **_decision(score, m["threshold"]),
                              "cv_auc": cv},
        "trace": trace,
        "circuit_svg": _circuit_svg(req.encoding, tuple(np.round(z[0][:8], 4)) if req.encoding != "amplitude"
                                    else tuple(np.round(z[0], 4))),
    }


@app.get("/api/privacy")
def privacy():
    r = runs()
    s = pd.concat([_csv(r["exp3"] / "summary.csv"), _csv(r["exp3_sensitivity"] / "summary.csv")], ignore_index=True)
    s["eps"] = ["inf" if str(e) in ("inf", "nan") else f"{float(e):g}" for e in s.target_epsilon]
    order = ["0.5", "1", "2", "4", "8", "inf"]
    series_spec = [
        ("vqc_sub", "VQC (quantum), 1.2k rows, PCA-4", "quantum"),
        ("logreg_sub", "LogReg, same rows & features, clip 1", "classical"),
        ("logreg_sub_clip10", "LogReg, same rows & features, clip 10 (sensitivity)", "classical"),
        ("logreg_full", "LogReg, full data, all 95 features", "classical"),
    ]
    series = []
    for arm, label, family in series_spec:
        d = s[s.arm == arm].set_index("eps")
        if d.empty:
            continue
        series.append({"arm": arm, "label": label, "family": family, "points": [
            {"eps": e, "auc": _clean(d.loc[e, "roc_auc_mean"]), "auc_std": _clean(d.loc[e, "roc_auc_std"]),
             "balanced_accuracy": _clean(d.loc[e, "balanced_accuracy_mean"]),
             "epsilon_spent": _clean(d.loc[e, "epsilon_spent_mean"])} for e in order if e in d.index]})
    h = _csv(r["holdout"] / "holdout_metrics.csv").set_index("model")
    nonpriv, dp4 = float(h.loc["logreg_full_all95", "roc_auc"]), float(h.loc["dp_logreg_full_all95_eps4", "roc_auc"])
    return {"epsilons": order, "delta": 1e-5, "series": series, "folds": 10,
            "headline": {"nonprivate_auc": nonpriv, "dp_eps4_auc": dp4, "cost": nonpriv - dp4,
                         "epsilon_spent": _clean(h.loc["dp_logreg_full_all95_eps4", "epsilon_spent"]),
                         "source": "locked hold-out (4,358 adolescents), LogReg on full data"}}


@app.get("/api/results")
def results():
    tables = RESULTS / "tables"
    t4 = _csv(tables / "table4_quantum_vs_classical.csv")
    t11 = _csv(tables / "table11_holdout.csv")
    t1 = _csv(tables / "table1_classical.csv")
    figs = [{"file": f, "url": f"/figures/{f}", "exists": (RESULTS / "figures" / f).exists()} for f in
            ("privacy_utility_curve.png", "roc_curves.png", "encoding_comparison.png", "federated_convergence.png")]
    return {"classical": _records(t1), "quantum_vs_classical": _records(t4[t4.Metric == "ROC-AUC"]),
            "quantum_vs_classical_all": _records(t4), "holdout": _records(t11), "figures": figs,
            "source": "results/tables (generated by src.m10_exp.final_report)"}


@app.get("/api/{rest:path}")
def unknown(rest: str):
    return JSONResponse(status_code=404, content={"error": f"unknown endpoint /api/{rest}"})


if (RESULTS / "figures").exists():
    app.mount("/figures", StaticFiles(directory=RESULTS / "figures"), name="figures")
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
