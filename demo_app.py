"""Architecture walkthrough demo — an Overview live trace plus one section per box of docs/architecture.jpg.

Start:  python demo_app.py        then open  http://127.0.0.1:8002/      (works offline)

The page reads demo/manifest.json (built from real results by scripts/build_demo_manifest.py) and one live endpoint.
POST /api/trace streams, stage by stage in diagram order, what the pipeline does to 50 consecutive rows of the locked
hold-out (hold-out indices 0-49, read from the NSCH file now): read -> de-identify/impute/scale -> 8 PCA angles ->
k-anonymity and local DP shown on one row -> 8-qubit statevectors -> QSVM kernel against the stored training states ->
risk bands. Inference only: nothing is trained or tuned, and no metric is computed from the 50 rows.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "demo" / "manifest.json"
CONFIG = ROOT / "configs" / "holdout_final.yaml"
app = FastAPI(title="Architecture walkthrough", docs_url=None, redoc_url=None)
_STATE: dict = {}
SHOW = ["sc_age_years", "sc_sex", "sc_racer", "sc_hispanic_r", "fpl_i1", "k2q01", "makefriend", "bullied_r"]
HEAD = {"sc_age_years": "Age", "sc_sex": "Sex", "sc_racer": "Race", "sc_hispanic_r": "Hispanic", "fpl_i1": "Poverty ratio",
        "k2q01": "General health", "makefriend": "Difficulty keeping friends", "bullied_r": "Bullied or excluded"}
BANDS = {"sc_age_years": {1: "12–13", 2: "14–15", 3: "16–17"},
         "fpl_i1": {1: "≤ 99 %", 2: "100–199 %", 3: "200–399 %", 4: "400 % (top code)"}}


def required_files() -> list[Path]:
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {"data": {"paths": []}}
    return [CONFIG, MANIFEST, ROOT / "models/config_models.pkl", ROOT / "models/screening_models.pkl",
            ROOT / "demo/assets/circuit_angle.svg", *[ROOT / p for p in cfg["data"]["paths"]],
            *[(ROOT / p).with_name("nsch_2024_topical.do") for p in cfg["data"]["paths"]]]


def preflight() -> None:
    """Fail loudly, before the demo, on any missing file the trace needs."""
    missing = [str(p.relative_to(ROOT)) for p in required_files() if not p.exists()]
    if missing:
        raise SystemExit("PRE-FLIGHT FAILED — missing: " + ", ".join(missing)
                         + "\nBuild with scripts/build_config_models.py, scripts/build_screening_models.py, "
                           "scripts/build_demo_manifest.py (see README).")


@app.exception_handler(Exception)
async def any_error(request, exc):
    return JSONResponse(status_code=503, content={"error": str(exc) or type(exc).__name__})


def _load() -> dict:
    if not _STATE:
        preflight()
        _STATE["manifest"] = json.loads(MANIFEST.read_text(encoding="utf-8"))
        _STATE["cache"] = pickle.load(open(ROOT / "models/config_models.pkl", "rb"))
        _STATE["screen"] = pickle.load(open(ROOT / "models/screening_models.pkl", "rb"))
        from src.m1_data.codebook import parse_codebook
        do_file = [f for f in required_files() if f.suffix == ".do"][0]
        _STATE["values"] = parse_codebook(do_file)[1]          # NSCH value labels: code -> meaning
    return _STATE


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    return v if isinstance(v, str) else f"{float(v):g}"


def _fingerprint(model) -> str:
    h = hashlib.sha256()
    for a in (model.svc_.dual_coef_, model.svc_.intercept_, model.train_states_):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def run_trace():
    """Yield one dict per stage, in diagram order. Inference only."""
    import pennylane as qml

    from src.m1_data.nsch import FEATURES
    from src.m4_encoding import make_encoding
    from src.m7_privacy.input_privacy import NSCH_QUASI_IDENTIFIERS, KAnonymizer, LaplaceLDP
    from src.m8_eval.screening import risk_band
    from src.m10_exp.common import holdout_split, load_config, load_dataset

    from src.m1_data.codebook import display_value

    s = _load()
    man, cache, screen = s["manifest"], s["cache"], s["screen"]

    def dv(col, v):
        """Codebook meaning shown; the loaded value and NSCH code go to the tooltip."""
        text, tip, _ = display_value(col, None if v is None or (isinstance(v, float) and np.isnan(v)) else v,
                                     FEATURES[col][0], s["values"])
        return {"v": text, "raw": tip}
    tr, labels = man["trace"], man["labels"]
    lab = lambda c: _short(labels.get(c, c))  # noqa: E731
    entry = cache["qsvm"][("angle", 8)]
    model, ft = entry["model"], entry["transformer"]
    fp_before = _fingerprint(model)
    t_all = time.perf_counter()

    # ---- Box 1: read the NSCH file, take hold-out indices start..start+49
    t0 = time.perf_counter()
    cfg = load_config(CONFIG)
    loader, frame = load_dataset(cfg)
    y = frame[loader.label_column].to_numpy(dtype=int)
    _, hold = holdout_split(y, float(cfg["split"]["holdout_fraction"]), int(cfg["project"]["seed"]))
    pos = list(range(tr["start"], tr["start"] + tr["count"]))
    idx = [int(hold[p]) for p in pos]
    if idx != tr["dataset_rows"]:
        raise RuntimeError("the hold-out split read now differs from demo/manifest.json — rebuild the manifest")
    rows = frame.iloc[idx]
    feats = rows[list(FEATURES)]
    yield {"box": 1, "ms": _ms(t0), "source": cfg["data"]["paths"][0],
           "detail": f"Read {len(frame):,} adolescents (12–17) from the NSCH 2024 file. Took hold-out indices {pos[0]} to {pos[-1]} "
                     f"in order. Each row has {len(FEATURES)} survey answers plus {len(loader.metadata_columns)} metadata columns; "
                     f"{len(SHOW)} answers shown, decoded with the NSCH 2024 codebook (hover a value for its raw code); — = missing.",
           "table": {"columns": ["Hold-out #", "Dataset row"] + [HEAD[c] for c in SHOW] + ["Missing"],
                     "rows": [[p, i] + [dv(c, rows.iloc[j][c]) for c in SHOW] + [int(feats.iloc[j].isna().sum())]
                              for j, (p, i) in enumerate(zip(pos, idx))]}}

    # ---- Box 2: drop identifiers, impute with training statistics, one-hot, standardise
    t0 = time.perf_counter()
    prep = cache["prep"]
    meta = [c for c in loader.metadata_columns if c in rows.columns]
    ct = prep.transformer
    fill = dict(zip(prep.numeric_columns_, ct.named_transformers_["numeric"].statistics_))
    fill.update(zip(prep.categorical_columns_, ct.named_transformers_["categorical"].named_steps["imputer"].statistics_))
    miss = feats.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    X95df = prep.transform(rows.drop(columns=meta).assign(**{c: np.nan for c in meta}), loader.label_column)
    X95 = X95df.to_numpy()
    sc = ft.scaler
    a = list(X95df.columns).index("sc_age_years")
    yield {"box": 2, "ms": _ms(t0), "source": "models/config_models.pkl (TabularPreprocessor and StandardScaler fitted on 1,200 training rows)",
           "kv": [["Dropped as identifiers / metadata", ", ".join(meta) + " (never model inputs)"],
                  ["Missing cells imputed", f"{int(miss.sum())} cells in {len(miss)} questions across the 50 rows"],
                  ["Encoded", f"{len(FEATURES)} answers → {X95.shape[1]} columns (categorical answers one-hot encoded)"],
                  ["Scaling", f"each of the {X95.shape[1]} columns standardised, z = (x − mean) / SD, with training-row mean and SD; "
                              f"e.g. Age: mean {sc.mean_[a]:.2f}, SD {sc.scale_[a]:.2f}"]],
           "table": {"columns": ["Question with missing answers", "Missing in these 50", "Filled with (training median / mode)"],
                     "rows": [[lab(c), int(n), dv(c, fill.get(c) if isinstance(fill.get(c), str) else float(fill.get(c)))]
                              for c, n in miss.head(8).items()]}}

    # ---- Box 3: the 8 model inputs = 8 PCA components; strongest questions in plain English
    t0 = time.perf_counter()
    Z = ft.transform(X95)
    comps, evr = ft.pca.components_, ft.pca.explained_variance_ratio_
    yield {"box": 3, "ms": _ms(t0), "source": "models/config_models.pkl (PCA fitted on 1,200 training rows); labels from the NSCH codebook",
           "detail": f"The quantum model's 8 inputs are 8 principal components of all {X95.shape[1]} columns "
                     f"({evr.sum():.1%} of training variance), one per qubit — not 8 hand-picked questions. "
                     "The two questions that weigh most in each:",
           "table": {"columns": ["Qubit", "Variance", "Strongest questions (plain English)"],
                     "rows": [[q, f"{evr[q]:.1%}", "; ".join(lab(X95df.columns[k]) for k in np.argsort(-np.abs(comps[q]))[:2])]
                              for q in range(8)]}}

    # ---- Box 4: k-anonymity on the 50 rows; local DP on one row's 8 angles (shown, not fed to the model)
    t0 = time.perf_counter()
    kan = KAnonymizer(k=tr["kanon"]["k"])
    kan.safe_keys_ = set(tr["kanon"]["safe_keys"])
    anon = kan.transform(rows)
    n_sup = kan.last_suppressed_
    ldp = LaplaceLDP(epsilon=tr["ldp"]["epsilon"], seed=tr["ldp"]["seed"])
    z_noisy = ldp.transform(Z[:1])[0]

    def gen(c, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "suppressed"
        return f"band {int(v)} ({BANDS[c][int(v)]})" if c in BANDS else dv(c, v)
    before_after = [[lab(c), dv(c, rows.iloc[0][c]), gen(c, anon.iloc[0][c])] for c in NSCH_QUASI_IDENTIFIERS]
    before_after += [[f"Angle, qubit {q}", f"{Z[0, q]:.3f}", f"{z_noisy[q]:.3f}"] for q in range(8)]
    yield {"box": 4, "ms": _ms(t0), "source": f"src/m7_privacy/input_privacy.py; safe groups fitted on {tr['kanon']['fit_rows']:,} training rows; "
                                              f"Experiment 5: {tr['ldp']['source']}",
           "kv": [["k-anonymity (k = 5, 7 quasi-identifiers)", f"{n_sup} of 50 rows had their quasi-identifiers suppressed "
                                                             f"(their generalised group has fewer than 5 training rows)"],
                  ["Local DP (Laplace, ε = " + f"{tr['ldp']['epsilon']:g})", f"noise scale {ldp.noise_scale(8):.3f} rad per angle; record-level, labels not touched"],
                  ["Used for the predictions below?", "No. The deployed QSVM was trained without input-stage privacy; Experiment 5 measured "
                                                      f"that regime: QSVM ROC-AUC {tr['ldp']['auc_k5_eps32']:.3f} with k = 5 and ε = 32 vs "
                                                      f"{tr['ldp']['auc_none']:.3f} without (17,431 development records, cross-validated)"]],
           "table": {"columns": [f"Hold-out # {pos[0]} (dataset row {idx[0]}): field", "Before", "After"], "rows": before_after}}

    # ---- Box 5: rotation angles for one row and its circuit
    yield {"box": 5, "ms": 0.0, "source": "angles computed in box 3; circuit drawn by scripts/build_demo_manifest.py for the same row",
           "detail": f"Hold-out # {pos[0]} (dataset row {idx[0]}): each angle θ is applied as RY(θ) on one qubit, |ψ⟩ = U(x)|0⟩. "
                     f"Angles for all 50 rows are computed the same way.",
           "kv": [[f"Qubit {q}", f"{Z[0, q]:.4f} rad"] for q in range(8)], "img": tr["circuit"]}

    # ---- Box 6: statevectors and the kernel circuit against the stored training states
    t0 = time.perf_counter()
    states = model._states(Z)
    K = model._kernel(states, model.train_states_)
    scores = model.svc_.decision_function(K)
    yield {"box": 6, "ms": _ms(t0), "source": "src/m5_models/quantum.py QuantumKernelSVC.decision_scores (same path); PennyLane default.qubit",
           "kv": [["Circuits simulated now", f"{len(states)} (one per row), 8 qubits, {states.shape[1]} amplitudes each"],
                  ["Kernel", f"k(x, x′) = |⟨ψ(x′)|ψ(x)⟩|² against the {model.train_states_.shape[0]:,} stored training states "
                             f"= {K.size:,} values"],
                  ["Decision", f"weighted by {len(model.svc_.support_)} support vectors; threshold {entry['threshold']:.3f} "
                               "(set on inner-validation rows)"],
                  ["Hold-out # " + str(pos[0]), f"‖ψ‖² = {np.sum(np.abs(states[0]) ** 2):.6f}; score {scores[0]:.3f}"]]}

    # ---- Box 7: inference only
    fp_after = _fingerprint(model)
    yield {"box": 7, "ms": 0.0, "source": "scripts/build_config_models.py (training, run once); models/config_models.pkl",
           "kv": [["Training", f"already happened, once, on {cache['n_train']:,} development rows (C = {entry['params']['C']}; "
                               "threshold from 300 inner-validation rows). The QSVM has no Adam/COBYLA loop; the VQC's loop is in Full evaluation."],
                  ["This run", "inference only — no loss evaluated, no parameter updated"],
                  ["Model fingerprint (SHA-256 of parameters)", f"before {fp_before} · after {fp_after} · "
                                                              + ("identical" if fp_before == fp_after else "CHANGED")]]}

    # ---- Box 8: where it ran
    yield {"box": 8, "ms": 0.0, "source": "this process",
           "kv": [["Simulator", f"PennyLane {qml.__version__}, default.qubit statevector, CPU"],
                  ["Wall-clock so far", f"{_ms(t_all):,.0f} ms for 50 rows, including reading the NSCH file"]]}

    # ---- Box 9: predictions and risk bands (real cut-points)
    t0 = time.perf_counter()
    cuts = screen["risk"]["cuts"]
    P = screen["risk"]["model"].predict_proba(X95)[:, 1]
    bands = [risk_band(float(p), cuts) for p in P]
    counts = {b: bands.count(b) for b in ("low", "moderate", "high")}
    flagged = int(np.sum(scores >= entry["threshold"]))
    yield {"box": 9, "ms": _ms(t0), "source": "models/screening_models.pkl (cut-points from src/m8_eval/screening.py risk_cutpoints)",
           "detail": f"Risk band cut-points {cuts['low_moderate']:.3f} / {cuts['moderate_high']:.3f} "
                     f"({cuts['sensitivity_target']:.0%} sensitivity / {cuts['specificity_target']:.0%} specificity on 300 inner-validation rows). "
                     f"Bands: {counts['low']} low · {counts['moderate']} moderate · {counts['high']} high. QSVM flagged {flagged} of 50. "
                     "No accuracy, recall, F1 or AUC is computed from these 50 rows.",
           "table": {"columns": ["Hold-out #", "Dataset row", "QSVM score", "QSVM", "Depression probability", "Risk band"],
                     "rows": [[p, i, f"{sc_:.3f}", "flagged" if sc_ >= entry["threshold"] else "—", f"{pr:.3f}", b.upper()]
                              for p, i, sc_, pr, b in zip(pos, idx, scores, P, bands)]},
           "done": True, "total_ms": _ms(t_all)}


def _short(label: str) -> str:
    """Codebook label without its bracketed footnote; long labels cut with an ellipsis (never reworded)."""
    import re
    t = re.sub(r"\s*\[[^\]]*\]", "", label)
    return t if len(t) <= 72 else t[:70].rstrip() + "…"


def _ms(t0: float) -> float:
    return round(1000 * (time.perf_counter() - t0), 1)


@app.post("/api/trace")
def trace():
    def stream():
        try:
            for stage in run_trace():
                yield json.dumps(stage, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"
        except Exception as exc:                                  # surfaced on screen, never hidden
            yield json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n"
    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/{rest:path}")
def unknown(rest: str):
    return JSONResponse(status_code=404, content={"error": f"unknown endpoint /api/{rest}"})


app.mount("/fonts", StaticFiles(directory=ROOT / "web" / "vendor" / "fonts"), name="fonts")
app.mount("/", StaticFiles(directory=ROOT / "demo", html=True), name="demo")

if __name__ == "__main__":
    import uvicorn

    preflight()
    print("Pre-flight OK. Architecture walkthrough: http://127.0.0.1:8002/")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="warning")
