"""Build demo/manifest.json (and demo/assets/) from REAL artifacts only. Nothing is trained here.

Every number in the walkthrough comes from a results/ artifact of a real run, the real NSCH 2024 file, or the
model caches in models/. If any required artifact is missing the script stops and names it; it never fills a gap.
Each sub-item of the 15 architecture boxes is either backed by real content (with its source file) or marked
"Not available on this dataset" with the reason.

Usage: python scripts/build_demo_manifest.py   ->  demo/manifest.json, demo/assets/*
"""

import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = ROOT / "results"
DEMO = ROOT / "demo"
ASSETS = DEMO / "assets"
RUNS = {
    "phase1": R / "phase1-nsch2024-depression-20261104-183743",
    "exp1": R / "exp1-nsch2024-encodings-final-20261108-160825",
    "exp1_basis": R / "exp1-nsch2024-encodings-with-basis-20261112",
    "exp2": R / "exp2-nsch2024-quantum-vs-classical-20261108-165426",
    "exp2_svm": R / "exp2-nsch2024-svm-baseline-20261112-121554",
    "exp2_cobyla": R / "exp2-nsch2024-vqc-cobyla-20261112-121718",
    "exp3": R / "exp3-nsch2024-privacy-utility-20261108-165618",
    "exp4": R / "exp4-nsch2024-federated-dp-20261108-170320",
    "exp5": R / "exp5-nsch2024-input-privacy-20261112-120542",
    "exp6": R / "exp6-nsch2024-membership-inference-20261112-120934",
    "exp7": R / "exp7-nsch2024-federated-cost-20261112-123743",
    "exp7b": R / "exp7b-nsch2024-federated-secure-aggregation-20261112-153004",
    "exp8": R / "exp8-nsch2024-hardware-noise-20261112-125925",
    "exp8b": R / "exp8b-nsch2024-hardware-noise-zz-vs-angle-4q-20261112-144158",
    "exp9": R / "exp9-nsch2024-multiclass-20261112-140242",
    "exp9b": R / "exp9b-nsch2024-multiclass-vqc-20261112-140937",
    "exp10": R / "exp10-nsch2024-umap-autoencoder-20261112-143551",
    "holdout": R / "holdout-20261108-173801",
    "holdout_mc": R / "holdout-multiclass-20261112-171606",
}
NA = "Not available on this dataset"


def need(path: Path) -> Path:
    if not Path(path).exists():
        raise SystemExit(f"MISSING ARTIFACT: {Path(path).relative_to(ROOT)} — run the experiment that produces it; "
                         "the demo never fills a gap.")
    return Path(path)


def rel(path: Path) -> str:
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


def csv(path):
    return pd.read_csv(need(path))


def r3(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)


def item(title, kind="kv", source=None, **kw):
    return {"title": title, "kind": kind, "source": source, **kw}


def na(title, reason):
    return {"title": title, "kind": "na", "reason": reason}


def figure(src: Path, name: str) -> str:
    ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copy(need(src), ASSETS / name)
    return f"assets/{name}"


def svg_circuit(draw_fn, name, *args):
    style = {"svg.fonttype": "none", "font.family": "DejaVu Sans Mono", "font.size": 13}
    with plt.rc_context(style):
        fig, _ = draw_fn(*args)
        fig.savefig(ASSETS / name, format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return f"assets/{name}"


def main() -> None:
    import pennylane as qml

    from src.m1_data.codebook import parse_codebook, plain_label
    from src.m1_data.nsch import FEATURES
    from src.m4_encoding import make_encoding
    from src.m5_models.quantum import _variational_layers
    from src.m7_privacy.input_privacy import NSCH_QUASI_IDENTIFIERS
    from src.m10_exp.common import holdout_split, load_config, load_dataset

    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir(parents=True)
    cfg = load_config(ROOT / "configs/holdout_final.yaml")
    seed = int(cfg["project"]["seed"])
    data_path = ROOT / cfg["data"]["paths"][0]
    var, vals = parse_codebook(need(data_path.with_name("nsch_2024_topical.do")))
    profile = json.loads(need(RUNS["phase1"] / "dataset_profile.json").read_text())
    cache = pickle.load(open(need(ROOT / "models/config_models.pkl"), "rb"))
    screen = pickle.load(open(need(ROOT / "models/screening_models.pkl"), "rb"))
    with pd.io.stata.StataReader(need(data_path)) as rdr:
        raw_cols = list(rdr.variable_labels())
    loader, frame = load_dataset(cfg)
    y = frame[loader.label_column].to_numpy(dtype=int)
    dev, hold = holdout_split(y, float(cfg["split"]["holdout_fraction"]), seed)
    m95 = list(cache["prep"].transform(frame.iloc[:1], loader.label_column).columns)
    lab = lambda c: plain_label(c, var, vals)  # noqa: E731
    sections = []

    # ------------------------------------------------------------ live row: first hold-out row (no selection)
    live_index = int(hold[0])
    live_raw = frame.iloc[live_index]
    live = {"row_index": live_index, "split": "locked hold-out (never used to train or tune any model)",
            "survey_label": int(live_raw[loader.label_column]),
            "raw": {c: (None if pd.isna(live_raw[c]) else (live_raw[c] if isinstance(live_raw[c], str) else float(live_raw[c])))
                    for c in list(FEATURES) + list(loader.metadata_columns)},
            "identifier_columns": list(loader.metadata_columns)}

    def answer(col, v):                                      # codebook meaning of a raw answer (never invented)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "missing"
        kind = FEATURES[col][0]
        code = int(v[1:]) if isinstance(v, str) and v.startswith("c") else (1 if v == 1 else 2) if kind == "binary" else int(v) if float(v).is_integer() else None
        meaning = vals.get(col, {}).get(code) if code is not None else None
        return f"{meaning} ({v if not isinstance(v, float) else f'{v:g}'})" if meaning else (v if isinstance(v, str) else f"{v:g}")
    live["display"] = {c: answer(c, live["raw"][c]) for c in FEATURES}
    X_live = cache["prep"].transform(frame.iloc[[live_index]], loader.label_column).to_numpy()

    # ------------------------------------------------------------ Box 1
    n_rows = profile["n_dev"] + profile["n_holdout"]
    sections.append({"box": 1, "items": [
        na("Clinical notes (de-identified)", "No clinical-notes dataset is accessible to this project."),
        na("Counselling records", "No counselling-records dataset is accessible to this project."),
        item("Survey responses", source=rel(RUNS["phase1"] / "dataset_profile.json"), rows=[
            ["Dataset", "National Survey of Children's Health (NSCH) 2024, topical public-use file"],
            ["Adolescents (rows)", f"{n_rows:,}"], ["Age range", "12–17 years"],
            ["Questions used", f"{profile['n_raw_features']} whitelisted answers → {len(m95)} model columns"],
            ["Positive rate (current depression diagnosis)", f"{profile['n_positive'] / n_rows:.2%} ({profile['n_positive']:,} adolescents)"],
            ["Split", f"{profile['n_dev']:,} development rows · {profile['n_holdout']:,} locked hold-out rows"]]),
        na("Social media text", "No social-media dataset linked to mental-health outcomes is accessible."),
        na("Wearable/IoT data (optional)", "No physiological sensor data is accessible; NSCH has none."),
        item("Public datasets", source="data/README.md", rows=[
            ["Source", "U.S. Census Bureau for HRSA Maternal and Child Health Bureau (public download, checksum recorded)"],
            ["Raw file", rel(data_path)]]),
        item("Ethical Use | Consent | De-identification", source="data/README.md", rows=[
            ["De-identification", "Public-use file is de-identified by the Census Bureau; identifier/metadata columns are never model inputs"],
            ["Consent", "Collected by the survey under its own consent process; the project runs no consent process of its own"],
            ["Use", "Research configuration only — not for clinical or individual decisions"]])]})

    # ------------------------------------------------------------ Box 2
    sections.append({"box": 2, "items": [
        item("Data cleaning", source="src/m1_data/nsch.py", rows=[
            ["Raw columns in the file", f"{len(raw_cols):,}"],
            ["Kept as model inputs", f"{len(FEATURES)} whitelisted survey answers"],
            ["Not used", f"{len(raw_cols) - len(FEATURES):,} columns (diagnosis/treatment items, administrative fields, all others)"],
            ["Age filter", "12–17 years"],
            ["Unknown labels", "rows with an undeterminable depression answer dropped (21,960 aged 12–17 → 21,789)"],
            ["Leakage guard", "any column starting k2q3/k2q60/sc_k2q/k4q2 or in the treatment list is refused (assert_no_leakage)"]]),
        na("Noise removal", "Not applicable: answers are categorical/ordinal survey codes; there is no signal noise to remove."),
        item("Tokenization / Encoding", source="src/m2_preprocess/preprocessing.py", rows=[
            ["Tokenization", f"{NA}: there is no text"],
            ["Encoding", f"{profile['n_categorical_features']} categorical questions one-hot encoded → {len(m95)} columns in total"]]),
        item("Anonymization (PII removal)", source="src/m2_preprocess/preprocessing.py", rows=[
            ["Identifier / metadata columns removed before modelling", ", ".join(loader.metadata_columns) + " (3 columns)"],
            ["k-anonymity (Experiment 5 only)", "k = 5 on 7 demographic quasi-identifiers — see box 4"]]),
        item("Normalization", source="src/m3_features/features.py", rows=[
            ["Missing values", f"medians / modes from training rows (missing-cell rate {profile['missing_cell_rate']:.2%})"],
            ["Scaling", "standardisation, then [0, π] rotation angles for quantum inputs — all fitted on training rows"]]),
        item("Train-test split", source="src/m10_exp/common.py · tests/test_m10_runner_no_leak.py", rows=[
            ["Locked hold-out", f"{profile['n_holdout']:,} rows (20 %, stratified), scored once at the end"],
            ["Cross-validation", "repeated stratified 5-fold on the development rows (10 or 15 folds)"],
            ["Fold-wise fitting rule", "imputation, scaling, feature selection and PCA are fitted inside each fold on its training rows only"]])]})

    # ------------------------------------------------------------ Box 3
    mi = csv(R / "feature_importance/mi_scores.csv")
    qi = set(NSCH_QUASI_IDENTIFIERS)
    demo_feats = [[lab(c), c] for c in FEATURES if c in qi]
    beh_feats = [[lab(c), c] for c in FEATURES if c not in qi]
    pca8 = cache["qsvm"][("angle", 8)]["transformer"]
    ev = float(np.sum(pca8.pca.explained_variance_ratio_))
    top8 = mi[mi.selected_top8].sort_values("rank")
    sections.append({"box": 3, "items": [
        {"title": "Text Features", "kind": "group", "items": [
            na("Transformer embeddings (e.g., BERT, RoBERTa)", "Needs free text; the data is a structured survey."),
            na("Sentiment & emotion features", "Needs free text."),
            na("Contextual representations", "Needs free text.")]},
        {"title": "Structured Features", "kind": "group", "items": [
            item(f"Behavioral indicators ({len(beh_feats)} questions)", kind="list", source="src/m1_data/nsch.py · official NSCH codebook labels",
                 entries=beh_feats),
            item(f"Demographic (anonymized) ({len(demo_feats)} questions)", kind="list", source="src/m7_privacy/input_privacy.py (quasi-identifiers)",
                 entries=demo_feats),
            na("Physiological signals (optional)", "NSCH records no physiological signals.")]},
        item("Dimensionality Reduction (e.g., PCA, UMAP, Autoencoder)", kind="figure",
             src=figure(R / "feature_importance/mi_ranking.png", "mi_ranking.png"),
             source="results/feature_importance/mi_scores.csv (scripts/build_feature_importance.py, 17,431 development rows)",
             rows=[["95 → 8 for the quantum model", f"PCA to 8 components (keeps {ev:.1%} of the variance on the 1,200 training rows), "
                    "each scaled to a rotation angle in [0, π] — one per qubit"],
                   ["Mutual-information alternative (8 kept)", "; ".join(top8.label.str.slice(0, 48))],
                   ["UMAP / autoencoder (Experiment 10 only)", "ROC-AUC with the QSVM: PCA 0.847 · autoencoder 0.837 · UMAP 0.774 "
                    "(results/exp10-…/reduction_report.md)"]])]})

    # ------------------------------------------------------------ Box 4
    e3 = csv(RUNS["exp3"] / "summary.csv")
    e3["eps"] = ["∞" if str(e) in ("inf", "nan") else f"{float(e):g}" for e in e3.target_epsilon]
    e5 = csv(RUNS["exp5"] / "summary.csv")
    e6 = csv(RUNS["exp6"] / "summary.csv")
    ka = e5[(e5.k_anon == "k=5") & (e5.epsilon_input.astype(str).isin(["inf", "nan"])) & (e5.arm == "logreg") & (e5.regime == "full")]
    ldp = e5[(e5.arm == "qsvm") & (e5.k_anon == "off")].copy()
    ldp["e"] = ["∞" if str(v) in ("inf", "nan") else f"{float(v):g}" for v in ldp.epsilon_input]
    lf = e3[e3.arm == "logreg_full"].set_index("eps")
    oh = csv(R / "secagg_overhead/overhead.csv")
    sections.append({"box": 4, "items": [
        item("Differential Privacy (add calibrated noise)", kind="table", source=rel(RUNS["exp3"] / "summary.csv"),
             columns=["Target ε", "ε spent (Opacus RDP accountant)", "Noise multiplier σ", "ROC-AUC (LogReg, full data)"],
             table=[[e, r3(lf.loc[e, "epsilon_spent_mean"]) if e != "∞" else "—", r3(lf.loc[e, "noise_multiplier"]) if e != "∞" else "—",
                     f"{lf.loc[e, 'roc_auc_mean']:.3f}"] for e in ["0.5", "1", "2", "4", "8", "∞"]],
             note="δ = 10⁻⁵. Local DP on inputs (Experiment 5, Laplace, ε_input separate from ε_training): QSVM ROC-AUC "
                  + " · ".join(f"ε_input {r.e}: {r.roc_auc_mean:.3f}" for _, r in ldp.iterrows())),
        item("Federated Simulation (distributed clients)", source=rel(RUNS["exp4"] / "summary.csv"), rows=[
            ["Algorithm", "FedAvg, 30 rounds, client-side DP-SGD at ε = 4"], ["Clients", "3, 5 and 10; IID or split by U.S. state"]]),
        item("Secure Aggregation (encrypted model updates)", source=rel(R / "secagg_overhead/overhead.csv"), rows=[
            ["Scheme", "pairwise Diffie–Hellman masks (RFC 3526 2048-bit group), fixed-point mod 2⁶⁴; server sees only the sum"],
            ["Effect on results (Experiment 7b)", "identical to unmasked FedAvg in 30 of 30 runs"],
            ["One-off key setup (3 / 5 / 10 clients)", " / ".join(f"{s:.2f} s" for s in oh.drop_duplicates("clients").setup_s_once)],
            ["Used in", "Experiments 7b and 7c only; the default federated runs (Experiment 4) do not use it"]]),
        na("Homomorphic Encryption (on simulated data)", "Not implemented: impractical at this scale, and a toy version would misrepresent its cost."),
        item("Privacy Risk Analysis (membership inference test)", kind="table", source=rel(RUNS["exp6"] / "summary.csv"),
             columns=["Model", "Target ε", "Attack ROC-AUC (0.5 = guessing)"],
             table=[[r.arm, "∞" if str(r.target_epsilon) in ("inf", "nan") else f"{float(r.target_epsilon):g}", f"{r.attack_auc_mean:.3f}"]
                    for _, r in e6.iterrows()]),
        item("k-anonymity (input stage, Experiment 5)", source=rel(RUNS["exp5"] / "summary.csv"), rows=[
            ["k", "5, on 7 demographic quasi-identifiers"],
            ["Rows suppressed (training)", f"{ka.suppression_rate_train.iloc[0]:.1%}"],
            ["LogReg ROC-AUC with k = 5", f"{ka.roc_auc_mean.iloc[0]:.3f} (without: {e5[(e5.arm == 'logreg') & (e5.regime == 'full') & (e5.k_anon == 'off') & (e5.epsilon_input.astype(str).isin(['inf', 'nan']))].roc_auc_mean.iloc[0]:.3f})"]])]})

    # ------------------------------------------------------------ Box 5
    res = csv(RUNS["exp1_basis"] / "circuit_resources.csv")
    circuits = []
    for enc in ("angle", "amplitude", "basis", "zz", "reupload"):
        e = make_encoding(enc, 8)
        if enc == "basis":
            ft = cache["qsvm"][("angle", 8)]["transformer"]
        else:
            ft = cache["qsvm"][(enc, 8)]["transformer"]
        x = e.prepare(ft.transform(X_live))[0]

        @qml.qnode(qml.device("default.qubit", wires=8))
        def circ(v, e=e):
            e.feature_map(v)
            return qml.state()
        src = svg_circuit(lambda v, c=circ, enc=enc: qml.draw_mpl(c, decimals=None if enc == "amplitude" else 2, level="device")(v),
                          f"circuit_{enc}.svg", x)
        rr = res[(res.encoding == enc) & (res.n_qubits == 8) & (res.n_layers == 0)]
        circuits.append({"encoding": enc, "src": src,
                         "gates": None if rr.empty else int(rr.n_gates.iloc[0]), "depth": None if rr.empty else int(rr.depth.iloc[0])})
    sections.append({"box": 5, "items": [
        item("Amplitude encoding · Angle encoding · Basis encoding · Quantum feature maps", kind="circuits",
             source=f"live record = hold-out row {live_index}; transforms from models/config_models.pkl; gate counts {rel(RUNS['exp1_basis'] / 'circuit_resources.csv')}",
             circuits=circuits, note="8 qubits. Amplitude: 95 values in 256 amplitudes. Basis: each angle binarised at π/2. "
                                     "Quantum feature maps: ZZ (entangling) and data re-uploading.")]})

    # ------------------------------------------------------------ Box 6
    s2 = csv(RUNS["exp2"] / "summary.csv")
    params = {m: s2[(s2.model == m) & (s2.regime == "subsample")].n_params_mean.iloc[0] for m in ("qsvm", "vqc", "hybrid")}
    vq = make_encoding("reupload", 4)

    @qml.qnode(qml.device("default.qubit", wires=4))
    def vqc_circ(x, w):
        _variational_layers(vq, x, w)
        return qml.expval(qml.PauliZ(0))
    xv = cache["qsvm"][("reupload", 4)]["transformer"].transform(X_live)[0]
    vqc_src = svg_circuit(lambda: qml.draw_mpl(vqc_circ, decimals=None, level="device")(xv, np.zeros((2, 4, 3))), "circuit_vqc.svg")
    qc = csv(R / "qiskit_crosscheck/crosscheck.csv")
    sections.append({"box": 6, "items": [
        item("Parameterized quantum gates · Entanglement layers", kind="figure", src=vqc_src,
             source=f"src/m5_models/quantum.py; structure of the Experiment 2 VQC drawn for hold-out row {live_index} (trained weights hidden)",
             rows=[["VQC", "re-uploading encoding, 4 qubits, 2 strongly-entangling layers (Rot gates + CNOT entanglers), readout ⟨Z₀⟩"],
                   ["Trained parameters per model (Experiment 2)", f"VQC {params['vqc']:.0f} · hybrid QNN {params['hybrid']:.0f} · "
                    f"QSVM {params['qsvm']:.0f} (dual coefficients) · LogReg 96"]]),
        item("Quantum simulators (Qiskit, PennyLane)", source="results/qiskit_crosscheck/crosscheck.csv", rows=[
            ["PennyLane", "all training and evaluation (default.qubit statevector; default.mixed for noise)"],
            ["Qiskit", f"cross-check only: re-scores the trained VQCs; max score difference {qc.max_abs_score_diff.max():.1e} over "
                       f"{int(qc.n_val.sum()):,} rows, predictions identical in {int(qc.same_predictions.sum())}/{len(qc)} folds"]])]})

    # ------------------------------------------------------------ Box 7
    rh = csv(RUNS["exp4"] / "round_history.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for arm, color, name in (("logreg_full", "#4269d0", "Logistic regression (full data)"), ("vqc_sub", "#3c9e86", "VQC (1.2k rows)")):
        h = rh[(rh.arm == arm) & (rh.n_clients == 3) & (rh.partition == "by_state")].groupby("round").val_auc.mean()
        ax.plot(h.index, h.values, color=color, lw=2.4, label=name)
    ax.set_xlabel("Training round"); ax.set_ylabel("Validation ROC-AUC (mean of 10 folds)")
    ax.set_title("Federated training, 3 clients, DP ε = 4", loc="left")
    ax.grid(color="#e4e4e7"); ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(ASSETS / "training_curve.png", dpi=150, facecolor="white"); plt.close(fig)
    cob = csv(RUNS["exp2_cobyla"] / "comparison.csv")
    cob = cob[cob.metric == "roc_auc"].iloc[0]
    sections.append({"box": 7, "items": [
        item("Classical Optimizer (e.g., Adam, COBYLA) ↔ Loss Function (Classification / Regression)", kind="loop",
             source="src/m5_models/quantum.py · configs/exp2_nsch2024.yaml · " + rel(RUNS["exp2_cobyla"] / "comparison.csv"),
             optimizer=["Adam, learning rate 0.01 or 0.05 (chosen on inner-validation rows), 30 epochs, batch 64",
                        f"COBYLA (gradient-free, 300 evaluations): ROC-AUC {cob.a_mean:.3f} vs Adam {cob.b_mean:.3f} — not significant (Holm p {cob.p_nb_t_holm:.2f})"],
             loss=["Classification: class-weighted binary cross-entropy", f"Regression: {NA} (no continuous target)"],
             note="This loop is where the learning happens: the circuit is evaluated, the loss computed, and the optimiser updates the gate angles."),
        item("Training curve", kind="figure", src="assets/training_curve.png", source=rel(RUNS["exp4"] / "round_history.csv"))]})

    # ------------------------------------------------------------ Box 8
    s1 = csv(RUNS["exp1"] / "summary.csv")
    e8 = csv(RUNS["exp8"] / "comparison.csv"); e8 = e8[e8.metric == "roc_auc"]
    e8b = csv(RUNS["exp8b"] / "comparison.csv"); e8b = e8b[e8b.metric == "roc_auc"]
    e4 = csv(RUNS["exp4"] / "summary.csv")
    cc = csv(RUNS["exp7"] / "communication_cost.csv")
    qcount = s1[s1.model == "qsvm"].groupby("n_qubits").roc_auc_mean.max()
    sections.append({"box": 8, "items": [
        item("Qiskit · PennyLane · Cirq · TensorFlow Quantum", kind="table", source="requirements.txt",
             columns=["Simulator", "Status"],
             table=[["PennyLane", "used for all training and evaluation"], ["Qiskit", "cross-check of the VQC only (box 6)"],
                    ["Cirq", f"{NA}: not used — redundant with PennyLane and Qiskit"], ["TensorFlow Quantum", f"{NA}: not used — redundant"]]),
        item("Noise simulation (realistic quantum hardware)", kind="table", source=f"{rel(RUNS['exp8'] / 'comparison.csv')} · {rel(RUNS['exp8b'] / 'comparison.csv')}",
             columns=["Kernel", "Setting", "ROC-AUC", "vs noise-free (Holm p)"],
             table=[["Angle, 8 qubits", r.a.split("/")[0].replace("qsvm_", ""), f"{r.a_mean:.3f}", f"{r.p_nb_t_holm:.2f}"] for _, r in e8[e8.b.str.startswith("qsvm/")].iterrows()]
                   + [["ZZ, 4 qubits", r.a.split("/")[0].replace("qsvm_zz_", ""), f"{r.a_mean:.3f}", f"{r.p_nb_t_holm:.2f}"]
                      for _, r in e8b[e8b.a.str.startswith("qsvm_zz_noise") & e8b.b.str.startswith("qsvm_zz_ideal")].iterrows()],
             note="No noisy setting differs significantly from its noise-free baseline under the primary test. No real hardware was used."),
        item("Different number of qubits", kind="table", source=rel(RUNS["exp1"] / "summary.csv"), columns=["Qubits", "Best QSVM ROC-AUC"],
             table=[[int(k), f"{v:.3f}"] for k, v in qcount.items()]),
        item("Multiple quantum circuit designs", source=rel(RUNS["exp1"] / "summary.csv"), rows=[
            ["Encodings", "angle, amplitude, ZZ, re-uploading (+ basis)"], ["Models", "QSVM, VQC, hybrid QNN"]]),
        item("Scalability experiments (number of clients) · Heterogeneous data simulation (non-IID)", kind="table", source=rel(RUNS["exp4"] / "summary.csv"),
             columns=["Model", "Clients", "ROC-AUC, IID split", "ROC-AUC, by U.S. state (non-IID)"],
             table=[[arm, int(n)] + [f"{g[g.partition == part].roc_auc_mean.iloc[0]:.3f}" if (g.partition == part).any() else "not run"
                                     for part in ("iid", "by_state")]
                    for (arm, n), g in e4[e4.setting == "federated"].groupby(["arm", "n_clients"])]),
        item("Communication constraints (simulated federated setup)", kind="table", source=rel(RUNS["exp7"] / "communication_cost.csv"),
             columns=["Model", "Clients", "Bytes per update", "kB per round", "Server ms per round"],
             table=[[r.arm, int(r.clients), int(r.payload_bytes_per_update), f"{r.payload_kB_per_round:.1f}", f"{r.aggregation_ms_per_round:.2f}"]
                    for _, r in cc.iterrows()], note="In-process simulation: no network, so network transfer time is not measured.")]})

    # ------------------------------------------------------------ Box 9
    rk = screen["risk"]
    hm = csv(RUNS["holdout_mc"] / "holdout_multiclass_metrics.csv").set_index("model")
    sections.append({"box": 9, "items": [
        item("Mental health state classification (e.g., normal, stress, anxiety, depression)", source=rel(RUNS["holdout_mc"] / "holdout_multiclass_metrics.csv"), rows=[
            ["What is predicted", "binary: currently diagnosed depression (yes/no); three classes: none / anxiety only / depression"],
            ["On whom", "adolescents aged 12–17 in NSCH 2024 (caregiver-reported diagnoses)"],
            ["Stress", f"{NA}: NSCH records no stress variable"],
            ["Three classes, locked hold-out macro ROC-AUC", f"LogReg {hm.loc['logreg_full_all95', 'three_class_macro_ovr_auc']:.3f} · "
             f"QSVM {hm.loc['qsvm_angle_pca8_1500rows', 'three_class_macro_ovr_auc']:.3f} · VQC {hm.loc['vqc_ovr_reupload_pca4_1500rows', 'three_class_macro_ovr_auc']:.3f}"]]),
        item("Risk level prediction", kind="table", source="src/m8_eval/screening.py · models/screening_models.pkl",
             columns=["Band", "Depression probability", "Inner-validation rows", "of which depressed"],
             table=[["Low", f"< {rk['cuts']['low_moderate']:.3f}", rk["inner_val_bands"]["low"]["rows"], rk["inner_val_bands"]["low"]["positives"]],
                    ["Moderate", f"{rk['cuts']['low_moderate']:.3f} – {rk['cuts']['moderate_high']:.3f}", rk["inner_val_bands"]["moderate"]["rows"], rk["inner_val_bands"]["moderate"]["positives"]],
                    ["High", f"≥ {rk['cuts']['moderate_high']:.3f}", rk["inner_val_bands"]["high"]["rows"], rk["inner_val_bands"]["high"]["positives"]]],
             note=f"Cut-points: 95 % sensitivity / 95 % specificity on {rk['n_inner_val']} inner-validation rows ({rk['n_inner_val_pos']} positive). "
                  "A screening indication, not a diagnosis."),
        na("Topic and trend analysis", "Needs free text; the survey is a single cross-sectional snapshot."),
        na("Emotion and sentiment analysis", "Needs free text."),
        na("Personalized insights (simulated)", "Not built; would need text or longitudinal data.")]})

    # ------------------------------------------------------------ Box 10 / 11
    fm = csv(RUNS["exp2"] / "fold_metrics.csv")
    fm = pd.concat([fm, csv(RUNS["exp2_svm"] / "fold_metrics.csv")], ignore_index=True)
    p1 = csv(RUNS["phase1"] / "summary.csv")
    rf = p1[p1.model == "random_forest"].sort_values("roc_auc_mean").iloc[-1]
    names = {"logreg": "Logistic Regression", "svm": "SVM (RBF)", "xgboost": "XGBoost", "mlp": "MLP",
             "qsvm": "QSVM (quantum kernel)", "vqc": "VQC", "hybrid": "Hybrid QNN"}
    rows11 = []
    keep11 = {("logreg", "full", "all95"), ("svm", "full", "all95"), ("xgboost", "full", "all95"), ("mlp", "full", "all95"),
              ("logreg", "subsample", "pca8"), ("svm", "subsample", "pca8"), ("qsvm", "subsample", "pca8"),
              ("logreg", "subsample", "pca4"), ("vqc", "subsample", "pca4"), ("logreg", "subsample", "all95"),
              ("hybrid", "subsample", "all95")}
    order11 = sorted(keep11, key=lambda k: (k[1] != "full", k[2], k[0] not in ("logreg", "svm", "xgboost", "mlp")))
    for (m, reg, fs) in order11:
        g = fm[(fm.model == m) & (fm.regime == reg) & (fm.feature_set == fs)]
        rows11.append([names[m], "1.2k rows" if reg == "subsample" else "full data", fs, f"{g.accuracy.mean():.3f}", f"{g.precision.mean():.3f}",
                       f"{g.recall.mean():.3f}", f"{g.f1.mean():.3f}", f"{g.roc_auc.mean():.3f}", f"{g.fit_seconds.mean():.2f}"])
    t = {m: fm[(fm.model == m) & (fm.regime == "subsample") & (fm.feature_set == fs)].roc_auc.mean()
         for m, fs in (("logreg", "pca8"), ("svm", "pca8"), ("qsvm", "pca8"), ("vqc", "pca4"), ("hybrid", "all95"))}
    full = {m: fm[(fm.model == m) & (fm.regime == "full") & (fm.feature_set == "all95")].roc_auc.mean() for m in ("logreg", "svm")}
    sections.append({"box": 10, "items": [
        item("Classical Baselines | Quantum Models", kind="twocol", source=rel(RUNS["exp2"] / "summary.csv") + " · " + rel(RUNS["phase1"] / "summary.csv"),
             left=[["Logistic Regression", f"ROC-AUC {full['logreg']:.3f} (full data) · {t['logreg']:.3f} (same 1.2k rows / 8 features as QSVM)"],
                   ["SVM", f"{full['svm']:.3f} (full data) · {t['svm']:.3f} (1.2k rows)"],
                   ["Random Forest", f"{rf.roc_auc_mean:.3f} (Phase 1, {rf.feature_method} {int(rf.n_features)}); excluded from later results on supervisor instruction"],
                   ["BERT + ML", f"{NA}: needs text"], ["Deep Learning (LSTM, CNN)", f"{NA}: needs sequence data; an MLP is used instead"]],
             right=[["VQC", f"{t['vqc']:.3f} (1.2k rows, 4 features)"], ["Quantum Neural Network", f"{t['hybrid']:.3f} — the hybrid QNN class (same class as the next row)"],
                    ["Hybrid Quantum-Classical", f"{t['hybrid']:.3f} (1.2k rows, 95 features compressed to 8 qubits)"],
                    ["QSVM (not in the diagram)", f"{t['qsvm']:.3f} (1.2k rows, 8 features) — ties LogReg {t['logreg']:.3f} (Holm p 0.70)"]],
             note="Matched comparison: same rows, same features, same 15 folds. No quantum model beats logistic regression.")]})
    sections.append({"box": 11, "items": [
        item("Accuracy · Precision · Recall · F1-score · Training time", kind="table", source=rel(RUNS["exp2"] / "fold_metrics.csv") + " · " + rel(RUNS["exp2_svm"] / "fold_metrics.csv"),
             columns=["Model", "Rows", "Features", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Fit s"], table=rows11,
             note="Mean over 15 folds; the 1.2k-row rows are matched pairs (same rows and features). 9.96 % of adolescents are positive, "
                  "so accuracy can be high while recall is low — read them together. Precision is low for every model at this prevalence."),
        item("Resource usage (qubits, gates)", kind="table", source=rel(RUNS["exp1"] / "circuit_resources.csv"),
             columns=["Encoding", "Qubits", "Circuit", "Depth", "Gates", "CNOTs"],
             table=[[r.encoding, int(r.n_qubits), r.circuit.split(" ")[0], int(r.depth), int(r.n_gates), int(r.n_cnot)]
                    for _, r in res[res.n_qubits.isin([4, 8]) & ((res.n_layers == 0) | (res.encoding == "reupload"))].iterrows()])]})

    # ------------------------------------------------------------ Box 12
    rid = csv(R / "reidentification/risk.csv")
    p_dp = csv(RUNS["exp3"] / "tests_private_vs_nonprivate.csv")
    pd4 = p_dp[(p_dp.arm == "logreg_full") & (p_dp.metric == "roc_auc") & (p_dp.epsilon.astype(str).isin(["4", "4.0"]))].iloc[0]
    sections.append({"box": 12, "items": [
        item("Membership inference risk", kind="table", source=rel(RUNS["exp6"] / "summary.csv"), columns=["Model", "Target ε", "Attack ROC-AUC"],
             table=[[r.arm, "∞" if str(r.target_epsilon) in ("inf", "nan") else f"{float(r.target_epsilon):g}", f"{r.attack_auc_mean:.3f}"] for _, r in e6.iterrows()],
             note="About 0.50 for every real model at every ε: no measurable leakage. The overfitted MLP is a positive control."),
        item("Re-identification risk", kind="table", source="results/reidentification/risk.csv", columns=["Quasi-identifier table", "Unique rows", "Max re-identification risk"],
             table=[[r.table, f"{r.uniqueness:.1%}", f"{r.max_prosecutor_risk:.2f}"] for _, r in rid.iterrows()]),
        item("Effect of privacy parameters (ε, noise level)", kind="table", source=rel(RUNS["exp3"] / "summary.csv"),
             columns=["ε", "σ (LogReg full)", "LogReg full ROC-AUC", "VQC ROC-AUC"],
             table=[[e, r3(lf.loc[e, "noise_multiplier"]) if e != "∞" else "—", f"{lf.loc[e, 'roc_auc_mean']:.3f}",
                     f"{e3[(e3.arm == 'vqc_sub') & (e3.eps == e)].roc_auc_mean.iloc[0]:.3f}"] for e in ["0.5", "1", "2", "4", "8", "∞"]]),
        item("Data utility trade-off", kind="figure", src=figure(R / "figures/privacy_utility_curve.png", "privacy_utility_curve.png"),
             source="results/figures/privacy_utility_curve.png · " + rel(RUNS["exp3"] / "tests_private_vs_nonprivate.csv"),
             rows=[["DP cost at ε = 4 (LogReg, full data)", f"{pd4.change:+.3f} ROC-AUC — NOT statistically significant (Holm p {pd4.p_nb_t_holm:.2f})"]])]})

    # ------------------------------------------------------------ Box 13
    f4 = csv(RUNS["exp4"] / "fold_metrics.csv")
    split = f4[(f4.setting == "federated")].groupby(["partition", "n_clients"]).agg(
        rows_min=("client_size_min", "min"), rows_max=("client_size_max", "max"),
        prev_min=("client_prevalence_min", "min"), prev_max=("client_prevalence_max", "max")).reset_index()
    sections.append({"box": 13, "items": [
        item("Varying number of clients", kind="table", source=rel(RUNS["exp4"] / "summary.csv"), columns=["Model", "Clients", "Split", "ROC-AUC", "Max client ε"],
             table=[[r.arm, int(r.n_clients), r.partition, f"{r.roc_auc_mean:.3f}", r3(r.epsilon_spent_max)] for _, r in e4.iterrows()]),
        item("Non-IID data distribution", kind="table", source=rel(RUNS["exp4"] / "fold_metrics.csv"),
             columns=["Split", "Clients", "Client rows (min–max)", "Depression rate per client (min–max)"],
             table=[[r.partition, int(r.n_clients), f"{int(r.rows_min):,}–{int(r.rows_max):,}", f"{r.prev_min:.1%}–{r.prev_max:.1%}"] for _, r in split.iterrows()]),
        item("Communication latency simulation", kind="table", source=rel(RUNS["exp7"] / "communication_cost.csv"),
             columns=["Model", "Clients", "Slowest client s/round", "Rounds to within 0.01 of best"],
             table=[[r.arm, int(r.clients), f"{r.client_seconds_per_round_slowest:.2f}", r["rounds_to_within_0.01_of_best"]] for _, r in cc.iterrows()],
             note="Compute time and payload are measured; network transfer time is not (in-process simulation)."),
        item("Server aggregation load", kind="table", source=rel(RUNS["exp7"] / "communication_cost.csv"), columns=["Model", "Clients", "Server ms per round", "p95 ms"],
             table=[[r.arm, int(r.clients), f"{r.aggregation_ms_per_round:.2f}", f"{r.aggregation_ms_p95:.2f}"] for _, r in cc.iterrows()])]})

    # ------------------------------------------------------------ Box 14
    st = qml.device("default.qubit", wires=8)
    ea = make_encoding("angle", 8)

    @qml.qnode(st)
    def state(v):
        ea.feature_map(v)
        return qml.state()
    probs = np.abs(np.asarray(state(cache["qsvm"][("angle", 8)]["transformer"].transform(X_live)[0]))) ** 2
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(np.arange(256), probs, color="#3c9e86", width=1.0)
    ax.set_xlabel("Basis state (0–255)"); ax.set_ylabel("Probability")
    ax.set_title(f"State vector of hold-out row {live_index}, angle encoding, 8 qubits", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(ASSETS / "state_vector.png", dpi=150, facecolor="white"); plt.close(fig)
    figs = [(R / "figures" / f, f) for f in ("encoding_comparison.png", "roc_curves.png", "federated_convergence.png")]
    sections.append({"box": 14, "items": [
        item("Performance comparison graphs", kind="figures", source="results/figures/",
             figures=[{"src": figure(p, n), "source": rel(p)} for p, n in figs]),
        item("Feature and attention visualization", kind="figure", src="assets/mi_ranking.png", source="results/feature_importance/mi_scores.csv",
             rows=[["Attention", f"{NA}: needs a transformer model over text"]]),
        item("Quantum state visualization (Bloch sphere, state vector)", kind="figures",
             source="results/state_visualisation/bloch_angle_vs_zz.png · computed from hold-out row by this script",
             figures=[{"src": figure(R / "state_visualisation/bloch_angle_vs_zz.png", "bloch.png"), "source": "results/state_visualisation/bloch_angle_vs_zz.png"},
                      {"src": "assets/state_vector.png", "source": f"hold-out row {live_index}, models/config_models.pkl"}]),
        item("Research report and insights", source="results/RESULTS.md", rows=[["Report", f"results/RESULTS.md ({len(need(R / 'RESULTS.md').read_text(encoding='utf-8').splitlines())} lines; sections 1–8, statistical methods, limitations)"]])]})

    # ------------------------------------------------------------ Box 15
    ho = csv(RUNS["holdout"] / "holdout_metrics.csv").set_index("model")
    sections.append({"box": 15, "items": [
        item("Feasibility of Quantum AI for mental health analysis", source=rel(RUNS["holdout"] / "holdout_metrics.csv"), rows=[
            ["Feasible on a simulator", f"QSVM hold-out ROC-AUC {ho.loc['qsvm_angle_pca8_1500rows', 'roc_auc']:.3f}; LogReg {ho.loc['logreg_full_all95', 'roc_auc']:.3f}"],
            ["Quantum advantage", "none: the best quantum model ties logistic regression on identical rows and features"],
            ["Why", "the survey signal is near-linear (logistic regression matches XGBoost and the MLP); qubit-sized inputs (4–8 features, "
                    "1.2k rows) cost accuracy; the winning angle-encoding kernel is classically computable"]]),
        item("Privacy-preserving insights", source=rel(RUNS["exp3"] / "tests_private_vs_nonprivate.csv"), rows=[
            ["DP-SGD at ε = 4", f"costs {abs(pd4.change):.3f} ROC-AUC on full-data LogReg; not statistically significant"],
            ["Federated learning", "matches centralised DP for logistic regression with 3–10 clients; the small VQC degrades as clients increase"],
            ["Membership inference", "attack ROC-AUC ≈ 0.50 at every ε"]]),
        item("Comparative performance results", source="results/RESULTS.md (Tables 1–4, 11, 12)", rows=[
            ["Best overall (hold-out)", f"Logistic regression, all 95 features: ROC-AUC {ho.loc['logreg_full_all95', 'roc_auc']:.3f}, recall {ho.loc['logreg_full_all95', 'recall']:.3f}"]]),
        item("Guidelines for future real-world deployment", source="results/RESULTS.md (Limitations)", rows=[
            ["Status", "limitations stated in RESULTS.md; no separate deployment-guideline document"],
            ["Label bias", "the label is caregiver-reported diagnosed depression — undiagnosed adolescents sit in the negative class"],
            ["Not for", "clinical or individual decisions"]]),
        item("Research publication / project report", source="results/RESULTS.md · CONFORMANCE.md", rows=[
            ["Project report", "results/RESULTS.md and CONFORMANCE.md"], ["Publication", f"{NA}: not submitted"]])]})

    manifest = {"dataset": {"rows": n_rows, "columns_one_hot": len(m95), "questions": profile["n_raw_features"],
                            "positive_rate": profile["n_positive"] / n_rows},
                "architecture": "assets/architecture.jpg", "live": live, "sections": sections,
                "labels": {c: lab(c) for c in list(FEATURES) + m95},
                "built_from": "scripts/build_demo_manifest.py"}
    shutil.copy(need(ROOT / "docs/architecture.jpg"), ASSETS / "architecture.jpg")
    (DEMO / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float), encoding="utf-8")
    print(f"wrote demo/manifest.json ({len(sections)} sections) and {len(list(ASSETS.iterdir()))} assets; live row = hold-out row {live_index}")


if __name__ == "__main__":
    main()
