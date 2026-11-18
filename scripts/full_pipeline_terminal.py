"""Whole-dataset run of the architecture (docs/architecture.jpg), boxes 1-15, printed step by step in the terminal.

    python scripts/full_pipeline_terminal.py --config configs/terminal_pipeline.yaml
    python scripts/full_pipeline_terminal.py --config configs/terminal_pipeline.yaml --pause   # Enter between boxes

All 21,789 records go through the pipeline. Splits are the experiments' own (seed 2026): locked hold-out 20 %,
then cross-validation fold r0f0 of the development rows. Every model is trained on that fold's training rows and
scored on its validation rows. The locked hold-out is never used for training or scoring.
Numbers are computed fresh on one fold, so they differ slightly from results/ (15-fold means, significance tests).
Quantum circuits are simulated on CPU; they are not a privacy mechanism. Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts.terminal_walkthrough import banner, para, table  # noqa: E402

PAUSE = False
T_START = time.perf_counter()


def box(n: int, title: str) -> None:
    if PAUSE and n > 1:
        input(f"\n[Enter] for box {n} ...")
    banner(f"BOX {n} — {title}   (elapsed {time.perf_counter() - T_START:,.0f} s)")


def step(msg: str) -> float:
    print(f"  > {msg} ...", flush=True)
    return time.perf_counter()


def done(t0: float) -> str:
    return f"{time.perf_counter() - t0:,.1f} s"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def kv(rows: list[tuple[str, object]]) -> None:
    for k, v in rows:
        para(f"{k}: {v}", "  ")


def main() -> None:
    global PAUSE
    p = argparse.ArgumentParser(description="Whole-dataset terminal run of boxes 1-15")
    p.add_argument("--config", required=True)
    p.add_argument("--pause", action="store_true", help="wait for Enter between boxes")
    a = p.parse_args()
    PAUSE = a.pause
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    import pandas as pd
    import pennylane as qml
    import sklearn
    import torch
    from sklearn.feature_selection import mutual_info_classif

    from src.m1_data.nsch import FEATURES
    from src.m2_preprocess.preprocessing import TabularPreprocessor
    from src.m3_features.features import FeatureTransformer
    from src.m5_models.classical import make_classical
    from src.m5_models.quantum import QuantumKernelSVC, VQCClassifier, vqc_resources
    from src.m6_federated.fedavg import fedavg_train, stratified_iid_partition
    from src.m7_privacy.dp_training import DPClassifier
    from src.m7_privacy.input_privacy import NSCH_QUASI_IDENTIFIERS, KAnonymizer, LaplaceLDP
    from src.m8_eval.evaluation import binary_metrics
    from src.m8_eval.screening import risk_band, risk_cutpoints
    from src.m8_eval.state_viz import bloch_vectors, class_separation, statevector
    from src.m9_audit.membership import loss_threshold_attack, margin_loss, per_example_loss
    from src.m9_audit.reidentification import reidentification_risk
    from src.m10_exp.common import cv_folds, holdout_split, load_config, load_dataset
    from src.m10_exp.exp2_compare import feature_matrices
    from src.m10_exp.tuning import inner_split, scores_of, select_and_fit, stratified_subsample, youden_threshold

    cfg = load_config(ROOT / a.config)
    seed = int(cfg["project"]["seed"])
    frac = float(cfg["tuning"]["inner_val_fraction"])
    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    torch.set_flush_denormal(True)
    results = []                                             # one dict per model, for boxes 10, 11, 14, 15

    # ================================================================== BOX 1
    box(1, "Data Sources (Adolescent Mental Health)")
    t0 = step("reading the NSCH 2024 file")
    loader, frame = load_dataset(cfg)
    y = frame[loader.label_column].to_numpy(dtype=int)
    dev, hold = holdout_split(y, float(cfg["split"]["holdout_fraction"]), seed)
    fold = cv_folds(dev, y, int(cfg["cv"]["n_splits"]), int(cfg["cv"]["n_repeats"]), seed)[0]
    ftr, fva = fold["train"], fold["val"]
    kv([("File", cfg["data"]["paths"][0] + f" (read in {done(t0)})"),
        ("Records (ages 12-17)", f"{len(frame):,}; current depression (caregiver-reported diagnosis): "
                                 f"{y.sum():,} ({y.mean():.1%})"),
        ("Inputs per record", f"{len(FEATURES)} survey answers + {len(loader.metadata_columns)} metadata columns "
                              f"({', '.join(loader.metadata_columns)})"),
        ("Locked hold-out", f"{len(hold):,} records — set aside, never used in this run"),
        ("Development rows", f"{len(dev):,} → fold r0f0: {len(ftr):,} training ({y[ftr].sum():,} positive), "
                             f"{len(fva):,} validation ({y[fva].sum():,} positive)"),
        ("Sources not used", "clinical notes, counselling records, social-media text, wearable/IoT data — "
                             "not in this dataset (survey responses only)")])

    # ================================================================== BOX 2
    box(2, "Data Preprocessing & Anonymization")
    sub = stratified_subsample(ftr, y, int(cfg["subsample"]["n"]), seed)
    s_tr, s_iv = inner_split(sub, y, frac, seed)              # quantum regime: 1,200 / 300
    f_tr, f_iv = inner_split(ftr, y, frac, seed)              # full regime
    t0 = step("fitting imputation + one-hot encoding on training rows only")
    prep_f = TabularPreprocessor(identifier_columns=loader.metadata_columns)
    Xf_in_df = prep_f.fit_transform(frame.iloc[f_tr], loader.label_column)
    Xf_in = Xf_in_df.to_numpy()
    Xf_iv = prep_f.transform(frame.iloc[f_iv], loader.label_column).to_numpy()
    Xf_va = prep_f.transform(frame.iloc[fva], loader.label_column).to_numpy()
    prep_s = TabularPreprocessor(identifier_columns=loader.metadata_columns)
    Xs_in = prep_s.fit_transform(frame.iloc[s_tr], loader.label_column).to_numpy()
    Xs_iv = prep_s.transform(frame.iloc[s_iv], loader.label_column).to_numpy()
    Xs_va = prep_s.transform(frame.iloc[fva], loader.label_column).to_numpy()
    feats = frame[list(FEATURES)]
    miss = feats.isna().mean().sort_values(ascending=False)
    kv([("Identifiers / metadata dropped", ", ".join(loader.metadata_columns) + " (never model inputs)"),
        ("Missing cells (all records)", f"{feats.isna().to_numpy().mean():.2%}; filled with training-row "
                                        f"median (numeric) / mode (categorical)"),
        ("Encoding", f"{len(FEATURES)} answers → {Xf_in.shape[1]} columns (categorical answers one-hot)"),
        ("Full regime", f"{len(f_tr):,} inner-train / {len(f_iv):,} inner-validation rows"),
        ("Quantum regime", f"stratified subsample {len(sub):,} → {len(s_tr):,} inner-train / {len(s_iv):,} "
                           f"inner-validation rows"),
        ("Time", done(t0))])
    print()
    table(["Most-missing answer", "Missing"], [[c, f"{v:.2%}"] for c, v in miss.head(5).items()])

    # ================================================================== BOX 3
    box(3, "Feature Representation")
    t0 = step("standardising, PCA and mutual information (fitted on training rows only)")
    sets = {}
    sets["all95_full"] = feature_matrices({"method": "all"}, Xf_in, y[f_tr], [Xf_iv, Xf_va], seed)
    sets["pca8_sub"] = feature_matrices({"method": "pca", "k": 8}, Xs_in, y[s_tr], [Xs_iv, Xs_va], seed)
    sets["pca4_sub"] = feature_matrices({"method": "pca", "k": 4}, Xs_in, y[s_tr], [Xs_iv, Xs_va], seed)
    ft8 = FeatureTransformer(d_prime=8, method="pca", output="angle", seed=seed).fit(Xs_in, y[s_tr])
    mi = mutual_info_classif(Xf_in, y[f_tr], random_state=seed)
    top = np.argsort(-mi)[:6]
    kv([("Text features", "Not available on this dataset (no free text)"),
        ("Structured features", f"{Xf_in.shape[1]} columns: behaviour, health, family, demographics"),
        ("PCA-8 (QSVM input)", f"{ft8.pca.explained_variance_ratio_.sum():.1%} of training variance, "
                               f"scaled to rotation angles in [0, π]"),
        ("PCA-4 (VQC input)", "4 components, same rows"),
        ("Time", done(t0))])
    print()
    table(["Top answers by mutual information with depression", "MI"],
          [[Xf_in_df.columns[i], f"{mi[i]:.4f}"] for i in top])

    # ================================================================== BOX 4
    box(4, "Privacy-Preserving Mechanisms")
    qi = list(NSCH_QUASI_IDENTIFIERS)
    t0 = step(f"k-anonymity (k = {cfg['privacy']['k_anonymity']}) fitted on training rows")
    kan = KAnonymizer(k=int(cfg["privacy"]["k_anonymity"])).fit(frame.iloc[ftr])
    kan.transform(frame.iloc[fva])                           # sets last_suppressed_ for the validation rows
    ldp = LaplaceLDP(epsilon=float(cfg["privacy"]["ldp_epsilon"]), seed=seed)
    A8, B8, C8 = sets["pca8_sub"]
    noisy = ldp.transform(C8)
    kv([("k-anonymity", f"{len(qi)} quasi-identifiers ({', '.join(qi)}); training: "
                        f"{kan.report_['suppression_rate']:.1%} of rows suppressed; validation: "
                        f"{kan.last_suppressed_ / len(fva):.1%} ({done(t0)})"),
        ("Local DP (Laplace)", f"ε = {ldp.epsilon:g} per record: noise scale {ldp.noise_scale(8):.3f} rad per angle; "
                               f"mean |change| on validation angles {np.abs(noisy - np.clip(C8, 0, np.pi)).mean():.3f} rad")])
    A95, B95, C95 = sets["all95_full"]
    dp = cfg["privacy"]
    dp_rows, dp_models = [], {}
    for eps in [*dp["dp_epsilons"], None]:
        t0 = step(f"DP-SGD logistic regression, ε = {'∞ (non-private)' if eps is None else eps}, "
                  f"{len(f_tr):,} rows")
        m = DPClassifier(kind="logreg", epsilon=eps, delta=float(dp["delta"]), epochs=int(dp["dp_epochs"]),
                         batch_size=int(dp["dp_batch_size"]), learning_rate=float(dp["dp_learning_rate"]),
                         seed=seed).fit(A95, y[f_tr])
        thr = youden_threshold(y[f_iv], m.decision_scores(B95))
        met = binary_metrics(y[fva], m.decision_scores(C95), thr)
        dp_models[eps] = m
        spent = "∞" if eps is None else f"{m.privacy_.epsilon:.2f}"
        dp_rows.append(["∞" if eps is None else f"{eps:g}", spent,
                        "—" if eps is None else f"{m.noise_multiplier_:.3f}", f"{met['roc_auc']:.3f}",
                        f"{met['recall']:.3f}", done(t0)])
    print()
    table(["Target ε", "ε spent (RDP)", "Noise multiplier", "ROC-AUC", "Recall", "Time"], dp_rows)
    fed = cfg["federated"]

    def fed_run(n_clients: int, secure: bool):
        parts = stratified_iid_partition(y[f_tr], n_clients, seed)

        def score(module, X):
            module.eval()
            with torch.no_grad():
                return module(torch.as_tensor(X, dtype=torch.float64)).reshape(-1).numpy()
        out = fedavg_train("logreg", {}, [(A95[q], y[f_tr][q]) for q in parts], rounds=int(fed["rounds"]),
                           local_epochs=int(fed["local_epochs"]), learning_rate=float(dp["dp_learning_rate"]),
                           sample_rate=int(dp["dp_batch_size"]) / len(f_tr), epsilon=4.0,
                           delta=float(dp["delta"]), max_grad_norm=1.0, seed=seed, secure_aggregation=secure)
        thr = youden_threshold(y[f_iv], score(out["module"], B95))
        return out, binary_metrics(y[fva], score(out["module"], C95), thr)

    nsec = int(fed["secure_aggregation_clients"])
    t0 = step(f"federated DP-SGD (FedAvg, {nsec} clients, ε = 4) WITH pairwise-masked secure aggregation")
    sec_out, sec_met = fed_run(nsec, True)
    kv([("Federated + DP + secure aggregation", f"{nsec} clients, {fed['rounds']} rounds: ROC-AUC "
                                                f"{sec_met['roc_auc']:.3f}; key setup {sec_out['secagg_setup_seconds']:.2f} s; "
                                                f"the server only sees masked sums ({done(t0)})"),
        ("Homomorphic encryption", "Not implemented in this project"),
        ("Privacy risk analysis", "membership inference and re-identification: box 12")])

    # ================================================================== BOX 5
    box(5, "Quantum Encoding")
    t0 = step("simulating the 8-qubit angle-encoded statevector of validation record 1")
    psi = statevector("angle", 8, C8[0])
    order = np.argsort(-np.abs(psi) ** 2)[:6]
    kv([("Encoding", "angle encoding, |ψ⟩ = U(x)|0⟩, one RY(θ) per qubit, 8 qubits, 2^8 = 256 amplitudes"),
        ("Record", f"dataset row {fva[0]} (validation fold), angles {np.round(C8[0], 3).tolist()}"),
        ("Norm", f"‖ψ‖² = {np.sum(np.abs(psi) ** 2):.6f} ({done(t0)})")])
    print()
    table(["Basis state", "Amplitude", "Probability"],
          [[f"|{k:08b}⟩", f"{psi[k].real:+.4f}", f"{abs(psi[k]) ** 2:.4f}"] for k in order])
    t0 = step(f"Bloch vectors of all 8 qubits for the {len(s_iv)} inner-validation records")
    bv = bloch_vectors("angle", 8, B8)
    sep = class_separation(bv, y[s_iv])
    print()
    table(["Qubit", "Mean |r| (1 = pure)", "Class separation (0 = none, 2 = max)"],
          [[q, f"{np.linalg.norm(bv[:, q], axis=1).mean():.3f}", f"{sep[q]:.3f}"] for q in range(8)])
    print(f"  ({done(t0)}; amplitude, basis and ZZ encodings are compared in Experiment 1, results/)")

    def record(name, family, rows_desc, feats_desc, model, X_iv, y_iv, X_va, fit_s, params, extra=""):
        thr = youden_threshold(y_iv, scores_of(model, X_iv))
        s = scores_of(model, X_va)
        results.append({"model": name, "family": family, "rows": rows_desc, "features": feats_desc,
                        **binary_metrics(y[fva], s, thr), "fit_s": fit_s, "params": params, "extra": extra,
                        "scores": s})

    # ================================================================== BOX 6
    box(6, "Quantum Model (Simulation)")
    q = cfg["quantum"]
    t0 = step(f"QSVM: quantum kernel |⟨ψ(x′)|ψ(x)⟩|² on {len(s_tr):,} × {len(s_tr):,} training pairs, "
              f"C ∈ {q['qsvm']['grid']['C']}")
    best = select_and_fit(lambda **pp: QuantumKernelSVC(encoding="angle", n_qubits=8, seed=seed, **pp),
                          q["qsvm"]["grid"], A8, y[s_tr], B8, y[s_iv])
    qsvm = best["model"]
    record("QSVM (angle, 8 qubits)", "quantum", "1.2k", "PCA-8", qsvm, B8, y[s_iv], C8, time.perf_counter() - t0,
           int(qsvm.svc_.dual_coef_.size), "8 qubits, 8 RY gates")
    kv([("QSVM", f"C = {best['params']['C']} (picked on inner-validation, AUC {best['inner_val_auc']:.3f}); "
                 f"{len(qsvm.svc_.support_)} support vectors; {done(t0)}")])
    A4, B4, C4 = sets["pca4_sub"]
    vq = q["vqc"]
    res = vqc_resources(vq["encoding"], vq["n_qubits"], vq["n_layers"])
    t0 = step(f"VQC: {vq['encoding']} encoding, {vq['n_qubits']} qubits, {vq['n_layers']} layers, "
              f"{vq['epochs']} epochs of Adam")
    vqc = VQCClassifier(encoding=vq["encoding"], n_qubits=vq["n_qubits"], n_layers=vq["n_layers"],
                        learning_rate=vq["learning_rate"], epochs=vq["epochs"], batch_size=vq["batch_size"],
                        init_scale=vq["init_scale"], seed=seed).fit(A4, y[s_tr], eval_set=(B4, y[s_iv]))
    record("VQC (re-upload, 4 qubits)", "quantum", "1.2k", "PCA-4", vqc, B4, y[s_iv], C4, time.perf_counter() - t0,
           vqc.n_parameters(), f"depth {res['depth']}, {res['n_gates']} gates")
    kv([("VQC", f"{vqc.n_parameters()} trainable rotation angles; circuit depth {res['depth']}; {done(t0)}"),
        ("Simulator", f"PennyLane {qml.__version__} default.qubit (exact statevector, CPU)")])

    # ================================================================== BOX 7
    box(7, "Hybrid Quantum-Classical Learning")
    kv([("Loop", "quantum circuit → loss (class-weighted binary cross-entropy) → Adam updates the angles → repeat"),
        ("Model kept", f"epoch {vqc.best_epoch_} (best inner-validation ROC-AUC)")])
    print()
    table(["Epoch", "Inner-validation ROC-AUC"],
          [[e, f"{auc:.3f}"] for e, auc in enumerate(vqc.history_, 1) if e == 1 or e % 5 == 0 or e == vqc.best_epoch_])

    # ================================================================== BOX 8
    box(8, "Simulation Environment")
    kv([("Simulator", f"PennyLane {qml.__version__}, default.qubit statevector (noise-free); noisy runs: Experiment 8"),
        ("Python / libraries", f"Python {platform.python_version()}, PyTorch {torch.__version__}, "
                               f"scikit-learn {sklearn.__version__}, NumPy {np.__version__}"),
        ("Machine", f"{platform.system()} {platform.release()}, {os.cpu_count()} CPU threads, no GPU, no quantum hardware"),
        ("Circuits used", "8 qubits (QSVM), 4 qubits (VQC)"),
        ("Federated clients simulated", "in-process, box 13")])

    # ================================================================== BOX 9 (classical models trained here, shown in 10-11)
    box(9, "Mental Health Analysis Tasks")
    print("  Classical baselines are trained first (needed for the risk bands below).")
    cl = cfg["classical"]
    Afl, Bfl, Cfl = A95, B95, C95
    for name, data, rows_desc, feats_desc, X_iv_, y_iv_, regime_y in [
            ("logreg", (A8, B8, C8), "1.2k", "PCA-8", B8, y[s_iv], y[s_tr]),
            ("logreg", (A4, B4, C4), "1.2k", "PCA-4", B4, y[s_iv], y[s_tr]),
            ("logreg", (Afl, Bfl, Cfl), f"{len(f_tr) / 1000:.1f}k", "all 95", Bfl, y[f_iv], y[f_tr]),
            ("xgboost", (Afl, Bfl, Cfl), f"{len(f_tr) / 1000:.1f}k", "all 95", Bfl, y[f_iv], y[f_tr]),
            ("mlp", (Afl, Bfl, Cfl), f"{len(f_tr) / 1000:.1f}k", "all 95", Bfl, y[f_iv], y[f_tr])]:
        spec = cl[name]
        t0 = step(f"{name} on {rows_desc} rows, {feats_desc}")
        b = select_and_fit(lambda name=name, spec=spec, ry=regime_y, **pp: make_classical(
            name, ry, seed=seed, n_jobs=n_jobs, **{**spec.get("base", {}), **pp}),
            spec.get("grid"), data[0], regime_y, data[1], y_iv_)
        m = b["model"]
        n_par = (int(m.coef_.size + m.intercept_.size) if hasattr(m, "coef_") else
                 int(sum(t.numel() for t in m.model_.parameters())) if hasattr(m, "model_") else None)
        record({"logreg": "LogReg", "xgboost": "XGBoost", "mlp": "MLP"}[name], "classical", rows_desc, feats_desc,
               m, data[1], y_iv_, data[2], time.perf_counter() - t0, n_par)
    lr_model_iv = make_classical("logreg", y[f_tr], seed=seed).fit(Afl, y[f_tr])
    cuts = risk_cutpoints(lr_model_iv.predict_proba(Bfl)[:, 1], y[f_iv], **cfg["risk_bands"])
    prob = lr_model_iv.predict_proba(Cfl)[:, 1]
    bands = np.array([risk_band(float(pv), cuts) for pv in prob])
    print()
    kv([("Task 1 — depression classification", f"binary, {len(fva):,} validation records; metrics in box 11"),
        ("Task 2 — risk level", f"logistic regression probability → low / moderate / high; cut-points "
                                f"{cuts['low_moderate']:.3f} / {cuts['moderate_high']:.3f} set on inner-validation "
                                f"rows ({cuts['sensitivity_target']:.0%} sensitivity, {cuts['specificity_target']:.0%} specificity)"),
        ("Not available on this dataset", "topic/trend, emotion and sentiment analysis (no text); "
                                          "three-class task: Experiment 9 in results/")])
    print()
    table(["Risk band", "Records", "Observed depression rate"],
          [[b_.upper(), int((bands == b_).sum()), f"{y[fva][bands == b_].mean():.1%}" if (bands == b_).any() else "—"]
           for b_ in ("low", "moderate", "high")])

    # ================================================================== BOX 10
    box(10, "Evaluation & Comparison")
    print("  Like-for-like pairs: same rows, same features, same fold; only the model differs.\n")

    def get(model, feats_):
        return next(r for r in results if r["model"].startswith(model) and r["features"] == feats_)
    pairs = [("QSVM", "LogReg", "PCA-8"), ("VQC", "LogReg", "PCA-4")]
    table(["Quantum", "Classical", "Rows / features", "Quantum AUC", "Classical AUC", "Difference"],
          [[get(qn, f)["model"], cn, f"1.2k / {f}", f"{get(qn, f)['roc_auc']:.3f}", f"{get(cn, f)['roc_auc']:.3f}",
            f"{get(qn, f)['roc_auc'] - get(cn, f)['roc_auc']:+.3f}"] for qn, cn, f in pairs])
    print("\n  One fold only: differences here are not significance-tested. The 15-fold tests (Nadeau-Bengio, Holm)")
    print("  are in results/RESULTS.md. Random forest, BERT and LSTM/CNN baselines: not run (no text; dropped).")

    # ================================================================== BOX 11
    box(11, "Performance Metrics")
    table(["Model", "Rows", "Features", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train s", "Params", "Resources"],
          [[r["model"], r["rows"], r["features"], f"{r['accuracy']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}",
            f"{r['f1']:.3f}", f"{r['roc_auc']:.3f}", f"{r['fit_s']:.1f}", "—" if r["params"] is None else f"{r['params']:,}",
            r["extra"] or "CPU"] for r in sorted(results, key=lambda r: -r["roc_auc"])])
    print(f"\n  Validation fold: {len(fva):,} records, {y[fva].sum():,} positive. Thresholds set on inner-validation rows.")
    print("  Parameter counts are not comparable across model types (SVM dual coefficients vs weights vs angles).")

    # ================================================================== BOX 12
    box(12, "Privacy Evaluation")
    t0 = step("loss-threshold membership inference (members = training rows, non-members = validation rows)")
    mia = []
    for eps, m in dp_models.items():
        lm = per_example_loss(y[f_tr], sigmoid(m.decision_scores(A95)))
        ln = per_example_loss(y[fva], sigmoid(m.decision_scores(C95)))
        r_ = loss_threshold_attack(lm, ln)
        mia.append([f"DP LogReg ε = {'∞' if eps is None else eps}", f"{r_['attack_auc']:.3f}", f"{r_['advantage']:.3f}",
                    f"{r_['tpr_at_1pct_fpr']:.3f}"])
    r_ = loss_threshold_attack(margin_loss(y[s_tr], qsvm.decision_scores(A8)), margin_loss(y[fva], qsvm.decision_scores(C8)))
    mia.append(["QSVM (no DP)", f"{r_['attack_auc']:.3f}", f"{r_['advantage']:.3f}", f"{r_['tpr_at_1pct_fpr']:.3f}"])
    table(["Model", "Attack AUC (0.5 = chance)", "Advantage", "TPR at 1% FPR"], mia)
    print(f"  ({done(t0)})")
    t0 = step("re-identification risk on the quasi-identifiers, training rows")
    before = reidentification_risk(frame.iloc[ftr], qi, k=kan.k)
    after = reidentification_risk(kan.transform(frame.iloc[ftr]), qi, k=kan.k)
    print()
    table(["Quasi-identifiers", "Unique records", f"In groups < {kan.k}", "Mean risk", "Max risk"],
          [[lab, f"{d['uniqueness']:.1%}", f"{d['k_violation_rate']:.1%}", f"{d['avg_prosecutor_risk']:.3f}",
            f"{d['max_prosecutor_risk']:.3f}"] for lab, d in (("raw", before), (f"after k = {kan.k}", after))])
    print("\n  Privacy-utility trade-off (ε vs ROC-AUC): box 4 table. Gradient inversion: not implemented.")

    # ================================================================== BOX 13
    box(13, "Scalability Analysis (Simulation)")
    rows13 = []
    for n in fed["clients"]:
        t0 = step(f"FedAvg with DP (ε = 4), {n} IID clients, {fed['rounds']} rounds")
        out, met = fed_run(int(n), False)
        h = out["history"]
        rows13.append([n, "plain", f"{met['roc_auc']:.3f}", f"{h['cumulative_bytes'].iloc[-1] / 1e3:,.1f}",
                       f"{h['client_seconds_max'].mean() * 1e3:.1f}", f"{h['aggregation_seconds'].mean() * 1e3:.2f}",
                       f"{out['epsilon_spent_max']:.2f}", done(t0)])
    h = sec_out["history"]
    rows13.append([nsec, "secure agg.", f"{sec_met['roc_auc']:.3f}", f"{h['cumulative_bytes'].iloc[-1] / 1e3:,.1f}",
                   f"{h['client_seconds_max'].mean() * 1e3:.1f}", f"{h['aggregation_seconds'].mean() * 1e3:.2f}",
                   f"{sec_out['epsilon_spent_max']:.2f}", "box 4"])
    table(["Clients", "Aggregation", "ROC-AUC", "Traffic total kB", "Client ms/round", "Server ms/round", "ε spent", "Time"],
          rows13)
    print("\n  Non-IID (by-state) partitions and communication latency: Experiments 4 and 7 in results/.")

    # ================================================================== BOX 14
    box(14, "Output & Visualization")
    print("  ROC-AUC on the validation fold (each █ = 0.02):\n")
    for r in sorted(results, key=lambda r: -r["roc_auc"]):
        print(f"  {r['model'] + ' · ' + r['features']:<34} {'█' * int(round(r['roc_auc'] / 0.02)):<50} {r['roc_auc']:.3f}")
    print("\n  Quantum state visualisation: statevector and Bloch vectors in box 5.")
    print("  Saved figures and the full report: results/results_report.pdf (this run writes nothing).")

    # ================================================================== BOX 15
    box(15, "Key Outcomes")
    bq = max((r for r in results if r["family"] == "quantum"), key=lambda r: r["roc_auc"])
    bc = max((r for r in results if r["family"] == "classical"), key=lambda r: r["roc_auc"])
    same = get("LogReg", bq["features"])
    dp4 = [r for r in dp_rows if r[0] == "4"][0][3]
    dpinf = [r for r in dp_rows if r[0] == "∞"][0][3]
    for line in [
            f"Best quantum model: {bq['model']} ROC-AUC {bq['roc_auc']:.3f}; logistic regression on the same rows and "
            f"features: {same['roc_auc']:.3f}. Best overall: {bc['model']} on {bc['features']} ({bc['roc_auc']:.3f}).",
            "No quantum advantage is claimed; this is one fold. See results/RESULTS.md for the 15-fold tests.",
            f"Differential privacy: ROC-AUC {dpinf} without DP → {dp4} at ε = 4 (DP-SGD, the project's privacy mechanism; "
            "quantum circuits add no privacy).",
            f"Federated training with secure aggregation reached ROC-AUC {sec_met['roc_auc']:.3f} with {nsec} clients.",
            "Limits: survey data only (no text, sensors or clinical notes); simulator only, no quantum hardware."]:
        para(line, "  • ")
    print(f"\nTotal run time {time.perf_counter() - T_START:,.0f} s. Locked hold-out untouched.")


if __name__ == "__main__":
    main()
