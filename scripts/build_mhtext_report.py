"""Phase 7: results report for the Mental Health Text dataset, one section per architecture box, beside NSCH.

Every number is read from a file under results/ at build time and recorded in the traceability appendix
(box, item, value, file). Nothing is computed from raw data and no model is run.

    python scripts/build_mhtext_report.py  -> results/mhtext/results_report_mhtext.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.m9_audit.disclosure import RULE, numeric  # noqa: E402

R = ROOT / "results"
MH = R / "mhtext"
NS_EXP2 = R / "exp2-nsch2024-quantum-vs-classical-20261108-165426"
NS_EXP4 = R / "exp4-nsch2024-federated-dp-20261108-170320"
NS_SWEEP = R / "exp1-nsch2024-encodings-with-basis-20261112"
NAMES = {"length_only_logreg": "Length only (log words + log characters), LogReg",
         "bert_pooled_logreg": "Frozen MiniLM sentence encoder (BERT family), whole post, LogReg",
         "bert_sentence_logreg": "Frozen MiniLM sentence encoder (BERT family), per-sentence mean + max, LogReg",
         "logreg": "All pooled features, LogReg", "xgboost": "All pooled features, XGBoost",
         "mlp": "All pooled features, MLP", "svm": "All pooled features, SVM (RBF)",
         "lstm": "LSTM over tokens (embedding trained from scratch)", "cnn": "CNN over tokens (embedding trained from scratch)"}

TRACE: list[tuple] = []
BOX = [""]


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def v(value, src: Path, item: str, fmt: str = "{:.3f}") -> str:
    """Format a number read from `src` and record it for the traceability appendix."""
    s = fmt.format(value) if isinstance(value, (int, float)) else str(value)
    TRACE.append((BOX[0], item, s, rel(src)))
    return s


def csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def js(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def section(L, n, title, does):
    BOX[0] = f"Box {n}"
    L += [f"## Box {n} — {title}", "", f"**What this step does:** {does}", ""]


def sources(L, *paths):
    L += ["Source: " + " · ".join(rel(p) for p in paths), ""]


def main() -> None:
    L = ["# Results report — Mental Health Text Classification dataset (second data source), beside NSCH 2024", "",
         "One section per architecture box. Every number is read from a results/ file at build time; the "
         "traceability appendix lists each one with its file. NSCH numbers are read from the NSCH results/ files and "
         "were not recomputed. Inputs are the frozen MiniLM sentence encoder (BERT family: "
         "sentence-transformers/all-MiniLM-L6-v2), never fine-tuned. Quantum circuits are simulated. Federated clients "
         "are simulated. The locked hold-out of this dataset has not been scored.", "",
         f"**Disclosure control.** {RULE} In this dataset's reports it applies to 10 cells of the Phase 5 "
         "re-identification tables (results/mhtext/phase5/phase5.md, which states which cells and why no second "
         "cell was needed).", ""]

    # ---------------------------------------------------------------- Box 1
    section(L, 1, "Data Sources (Adolescent Mental Health)", "Adds a second, unrelated data source: public social-media "
            "posts labelled Normal / Anxiety / Depression / Suicidal.")
    m1 = MH / "phase1/manifest.json"
    man = js(m1)
    raw = man["stages"]["raw_all"]
    L += ["| Item | Value |", "|---|---|",
          "| Dataset | Mukherjee (2025), Mental Health Text Classification Dataset (4-class), Hugging Face, CC BY 4.0 (data/README.md) |",
          f"| Raw rows | {v(sum(c['n'] for c in raw.values()), m1, 'raw rows', '{:,}')} |",
          f"| After deduplication (exact + near ≥ {man['dedup']['threshold']}) | {v(man['dedup']['rows_out'], m1, 'rows after dedup', '{:,}')} |",
          f"| Processed rows (subsample + quarantined block) | {v(sum(sum(c['rows'] for c in s.values()) for s in [man['views']['4-class']]), m1, 'processed rows 4-class', '{:,}')} |",
          "| Who wrote it | Adults and adolescents, unknown ages; NOT an adolescent sample (see Box 9) |", ""]
    L += ["*Table 1.2. The six sources in the diagram — coverage now*", "",
          "| Source in the diagram | NSCH | Mental Health Text |", "|---|---|---|",
          "| Clinical notes (de-identified) | No | No: not available, no clinical-notes dataset is part of this project |",
          "| Counselling records | No | No: not available, no counselling-records dataset is part of this project |",
          "| Survey responses | Yes | No |",
          "| Social media text | No | **Yes: BUILT** |",
          "| Wearable/IoT data (optional) | No | No: not available, no sensor dataset is part of this project |",
          "| Public datasets | Yes | Yes |", ""]
    sources(L, m1, ROOT / "data/README.md")

    # ---------------------------------------------------------------- Box 2
    section(L, 2, "Data Preprocessing & Anonymization", "Deduplicates before any split, drops conflicting labels, "
            "quarantines the block without IDs, locks a hold-out, blocks keyword-leakage columns.")
    d = man["dedup"]
    L += ["| Step | Rows |", "|---|---|",
          f"| Exact duplicates dropped | {v(d['exact_duplicate_rows_dropped'], m1, 'exact duplicates dropped', '{:,}')} |",
          f"| Exact conflicting-label rows dropped | {v(d['exact_conflict_rows_dropped'], m1, 'exact conflict rows', '{:,}')} |",
          f"| Near duplicates dropped (TF-IDF cosine ≥ 0.90) | {v(d['near_duplicate_rows_dropped'], m1, 'near duplicates dropped', '{:,}')} |",
          f"| Near conflicting-label rows dropped | {v(d['near_conflict_rows_dropped'], m1, 'near conflict rows', '{:,}')} |",
          f"| Confirmation pass on subsample + quarantine, rows dropped | {v(d['confirm_pass_main_rows_dropped'] + d['confirm_pass_quarantine_rows_dropped'], m1, 'confirmation pass dropped', '{:,}')} |",
          f"| Locked hold-out fraction | {v(man['holdout_fraction'], m1, 'hold-out fraction', '{:.0%}')} (not scored) |",
          f"| Blocked columns (label leakage) | {', '.join(man['blocked_columns_removed'])} |", "",
          "Anonymisation: there are no quasi-identifier columns to generalise, so k-anonymity does not apply. Direct "
          "identifiers can be scrubbed from the text (URLs, e-mails, handles, communities, numbers); Box 12 shows this "
          "does not reduce re-identification.", ""]
    sources(L, m1)

    # ---------------------------------------------------------------- Box 3
    section(L, 3, "Feature Representation", "Turns each post into numbers with frozen models only (nothing fine-tuned).")
    m2 = MH / "phase2/manifest.json"
    p2 = js(m2)
    fam = csv(MH / "phase2/mi_by_family.csv")
    f4 = fam[fam.view == "4class"].set_index("family")
    L += ["*Table 3.1. Text features — coverage now*", "", "| Diagram item | NSCH | Mental Health Text |", "|---|---|---|",
          f"| Transformer embeddings (e.g., BERT, RoBERTa) | Not available | **BUILT**: frozen MiniLM sentence encoder (BERT family), "
          f"{v(384, m2, 'embedding dims', '{}')} dims, whole post and per-sentence mean + max |",
          f"| Sentiment & emotion features | Not available | **BUILT**: frozen emotion classifier ({p2['emotion']['model']}), "
          f"{v(len(p2['emotion']['labels']), m2, 'emotion labels', '{}')} emotion probabilities |",
          "| Contextual representations | Not available | **BUILT**: the sentence-encoder embedding is contextual |",
          "| Physiological signals (optional) | Not available | Not available: no sensor data |", "",
          "*Table 3.2. Mutual information with the 4-class label, by feature family*", "",
          "| Family | Columns | Max MI | Mean MI | Best rank |", "|---|---|---|---|---|"]
    for f, x in f4.iterrows():
        L.append(f"| {f.replace('pooled embedding', 'MiniLM pooled embedding').replace('sentence ', 'MiniLM sentence ')} | {int(x.columns)} | "
                 f"{v(x.max_mi, MH / 'phase2/mi_by_family.csv', f'max MI {f}')} | {v(x.mean_mi, MH / 'phase2/mi_by_family.csv', f'mean MI {f}')} | {int(x.best_rank)} |")
    L += ["", "PCA vs mutual-information selection to reach 8 quantum inputs: see Box 15 (P4) — it does not replicate "
          "across the two datasets.", ""]
    sources(L, m2, MH / "phase2/mi_by_family.csv")

    # ---------------------------------------------------------------- Box 4
    section(L, 4, "Privacy-Preserving Mechanisms", "Applies the NSCH mechanisms with their NSCH settings: DP-SGD during "
            "training, Laplace local DP on the encoder inputs, federated training (Box 13); identifier scrubbing for text.")
    p5 = MH / "phase5"
    L += ["| Mechanism | NSCH | Mental Health Text |", "|---|---|---|",
          "| k-anonymity on quasi-identifiers | Used (k = 5) | Not applicable: free text has no quasi-identifier columns |",
          "| Identifier scrubbing (URLs, e-mails, handles, communities, numbers) | n/a | BUILT (Box 12: no effect on re-identification) |",
          "| DP-SGD (Opacus, ε 0.5-8, δ 1e-5) | Used | BUILT (same settings) |",
          "| Laplace local DP on the 8 encoder inputs (ε_input 4-32) | Used | BUILT (same settings) |",
          "| Federated averaging with client DP (ε = 4) | Used | BUILT, simulated clients (Box 13) |", ""]
    sources(L, p5 / "dp/run.json", p5 / "ldp/run.json")

    # ---------------------------------------------------------------- Box 5
    section(L, 5, "Quantum Encoding", "Maps 4-8 reduced inputs onto qubits with five encodings (exp1 method, 1,200 rows, 10 folds).")
    sw = csv(MH / "phase4/binary/sweep_summary.csv")
    L += ["| Model | Features | Encoding | 4 qubits | 6 qubits | 8 qubits |", "|---|---|---|---|---|---|"]
    for (mm, fm, enc), g in sw[sw.model != "logreg"].groupby(["model", "feature_method", "encoding"]):
        g = g.set_index("n_qubits").roc_auc
        L.append(f"| {mm.upper()} | {fm} | {enc} | " + " | ".join(
            v(g[q], MH / "phase4/binary/sweep_summary.csv", f"sweep {mm} {fm} {enc} {q}q") for q in (4, 6, 8)) + " |")
    L += ["", "Best encodings on both datasets: angle and re-upload on PCA inputs; basis is worst.", ""]
    sources(L, MH / "phase4/binary/sweep_summary.csv")

    # ---------------------------------------------------------------- Box 6 / 7
    s4 = csv(MH / "phase4/binary/summary.csv").set_index(["model", "regime", "feature_set"])
    S4 = MH / "phase4/binary/summary.csv"
    section(L, 6, "Quantum Model (Simulation)", "Trains QSVM, VQC and hybrid QNN on the binary Depression vs Suicidal view, "
            "1,200 rows per fold, 15 folds (exp2 method).")
    L += ["| Model | Input | ROC AUC | Balanced acc. | Recall | Accuracy | Quarantine AUC |", "|---|---|---|---|---|---|---|"]
    for q, fs, lab in (("qsvm", "pca8", "QSVM, angle, 8 qubits"), ("vqc", "pca4", "VQC, re-upload, 4 qubits"),
                       ("hybrid", "all384", "Hybrid QNN, angle, 8 qubits")):
        x = s4.loc[(q, "subsample", fs)]
        L.append(f"| {lab} | {fs} | {v(x.roc_auc_mean, S4, f'{q} ROC AUC')} | {v(x.balanced_accuracy_mean, S4, f'{q} bal acc')} | "
                 f"{v(x.recall_mean, S4, f'{q} recall')} | {v(x.accuracy_mean, S4, f'{q} accuracy')} | {v(x.extra_roc_auc_mean, S4, f'{q} quarantine AUC')} |")
    L += ["", "Four-class and three-class quantum views: deliberately not run (reason in results/mhtext/phase4/binary/phase4.md).", ""]
    sources(L, S4)
    section(L, 7, "Hybrid Quantum-Classical Learning", "Trains a classical layer (384 → 8) and an 8-qubit circuit end to end.")
    hy = s4.loc[("hybrid", "subsample", "all384")]
    L += [f"Hybrid QNN ROC AUC {v(hy.roc_auc_mean, S4, 'hybrid ROC AUC')} ± {v(hy.roc_auc_std, S4, 'hybrid ROC AUC sd')}, "
          f"against MLP on the same rows and all 384 dims {v(s4.loc[('mlp', 'subsample', 'all384')].roc_auc_mean, S4, 'mlp all384 sub AUC')}. "
          "It is the weakest quantum model here, as on NSCH (Box 15, P6).", ""]
    sources(L, S4)

    # ---------------------------------------------------------------- Box 8
    section(L, 8, "Simulation Environment", "Records the software, the simulator and the wall time of each run.")
    env = js(MH / "phase4/binary/run.json")["environment"]
    runs = [("Phase 4 quantum sweep + comparison, 642 tasks of the second launch (whole run 11:15-13:37 incl. restart)", MH / "phase4/binary/run.json", "wall_seconds_this_launch"),
            ("Phase 4 all-rows classical ceiling", MH / "phase4/binary_ceiling/run.json", "wall_seconds"),
            ("Phase 5 DP-SGD", p5 / "dp/run.json", "wall_seconds_this_launch"),
            ("Phase 5 membership inference", p5 / "mia/run.json", "wall_seconds_this_launch"),
            ("Phase 5 input local DP", p5 / "ldp/run.json", "wall_seconds_this_launch"),
            ("Phase 5 re-identification", p5 / "reid/run.json", "wall_seconds"),
            ("Phase 6 federated (120 tasks)", MH / "phase6/run.json", "wall_seconds_this_launch")]
    L += ["| Run | Wall time (min) |", "|---|---|"]
    for name, p, k in runs:
        L.append(f"| {name} | {v(js(p)[k] / 60, p, f'wall time {name}', '{:.0f}')} |")
    L += ["", f"Python {env['python'].split()[0]}, statevector simulation, 6 CPU workers. Packages: "
          + ", ".join(f"{k} {val}" for k, val in sorted(env.get('packages', {}).items())) + ".", "",
          "*Table 8.3. Qubit count 4 → 8 (paired NB-t, Holm over each dataset's table)*: see Box 15, P3.", ""]
    sources(L, *[p for _, p, _ in runs])

    # ---------------------------------------------------------------- Box 9
    section(L, 9, "Mental Health Analysis Tasks", "Defines what is predicted on whom.")
    L += ["| Task | NSCH | Mental Health Text |", "|---|---|---|",
          "| Binary | provider-diagnosed depression, adolescents 12-17 | Depression vs Suicidal (the hard, length-free boundary) |",
          f"| Multi-class | none / anxiety only / depression | 4-class (Normal, Anxiety, Depression, Suicidal) and 3-class (Normal dropped) |",
          "| Stress | Not available | Not available: no stress label |",
          "| Topic and trend analysis | Not available | Not available: posts carry no timestamps |",
          "| Emotion and sentiment analysis | Not available | **BUILT** as features (Box 3); not a prediction target |",
          "| Personalized insights (simulated) | Not available | Not available: no per-person history |", "",
          "Population limit: authors' ages are unknown; regex counts find an age of 12-17 stated in "
          f"{v(csv(p5 / 'reid/self_disclosures.csv').set_index('corpus').loc['dev', 'states_age_12_17'], p5 / 'reid/self_disclosures.csv', 'share stating age 12-17', '{:.1%}')} "
          "of development posts. This dataset is not an adolescent sample.", ""]
    sources(L, p5 / "reid/self_disclosures.csv")

    # ---------------------------------------------------------------- Box 10 / 11
    section(L, 10, "Evaluation & Comparison", "Compares every model against the length-only reference (multi-class) and "
            "quantum against classical on the same rows and inputs (binary). NB-t corrected resampled t-test, Holm.")
    s3 = csv(MH / "phase3/summary.csv")
    S3 = MH / "phase3/summary.csv"
    for view in ("4-class", "3-class"):
        t = s3[(s3["view"] == view) & (s3["set"] == "cv")].sort_values("macro_auc_mean", ascending=False)
        L += [f"*{view}, cross-validation (15 folds)*", "", "| Model | Macro AUC | Macro-F1 | Balanced acc. | Accuracy | Quarantine macro AUC |",
              "|---|---|---|---|---|---|"]
        q = s3[(s3["view"] == view) & (s3["set"] == "quarantine")].set_index("model")
        for _, x in t.iterrows():
            qa = q.macro_auc_mean.get(x.model, float("nan"))
            L.append(f"| {NAMES.get(x.model, x.model)} | {v(x.macro_auc_mean, S3, f'{view} {x.model} macro AUC')} | "
                     f"{v(x.macro_f1_mean, S3, f'{view} {x.model} macro F1')} | {v(x.balanced_accuracy_mean, S3, f'{view} {x.model} bal acc')} | "
                     f"{v(x.accuracy_mean, S3, f'{view} {x.model} accuracy')} | {v(qa, S3, f'{view} {x.model} quarantine macro AUC')} |")
        L.append("")
    L += ["*Diagram items — coverage now*", "", "| Diagram item | NSCH | Mental Health Text |", "|---|---|---|",
          "| BERT + ML | Not available | **BUILT**: frozen MiniLM sentence encoder (BERT family) + LogReg / XGBoost / MLP / SVM |",
          "| Deep Learning (LSTM, CNN) | Not available (MLP instead) | **BUILT**: LSTM and CNN over token sequences |",
          "| Quantum Neural Network | hybrid QNN | hybrid QNN (Box 7) |", ""]
    t4 = csv(MH / "phase4/binary/tests_same_rows_features.csv")
    T4 = MH / "phase4/binary/tests_same_rows_features.csv"
    L += ["*Binary view: quantum vs classical on the same rows and inputs (ROC AUC)*", "",
          "| Quantum | Classical | Quantum | Classical | Diff | Holm p |", "|---|---|---|---|---|---|"]
    for _, x in t4[t4.metric == "roc_auc"].iterrows():
        L.append(f"| {x.quantum} | {x.classical} | {v(x.quantum_mean, T4, f'{x.quantum} vs {x.classical} q')} | "
                 f"{v(x.classical_mean, T4, f'{x.quantum} vs {x.classical} c')} | {x.mean_diff:+.3f} | {v(x.p_nb_t_holm, T4, f'{x.quantum} vs {x.classical} p')} |")
    ceil = s4.loc[("logreg", "all_rows", "all384")]
    L += ["", f"Length-only floor {v(s4.loc[('logreg', 'subsample', 'length')].roc_auc_mean, S4, 'length floor AUC')}; classical ceiling, "
          f"5,044 rows {v(s4.loc[('svm', 'full', 'all384')].roc_auc_mean, S4, 'dev-rows ceiling AUC')}; classical ceiling, all rows "
          f"(≈ 17,600 per fold) {v(ceil.roc_auc_mean, S4, 'all-rows ceiling AUC')}.", ""]
    sources(L, S3, T4, S4)
    section(L, 11, "Performance Metrics", "Reports accuracy only beside class-balanced metrics.")
    L += ["Every table in Boxes 6 and 10 reports accuracy beside balanced accuracy, recall, macro-F1 and ROC AUC; "
          "per-class precision, recall, F1 and AUC are in results/mhtext/phase3/summary.csv.", ""]
    sources(L, S3)

    # ---------------------------------------------------------------- Box 12
    section(L, 12, "Privacy Evaluation", "Measures re-identification, membership inference and the utility cost of DP, beside NSCH.")
    p5md = (p5 / "phase5.md").read_text(encoding="utf-8")
    summ = p5md.split("## Summary", 1)[1].split("\n## ", 1)[0].strip()
    L += ["*Summary carried from results/mhtext/phase5/phase5.md (every figure there is read from the files below)*", "", summ, ""]
    fr = csv(p5 / "reid/fragment_risk.csv")
    FR = p5 / "reid/fragment_risk.csv"
    ns_r = csv(R / "reidentification/risk.csv").set_index("table")
    dv = fr[(fr.corpus == "dev") & (fr.text == "as released") & fr.attacker_knows.str.startswith("a ")].set_index("span_words")
    lk = csv(p5 / "reid/linkage.csv")
    for c in ("top1", "top5", "top10"):
        lk[c] = numeric(lk[c])          # suppressed cells -> NaN; none of them is shown here
    LK = p5 / "reid/linkage.csv"
    pca12 = lk[(lk.released == "PCA-8 encoder input") & (lk.fragment_words == 12) & (lk.epsilon_input.astype(str) == "inf")].iloc[0]
    mia = csv(p5 / "mia/summary.csv")
    mia["target_epsilon"] = mia.target_epsilon.map(lambda e: "inf" if str(e) in ("inf", "nan") else f"{float(e):g}")
    mia = mia.set_index(["arm", "target_epsilon"])
    MIA = p5 / "mia/summary.csv"
    nmia = csv(R / "exp6-nsch2024-membership-inference-20261112-120934/summary.csv")
    nmia["target_epsilon"] = nmia.target_epsilon.map(lambda e: "inf" if str(e) in ("inf", "nan") else f"{float(e):g}")
    nmia = nmia.set_index(["arm", "target_epsilon"])
    NMIA = R / "exp6-nsch2024-membership-inference-20261112-120934/summary.csv"
    L += ["| Measure | NSCH | Mental Health Text |", "|---|---|---|",
          f"| Unique rows, most identifying table (NSCH: 7 demographics + state; text: 5-word fragment) | "
          f"{v(ns_r.loc['7 demographic QIs, raw values + U.S. state', 'uniqueness'], R / 'reidentification/risk.csv', 'NSCH unique 7QI+state', '{:.1%}')} | "
          f"{v(dv.loc[5, 'uniqueness'], FR, 'text unique 5-word', '{:.1%}')} |",
          f"| Unique rows after the pipeline's protection | {v(ns_r.loc['7 demographic QIs after k-anonymity (k = 5, as used in the pipeline)', 'uniqueness'], R / 'reidentification/risk.csv', 'NSCH unique after k-anon', '{:.1%}')} (k-anonymity) | "
          f"{v(fr[(fr.corpus == 'dev') & (fr.text != 'as released') & (fr.span_words == 5) & fr.attacker_knows.str.startswith('a ')].uniqueness.iloc[0], FR, 'text unique 5-word scrubbed', '{:.1%}')} (scrubbing) |",
          f"| Linkage through the 8 quantum inputs (12-word fragment, projection known) | n/a | {v(pca12.top1, LK, 'linkage PCA-8 12 words', '{:.1%}')} |",
          f"| Overfit MLP control, membership attack AUC | {v(nmia.loc[('mlp_overfit', 'inf'), 'attack_auc_mean'], NMIA, 'NSCH mlp_overfit MIA')} | {v(mia.loc[('mlp_overfit', 'inf'), 'attack_auc_mean'], MIA, 'text mlp_overfit MIA')} |",
          f"| QSVM, membership attack AUC | {v(nmia.loc[('qsvm', 'inf'), 'attack_auc_mean'], NMIA, 'NSCH qsvm MIA')} | {v(mia.loc[('qsvm', 'inf'), 'attack_auc_mean'], MIA, 'text qsvm MIA')} |",
          f"| VQC (no DP), membership attack AUC | {v(nmia.loc[('dp_vqc', 'inf'), 'attack_auc_mean'], NMIA, 'NSCH vqc MIA')} | {v(mia.loc[('dp_vqc', 'inf'), 'attack_auc_mean'], MIA, 'text vqc MIA')} |", ""]
    sources(L, p5 / "phase5.md", FR, LK, MIA, R / "reidentification/risk.csv", NMIA)

    # ---------------------------------------------------------------- Box 13
    section(L, 13, "Scalability Analysis (Simulation)", "FedAvg with client DP-SGD (ε = 4) over 3, 5 and 10 SIMULATED clients (exp4 method).")
    fm = csv(MH / "phase6/fold_metrics.csv")
    FM = MH / "phase6/fold_metrics.csv"
    ns4 = csv(NS_EXP4 / "summary.csv")
    L += ["| Arm | Partition | 3 clients | 5 clients | 10 clients | NSCH (iid) 3 / 5 / 10 |", "|---|---|---|---|---|---|"]
    for arm in ("vqc_sub", "logreg_full"):
        for part in ("iid", "by_length"):
            cells = [v(fm[(fm.arm == arm) & (fm.partition == part) & (fm.n_clients == n)].roc_auc.mean(), FM, f"fed {arm} {part} {n}") for n in (3, 5, 10)]
            ncl = " / ".join(v(ns4[(ns4.arm == arm) & (ns4.partition == "iid") & (ns4.n_clients == n)].roc_auc_mean.iloc[0],
                               NS_EXP4 / "summary.csv", f"NSCH fed {arm} iid {n}") for n in (3, 5, 10))
            L.append(f"| {arm} | {part} | " + " | ".join(cells) + f" | {ncl} |")
    L += ["", f"Communication per FedAvg round: VQC {v(fm[fm.arm == 'vqc_sub'].bytes_per_round.mean() / 1024, FM, 'vqc kB per round', '{:.2f}')} kB, "
          f"LogReg {v(fm[fm.arm == 'logreg_full'].bytes_per_round.mean() / 1024, FM, 'logreg kB per round', '{:.2f}')} kB (mean over clients and folds). "
          "by_length is the non-IID substitute for NSCH's by_state (posts carry no site).", ""]
    sources(L, FM, NS_EXP4 / "summary.csv", MH / "phase6/phase6.md")

    # ---------------------------------------------------------------- Box 14
    section(L, 14, "Output & Visualization", "Lists what this dataset adds to the visual outputs.")
    L += ["| Diagram item | NSCH | Mental Health Text |", "|---|---|---|",
          "| Feature and attention visualization: attention | Not available | Not produced: the encoder is used frozen and "
          "attention maps were not extracted; available in principle, not built |",
          "| Quantum state visualization | Built (Box 5 of the NSCH report) | Not repeated: same circuits, different inputs |",
          "| Results tables | Built | Built (this report and the phase reports) |",
          f"| Results figures | Built | Built: {len(FIGMAN['figures'])} figures in Boxes "
          f"{', '.join(str(b) for b in sorted({f['box'] for f in FIGMAN['figures']}))} (index below) |", "",
          "Figures are drawn by scripts/build_mhtext_figures.py from committed files only; nothing is recomputed there. "
          "Every plotted column passes a suppression guard that stops the build if any cell is a suppressed small count "
          f"({len(FIGMAN['plotted_columns_checked_for_suppression'])} plotted columns checked, none suppressed; list in "
          "results/mhtext/report_figures/figures.json). No cell shown as \"< 5 posts (suppressed)\" in a table appears as a "
          "point, bar or label in any figure: the linkage figure plots only the rows without local DP, and no figure "
          "plots the self-disclosure table. Figures contain no post text: axis labels and legends are model, feature-"
          "column and setting names only. All figures print in black and white (markers, line styles, hatching).", "",
          "<<FIGINDEX>>", ""]
    sources(L, FIGDIR / "figures.json")

    # ---------------------------------------------------------------- Box 15
    section(L, 15, "Key Outcomes", "States which findings replicate across two unrelated datasets, and the limits.")
    pc = MH / "phase7/pattern_checks"
    q3 = csv(pc / "p3_qubits_8_vs_4.csv")
    qq = q3[q3.model != "logreg"]
    nq, tq = qq[qq.dataset == "NSCH 2024"], qq[qq.dataset == "Mental Health Text"]
    rep = csv(MH / "phase6/tests_replication_clients.csv")
    REP = MH / "phase6/tests_replication_clients.csv"
    r_t = rep[(rep.dataset == "Mental Health Text") & (rep.partition == "iid")].set_index("quantity")
    r_n = rep[(rep.dataset == "NSCH 2024") & (rep.partition == "iid")].set_index("quantity")
    ns_t = csv(NS_EXP2 / "tests_same_rows_features.csv")
    ns_s = csv(NS_EXP2 / "summary.csv").set_index(["model", "regime", "feature_set"])

    def nsp(qm, cm):
        r = ns_t[ns_t.quantum.str.startswith(qm + "/") & ns_t.classical.str.startswith(cm + "/") & (ns_t.metric == "roc_auc")]
        return r.p_nb_t_holm.iloc[0]

    tp = t4[t4.metric == "roc_auc"].set_index(["quantum", "classical"])
    COB = R / "exp2-nsch2024-vqc-cobyla-20261112-121718/comparison.csv"
    cob = csv(COB).query("metric == 'roc_auc'").a_mean.iloc[0]
    L += ["*Table 15.1. Patterns tested on both datasets (same method, same statistics)*", "",
          "| # | Pattern | NSCH 2024 | Mental Health Text | Status |", "|---|---|---|---|---|",
          f"| P1 | QSVM ties the best classical model on matched rows | {v(ns_s.loc[('qsvm', 'subsample', 'pca8')].roc_auc_mean, NS_EXP2 / 'summary.csv', 'NSCH qsvm AUC')} vs "
          f"{v(ns_s.loc[('logreg', 'subsample', 'pca8')].roc_auc_mean, NS_EXP2 / 'summary.csv', 'NSCH logreg pca8 sub AUC')}, p {v(nsp('qsvm', 'logreg'), NS_EXP2 / 'tests_same_rows_features.csv', 'NSCH qsvm vs logreg p')} | "
          f"{v(s4.loc[('qsvm', 'subsample', 'pca8')].roc_auc_mean, S4, 'text qsvm AUC')} vs {v(tp.loc[('qsvm/subsample/pca8', 'logreg/subsample/pca8'), 'classical_mean'], T4, 'text logreg pca8 AUC')}, "
          f"p {v(tp.loc[('qsvm/subsample/pca8', 'logreg/subsample/pca8'), 'p_nb_t_holm'], T4, 'text qsvm vs logreg p')} | Replicates |",
          f"| P2 | VQC loses by about 0.02, significantly | {v(ns_s.loc[('vqc', 'subsample', 'pca4')].roc_auc_mean, NS_EXP2 / 'summary.csv', 'NSCH vqc AUC')} vs "
          f"{v(ns_s.loc[('logreg', 'subsample', 'pca4')].roc_auc_mean, NS_EXP2 / 'summary.csv', 'NSCH logreg pca4 AUC')}, p {v(nsp('vqc', 'logreg'), NS_EXP2 / 'tests_same_rows_features.csv', 'NSCH vqc vs logreg p')} | "
          f"{v(s4.loc[('vqc', 'subsample', 'pca4')].roc_auc_mean, S4, 'text vqc AUC')} vs {v(tp.loc[('vqc/subsample/pca4', 'logreg/subsample/pca4'), 'classical_mean'], T4, 'text logreg pca4 AUC')}, "
          f"p {v(tp.loc[('vqc/subsample/pca4', 'logreg/subsample/pca4'), 'p_nb_t_holm'], T4, 'text vqc vs logreg p')} | Replicates |",
          f"| P3 | 4 → 8 qubits gives no significant gain for any quantum model | {v(int(nq.significant.sum()), pc / 'p3_qubits_8_vs_4.csv', 'NSCH significant qubit cells', '{}')} of {len(nq)} cells significant either way; "
          f"headline configs QSVM {nq[(nq.model == 'qsvm') & (nq.feature_method == 'pca') & (nq.encoding == 'angle')].mean_diff.iloc[0]:+.3f}, VQC re-upload "
          f"{nq[(nq.model == 'vqc') & (nq.feature_method == 'pca') & (nq.encoding == 'reupload')].mean_diff.iloc[0]:+.3f} | "
          f"0 of {len(tq)} cells gain significantly; {v(int(((tq.mean_diff < 0) & tq.significant).sum()), pc / 'p3_qubits_8_vs_4.csv', 'text significant qubit drops', '{}')} drop (VQC, weak encodings); headline QSVM "
          f"{tq[(tq.model == 'qsvm') & (tq.feature_method == 'pca') & (tq.encoding == 'angle')].mean_diff.iloc[0]:+.3f}, VQC re-upload "
          f"{tq[(tq.model == 'vqc') & (tq.feature_method == 'pca') & (tq.encoding == 'reupload')].mean_diff.iloc[0]:+.3f} | Replicates, in this form |",
          f"| P5 | Federated VQC falls from 3 to 10 clients; LogReg stays flat | VQC {r_n.loc['VQC: 10 clients minus 3', 'mean']:+.3f}, p {v(r_n.loc['VQC: 10 clients minus 3', 'p_nb_t_holm'], REP, 'NSCH fed VQC drop p')}; "
          f"LogReg {r_n.loc['LogReg: 10 clients minus 3', 'mean']:+.3f}, p {v(r_n.loc['LogReg: 10 clients minus 3', 'p_nb_t_holm'], REP, 'NSCH fed LogReg p')} | "
          f"VQC {r_t.loc['VQC: 10 clients minus 3', 'mean']:+.3f}, p {v(r_t.loc['VQC: 10 clients minus 3', 'p_nb_t_holm'], REP, 'text fed VQC drop p')}; "
          f"LogReg {r_t.loc['LogReg: 10 clients minus 3', 'mean']:+.3f}, p {v(r_t.loc['LogReg: 10 clients minus 3', 'p_nb_t_holm'], REP, 'text fed LogReg p')} | Replicates |",
          f"| P6 | Hybrid QNN is the lowest quantum model on ROC AUC | {v(ns_s.loc[('hybrid', 'subsample', 'all95')].roc_auc_mean, NS_EXP2 / 'summary.csv', 'NSCH hybrid AUC')} "
          f"(VQC {ns_s.loc[('vqc', 'subsample', 'pca4')].roc_auc_mean:.3f}, VQC-COBYLA {v(cob, COB, 'NSCH VQC COBYLA AUC')}, QSVM {ns_s.loc[('qsvm', 'subsample', 'pca8')].roc_auc_mean:.3f}) | "
          f"{v(hy.roc_auc_mean, S4, 'text hybrid AUC')} (VQC {s4.loc[('vqc', 'subsample', 'pca4')].roc_auc_mean:.3f}, QSVM {s4.loc[('qsvm', 'subsample', 'pca8')].roc_auc_mean:.3f}) | "
          "Replicates on ROC AUC; on NSCH the hybrid edges the VQC on accuracy, precision and F1 |", ""]
    L += [f"P3 is stated in its measured form. 'Changes almost nothing' holds for the headline encodings (angle and "
          f"re-upload on PCA). For weak encodings the means move more (up to {v(nq.range_4_6_8.max(), pc / 'p3_qubits_8_vs_4.csv', 'NSCH max qubit range', '{:.3f}')} "
          f"on NSCH, {v(tq.range_4_6_8.max(), pc / 'p3_qubits_8_vs_4.csv', 'text max qubit range', '{:.3f}')} on text across 4, 6, 8 qubits), "
          "not significantly on NSCH and as two significant drops on text. On neither dataset do more qubits help.", ""]
    m4t = csv(pc / "p4_mi_vs_pca_text.csv")
    m4n = csv(pc / "p4_mi_vs_pca_nsch_stored.csv")
    tw, tl = (m4t.mean_diff < 0), (m4t.mean_diff > 0)
    L += ["Table 15.1 holds four replications from the list of candidates (P1, P2, P3, P6) plus the federated one (P5): "
          "five in total. P4 (PCA vs mutual-information selection) was a candidate and is not in the table; it is "
          "reported below as a disagreement.", "",
          "### Where the two datasets disagree", "",
          "**P4 — the better way to reduce inputs to 4-8 qubits depends on the kind of data.** Same comparison on both "
          "datasets (exp1 sweep: PCA vs mutual-information column selection, every model x encoding x qubit cell, paired "
          "NB-t, Holm with NSCH's families: non-basis encodings in one family, basis in another).", "",
          "| | NSCH 2024 (95 survey codes) | Mental Health Text (384-dim sentence embedding) |", "|---|---|---|",
          f"| Cells compared | {len(m4n)} | {len(m4t)} |",
          f"| PCA significantly better | {v(int(((m4n.mean_diff < 0) & m4n.significant).sum()), pc / 'p4_mi_vs_pca_nsch_stored.csv', 'NSCH PCA sig wins', '{}')} | "
          f"{v(int((tw & m4t.significant).sum()), pc / 'p4_mi_vs_pca_text.csv', 'text PCA sig wins', '{}')} |",
          f"| MI significantly better | {v(int(((m4n.mean_diff > 0) & m4n.significant).sum()), pc / 'p4_mi_vs_pca_nsch_stored.csv', 'NSCH MI sig wins', '{}')} (all basis-encoding cells) | "
          f"{v(int((tl & m4t.significant).sum()), pc / 'p4_mi_vs_pca_text.csv', 'text MI sig wins', '{}')} |",
          f"| PCA higher mean, any significance | {v(int((m4n.mean_diff < 0).sum()), pc / 'p4_mi_vs_pca_nsch_stored.csv', 'NSCH PCA higher mean', '{}')} | "
          f"{v(int(tw.sum()), pc / 'p4_mi_vs_pca_text.csv', 'text PCA higher mean', '{}')} |", "",
          "Finding: the better reduction depends on the kind of data. On sentence embeddings PCA is clearly better; on "
          "survey codes PCA never wins significantly and MI selection wins outright for the basis encoding. Anyone doing "
          "quantum ML should choose the reduction for the data type, not carry one choice from tabular to text inputs or "
          "back. Why the two differ is not tested here (one candidate explanation: information spread across many "
          "correlated embedding dimensions versus concentrated in a few survey columns).", "",
          "**Cell-level caveat (a correction to an earlier summary of ours).** An earlier summary said \"PCA beats MI for "
          "every model\" on text. That was a per-model average. At cell level PCA has the higher mean in "
          f"{int(tw.sum())} of {len(m4t)} text cells, not all: MI has the higher mean in {int(tl.sum())}, for example "
          + ", ".join(f"{r.model.upper()} {r.encoding} at {int(r.n_qubits)} qubits ({r.auc_mi:.3f} vs {r.auc_pca:.3f})"
                      for _, r in m4t[tl].head(3).iterrows())
          + "; none of these is significant.", "",
          "NSCH's separate comparison of PCA against UMAP and an autoencoder (NSCH report, Box 3, Table 3.3) is a different "
          "comparison and is not part of P4.", "",
          "**On P5, plainly:** NSCH's federated VQC drop has Holm p "
          f"{r_n.loc['VQC: 10 clients minus 3', 'p_nb_t_holm']:.3f}; it only just clears the threshold, and on its own that would be weak "
          f"evidence. The text drop is a third the size ({r_t.loc['VQC: 10 clients minus 3', 'mean']:+.3f} vs "
          f"{r_n.loc['VQC: 10 clients minus 3', 'mean']:+.3f}) but far tighter (95 % CI {r_t.loc['VQC: 10 clients minus 3', 'ci95_low']:+.3f} to "
          f"{r_t.loc['VQC: 10 clients minus 3', 'ci95_high']:+.3f}) and clearly significant (p {r_t.loc['VQC: 10 clients minus 3', 'p_nb_t_holm']:.3f}). "
          "Neither dataset alone would settle it; the replication is what makes it solid.", "",
          "**What is new on this dataset, and does not replicate because NSCH cannot show it:** a free-text post is far "
          "more identifying than a row of survey codes, and the 8 numbers fed to the quantum circuit still identify the "
          "post about a third of the time (Box 12). The small quantum models leak little, and that comes from their low "
          "capacity, not from anything quantum.", "",
          "**Limits.** Authors' ages are unknown, so this is not an adolescent sample. Labels come from the source corpora, "
          "not from clinicians. Clients are simulated. Circuits are simulated, not run on hardware. Quantum results use the "
          "binary view only. The locked hold-out has not been scored.", ""]
    sources(L, pc / "p3_qubits_8_vs_4.csv", pc / "p4_mi_vs_pca_text.csv", pc / "p4_mi_vs_pca_nsch_stored.csv", REP,
            NS_EXP2 / "summary.csv", NS_EXP2 / "tests_same_rows_features.csv", S4, T4)

    # ---------------------------------------------------------------- coverage
    BOX[0] = "Coverage"
    L += ["## Coverage update: what this dataset added to the architecture", "",
          "| Box | Diagram item | Before (NSCH only) | Now |", "|---|---|---|---|",
          "| 1 | Social media text | Not available on this dataset | **BUILT** |",
          "| 3 | Transformer embeddings (e.g., BERT, RoBERTa) | Not available on this dataset | **BUILT** (frozen MiniLM, BERT family) |",
          "| 3 | Sentiment & emotion features | Not available on this dataset | **BUILT** (frozen emotion classifier) |",
          "| 3 | Contextual representations | Not available on this dataset | **BUILT** (contextual sentence embeddings) |",
          "| 9 | Emotion and sentiment analysis | Not available on this dataset | **BUILT** as features (not a target) |",
          "| 10 | BERT + ML | Not available on this dataset | **BUILT** |",
          "| 10 | Deep Learning (LSTM, CNN) | Not available on this dataset | **BUILT** |",
          "| 12 | Re-identification of free text | not applicable (no text) | **BUILT** (fragment risk, linkage) |", "",
          "Still unavailable, with the reason:", "",
          "| Box | Diagram item | Reason |", "|---|---|---|",
          "| 1 | Clinical notes (de-identified) | No clinical-notes dataset is part of this project; access needs a data-use agreement with a health provider. |",
          "| 1 | Counselling records | No counselling-records dataset is part of this project; such records are not public. |",
          "| 1 | Wearable/IoT data (optional) | No sensor dataset is part of this project. |",
          "| 3 | Physiological signals (optional) | Needs sensor data (above). |",
          "| 9 | Stress | Neither dataset has a stress label. |",
          "| 9 | Topic and trend analysis | Needs text over time; the posts carry no timestamps. |",
          "| 9 | Personalized insights (simulated) | Needs a per-person history; posts are not linked to authors. |",
          "| 14 | Attention visualization | Available in principle (frozen transformer); not extracted. |", ""]

    L = insert_figures(L)
    L += ["## Traceability appendix", "", "Every number and figure in the report, with the file it came from.", "",
          "| Box | Item | Value | File |", "|---|---|---|---|"]
    L += [f"| {b} | {i} | {val} | {f} |" for b, i, val, f in TRACE]
    write_outputs(L)


# ---------------------------------------------------------------- figures and printing
FIGDIR = MH / "report_figures"
FIGMAN = json.loads((FIGDIR / "figures.json").read_text(encoding="utf-8")) if (FIGDIR / "figures.json").exists() else {
    "figures": [], "plotted_columns_checked_for_suppression": []}
CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
NUMBERED: list[dict] = []


def insert_figures(L: list[str]) -> list[str]:
    """Place each figure at the end of its box as Figure <box>.<k>, and trace its source files."""
    by_box: dict[int, list[dict]] = {}
    for f in FIGMAN["figures"]:
        by_box.setdefault(f["box"], []).append(f)
    out, current = [], None

    def flush(box):
        for k, f in enumerate(by_box.get(box, []), 1):
            n = f"{box}.{k}"
            NUMBERED.append({**f, "n": n})
            out.extend([f"![Figure {n}](report_figures/{f['file']})", "",
                        f"*Figure {n}. {f['caption']} Source: {'; '.join(f['sources'])}*", ""])
            for s in f["sources"]:
                TRACE.append((f"Box {box}", f"Figure {n} (plotted values)", "all plotted points", s.split(" (copy of")[0]))

    for line in L:
        if line.startswith("## "):
            if current is not None:
                flush(current)
            current = int(line.split()[2]) if line.startswith("## Box ") else None
        out.append(line)
    if current is not None:
        flush(current)
    return out


def fig_index(pages: dict[str, str] | None) -> list[str]:
    rows = ["| Figure | Box | Caption | Page |", "|---|---|---|---|"]
    for f in NUMBERED:
        cap = f["caption"][:90] + ("…" if len(f["caption"]) > 90 else "")
        rows.append(f"| Figure {f['n']} | Box {f['box']} | {cap} | {(pages or {}).get(f['n'], '–')} |")
    return ["*Index of figures*", ""] + rows


CSS = """
@page { size: A4 portrait; margin: 16mm 15mm 18mm 15mm;
  @bottom-center { content: "Page " counter(page); font: 9pt "Segoe UI", Arial, sans-serif; color: #000; } }
body { font: 10.5pt/1.4 "Segoe UI", Arial, sans-serif; color: #000; background: #fff; }
h1 { font-size: 17pt; margin: 0 0 4mm; } h3 { font-size: 11.5pt; margin: 4mm 0 2mm; }
h2 { font-size: 14pt; margin: 0 0 3mm; border-bottom: 1.5pt solid #000; padding-bottom: 1mm; break-before: page; }
table { border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 1mm 0 2mm; }
th, td { border: 0.6pt solid #000; padding: 0.6mm 1mm; text-align: left; vertical-align: top; overflow-wrap: anywhere; }
th { background: #e6e6e6; } tr { break-inside: avoid; }
p.src { font-size: 8pt; color: #333; }
figure { margin: 2mm 0 4mm; break-inside: avoid; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 200mm; display: block; margin: 0 auto; }
figcaption { font-size: 8.8pt; margin-top: 1.5mm; }
.appendix td { font-size: 7.5pt; }
ul { margin: 1mm 0 2mm 5mm; padding-left: 4mm; }
"""


def _inline(s: str) -> str:
    import html
    import re
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def md_to_html(lines: list[str]) -> str:
    """Converter for the subset of markdown this builder writes (headings, tables, figures, lists, paragraphs)."""
    import html
    import re
    out, i, first_h2, appendix = [], 0, True, False
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("# "):
            out.append(f"<h1>{_inline(ln[2:])}</h1>")
        elif ln.startswith("## "):
            appendix = ln.startswith("## Traceability")
            style = " style='break-before:auto'" if first_h2 else ""
            out.append(f"<h2{style}>{_inline(ln[3:])}</h2>")
            first_h2 = False
        elif ln.startswith("### "):
            out.append(f"<h3>{_inline(ln[4:])}</h3>")
        elif ln.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", ln)
            cap = lines[i + 2] if i + 2 < len(lines) else ""
            cap = cap.strip("*")
            num, _, rest = cap.partition(". ")
            text, _, src = rest.partition(" Source: ")
            out.append(f"<figure><img src='{html.escape(m.group(2))}' alt='{html.escape(m.group(1))}'>"
                       f"<figcaption><b>{_inline(num)}.</b> {_inline(text)} <span class='src'>Source: {_inline(src)}</span>"
                       "</figcaption></figure>")
            i += 3
            continue
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split(" | ")])
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= set("-|: ")]
            out.append(f"<table{' class=appendix' if appendix else ''}><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head)
                       + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table>")
            continue
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        else:
            out.append(f"<p{' class=src' if ln.startswith('Source: ') else ''}>{_inline(ln)}</p>")
        i += 1
    return ("<!doctype html><html><head><meta charset='utf-8'><title>Mental Health Text results report</title>"
            f"<style>{CSS}</style></head><body>" + "\n".join(out) + "</body></html>")


def print_pdf(html_path: Path, pdf_path: Path) -> int:
    import re
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()], check=True, capture_output=True, timeout=180)
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))


def write_outputs(L: list[str]) -> None:
    def with_index(pages):
        j = L.index("<<FIGINDEX>>")
        return L[:j] + fig_index(pages) + L[j + 1:]
    # page of each figure: print the report cut just after that figure's caption and count the pages
    pages, probe_h, probe_p = {}, MH / "_probe.html", MH / "_probe.pdf"
    full = with_index(None)
    for f in NUMBERED:
        cut = next(k for k, ln in enumerate(full) if ln.startswith(f"*Figure {f['n']}. "))
        probe_h.write_text(md_to_html(full[:cut + 1]), encoding="utf-8")
        pages[f["n"]] = f"{print_pdf(probe_h, probe_p)}"
    probe_h.unlink(missing_ok=True)
    probe_p.unlink(missing_ok=True)
    final = with_index(pages)
    md = MH / "results_report_mhtext.md"
    md.write_text("\n".join(final) + "\n", encoding="utf-8")
    hp = MH / "results_report_mhtext.html"
    hp.write_text(md_to_html(final), encoding="utf-8")
    n = print_pdf(hp, MH / "results_report_mhtext.pdf")
    print(f"wrote {md.name}, {hp.name}, results_report_mhtext.pdf ({n} pages, {len(NUMBERED)} figures, "
          f"{len(TRACE)} traced items); figure pages {pages}")


if __name__ == "__main__":
    main()
