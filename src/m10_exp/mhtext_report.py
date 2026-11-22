"""Phase 3 report for the Mental Health Text baselines: every number read from the CSVs the runner wrote."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.m1_data.mhtext import CLASSES_3, CLASSES_4

NAMES = {"length_only_logreg": "Length only (LogReg on log words, log characters of the text)",
         "bert_pooled_logreg": "Frozen MiniLM sentence encoder (BERT family), whole text + LogReg",
         "bert_sentence_logreg": "Frozen MiniLM sentence encoder (BERT family), per sentence mean+max + LogReg",
         "logreg": "LogReg, all pooled features", "xgboost": "XGBoost, all pooled features",
         "mlp": "MLP, all pooled features", "svm": "SVM (RBF), all pooled features",
         "lstm": "BiLSTM on token sequences", "cnn": "1-D CNN on token sequences"}


def _ms(s: pd.DataFrame, r: pd.Series, m: str) -> str:
    return f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_std']:.3f}"


def write_phase3_report(out: Path, cfg: dict) -> None:
    s = pd.read_csv(out / "summary.csv")
    sig = pd.read_csv(out / "tests_vs_length_only.csv")
    run = json.loads((out / "run.json").read_text())
    meta = run["meta"]
    L = ["# Mental Health Text — Phase 3 (box 10): baselines", "",
         f"Same repeated stratified folds for every model within a view ({cfg['cv']['n_splits']} x {cfg['cv']['n_repeats']} "
         "on development rows); hyperparameters and early stopping on an inner split of each fold's training rows; "
         "scalers fitted on inner-train rows only. The locked hold-out is not used. Mean ± SD over folds.",
         "Views: " + "; ".join(f"{v}-class: {m['dev_rows']:,} development rows, {m['folds']} folds" for v, m in meta.items()) + ".",
         "Accuracy is shown only beside recall, F1 and ROC-AUC. Labels whose source posts were filed under two classes "
         "were dropped in Phase 1; that soft Depression/Suicidal boundary caps the achievable score between those two "
         "classes, so a middling per-class score there is not by itself a modelling failure.", ""]
    for view, classes in (("4-class", CLASSES_4), ("3-class", CLASSES_3)):
        cv = s[(s["view"] == view) & (s["set"] == "cv")].sort_values("macro_auc_mean", ascending=False)
        if cv.empty:
            continue
        L += [f"## {view}: macro metrics (cross-validation), sorted by macro ROC-AUC", "",
              "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Macro ROC-AUC |", "|---|---|---|---|---|---|"]
        for _, r in cv.iterrows():
            L.append(f"| {NAMES.get(r['model'], r['model'])} | {_ms(s, r, 'accuracy')} | {_ms(s, r, 'macro_precision')} | "
                     f"{_ms(s, r, 'macro_recall')} | {_ms(s, r, 'macro_f1')} | {_ms(s, r, 'macro_auc')} |")
        L += ["", f"### {view}: per class (recall / F1 / one-vs-rest ROC-AUC, means)", "",
              "| Model | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
        for _, r in cv.iterrows():
            L.append(f"| {NAMES.get(r['model'], r['model'])} | " + " | ".join(
                f"{r[f'recall_{c}_mean']:.3f} / {r[f'f1_{c}_mean']:.3f} / {r[f'auc_{c}_mean']:.3f}" for c in classes) + " |")
        L.append("")
    L += ["## Each model against the length-only baseline (NB-t, Holm within view x metric)", "",
          "| View | Metric | Model | Model | Length only | Difference [95% CI] | Holm p | Significant |", "|---|---|---|---|---|---|---|---|"]
    for _, r in sig.sort_values(["view", "metric", "model_mean"], ascending=[True, True, False]).iterrows():
        L.append(f"| {r['view']} | {r['metric']} | {NAMES.get(r['model'], r['model'])} | {r['model_mean']:.3f} | "
                 f"{r['reference_mean']:.3f} | {r['mean_diff']:+.3f} [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}] | "
                 f"{r['p_holm']:.3g} | {'yes' if r['significant'] else 'no'} |")
    both = s[s["set"] == "cv"].pivot_table(index="model", columns="view", values=["macro_auc_mean", "macro_f1_mean"])
    if ("macro_auc_mean", "3-class") in both.columns and ("macro_auc_mean", "4-class") in both.columns:
        L += ["", "## Four-class against three-class (the length-confound sensitivity run)", "",
              "| Model | 4-class macro AUC | 3-class macro AUC | 4-class macro F1 | 3-class macro F1 |", "|---|---|---|---|---|"]
        for m, r in both.sort_values(("macro_auc_mean", "4-class"), ascending=False).iterrows():
            L.append(f"| {NAMES.get(m, m)} | {r[('macro_auc_mean', '4-class')]:.3f} | {r[('macro_auc_mean', '3-class')]:.3f} | "
                     f"{r[('macro_f1_mean', '4-class')]:.3f} | {r[('macro_f1_mean', '3-class')]:.3f} |")
        L += ["", "Chance level for macro one-vs-rest ROC-AUC is 0.5 in both views. Macro F1 is not comparable across "
              "views at face value (four classes vs three, different class shares)."]
    q = s[s["set"] == "quarantine"]
    if not q.empty:
        L += ["", "## Sensitivity: the quarantined 9,582-row block (different provenance), scored by every fold's model", "",
              "| View | Model | Macro AUC, cross-validation | Macro AUC, quarantined block | Macro F1, quarantined block |",
              "|---|---|---|---|---|"]
        cvm = s[s["set"] == "cv"].set_index(["view", "model"])
        for _, r in q.sort_values(["view", "macro_auc_mean"], ascending=[True, False]).iterrows():
            L.append(f"| {r['view']} | {NAMES.get(r['model'], r['model'])} | {cvm.loc[(r['view'], r['model']), 'macro_auc_mean']:.3f} | "
                     f"{r['macro_auc_mean']:.3f} | {r['macro_f1_mean']:.3f} |")
        L += ["", "The block has no shipped numeric features (no Unique_ID rows); for models that use them they enter "
              "as the training mean after scaling. Its Normal posts are longer (median 31 words vs 10)."]
    L += ["", f"Run time {run['seconds'] / 60:.0f} min. Files: fold_metrics.csv, summary.csv, tests_vs_length_only.csv, run.json."]
    (out / "phase3.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
