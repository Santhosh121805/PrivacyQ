"""Fit, once, the two models behind the Inspect page's "Screening indication" (box 9).

Same rows and preprocessing as the Inspect cache (scripts/build_config_models.py): the 1,500-row stratified dev
subsample (seed 2026), inner-train 1,200 / inner-val 300, and the TabularPreprocessor already fitted on the
inner-train rows (read from models/config_models.pkl, so the 95 columns are identical). The demo records the page
shows are dev rows outside that subsample, so neither model has seen them.

- Three-class model: class-weighted multinomial logistic regression on the 95 columns, classes none / anxiety only /
  depression (Experiment 9 labels; rows with undetermined anxiety excluded), C chosen on inner-val macro one-vs-rest
  ROC-AUC.
- Risk model: class-weighted binary logistic regression on the same 95 columns, C chosen on inner-val ROC-AUC.
  Low / moderate / high cut-points chosen on the inner-val probabilities (src/m8_eval/screening.py): 95 %
  sensitivity for the lower cut (rule-out), 95 % specificity for the upper cut (rule-in). A 90 % / 90 % rule was
  tried first and left only 5 of the 300 inner-val rows in the moderate band, so the conventional 95 % / 95 %
  screening rule is used; both are recorded in the cache.

Usage: python scripts/build_screening_models.py   ->  models/screening_models.pkl
"""

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.m1_data.nsch import three_class_labels  # noqa: E402
from src.m5_models.classical import make_classical  # noqa: E402
from src.m5_models.multiclass import CLASS_NAMES, MulticlassLogReg  # noqa: E402
from src.m8_eval import multiclass_metrics  # noqa: E402
from src.m8_eval.screening import band_summary, risk_cutpoints  # noqa: E402
from src.m10_exp.common import holdout_split, load_config, load_dataset  # noqa: E402
from src.m10_exp.tuning import inner_split, select_and_fit, stratified_subsample  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs/holdout_final.yaml")
    seed = int(cfg["project"]["seed"])
    loader, frame = load_dataset(cfg)
    labels = frame[loader.label_column].to_numpy(dtype=int)
    dev, _ = holdout_split(labels, float(cfg["split"]["holdout_fraction"]), seed)
    sub = stratified_subsample(dev, labels, 1500, seed)
    tr, iv = inner_split(sub, labels, float(cfg["tuning"]["inner_val_fraction"]), seed)
    with open(ROOT / "models/config_models.pkl", "rb") as fh:
        cache = pickle.load(fh)
    prep, pool = cache["prep"], cache["pool"]
    assert not set(pool.index) & set(sub.tolist()), "demo pool overlaps the training subsample"
    X = lambda rows: prep.transform(frame.iloc[rows], loader.label_column).to_numpy()  # noqa: E731

    # three classes
    y3 = three_class_labels(loader)
    det = ~np.isnan(y3)
    tr3, iv3 = tr[det[tr]], iv[det[iv]]
    yt, yv = y3[tr3].astype(int), y3[iv3].astype(int)
    best3 = None
    for C in (0.01, 0.1, 1.0, 10.0):
        m = MulticlassLogReg(C=C, seed=seed).fit(X(tr3), yt)
        auc = multiclass_metrics(yv, m.class_scores(X(iv3)), m.predict(X(iv3)), CLASS_NAMES)["macro_ovr_auc"]
        if best3 is None or auc > best3[0]:
            best3 = (auc, m, C)
    print(f"three-class LogReg: C={best3[2]}, inner-val macro AUC {best3[0]:.3f} "
          f"(train {len(tr3)} rows, inner-val {len(iv3)}; excluded {len(tr) - len(tr3) + len(iv) - len(iv3)} undetermined)")

    # binary risk model and bands
    best = select_and_fit(lambda **p: make_classical("logreg", labels[tr], seed=seed, n_jobs=1, **p),
                          {"C": [0.01, 0.1, 1.0, 10.0]}, X(tr), labels[tr], X(iv), labels[iv])
    p_iv = best["model"].predict_proba(X(iv))[:, 1]
    cuts = risk_cutpoints(p_iv, labels[iv], 0.95, 0.95)
    cuts_90 = risk_cutpoints(p_iv, labels[iv], 0.90, 0.90)
    bands = band_summary(p_iv, labels[iv], cuts)
    print(f"risk LogReg: C={best['params']['C']}, inner-val AUC {roc_auc_score(labels[iv], p_iv):.3f}; cuts {cuts}; bands {bands}")

    pool_y3 = {int(i): (None if np.isnan(v) else CLASS_NAMES[int(v)]) for i, v in zip(pool.index, y3[pool.index])}
    with open(ROOT / "models/screening_models.pkl", "wb") as fh:
        pickle.dump({"three_class": {"model": best3[1], "C": best3[2], "inner_val_macro_auc": best3[0],
                                     "n_train": len(tr3), "n_inner_val": len(iv3)},
                     "risk": {"model": best["model"], "C": best["params"]["C"], "cuts": cuts, "inner_val_bands": bands,
                              "inner_val_auc": float(roc_auc_score(labels[iv], p_iv)), "rule_tried_first": {"cuts": cuts_90, "bands": band_summary(p_iv, labels[iv], cuts_90)}, "n_train": len(tr),
                              "n_inner_val": len(iv), "n_inner_val_pos": int(labels[iv].sum())},
                     "pool_three_class": pool_y3, "source": loader.source_tag()}, fh)
    print("cached models/screening_models.pkl")


if __name__ == "__main__":
    main()
