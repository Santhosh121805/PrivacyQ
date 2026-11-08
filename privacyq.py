"""PrivacyQ measured-configuration catalogue.

Every configuration the tool can report is a row read from the experiment
result files (results/*/summary.csv). Nothing is interpolated: a request for a
combination that was never run returns the nearest measured row together with
an explicit list of what differs.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
EPS_ORDER = ["0.5", "1", "2", "4", "8", "inf"]
SITES = [1, 3, 5, 10]
MODELS = {"logreg": "Logistic regression", "qsvm": "QSVM", "vqc": "VQC", "hybrid": "Hybrid QNN"}
ENCODINGS = {"angle": "Angle", "amplitude": "Amplitude", "zz": "ZZ", "reupload": "Re-uploading"}


class CatalogueError(Exception):
    pass


def _eps(v) -> str:
    return "inf" if str(v) in ("inf", "nan", "None") else f"{float(v):g}"


@lru_cache(maxsize=1)
def runs() -> dict[str, Path]:
    cfg = ROOT / "configs" / "final_report.yaml"
    if not cfg.exists():
        raise CatalogueError("configs/final_report.yaml not found; the result directories cannot be located.")
    return {k: ROOT / v for k, v in yaml.safe_load(cfg.read_text(encoding="utf-8"))["runs"].items()}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise CatalogueError(f"Result file missing: {path.relative_to(ROOT)}. Re-run that experiment or the final report.")
    return pd.read_csv(path)


def _row(**kw) -> dict:
    base = dict(encoding="none", rows="1.2k", eps="inf", sites=1, partition="central", clip=None,
                feature_note="", qubits=None, depth=None, gates=None, run_note="")
    base.update(kw)
    return base


@lru_cache(maxsize=1)
def catalogue() -> pd.DataFrame:
    r = runs()
    res = read_csv(r["exp1"] / "circuit_resources.csv")

    def circuit(enc, n, model):
        if model == "logreg":
            return None, None
        if model == "qsvm":
            sel = res[(res.encoding == enc) & (res.n_qubits == n) & (res.n_layers == 0)]
        else:
            sel = res[(res.encoding == enc) & (res.n_qubits == n) & (res.n_layers == 2)]
        return (int(sel.depth.iloc[0]), int(sel.n_gates.iloc[0])) if len(sel) else (None, None)

    rows = []
    # Experiment 1: encoding sweep (10 folds, 1.2k training rows, PCA features, non-private)
    e1 = read_csv(r["exp1"] / "summary.csv")
    for _, s in e1[e1.feature_method == "pca"].iterrows():
        model, enc, n = s.model, (s.encoding if s.model != "logreg" else "none"), int(s.n_qubits)
        depth, gates = circuit(enc, n, model)
        rows.append(_row(model=model, encoding=enc, features=n, auc=s.roc_auc_mean, auc_std=s.roc_auc_std,
                         bal_acc=s.balanced_accuracy_mean, recall=s.recall_mean, folds=int(s.n_folds),
                         fit_s=s.fit_seconds_mean, qubits=None if model == "logreg" else n, depth=depth, gates=gates,
                         feature_note=f"PCA-{n}" if enc != "amplitude" else f"{n} qubits = {min(2**n - 1, 95)} PCA components",
                         source="Experiment 1 (encoding sweep)", priority=1))
    # Experiment 2: matched comparison (15 folds); preferred where it overlaps Experiment 1
    e2 = read_csv(r["exp2"] / "summary.csv")
    spec = {("qsvm", "pca8"): ("angle", 8), ("vqc", "pca4"): ("reupload", 4), ("hybrid", "all95"): ("angle", 8)}
    for _, s in e2.iterrows():
        if s.model in ("qsvm", "vqc", "hybrid"):
            if (s.model, s.feature_set) not in spec:
                continue
            enc, n = spec[(s.model, s.feature_set)]
            depth, gates = circuit(enc, n, "vqc" if s.model == "hybrid" else s.model)
            note = "learned linear compression of all 95 features to 8 qubits" if s.model == "hybrid" else f"PCA-{n}"
            rows.append(_row(model=s.model, encoding=enc, features=n, auc=s.roc_auc_mean, auc_std=s.roc_auc_std,
                             bal_acc=s.balanced_accuracy_mean, recall=s.recall_mean, folds=int(s.n_folds),
                             fit_s=s.fit_seconds_mean, qubits=n, depth=depth, gates=gates, feature_note=note,
                             source="Experiment 2 (quantum vs classical)", priority=0))
        elif s.model == "logreg":
            n = 95 if s.feature_set == "all95" else int(s.feature_set[3:])
            rows.append(_row(model="logreg", features=n, rows="1.2k" if s.regime == "subsample" else "~11k",
                             auc=s.roc_auc_mean, auc_std=s.roc_auc_std, bal_acc=s.balanced_accuracy_mean,
                             recall=s.recall_mean, folds=int(s.n_folds), fit_s=s.fit_seconds_mean,
                             feature_note="all 95 features" if n == 95 else f"PCA-{n}",
                             source="Experiment 2 (quantum vs classical)", priority=0))
    # Experiment 3: DP-SGD (10 folds) + clipping sensitivity
    arms = {"vqc_sub": ("vqc", "reupload", 4, "1.2k", 1.0), "logreg_sub": ("logreg", "none", 4, "1.2k", 1.0),
            "logreg_full": ("logreg", "none", 95, "~11k", 1.0), "logreg_sub_clip5": ("logreg", "none", 4, "1.2k", 5.0),
            "logreg_sub_clip10": ("logreg", "none", 4, "1.2k", 10.0)}
    e3 = pd.concat([read_csv(r["exp3"] / "summary.csv"), read_csv(r["exp3_sensitivity"] / "summary.csv")])
    for _, s in e3.iterrows():
        if s.arm not in arms or _eps(s.target_epsilon) == "inf" and s.arm != "vqc_sub":
            continue  # non-private LogReg rows are covered by Experiment 2 (15 folds)
        model, enc, n, nrows, clip = arms[s.arm]
        depth, gates = circuit(enc, n, model)
        rows.append(_row(model=model, encoding=enc, features=n, rows=nrows, eps=_eps(s.target_epsilon), clip=clip,
                         auc=s.roc_auc_mean, auc_std=s.roc_auc_std, bal_acc=s.balanced_accuracy_mean, recall=s.recall_mean,
                         folds=int(s.n_folds), fit_s=s.fit_seconds_mean, qubits=n if model != "logreg" else None,
                         depth=depth, gates=gates, epsilon_spent=s.epsilon_spent_mean,
                         feature_note="all 95 features" if n == 95 else f"PCA-{n}",
                         source="Experiment 3 (privacy–utility)" + (" — clipping sensitivity, added post hoc" if clip != 1.0 else ""),
                         priority=2 if (s.arm == "vqc_sub" and _eps(s.target_epsilon) == "inf") else (0 if clip == 1.0 else 3)))
    # Experiment 4: federated + DP at eps = 4 (10 folds)
    e4 = read_csv(r["exp4"] / "summary.csv")
    for _, s in e4[e4.setting == "federated"].iterrows():
        model, enc, n, nrows, clip = arms[s.arm]
        depth, gates = circuit(enc, n, model)
        rows.append(_row(model=model, encoding=enc, features=n, rows=nrows, eps=_eps(s.target_epsilon), clip=clip,
                         sites=int(s.n_clients), partition=s.partition, auc=s.roc_auc_mean, auc_std=s.roc_auc_std,
                         bal_acc=s.balanced_accuracy_mean, recall=s.recall_mean, folds=int(s.n_folds),
                         fit_s=s.fit_seconds_mean, qubits=n if model != "logreg" else None, depth=depth, gates=gates,
                         epsilon_spent=s.epsilon_spent_max, comm_kb=s.total_kB,
                         feature_note="all 95 features" if n == 95 else f"PCA-{n}",
                         source="Experiment 4 (federated + DP)", priority=0 if s.partition == "by_state" else 1))
    cat = pd.DataFrame(rows)
    cat = cat.sort_values("priority").drop_duplicates(
        subset=["model", "encoding", "features", "rows", "eps", "sites", "partition", "clip"], keep="first")
    return cat.reset_index(drop=True)


@lru_cache(maxsize=1)
def baseline() -> dict:
    """Unprivatised classical baseline: LogReg, all 95 features, full data."""
    r = runs()
    e2 = read_csv(r["exp2"] / "summary.csv")
    s = e2[(e2.model == "logreg") & (e2.regime == "full") & (e2.feature_set == "all95")].iloc[0]
    h = read_csv(r["holdout"] / "holdout_metrics.csv").set_index("model")
    return {"cv_auc": float(s.roc_auc_mean), "cv_folds": int(s.n_folds),
            "holdout_auc": float(h.loc["logreg_full_all95", "roc_auc"]),
            "dp4_holdout_auc": float(h.loc["dp_logreg_full_all95_eps4", "roc_auc"])}


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v


def _default_clip(c) -> bool:
    return c is None or (isinstance(c, float) and math.isnan(c)) or c == 1.0


def _distance(row, req) -> float:
    d = 0.0
    if row["eps"] != req["eps"]:
        a, b = EPS_ORDER.index(row["eps"]), EPS_ORDER.index(req["eps"])
        d += 100 + 10 * abs(a - b)
    if row["sites"] != req["sites"]:
        d += 60 + 10 * abs(SITES.index(row["sites"]) - SITES.index(req["sites"]))
    if req["model"] != "logreg" and row["encoding"] != req["encoding"]:
        d += 25
    if row["features"] != req["features"]:
        d += 8 * abs(row["features"] - req["features"]) / 2 if row["features"] <= 8 else 30
    if row["rows"] != "1.2k":
        d += 5
    if row["partition"] == "iid":
        d += 1
    if not _default_clip(row["clip"]):
        d += 2
    return d


def describe_differences(row, req) -> list[str]:
    out = []
    if row["eps"] != req["eps"]:
        out.append(f"privacy budget ε = {'∞' if row['eps'] == 'inf' else row['eps']} instead of "
                   f"{'∞' if req['eps'] == 'inf' else req['eps']}")
    if row["sites"] != req["sites"]:
        out.append(f"{row['sites']} site{'s' if row['sites'] > 1 else ''} instead of {req['sites']}")
    if req["model"] != "logreg" and row["encoding"] != req["encoding"]:
        out.append(f"{ENCODINGS[row['encoding']]} encoding instead of {ENCODINGS[req['encoding']]}")
    if row["features"] != req["features"]:
        out.append(f"{row['features']} features instead of {req['features']}")
    if row["rows"] != "1.2k":
        out.append("trained on the full ~11k rows, not the 1.2k-row matched subsample")
    if not _default_clip(row["clip"]):
        out.append(f"clipping norm {row['clip']:g} (post-hoc sensitivity run)")
    return out


def verdict(row, req, exact: bool, base: dict, cat: pd.DataFrame) -> str:
    gap = base["cv_auc"] - row["auc"]
    eps_txt = "∞" if row["eps"] == "inf" else row["eps"]
    if row["sites"] > 1 and row["model"] == "vqc":
        central = cat[(cat.model == "vqc") & (cat.eps == row["eps"]) & (cat.sites == 1)]
        if len(central):
            drop = float(central.auc.iloc[0] - row["auc"])
            if drop > 0.03:
                return (f"Too few rows per site (about {1200 // row['sites']}) for the noise at ε = {eps_txt}: "
                        f"{drop:.3f} below the same model trained centrally with DP.")
    if row["sites"] > 1 and row["model"] == "logreg":
        central = cat[(cat.model == "logreg") & (cat.features == 95) & (cat.eps == row["eps"]) & (cat.sites == 1)]
        if len(central) and abs(float(central.auc.iloc[0] - row["auc"])) < 0.01:
            return (f"Federation costs nothing measurable here: within 0.01 of centralised DP at ε = {eps_txt} "
                    f"(~{11155 // row['sites']:,} rows per site).")
    if row["eps"] not in ("inf", "8", "4") and row["auc_std"] > 0.05:
        return f"Unstable at ε = {eps_txt}: results vary by ± {row['auc_std']:.3f} across folds."
    if gap <= 0.005:
        return "Within 0.005 AUC of the unprivatised classical baseline."
    causes = []
    if row["features"] < 95:
        causes.append(f"{row['features']} of 95 features")
    if row["rows"] == "1.2k":
        causes.append("1.2k training rows")
    if row["eps"] != "inf":
        causes.append(f"DP noise at ε = {eps_txt}")
    if row["sites"] > 1:
        causes.append(f"{row['sites']} sites")
    size = "a modest cost" if gap <= 0.04 else "a substantial cost"
    return f"{gap:.3f} below the baseline, {size}; this configuration uses " + ", ".join(causes) + "."


def resolve(model: str, encoding: str, features: int, eps: str, sites: int) -> dict:
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}")
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {list(ENCODINGS)}")
    if features not in (4, 6, 8):
        raise ValueError("features must be 4, 6 or 8")
    eps = _eps(eps)
    if eps not in EPS_ORDER:
        raise ValueError(f"ε must be one of {EPS_ORDER}")
    if sites not in SITES:
        raise ValueError(f"sites must be one of {SITES}")
    req = {"model": model, "encoding": encoding, "features": features, "eps": eps, "sites": sites}
    cat, base = catalogue(), baseline()
    notice = None
    pool = cat[cat.model == model]
    if model == "qsvm" and (eps != "inf" or sites > 1):
        notice = ("No private or federated QSVM exists: the kernel SVM is not trained by gradient descent, "
                  "so DP-SGD and FedAvg do not apply to it. Showing the non-private, centralised QSVM instead.")
    best = min(pool.to_dict("records"), key=lambda r: _distance(r, req))
    differences = describe_differences(best, req)
    exact = not differences
    out = {k: _clean(v) for k, v in best.items() if k != "priority"}
    centralised_same = cat[(cat.model == best["model"]) & (cat.encoding == best["encoding"]) &
                           (cat.features == best["features"]) & (cat.rows == best["rows"]) & (cat.eps == "inf") &
                           (cat.sites == 1)]
    sensitivity = None
    if best["model"] == "logreg" and best["eps"] != "inf" and _default_clip(best["clip"]):
        alt = cat[(cat.model == "logreg") & (cat.features == best["features"]) & (cat.rows == best["rows"]) &
                  (cat.eps == best["eps"]) & (cat.sites == best["sites"]) & (cat["clip"] == 10.0)]
        if len(alt):
            sensitivity = (f"With clipping norm 10 (a post-hoc sensitivity run) the same configuration reaches "
                           f"{float(alt.auc.iloc[0]):.3f} ± {float(alt.auc_std.iloc[0]):.3f}; the pre-declared clipping "
                           "norm of 1 is shown above.")
    return {
        "requested": req,
        "sensitivity": sensitivity,
        "exact": exact,
        "notice": notice,
        "differences": differences,
        "measured": out,
        "baseline": base,
        "gap": base["cv_auc"] - float(best["auc"]),
        "nonprivate_auc": _clean(float(centralised_same.auc.iloc[0])) if len(centralised_same) else None,
        "verdict": verdict(best, req, exact, base, cat),
        "encoding_applies": model != "logreg",
    }
