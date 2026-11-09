"""Data for the "three schools" story demo, from the real Experiment 4 run.

1. Replay: the federated run of Experiment 4 with logistic regression, 3 clients split by
   U.S. state (non-IID), client-side DP-SGD at epsilon = 4, outer fold r0f0. Per-round
   validation ROC-AUC comes from that run's round_history.csv. Per-round epsilon is
   recomputed with Opacus's RDP accountant from the run's own noise multiplier, sampling
   rate and step count (only the final epsilon was logged), and checked against the log.
2. "Trained alone": each of the three clients trains logistic regression on ONLY its own
   rows (its own imputation, scaling, DP-SGD at the same epsilon = 4, and without DP),
   scored on the same outer validation fold. Repeated for all 10 Experiment 4 folds.

Usage: python scripts/build_story_data.py   -> results/story/story.json
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from opacus.accountants.utils import get_noise_multiplier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.m2_preprocess.preprocessing import TabularPreprocessor  # noqa: E402
from src.m6_federated.fedavg import group_partition  # noqa: E402
from src.m7_privacy.accounting import epsilon_for  # noqa: E402
from src.m7_privacy.dp_training import DPClassifier  # noqa: E402
from src.m10_exp.common import cv_folds, holdout_split, limit_worker_threads, load_config, load_dataset  # noqa: E402
from src.m10_exp.tuning import inner_split  # noqa: E402

N_CLIENTS, PARTITION, EPS, DELTA = 3, "by_state", 4.0, 1e-5


def alone_task(frame_rows, labels, metadata, label_col, val_frame, y_val, arm, seed, school, rep, fold, n_total):
    import torch
    torch.set_num_threads(1)
    prep = TabularPreprocessor(identifier_columns=metadata)          # the school's own preprocessing
    X = prep.fit_transform(frame_rows, label_col).to_numpy()
    scaler = StandardScaler().fit(X)
    Xs, Xv = scaler.transform(X), scaler.transform(prep.transform(val_frame, label_col).to_numpy())
    batch = max(1, int(round(arm["batch_size"] / n_total * len(labels))))   # same sampling rate as in FedAvg
    out = []
    for eps in (EPS, None):
        m = DPClassifier(kind="logreg", epsilon=eps, delta=DELTA, max_grad_norm=arm.get("max_grad_norm", 1.0),
                         epochs=30, batch_size=batch, learning_rate=arm["learning_rate"], seed=seed).fit(Xs, labels)
        out.append({"repeat": rep, "fold": fold, "school": school, "epsilon": "4" if eps else "inf",
                    "n_rows": int(len(labels)), "n_pos": int(labels.sum()),
                    "roc_auc": float(roc_auc_score(y_val, m.decision_scores(Xv))),
                    "epsilon_spent": m.privacy_.epsilon})
    return out


def main() -> None:
    t0 = time.time()
    limit_worker_threads()
    final = load_config(ROOT / "configs/final_report.yaml")
    exp4_dir = ROOT / final["runs"]["exp4"]
    cfg = load_config(exp4_dir / "config.yaml")
    arm = cfg["experiment"]["arms"]["logreg_full"]
    seed0 = int(cfg["project"]["seed"])
    loader, frame = load_dataset(cfg)
    labels = frame[loader.label_column].to_numpy(dtype=int)
    groups = frame[loader.group_column].to_numpy()
    dev, holdout = holdout_split(labels, float(cfg["split"]["holdout_fraction"]), seed0)
    folds = cv_folds(dev, labels, int(cfg["cv"]["n_splits"]), int(cfg["cv"]["n_repeats"]), seed0)

    fm = pd.read_csv(exp4_dir / "fold_metrics.csv")
    fed = fm[(fm.arm == "logreg_full") & (fm.n_clients == N_CLIENTS) & (fm.partition == PARTITION)]
    hist = pd.read_csv(exp4_dir / "round_history.csv")
    hist = hist[(hist.arm == "logreg_full") & (hist.n_clients == N_CLIENTS) & (hist.partition == PARTITION)]
    exp3 = pd.read_csv(ROOT / final["runs"]["exp3"] / "fold_metrics.csv")
    central_dp = exp3[(exp3.arm == "logreg_full") & (exp3.target_epsilon.astype(str).isin(["4", "4.0"]))]

    tasks, replay = [], None
    for f in folds:
        seed = seed0 + 1000 * f["repeat"] + f["fold"]
        tr, _ = inner_split(f["train"], labels, float(cfg["tuning"]["inner_val_fraction"]), seed)
        parts = group_partition(groups[tr], N_CLIENTS, seed)                 # identical to Experiment 4
        val_frame, y_val = frame.iloc[f["val"]], labels[f["val"]]
        for school, p in enumerate(parts):
            rows = tr[p]
            tasks.append(delayed(alone_task)(frame.iloc[rows], labels[rows], loader.metadata_columns,
                                             loader.label_column, val_frame, y_val, arm, seed, school,
                                             f["repeat"], f["fold"], len(tr)))
        if f["repeat"] == 0 and f["fold"] == 0:
            sizes = [len(p) for p in parts]
            states = [sorted({int(s) for s in groups[tr][p]}) for p in parts]
            sample_rate = arm["batch_size"] / len(tr)
            per_client = []
            for n in sizes:
                batch = max(1, int(round(sample_rate * n)))
                steps_per_round = int(np.ceil(n / batch))
                q = 1.0 / steps_per_round
                sigma = float(get_noise_multiplier(target_epsilon=EPS, target_delta=DELTA, sample_rate=q,
                                                   steps=30 * steps_per_round, accountant="rdp"))
                per_client.append({"rows": n, "batch": batch, "steps_per_round": steps_per_round, "sigma": sigma,
                                   "eps_by_round": [epsilon_for(sigma, q, r * steps_per_round, DELTA) for r in range(1, 31)]})
            run = fed[(fed["repeat"] == 0) & (fed["fold"] == 0)].iloc[0]
            h = hist[(hist["repeat"] == 0) & (hist["fold"] == 0)].sort_values("round")
            replay = {
                "fold": "r0f0", "n_val": int(len(f["val"])), "n_val_pos": int(y_val.sum()),
                "schools": [{"rows": s, "positives": int(labels[tr][p].sum()), "n_states": len(st)}
                            for s, p, st in zip(sizes, parts, states)],
                "rounds": [{"round": int(r["round"]), "val_auc": float(r["val_auc"]), "bytes_round": int(r["bytes"]),
                            "eps_spent_max": max(c["eps_by_round"][int(r["round"]) - 1] for c in per_client)}
                           for _, r in h.iterrows()],
                "n_params": int(run.n_params), "bytes_per_update": int(run.n_params) * 8,
                "sigma_per_school": [c["sigma"] for c in per_client],
                "logged_noise_multiplier_mean": float(run.noise_multiplier_mean),
                "logged_epsilon_spent": float(run.epsilon_spent),
                "final_auc": float(run.roc_auc), "final_balanced_accuracy": float(run.balanced_accuracy),
                "final_recall": float(run.recall),
            }

    alone = pd.DataFrame([r for rows in Parallel(n_jobs=6)(tasks) for r in rows])
    alone.to_csv(ROOT / "results" / "story_alone_runs.csv", index=False)
    a4 = alone[alone.epsilon == "4"]
    per_fold_alone = a4.groupby(["repeat", "fold"]).roc_auc.mean()
    fed_by_fold = fed.set_index(["repeat", "fold"]).roc_auc
    idx = per_fold_alone.index.intersection(fed_by_fold.index)
    from src.m8_eval.stats import wilcoxon_p
    replay_alone = a4[(a4["repeat"] == 0) & (a4["fold"] == 0)].sort_values("school")
    replay_alone_np = alone[(alone.epsilon == "inf") & (alone["repeat"] == 0) & (alone["fold"] == 0)].sort_values("school")
    summary = {
        "replay": replay,
        "replay_alone_auc_eps4": replay_alone.roc_auc.tolist(),
        "replay_alone_auc_nonprivate": replay_alone_np.roc_auc.tolist(),
        "replay_alone_eps_spent": replay_alone.epsilon_spent.tolist(),
        "ten_folds": {
            "alone_eps4_mean": float(a4.roc_auc.mean()), "alone_eps4_std": float(a4.roc_auc.std()),
            "alone_eps4_min": float(a4.roc_auc.min()), "alone_eps4_max": float(a4.roc_auc.max()),
            "alone_nonprivate_mean": float(alone[alone.epsilon == "inf"].roc_auc.mean()),
            "federated_eps4_mean": float(fed.roc_auc.mean()), "federated_eps4_std": float(fed.roc_auc.std()),
            "centralised_eps4_mean": float(central_dp.roc_auc.mean()),
            "federated_minus_alone_mean": float((fed_by_fold[idx] - per_fold_alone[idx]).mean()),
            "federated_beats_alone_folds": int((fed_by_fold[idx] > per_fold_alone[idx]).sum()), "k": int(len(idx)),
            "wilcoxon_p": wilcoxon_p(fed_by_fold[idx].to_numpy(), per_fold_alone[idx].to_numpy()),
        },
        "quantum_vs_classical": None,
        "holdout_size": int(len(holdout)),
        "source": {"exp4_run": str(final["runs"]["exp4"]), "data": loader.source_tag()},
    }
    t4 = pd.read_csv(ROOT / final["runs"]["exp2"] / "tests_same_rows_features.csv")
    t = t4[(t4.metric == "roc_auc") & (t4.quantum == "qsvm/subsample/pca8") & (t4.classical == "logreg/subsample/pca8")].iloc[0]
    summary["quantum_vs_classical"] = {"quantum_auc": float(t.quantum_mean), "classical_auc": float(t.classical_mean),
                                       "holm_p": float(t.wilcoxon_p_holm), "k": int(t.k)}
    out = ROOT / "results" / "story"
    out.mkdir(parents=True, exist_ok=True)
    (out / "story.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("replay_alone_auc_eps4", "replay_alone_auc_nonprivate", "ten_folds")}, indent=2))
    r = summary["replay"]
    print("replay: schools", r["schools"], "| final AUC", round(r["final_auc"], 4),
          "| logged eps", round(r["logged_epsilon_spent"], 4), "vs recomputed", round(r["rounds"][-1]["eps_spent_max"], 4),
          "| logged sigma mean", round(r["logged_noise_multiplier_mean"], 4), "vs recomputed", [round(s, 4) for s in r["sigma_per_school"]])
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
