"""Live federated training in the terminal — the computation behind the Overview "How it works" demo.

Runs, for real, the run that demo replays: Experiment 4, fold r0f0, logistic regression, 3 clients
("schools") split by U.S. state, client-side DP-SGD at target ε = 4, 30 FedAvg rounds. It uses the same
code path as Experiment 4 (src/m10_exp/exp4_federated.py → src/m6_federated/fedavg.py) and prints what
happens each round. At the end it compares its numbers with the recorded run (results/story/story.json).

Usage:  python scripts/federated_live.py            (about 0.6 s pause per round, for presenting)
        python scripts/federated_live.py --pause 0
"""

import argparse
import warnings
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

SCHOOLS = ["Northside High", "Riverside Academy", "Eastgate School"]
B, D, G, Y, R = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"


def main(pause: float) -> None:
    warnings.filterwarnings("ignore")                       # library deprecation notices, not part of the run
    if sys.platform == "win32":
        import os
        os.system("")                                       # enable ANSI colours in the Windows console
    print(f"{B}Loading NSCH 2024 and building the Experiment 4 task (fold r0f0) ...{R}", flush=True)
    import torch
    from sklearn.metrics import roc_auc_score

    from src.m5_models.quantum_dp import prepare_inputs
    from src.m6_federated.fedavg import fedavg_train, group_partition
    from src.m10_exp.common import cv_folds, holdout_split, limit_worker_threads, load_config, load_dataset
    from src.m10_exp.exp2_compare import feature_matrices
    from src.m10_exp.exp4_federated import build_tasks

    limit_worker_threads()
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    exp4_dir = ROOT / load_config(ROOT / "configs/final_report.yaml")["runs"]["exp4"]
    cfg = load_config(exp4_dir / "config.yaml")
    seed0 = int(cfg["project"]["seed"])
    loader, frame = load_dataset(cfg)
    labels = frame[loader.label_column].to_numpy(dtype=int)
    dev, _ = holdout_split(labels, float(cfg["split"]["holdout_fraction"]), seed0)
    folds = [f for f in cv_folds(dev, labels, int(cfg["cv"]["n_splits"]), int(cfg["cv"]["n_repeats"]), seed0)
             if f["repeat"] == 0 and f["fold"] == 0]
    task = next(t for t in build_tasks(cfg, loader, frame, folds, labels)
                if t["arm_name"] == "logreg_full" and t["n_clients"] == 3 and t["partition"] == "by_state"
                and str(t["epsilon"]) in ("4", "4.0"))
    arm, exp = task["arm"], task["exp"]
    A, _, C = feature_matrices(task["feature_spec"], task["X_in"], task["y_in"], [task["X_iv"], task["X_ov"]], task["seed"])
    params = arm.get("model_params", {})
    prepare = lambda X: prepare_inputs(arm["kind"], params.get("encoding"), params.get("n_qubits"), X)  # noqa: E731
    y = task["y_in"]
    parts = group_partition(task["groups_in"], task["n_clients"], task["seed"])
    delta, clip = float(exp["delta"]), arm.get("max_grad_norm", 1.0)

    print(f"\n{B}Setup{R}  {len(A):,} training rows, {A.shape[1]} features; validation {len(C):,} rows "
          f"(never used for training); DP-SGD target ε = {task['epsilon']}, δ = {delta}, clipping norm {clip}; "
          f"{exp['rounds']} rounds × {exp['local_epochs']} local epoch")
    for i, p in enumerate(parts):
        states = len(set(task["groups_in"][p].tolist()))
        print(f"  {SCHOOLS[i]:<18} {len(p):>6,} student records ({int(y[p].sum())} with depression), "
              f"{states} U.S. states  — {B}records stay at the school{R}")
    print()

    def evaluate(module):
        module.eval()
        with torch.no_grad():
            s = module(torch.as_tensor(prepare(C), dtype=torch.float64)).reshape(-1).numpy()
        return {"val_auc": float(roc_auc_score(task["y_ov"], s))}

    live = []

    def on_round(entry, workers, sizes):
        r = entry["round"]
        eps = [w.epsilon(delta) for w in workers]
        weights = sizes / sizes.sum()
        if r == 1:
            for i, w in enumerate(workers):
                print(f"  {D}{SCHOOLS[i]}: noise multiplier σ = {w.noise_multiplier:.4f} "
                      f"(calibrated by Opacus for all its steps over {exp['rounds']} rounds){R}")
            print()
        print(f"{B}Round {r:>2}{R}")
        print(f"  local training   each school: DP-SGD on its own rows — per-example gradients clipped to norm {clip}, "
              f"Gaussian noise σ·{clip} added ({entry['client_seconds_max']:.2f} s slowest school)")
        print(f"  sent to server   3 model updates × {entry['payload_bytes_per_update']:,} B  "
              f"= {entry['payload_bytes_per_update'] * 3:,} B   {G}{B}student records sent: 0{R}")
        print(f"  server           weighted average  w = " + " · ".join(f"{x:.3f}" for x in weights)
              + f"  (each school's share of rows)   {entry['aggregation_seconds'] * 1000:.1f} ms")
        print(f"  privacy          ε spent so far (RDP accountant): " + " · ".join(f"{e:.2f}" for e in eps)
              + f"   max {Y}{max(eps):.2f}{R} of {task['epsilon']}")
        print(f"  shared model     validation ROC-AUC {B}{entry['val_auc']:.4f}{R}\n", flush=True)
        live.append({"round": r, "val_auc": entry["val_auc"], "eps": max(eps)})
        if pause:
            time.sleep(pause)

    t0 = time.time()
    out = fedavg_train(arm["kind"], params, [(A[p], y[p]) for p in parts], rounds=exp["rounds"],
                       local_epochs=exp["local_epochs"], learning_rate=arm["learning_rate"],
                       sample_rate=arm["batch_size"] / len(y), epsilon=task["epsilon"], delta=delta,
                       max_grad_norm=clip, seed=task["seed"], prepare=prepare, evaluate=evaluate, on_round=on_round)
    print(f"{B}Done{R} in {time.time() - t0:.1f} s: final validation ROC-AUC {out['history'].val_auc.iloc[-1]:.4f}, "
          f"ε spent {out['epsilon_spent_max']:.3f} (target {task['epsilon']}), {G}{B}student records sent: 0{R}")

    story = ROOT / "results" / "story" / "story.json"
    if story.exists():
        rec = json.loads(story.read_text(encoding="utf-8"))["replay"]["rounds"]
        d_auc = max(abs(a["val_auc"] - b["val_auc"]) for a, b in zip(live, rec))
        d_eps = max(abs(a["eps"] - b["eps_spent_max"]) for a, b in zip(live, rec))
        same = d_auc < 1e-9 and d_eps < 1e-6
        print(f"Compared with the recorded run shown on the web page: max |ΔAUC| {d_auc:.2e}, max |Δε| {d_eps:.2e} "
              f"→ {'identical' if same else 'DIFFERENT'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pause", type=float, default=0.6, help="seconds to pause after each round")
    main(ap.parse_args().pause)
