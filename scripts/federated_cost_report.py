"""Communication and server-load measurements for federated training (box 13).

Reads the round histories of an instrumented federated run (src/m6_federated/fedavg.py records,
per round: measured serialised update size, client local-training wall-clock time, server
aggregation wall-clock time) and reports per arm and number of clients:

- payload: measured serialised bytes per client update, and per round (down + up, all clients),
- client local training seconds per round (slowest client; the round cannot finish earlier),
- server aggregation seconds per round (deserialise all updates + weighted average),
- rounds to converge: first round whose validation ROC-AUC is within 0.01 of that run's best,
  and within 0.01 of its final round (both from the recorded per-round AUC).

Network transfer time is NOT reported: training runs in one process with no network, and
converting bytes to seconds with an assumed bandwidth would be a simulation, not a measurement.

Usage: python scripts/federated_cost_report.py results/<instrumented exp4-style run>
"""

import sys
from pathlib import Path

import pandas as pd


def rounds_to(h: pd.DataFrame, target: float, tol: float = 0.01) -> int:
    hit = h[h.val_auc >= target - tol]
    return int(hit["round"].min())


def main(run_dir: str) -> None:
    d = Path(run_dir)
    hist = pd.read_csv(d / "round_history.csv")
    rows = []
    for (arm, n, part), g in hist.groupby(["arm", "n_clients", "partition"]):
        per_run = []
        for _, h in g.groupby(["repeat", "fold"]):
            h = h.sort_values("round")
            per_run.append({
                "rounds_to_best_minus_0.01": rounds_to(h, h.val_auc.max()),
                "rounds_to_final_minus_0.01": rounds_to(h, h.val_auc.iloc[-1]),
            })
        pr = pd.DataFrame(per_run)
        rows.append({
            "arm": arm, "clients": n, "partition": part, "runs": len(pr),
            "payload_bytes_per_update": int(g.payload_bytes_per_update.median()),
            "payload_kB_per_round": g.payload_bytes_round.median() / 1e3,
            "client_seconds_per_round_slowest": g.client_seconds_max.mean(),
            "client_seconds_per_round_all": g.client_seconds_total.mean(),
            "aggregation_ms_per_round": 1000 * g.aggregation_seconds.mean(),
            "aggregation_ms_p95": 1000 * g.aggregation_seconds.quantile(0.95),
            "rounds_to_within_0.01_of_best": f"{pr['rounds_to_best_minus_0.01'].median():.0f} (range {pr['rounds_to_best_minus_0.01'].min()}–{pr['rounds_to_best_minus_0.01'].max()})",
            "rounds_to_within_0.01_of_final": f"{pr['rounds_to_final_minus_0.01'].median():.0f} (range {pr['rounds_to_final_minus_0.01'].min()}–{pr['rounds_to_final_minus_0.01'].max()})",
        })
    out = pd.DataFrame(rows)
    out.to_csv(d / "communication_cost.csv", index=False)
    lines = ["# Communication cost and server load (measured)", "",
             f"Run: `{d}`. In-process simulation: no network, so network transfer time is not measured. "
             "Timings are wall-clock on the development laptop.", "",
             "| Arm | Clients | Payload / update (B) | Payload / round (kB) | Slowest client (s/round) | Server aggregation (ms/round) | Rounds to within 0.01 of best | Rounds to within 0.01 of final |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {r.arm} | {r.clients} | {r.payload_bytes_per_update:,} | {r.payload_kB_per_round:.1f} | "
                     f"{r.client_seconds_per_round_slowest:.2f} | {r.aggregation_ms_per_round:.2f} (p95 {r.aggregation_ms_p95:.2f}) | "
                     f"{r['rounds_to_within_0.01_of_best']} | {r['rounds_to_within_0.01_of_final']} |")
    (d / "communication_cost.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
