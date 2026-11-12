"""Head-to-head server and client cost of plain vs secure aggregation, interleaved in one process.

Uses the real parameter shapes of the Experiment 7 models (LogReg on 95 features; VQC re-upload 4 qubits,
2 layers) and the same server steps as src/m6_federated/fedavg.py: per-update serialise/deserialise
round-trip, then weighted average (plain) or sum mod 2^64 + decode (secure). Plain and secure repetitions
alternate, so both see the same machine state. Writes results/secagg_overhead/overhead.md.

Usage: python scripts/secagg_overhead.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.m6_federated.fedavg import _deserialise, _serialise  # noqa: E402
from src.m7_privacy.dp_training import build_module  # noqa: E402
from src.m7_privacy.secure_aggregation import setup, unmask_sum  # noqa: E402

MODELS = {"logreg_full": ("logreg", 95, {}),
          "vqc_sub": ("vqc", 4, {"encoding": "reupload", "n_qubits": 4, "n_layers": 2})}
REPS = 200


def main():
    rows = []
    for arm, (kind, d, params) in MODELS.items():
        state = build_module(kind, d, params).state_dict()
        keys = list(state)
        for n in (3, 5, 10):
            rng = np.random.default_rng(n)
            updates = [{k: torch.as_tensor(rng.normal(size=v.shape), dtype=v.dtype) for k, v in state.items()}
                       for _ in range(n)]
            sizes = rng.integers(500, 3000, n).astype(float)
            t0 = time.perf_counter(); clients = setup(n); setup_s = time.perf_counter() - t0
            plain_t, secure_t, mask_t = [], [], []
            for r in range(REPS):
                t0 = time.perf_counter()
                rec = [_deserialise(_serialise(u)) for u in updates]
                {k: sum(u[k] * (s / sizes.sum()) for u, s in zip(rec, sizes)) for k in keys}
                plain_t.append(time.perf_counter() - t0)
                flat = [torch.cat([u[k].reshape(-1) for k in keys]).numpy() for u in updates]
                t0 = time.perf_counter()
                seen = [c.masked(np.append(s * v, s), r) for c, v, s in zip(clients, flat, sizes)]
                mask_t.append((time.perf_counter() - t0) / n)
                blobs = [{"y": torch.from_numpy(y.view(np.int64))} for y in seen]
                t0 = time.perf_counter()
                tot = unmask_sum([_deserialise(_serialise(b))["y"].numpy().view(np.uint64) for b in blobs])
                avg, pos = tot[:-1] / tot[-1], 0
                for k in keys:
                    m = state[k].numel()
                    torch.as_tensor(avg[pos:pos + m]).reshape(state[k].shape)
                    pos += m
                secure_t.append(time.perf_counter() - t0)
            rows.append({"arm": arm, "clients": n, "n_params": sum(v.numel() for v in state.values()),
                         "plain_server_ms": 1000 * np.median(plain_t), "secure_server_ms": 1000 * np.median(secure_t),
                         "mask_ms_per_client": 1000 * np.median(mask_t), "setup_s_once": setup_s,
                         "plain_payload_B": len(_serialise(updates[0])), "secure_payload_B": len(_serialise(blobs[0]))})
    out = pd.DataFrame(rows)
    d = ROOT / "results/secagg_overhead"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "overhead.csv", index=False)
    lines = ["# Secure aggregation overhead (measured, interleaved with plain aggregation)", "",
             f"Median of {REPS} interleaved repetitions per row, one process, development laptop. "
             "Setup = Diffie-Hellman key generation + all pairwise agreements, once per training run.", "",
             "| Model | Clients | Params | Server plain (ms/round) | Server secure (ms/round) | Client masking (ms/round) | Setup once (s) | Payload plain (B) | Payload secure (B) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {r.arm} | {r.clients} | {r.n_params} | {r.plain_server_ms:.2f} | {r.secure_server_ms:.2f} | "
                     f"{r.mask_ms_per_client:.2f} | {r.setup_s_once:.2f} | {r.plain_payload_B:,} | {r.secure_payload_B:,} |")
    (d / "overhead.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
