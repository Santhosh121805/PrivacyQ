"""Secure aggregation by pairwise additive masking (the masking core of Bonawitz et al., CCS 2017).

What it provides: the server receives, from each client, only a masked vector. The masks come from a
cryptographic PRG (SHAKE-256) keyed by Diffie-Hellman secrets the server does not know, so each masked
vector is computationally indistinguishable from random to the server (checked statistically by the
tests, which cannot prove it). Updates are encoded as fixed-point integers and masked and summed
modulo 2^64, so the masks cancel exactly (bit for bit) in the sum: the server learns the weighted
sum of client updates (and the total row count) and, under the cryptographic assumptions, nothing
else about any one client, as long as it does not collude with all other clients. A coalition of the server and some clients learns
only the sum of the remaining honest clients' updates (tested in ``tests/test_secure_aggregation.py``).

Protocol, per run and per round:
1. Every client draws a Diffie-Hellman key pair in the 2048-bit MODP group of RFC 3526 (group 14,
   generator 2) and publishes its public key through the server.
2. Each pair (i, j) derives a shared secret g^(a_i a_j) mod p; a per-round seed is
   SHA-256(secret || i || j || round), expanded to a mask vector with SHAKE-256.
3. Client i encodes [n_i * w_i, n_i] in fixed point (``SCALE_BITS`` fractional bits, two's
   complement mod 2^64) and sends y_i = enc_i + sum_{j>i} m_ij - sum_{j<i} m_ji  (mod 2^64).
4. The server adds all y_i mod 2^64; the masks cancel, leaving sum_i enc_i, which decodes to
   sum_i n_i w_i and sum_i n_i; their ratio is the FedAvg model.

Not implemented (stated, not hidden):
- client dropout recovery (Bonawitz et al. secret-share the seeds with Shamir; here every client
  must report every round, or the sum cannot be unmasked);
- authentication of public keys (a malicious server could substitute keys: honest-but-curious
  server only);
- protection against malicious clients sending malformed updates.
Fixed-point encoding changes the averaged weights by at most 2^-SCALE_BITS per coordinate.

Secure aggregation hides individual updates from the server; it is not differential privacy. The
formal privacy guarantee of this project still comes from DP-SGD. Quantum circuits play no part.
"""

from __future__ import annotations

import hashlib
import secrets

import numpy as np

# RFC 3526, 2048-bit MODP group (group 14). tests/test_secure_aggregation.py checks it against the
# RFC's closed form 2^2048 - 2^1984 - 1 + 2^64 * (floor(2^1918 pi) + 124476).
MODP_2048_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
MODP_2048_G = 2
SCALE_BITS = 24
_LIMIT = 2.0 ** (63 - SCALE_BITS)        # |encoded sum| must stay below 2^63


def encode(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if np.any(np.abs(v) >= _LIMIT):
        raise OverflowError("value too large for the fixed-point range")
    return np.round(v * 2.0**SCALE_BITS).astype(np.int64).view(np.uint64)


def decode(field: np.ndarray) -> np.ndarray:
    return np.asarray(field, dtype=np.uint64).view(np.int64).astype(float) / 2.0**SCALE_BITS


class SecAggClient:
    def __init__(self, client_id: int, rng: secrets.SystemRandom | None = None):
        self.id = int(client_id)
        self._sk = (rng or secrets.SystemRandom()).randrange(2, MODP_2048_P - 2)
        self.public_key = pow(MODP_2048_G, self._sk, MODP_2048_P)
        self._secrets: dict[int, bytes] = {}

    def agree(self, public_keys: dict[int, int]) -> None:
        """Derive the shared secret with every other client from the published public keys."""
        for j, pk in public_keys.items():
            if j != self.id:
                if not 1 < pk < MODP_2048_P - 1:
                    raise ValueError("invalid public key")
                shared = pow(pk, self._sk, MODP_2048_P)
                self._secrets[j] = shared.to_bytes(256, "big")

    def _mask(self, j: int, round_: int, n: int) -> np.ndarray:
        lo, hi = min(self.id, j), max(self.id, j)
        seed = hashlib.sha256(self._secrets[j] + lo.to_bytes(4, "big") + hi.to_bytes(4, "big")
                              + int(round_).to_bytes(4, "big")).digest()
        return np.frombuffer(hashlib.shake_256(seed).digest(8 * n), dtype=np.uint64).copy()

    def masked(self, values: np.ndarray, round_: int) -> np.ndarray:
        """Masked fixed-point encoding of ``values`` (what the server receives)."""
        y = encode(values)
        for j in self._secrets:
            m = self._mask(j, round_, y.size)
            y = y + m if self.id < j else y - m          # uint64 arithmetic wraps mod 2^64
        return y


def unmask_sum(masked_updates: list[np.ndarray]) -> np.ndarray:
    """Server side: sum of all masked vectors mod 2^64, decoded. Only the sum is recoverable."""
    total = np.zeros_like(masked_updates[0])
    for y in masked_updates:
        total = total + y
    return decode(total)


def setup(n_clients: int, rng: secrets.SystemRandom | None = None) -> list[SecAggClient]:
    """Key generation and pairwise agreement for ``n_clients`` (public keys relayed by the server)."""
    clients = [SecAggClient(i, rng) for i in range(n_clients)]
    keys = {c.id: c.public_key for c in clients}
    for c in clients:
        c.agree(keys)
    return clients


def secure_weighted_average(clients: list[SecAggClient], vectors: list[np.ndarray], sizes: list[int],
                            round_: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """FedAvg through secure aggregation: each client masks [n_i * w_i, n_i]; the server unmasks
    only the sums. Returns (weighted average, the masked vectors the server saw)."""
    seen = [c.masked(np.append(float(n) * np.asarray(v, dtype=float), float(n)), round_)
            for c, v, n in zip(clients, vectors, sizes)]
    total = unmask_sum(seen)
    return total[:-1] / total[-1], seen


def secure_aggregation_status(used: bool) -> dict[str, object]:
    """Run-record entry: whether this run aggregated through secure aggregation."""
    return {"status": "used" if used else "not used in this run",
            "implementation": "src/m7_privacy/secure_aggregation.py (pairwise DH masks; no dropout recovery)"}
