import random

import mpmath
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.m6_federated.fedavg import fedavg_train, stratified_iid_partition
from src.m7_privacy.secure_aggregation import (
    MODP_2048_G, MODP_2048_P, SCALE_BITS, decode, encode, secure_weighted_average, setup, unmask_sum,
)


def _vectors(seed=0, n_clients=4, dim=50):
    rng = np.random.default_rng(seed)
    return [rng.normal(scale=2.0, size=dim) for _ in range(n_clients)], [int(s) for s in rng.integers(100, 2000, n_clients)]


def test_group_is_rfc3526_modp2048():
    mpmath.mp.prec = 4000
    closed = 2**2048 - 2**1984 - 1 + 2**64 * (int(mpmath.floor(mpmath.mpf(2) ** 1918 * mpmath.pi)) + 124476)
    assert MODP_2048_P == closed and MODP_2048_G == 2
    rng = random.Random(0)
    for _ in range(8):                                     # Fermat test on p and on q = (p - 1) / 2 (safe prime)
        a = rng.randrange(2, MODP_2048_P - 2)
        assert pow(a, MODP_2048_P - 1, MODP_2048_P) == 1
        assert pow(a, (MODP_2048_P - 1) // 2 - 1, (MODP_2048_P - 1) // 2) == 1


def test_masks_cancel_and_sum_equals_plain_weighted_average():
    vecs, sizes = _vectors()
    clients = setup(len(vecs))
    avg, _ = secure_weighted_average(clients, vecs, sizes, round_=1)
    plain = sum(n * v for v, n in zip(vecs, sizes)) / sum(sizes)
    assert np.max(np.abs(avg - plain)) < 2.0 ** -(SCALE_BITS - 1)
    assert np.allclose(decode(encode(np.array([-3.5, 0.0, 7.25]))), [-3.5, 0.0, 7.25])


def test_server_view_of_one_client_reveals_nothing_about_its_update():
    vecs, sizes = _vectors(dim=2000)
    clients = setup(len(vecs))
    _, seen = secure_weighted_average(clients, vecs, sizes, round_=1)
    y0 = seen[0]
    truth = encode(np.append(sizes[0] * vecs[0], sizes[0]))
    # A naive decode of what the server holds is nowhere near the client's true contribution.
    assert np.median(np.abs(decode(y0) - decode(truth))) > 1e9
    # Statistical check (not a proof): every bit of the masked words is balanced, as for random words.
    bits = np.unpackbits(y0.view(np.uint8)).reshape(-1, 64).mean(axis=0)
    assert np.all(np.abs(bits - 0.5) < 0.06)
    # And uncorrelated with the true update.
    assert abs(np.corrcoef(decode(y0)[:-1], vecs[0])[0, 1]) < 0.1


def test_identical_server_view_is_consistent_with_any_update():
    # For client 0's masked vector y0 and ANY alternative update x', there is a mask (a different
    # pairwise secret) giving the same y0: y0 alone does not pin down the update (the mask is unknown).
    vecs, sizes = _vectors(n_clients=2, dim=10)
    clients = setup(2)
    _, seen = secure_weighted_average(clients, vecs, sizes, round_=3)
    alternative = encode(np.append(sizes[0] * np.full(10, 123.0), sizes[0]))
    implied_mask = seen[0] - alternative                    # uint64 wraps mod 2^64
    assert np.array_equal(alternative + implied_mask, seen[0])


def test_colluding_server_and_clients_learn_only_the_honest_clients_sum():
    vecs, sizes = _vectors(n_clients=5, dim=500)
    clients = setup(5)
    _, seen = secure_weighted_average(clients, vecs, sizes, round_=2)
    honest, colluders = [0, 1], [2, 3, 4]
    # Colluders remove every mask they know (all masks involving them) from the honest clients' vectors.
    stripped = {}
    for h in honest:
        y = seen[h].copy()
        for c in colluders:
            m = clients[h]._mask(c, 2, y.size)
            y = y - m if h < c else y + m
        stripped[h] = y
    # Honest 0 is still hidden behind the 0-1 mask ...
    assert np.median(np.abs(decode(stripped[0])[:-1] - sizes[0] * vecs[0])) > 1e9
    # ... but the two honest clients' sum is exposed: that is the guarantee's limit, as documented.
    both = decode(stripped[0] + stripped[1])
    assert np.allclose(both[:-1], sizes[0] * vecs[0] + sizes[1] * vecs[1], atol=1e-5)


def test_fresh_masks_every_round():
    vecs, sizes = _vectors(n_clients=3, dim=20)
    clients = setup(3)
    _, r1 = secure_weighted_average(clients, vecs, sizes, round_=1)
    _, r2 = secure_weighted_average(clients, vecs, sizes, round_=2)
    assert not np.array_equal(r1[0], r2[0])                 # same update, different masked vector
    assert np.allclose(unmask_sum(r1), unmask_sum(r2))


@pytest.mark.parametrize("epsilon", [None, 2.0])
def test_fedavg_with_secure_aggregation_matches_plain_fedavg(epsilon):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] > 0.5).astype(int)
    parts = stratified_iid_partition(y, 3, 0)

    def evaluate(module):
        import torch
        with torch.no_grad():
            return {"auc": roc_auc_score(y, module(torch.as_tensor(X)).reshape(-1).numpy())}

    kw = dict(rounds=3, local_epochs=1, learning_rate=0.05, sample_rate=0.1, epsilon=epsilon, delta=1e-5,
              max_grad_norm=1.0, seed=0, evaluate=evaluate)
    plain = fedavg_train("logreg", {}, [(X[p], y[p]) for p in parts], **kw)
    secure = fedavg_train("logreg", {}, [(X[p], y[p]) for p in parts], secure_aggregation=True, **kw)
    assert secure["secure_aggregation"] and secure["secagg_setup_seconds"] > 0
    assert np.allclose(plain["history"].auc, secure["history"].auc, atol=1e-4)
    for a, b in zip(plain["module"].state_dict().values(), secure["module"].state_dict().values()):
        assert np.max(np.abs(a.numpy() - b.numpy())) < 1e-5
    assert (secure["history"].masking_seconds_max > 0).all()


def test_masks_cancel_exactly_in_the_integer_domain():
    # Fixed point mod 2^64: the sum of the masked words equals the sum of the unmasked encodings bit for bit.
    vecs, sizes = _vectors(n_clients=6, dim=300)
    clients = setup(6)
    plain = [encode(np.append(n * v, n)) for v, n in zip(vecs, sizes)]
    masked = [c.masked(np.append(n * v, n), round_=7) for c, v, n in zip(clients, vecs, sizes)]
    assert masked[0].dtype == np.uint64
    assert not any(np.array_equal(m, p) for m, p in zip(masked, plain))
    total_masked, total_plain = np.zeros(301, np.uint64), np.zeros(301, np.uint64)
    for m, p in zip(masked, plain):
        total_masked += m
        total_plain += p
    assert np.array_equal(total_masked, total_plain)


# Measured in Experiment 7c (results/exp7c-nsch2024-secagg-bound-measurement-20261112-164738, a deterministic
# re-run of Experiment 7 that reproduced its AUC exactly): largest number of clients 10, largest client size
# 3,786 rows, largest |weight| in any client update 8.936 (largest |n_i * w| sent 33,435; largest update L2 10.24).
EXP7_MAX_CLIENTS, EXP7_MAX_CLIENT_SIZE, EXP7_MAX_ABS_WEIGHT = 10, 3786, 8.936
REQUIRED_MARGIN_BITS = 8          # stated safety margin: worst-case sum at least 2^8 = 256x below 2^63


def test_negative_values_round_trip_in_twos_complement():
    from src.m7_privacy.secure_aggregation import SCALE_BITS

    exact = np.array([-1.0, -0.5, -2.0**-24, -3.75, -(2.0**38), 2.0**38, 0.0, -123456.0])   # multiples of 2^-24
    assert np.array_equal(decode(encode(exact)), exact)
    enc = encode(np.array([-1.0, -(2.0**38)]))
    assert enc.dtype == np.uint64 and enc[0] == np.uint64(2**64 - 2**24)          # two's complement of -2^24
    rng = np.random.default_rng(5)
    x = -np.abs(rng.normal(scale=50.0, size=1000))
    assert np.max(np.abs(decode(encode(x)) - x)) <= 2.0 ** -(SCALE_BITS + 1)       # rounding only
    vecs = [-np.abs(rng.normal(size=40)) for _ in range(4)]                      # all-negative updates, masked
    clients = setup(4)
    masked = [c.masked(v, round_=1) for c, v in zip(clients, vecs)]
    assert np.array_equal(unmask_sum(masked), decode(sum(encode(v) for v in vecs)))


def test_exp7_worst_case_cannot_overflow_with_the_stated_margin():
    from src.m7_privacy.secure_aggregation import check_capacity, headroom_bits

    worst_entry = EXP7_MAX_CLIENT_SIZE * EXP7_MAX_ABS_WEIGHT
    room = check_capacity(EXP7_MAX_CLIENTS, worst_entry, margin_bits=REQUIRED_MARGIN_BITS)
    assert room == pytest.approx(headroom_bits(EXP7_MAX_CLIENTS, worst_entry)) and room > 20
    # End to end at the extreme: every client sends the largest magnitude in every coordinate, same sign
    # (the worst case for the sum), then mixed signs; the masked sum must decode exactly.
    for signs in (np.ones(64), np.where(np.arange(64) % 2, 1.0, -1.0)):
        vec = signs * worst_entry
        clients = setup(EXP7_MAX_CLIENTS, max_abs_entry=worst_entry)
        masked = [c.masked(vec, round_=3) for c in clients]
        total = unmask_sum(masked)
        assert np.array_equal(total, decode(encode(vec)) * EXP7_MAX_CLIENTS)
        assert np.allclose(total, vec * EXP7_MAX_CLIENTS, rtol=0, atol=EXP7_MAX_CLIENTS * 2.0 ** -25)
    # The bound FedAvg declares by default (|w| <= 1e3, largest client size) also keeps the margin.
    assert check_capacity(EXP7_MAX_CLIENTS, EXP7_MAX_CLIENT_SIZE * 1e3) >= REQUIRED_MARGIN_BITS


def test_code_raises_when_the_bound_would_be_exceeded():
    from src.m7_privacy.secure_aggregation import check_capacity

    with pytest.raises(OverflowError):                       # declared bound too large for 63 bits + margin
        check_capacity(10, 2.0 ** 33, margin_bits=REQUIRED_MARGIN_BITS)
    with pytest.raises(OverflowError):
        setup(10, max_abs_entry=2.0 ** 33)
    clients = setup(3, max_abs_entry=100.0)                  # a client value above its declared bound
    with pytest.raises(OverflowError):
        clients[0].masked(np.array([1.0, -100.5]), round_=1)
    clients[0].masked(np.array([1.0, -100.0]), round_=1)    # at the bound is allowed
    with pytest.raises(OverflowError):                       # single value outside the fixed-point range
        encode(np.array([2.0 ** 39]))


def test_fedavg_refuses_a_weight_bound_that_could_overflow():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] > 0.5).astype(int)
    parts = stratified_iid_partition(y, 3, 0)
    with pytest.raises(OverflowError):
        fedavg_train("logreg", {}, [(X[p], y[p]) for p in parts], rounds=1, local_epochs=1, learning_rate=0.05,
                     sample_rate=0.1, epsilon=None, delta=1e-5, max_grad_norm=1.0, seed=0,
                     secure_aggregation=True, secagg_max_abs_weight=1e12)
    # A real update larger than the declared bound: with learning rate 1.0 the weights reach |w| ~ 12 in round 1
    # (measured), above a declared |w| <= 1, so the clients refuse to send.
    with pytest.raises(OverflowError):
        fedavg_train("logreg", {}, [(X[p], y[p]) for p in parts], rounds=1, local_epochs=2, learning_rate=1.0,
                     sample_rate=0.1, epsilon=None, delta=1e-5, max_grad_norm=1.0, seed=0,
                     secure_aggregation=True, secagg_max_abs_weight=1.0)
