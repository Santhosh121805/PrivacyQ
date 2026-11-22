import numpy as np
import pandas as pd

from src.m9_audit.disclosure import SUPPRESSED, numeric, suppress_shares


def test_small_counts_and_their_shares_are_suppressed_zero_and_large_kept():
    df = pd.DataFrame({"n": [1000, 1000, 1000, 1000], "a": [0.001, 0.004, 0.0, 0.005]})
    out, log = suppress_shares(df, "n", ["a"])
    assert out.a.tolist()[:2] == [SUPPRESSED, SUPPRESSED]     # 1 and 4 posts
    assert out.a.iloc[2] == 0.0 and out.a.iloc[3] == 0.005    # 0 kept, 5 kept
    assert [e["row"] for e in log] == [0, 1]


def test_numeric_hides_suppressed_values():
    s = pd.Series([0.2, SUPPRESSED])
    v = numeric(s)
    assert v.iloc[0] == 0.2 and np.isnan(v.iloc[1])
