"""Small-cell suppression for published tables (statistical disclosure control).

A share of n rows whose implied count k = round(share * n) is between 1 and THRESHOLD - 1 is replaced, together with
its share, by SUPPRESSED - the count alone is not enough, because share x n gives it back. Zero cells are kept (they
identify no one). Callers are responsible for checking complementary disclosure (whether a suppressed value can be
recovered by arithmetic from cells that remain) and for secondary suppression where it can.
"""

from __future__ import annotations

import pandas as pd

THRESHOLD = 5
SUPPRESSED = "< 5 posts (suppressed)"
RULE = ("Counts below 5 are suppressed where the cell describes people or their attributes, following standard "
        "statistical disclosure practice; the cell shows "
        f"\"{SUPPRESSED}\" and its share is removed with it (share x rows would give the count back). A second cell "
        "is suppressed where the first could be recovered by arithmetic. Counts describing pipeline operations - rows "
        "removed by deduplication and similar - are not suppressed, because they cannot single out an individual. "
        "'< 5' is a policy, not missing data.")


def suppress_shares(df: pd.DataFrame, n_col: str, share_cols: list[str], threshold: int = THRESHOLD
                    ) -> tuple[pd.DataFrame, list[dict]]:
    out = df.copy().astype({c: object for c in share_cols})
    log = []
    for i, r in df.iterrows():
        for c in share_cols:
            if pd.isna(r[c]):
                continue
            k = int(round(float(r[c]) * float(r[n_col])))
            if 1 <= k < threshold:
                out.at[i, c] = SUPPRESSED
                log.append({"row": int(i), "column": c})
    return out, log


def is_suppressed(x) -> bool:
    return isinstance(x, str) and x == SUPPRESSED


def numeric(s: pd.Series) -> pd.Series:
    """Suppressed cells -> NaN, so a column can be used in arithmetic that must not see the hidden value."""
    return pd.to_numeric(s.where(~s.map(is_suppressed)), errors="coerce")
