"""Re-identification risk of a de-identified table, measured on its quasi-identifiers.

For a set of quasi-identifier (QI) columns, rows sharing the same QI values form an
equivalence class. An attacker who knows a person's QI values (e.g. from another source)
and that the person is in the table can single them out with probability 1 / class size
("prosecutor" risk, El Emam 2008). Reported:

- uniqueness: fraction of rows alone in their class (re-identifiable with certainty),
- k-violation rate: fraction of rows in classes smaller than k,
- average and maximum prosecutor risk (mean and max of 1 / class size),
- number of classes and the smallest class size.

Missing QI values (including suppressed cells) are treated as their own value, so a fully
suppressed row joins the class of other fully suppressed rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reidentification_risk(frame: pd.DataFrame, quasi_identifiers: list[str], k: int = 5) -> dict:
    missing = set(quasi_identifiers) - set(frame.columns)
    if missing:
        raise ValueError(f"unknown quasi-identifiers: {sorted(missing)}")
    if frame.empty:
        raise ValueError("empty frame")
    keys = frame[quasi_identifiers].astype(str).agg("|".join, axis=1)
    size = keys.map(keys.value_counts()).to_numpy()
    return {
        "n_rows": int(len(frame)),
        "n_quasi_identifiers": len(quasi_identifiers),
        "n_classes": int(keys.nunique()),
        "uniqueness": float(np.mean(size == 1)),
        "k": k,
        "k_violation_rate": float(np.mean(size < k)),
        "min_class_size": int(size.min()),
        "avg_prosecutor_risk": float(np.mean(1.0 / size)),
        "max_prosecutor_risk": float(np.max(1.0 / size)),
    }
