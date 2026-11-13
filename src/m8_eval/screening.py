"""Graded risk bands for a screening indication (not a diagnosis).

Cut-points come from a validation set of predicted probabilities and labels, by a stated rule:
- low / moderate: the highest cut that still keeps at least ``sensitivity`` of the validation positives
  at or above it (so at most 1 - sensitivity of positives would be called "low");
- moderate / high: the cut below which at least ``specificity`` of the validation negatives fall
  (so at most 1 - specificity of negatives would be called "high").
Nothing here is calibrated to population prevalence; the bands only order records by model probability.
"""

from __future__ import annotations

import numpy as np

BANDS = ("low", "moderate", "high")


def risk_cutpoints(prob, y, sensitivity: float = 0.90, specificity: float = 0.90) -> dict:
    p, y = np.asarray(prob, float), np.asarray(y).astype(int)
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need both classes in the validation rows")
    k = int(np.floor((1 - sensitivity) * len(pos)))          # positives allowed below the low cut
    low = float(pos[k])
    j = int(np.ceil(specificity * len(neg))) - 1              # this many negatives (0-based index) fall below/at
    high = float(np.nextafter(neg[min(j, len(neg) - 1)], np.inf))
    if high < low:                                            # the two rules overlap: no moderate band is possible
        high = low
    return {"low_moderate": low, "moderate_high": high, "sensitivity_target": sensitivity,
            "specificity_target": specificity}


def risk_band(prob: float, cuts: dict) -> str:
    if prob >= cuts["moderate_high"]:
        return "high"
    if prob >= cuts["low_moderate"]:
        return "moderate"
    return "low"


def band_summary(prob, y, cuts: dict) -> dict:
    """Rows and positives per band on the given (validation) rows."""
    p, y = np.asarray(prob, float), np.asarray(y).astype(int)
    bands = np.array([risk_band(v, cuts) for v in p])
    return {b: {"rows": int((bands == b).sum()), "positives": int(y[bands == b].sum())} for b in BANDS}
