"""Paired significance tests over cross-validation folds.

Fold scores from (repeated) K-fold CV are not independent: training sets
overlap, so a plain paired t-test is anti-conservative. The primary test here
is the Nadeau-Bengio corrected resampled t-test, which inflates the variance by
(1/k + n_test/n_train). The Wilcoxon signed-rank test is reported alongside as
a non-parametric check. Holm's method controls the family-wise error rate when
several comparisons are reported together.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def corrected_resampled_ttest(a, b, test_train_ratio: float) -> dict[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("need two equal-length score arrays with at least 2 folds")
    d = a - b
    k = d.size
    mean, var = d.mean(), d.var(ddof=1)
    se = np.sqrt((1.0 / k + test_train_ratio) * var)
    if se == 0:
        t, p = (0.0, 1.0) if mean == 0 else (np.inf * np.sign(mean), 0.0)
    else:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), df=k - 1))
    half = float(stats.t.ppf(0.975, df=k - 1) * se) if se > 0 else 0.0
    return {"mean_diff": float(mean), "t": float(t), "p": float(p),
            "ci95_low": float(mean - half), "ci95_high": float(mean + half), "k": int(k)}


def wilcoxon_p(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="zsplit").pvalue)


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted
