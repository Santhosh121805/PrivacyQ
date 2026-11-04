from __future__ import annotations

import numpy as np


def loss_threshold_membership_inference(losses: np.ndarray | list[float], threshold: float | None = None) -> np.ndarray:
    values = np.asarray(losses, dtype=float)
    if threshold is None:
        threshold = float(np.quantile(values, 0.1))
    return (values <= threshold).astype(int)


def membership_inference(*args, **kwargs):
    return loss_threshold_membership_inference(*args, **kwargs)


def gradient_inversion_stub(*args, **kwargs) -> dict[str, object]:
    return {
        "status": "stub",
        "message": "gradient inversion is not implemented in this phase",
    }
