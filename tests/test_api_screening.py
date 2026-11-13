import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "models/screening_models.pkl").exists() or not (ROOT / "models/config_models.pkl").exists(),
                                reason="Inspect / screening model caches not built")


def test_inspect_returns_three_class_probabilities_and_a_consistent_risk_band():
    import api
    from src.m8_eval.screening import risk_band

    rid = int(api.models()["pool"].index[0])
    out = api.inspect(api.InspectRequest(record_id=rid, model="qsvm", encoding="angle", features=8))
    s = out["screening"]
    probs = s["three_class"]["probabilities"]
    assert set(probs) == {"none", "anxiety_only", "depression"}
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)
    assert s["three_class"]["predicted"] == max(probs, key=probs.get)
    r = s["risk"]
    assert 0 <= r["probability"] <= 1 and r["band"] == risk_band(r["probability"], r["cuts"])
    assert r["cuts"]["sensitivity_target"] == 0.95 and r["cuts"]["specificity_target"] == 0.95
    assert sum(b["rows"] for b in r["inner_val_bands"].values()) == r["n_inner_val"]
