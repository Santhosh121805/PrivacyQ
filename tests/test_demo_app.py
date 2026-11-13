"""Architecture walkthrough: manifest shape and the live prediction endpoint (skipped when artifacts are absent)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "demo" / "manifest.json"
NEEDED = [MANIFEST, ROOT / "models/config_models.pkl", ROOT / "models/screening_models.pkl"]

pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDED),
                                reason="demo manifest or model caches not built (real-data artifacts)")


def test_manifest_has_15_boxes_in_order_and_sources():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert [s["box"] for s in m["sections"]] == list(range(1, 16))

    def walk(items):
        for it in items:
            yield it
            if it["kind"] == "group":
                yield from walk(it["items"])
    for s in m["sections"]:
        for it in walk(s["items"]):
            if it["kind"] == "na":
                assert it["reason"].strip(), it["title"]
            elif it["kind"] in ("figure", "table", "kv", "list"):
                assert it.get("source"), f"box {s['box']} item {it['title']} has no source"
    assert m["live"]["split"].startswith("locked hold-out")


def test_live_prediction_runs_on_the_holdout_row():
    import demo_app
    d = json.loads(json.dumps(demo_app.live()))              # the POST /api/live handler, called directly
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert d["row_index"] == m["live"]["row_index"]
    assert d["band"] in ("low", "moderate", "high")
    assert [s["title"] for s in d["steps"]] == ["Real record", "De-identify", "Preprocess", "Reduce to 8 angles",
                                                "Encode (8 qubits)", "Predict"]
    assert "survey_weight" in d["steps"][1]["rows"][0][1]
