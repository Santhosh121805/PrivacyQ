"""Architecture walkthrough: manifest shape, pre-flight, and the 50-record live trace (skipped when artifacts are absent)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "demo" / "manifest.json"
NEEDED = [MANIFEST, ROOT / "models/config_models.pkl", ROOT / "models/screening_models.pkl",
          ROOT / "data/raw/nsch_2024/nsch_2024e_topical.dta"]

pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDED),
                                reason="real-data artifacts not built (manifest, model caches, NSCH file)")


def test_manifest_has_15_boxes_in_order_sources_and_headings():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert [s["box"] for s in m["sections"]] == list(range(1, 16))

    def walk(items):
        for it in items:
            yield it
            if it["kind"] == "group":
                yield from walk(it["items"])
    for s in m["sections"]:
        assert s["full_heading"]
        for it in walk(s["items"]):
            if it["kind"] == "na":
                assert it["reason"].strip(), it["title"]
            elif it["kind"] in ("figure", "table", "kv", "list"):
                assert it.get("source"), f"box {s['box']} item {it['title']} has no source"
    t = m["trace"]
    assert t["count"] == 50 and len(t["dataset_rows"]) == 50


def test_preflight_fails_loudly_on_a_missing_file(monkeypatch):
    import demo_app
    monkeypatch.setattr(demo_app, "MANIFEST", ROOT / "demo" / "no-such-manifest.json")
    with pytest.raises(SystemExit, match="no-such-manifest.json"):
        demo_app.preflight()


def test_trace_runs_every_stage_in_diagram_order_inference_only():
    import demo_app
    stages = [json.loads(json.dumps(s, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
              for s in demo_app.run_trace()]
    assert [s["box"] for s in stages] == list(range(1, 10))
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert [r[1] for r in stages[0]["table"]["rows"]] == m["trace"]["dataset_rows"]      # the same 50 rows, in order
    assert len(stages[-1]["table"]["rows"]) == 50
    assert {r[-1] for r in stages[-1]["table"]["rows"]} <= {"LOW", "MODERATE", "HIGH"}
    assert stages[6]["kv"][-1][1].endswith("identical")                                  # no parameter changed
    text = json.dumps(stages).lower()
    for metric in ("accuracy:", "recall:", "f1:", "roc-auc:"):                             # no metric from the 50 rows
        assert metric not in text
