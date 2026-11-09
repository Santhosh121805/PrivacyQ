"""Three-schools story demo: a replay of a real federated run (Experiment 4).

Run:   uvicorn story:app --port 8001      then open  http://127.0.0.1:8001/

Every number is read from results/story/story.json, produced from the real
experiment logs by `python scripts/build_story_data.py`. Nothing is trained or
simulated at launch; if the file is missing, each screen says so.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "story" / "story.json"

app = FastAPI(title="Three schools", docs_url=None, redoc_url=None)


@app.exception_handler(Exception)
async def any_error(request, exc):  # never a bare 500
    return JSONResponse(status_code=503, content={"error": str(exc) or type(exc).__name__})


@app.get("/api/story")
def story():
    if not DATA.exists():
        return JSONResponse(status_code=503, content={
            "error": "Replay data not found (results/story/story.json). Run `python scripts/build_story_data.py` once."})
    return json.loads(DATA.read_text(encoding="utf-8"))


app.mount("/vendor", StaticFiles(directory=ROOT / "web" / "vendor"), name="vendor")
app.mount("/", StaticFiles(directory=ROOT / "story", html=True), name="story")
