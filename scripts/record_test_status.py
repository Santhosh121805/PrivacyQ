"""Run the test suite and record the outcome for the PrivacyQ Overview page.

Usage: python scripts/record_test_status.py   -> results/test_status.json
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "-p", "no:cacheprovider"],
                      cwd=ROOT, capture_output=True, text=True)
summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|errors?|skipped)", summary)}
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
status = {"passed": counts.get("passed", 0), "failed": counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0),
          "summary": summary, "exit_code": proc.returncode, "git_commit": commit,
          "recorded": datetime.now().isoformat(timespec="seconds")}
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "test_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
print(json.dumps(status, indent=2))
