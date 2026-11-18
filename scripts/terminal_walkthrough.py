"""Terminal walkthrough of the architecture diagram (docs/architecture.jpg), one box at a time.

    python scripts/terminal_walkthrough.py --box 3        # one box
    python scripts/terminal_walkthrough.py --box 1 2 3    # several
    python scripts/terminal_walkthrough.py --all          # boxes 1-15 in order
    python scripts/terminal_walkthrough.py --list         # box titles
    options: --rows 50 (all live rows; default 10)   --no-live (results/ only, no trace)

Boxes 1-9 first run the live trace that the web demo runs: 50 locked hold-out records (indices 0-49), read from the
NSCH file now and sent through the stored models. It is inference only and computes no metric. Every box then prints
its section of results/results_report.md: the full-evaluation numbers, each read from a file in results/. Nothing
is trained, tuned or estimated here.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / "results" / "results_report.md"
WIDTH = min(shutil.get_terminal_size((120, 40)).columns, 140)


def _txt(v) -> str:
    return str(v["v"]) if isinstance(v, dict) else str(v)


def table(columns: list, rows: list, max_col: int = 42) -> None:
    cells = [[_txt(c) for c in columns]] + [[_txt(c) for c in r] for r in rows]
    n = len(cells[0])
    widths = [min(max_col, max(len(r[i]) for r in cells)) for i in range(n)]
    while sum(widths) + 3 * (n - 1) > WIDTH and max(widths) > 10:   # shrink the widest column until it fits
        widths[widths.index(max(widths))] -= 1
    num = re.compile(r"^[−+-]?[\d,.]+%?$")

    def line(r):
        wrapped = [textwrap.wrap(c, w) or [""] for c, w in zip(r, widths)]
        for k in range(max(len(x) for x in wrapped)):
            parts = []
            for i, x in enumerate(wrapped):
                s = x[k] if k < len(x) else ""
                parts.append(s.rjust(widths[i]) if num.match(r[i]) else s.ljust(widths[i]))
            print("   ".join(parts).rstrip())

    line(cells[0])
    print("   ".join("-" * w for w in widths))
    for r in cells[1:]:
        line(r)


def para(text: str, indent: str = "") -> None:
    print(textwrap.fill(text, WIDTH, initial_indent=indent, subsequent_indent=indent))


def banner(text: str) -> None:
    print("\n" + "=" * WIDTH + "\n" + text + "\n" + "=" * WIDTH)


def report_sections() -> dict[int, tuple[str, list[str]]]:
    if not REPORT.exists():
        raise SystemExit(f"Missing {REPORT.relative_to(ROOT)} — build it with: python scripts/build_results_report.py")
    out, cur = {}, None
    for ln in REPORT.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## Box (\d+) — (.*)$", ln)
        if m:
            cur = int(m.group(1))
            out[cur] = (m.group(2), [])
        elif ln.startswith("## "):
            cur = None
        elif cur is not None:
            out[cur][1].append(ln)
    return out


def print_markdown(lines: list[str]) -> None:
    """Plain-text rendering of one report section: paragraphs, tables, figure paths."""
    strip = lambda s: re.sub(r"\*\*?([^*]+)\*\*?", r"\1", s).replace("`", "")  # noqa: E731
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append([strip(c.strip()) for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body = [r for r in block[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            table(block[0], body)
            continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln)
        if m:
            print(f"[figure] {m.group(2)}")
        elif ln.startswith("<"):
            pass                                                    # HTML layout hints in the report
        elif ln:
            para(strip(ln))
        else:
            print()
        i += 1


def live_stages(upto: int) -> dict[int, dict]:
    """Run the demo's live trace once, up to box ``upto`` (1-9); return its stages by box."""
    import demo_app

    demo_app.preflight()
    print("Running the live trace: 50 locked hold-out records (indices 0-49), inference only ...", flush=True)
    out = {}
    for stage in demo_app.run_trace():
        out[stage["box"]] = stage
        if stage["box"] >= upto:
            break
    return out


def show_live(b: int, stage: dict, n_rows: int) -> None:
    banner(f"BOX {b} — {TITLES[b]}   ·   LIVE RUN (50 real hold-out records, {stage['ms']:,.0f} ms)")
    if stage.get("detail"):
        para(stage["detail"].replace(" (hover a value for its raw code)", ""))
    for k, v in stage.get("kv", []):
        para(f"{k}: {_txt(v)}", "  ")
    if stage.get("img"):
        print(f"[figure] demo/{stage['img']}")
    if stage.get("table"):
        print()
        rows = stage["table"]["rows"]
        table(stage["table"]["columns"], rows[:n_rows])
        if len(rows) > n_rows:
            print(f"... first {n_rows} of {len(rows)} rows shown (use --rows {len(rows)} for all)")
    print(f"\nSource: {stage['source']}")


def full(b: int, sections: dict) -> None:
    title, lines = sections[b]
    banner(f"BOX {b} — {title}   ·   FULL EVALUATION (from results/)")
    print_markdown(lines)


TITLES: dict[int, str] = {}


def main() -> None:
    p = argparse.ArgumentParser(description="Show the architecture boxes in the terminal")
    p.add_argument("--box", type=int, nargs="+", choices=range(1, 16), metavar="N", help="box number(s), 1-15")
    p.add_argument("--all", action="store_true", help="all 15 boxes")
    p.add_argument("--list", action="store_true", help="list the box titles")
    p.add_argument("--rows", type=int, default=10, help="live-table rows to print (default 10 of 50)")
    p.add_argument("--no-live", action="store_true", help="skip the live trace (boxes 1-9 show results/ only)")
    a = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    sections = report_sections()
    TITLES.update({k: v[0] for k, v in sections.items()})
    if a.list:
        for k in sorted(TITLES):
            print(f"{k:>2}  {TITLES[k]}" + ("   (live + results)" if k <= 9 else "   (results)"))
        return
    boxes = sorted(set(range(1, 16)) if a.all else set(a.box or []))
    if not boxes:
        p.error("give --box N, --all or --list")
    want_live = [] if a.no_live else [b for b in boxes if b <= 9]
    stages = live_stages(max(want_live)) if want_live else {}
    for b in boxes:
        if b in stages:
            show_live(b, stages[b], a.rows)
        full(b, sections)


if __name__ == "__main__":
    main()
