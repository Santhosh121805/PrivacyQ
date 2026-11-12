"""Render CONFORMANCE.md to a self-contained, print-ready HTML file (A4 landscape) without extra dependencies.

Handles exactly the Markdown the document uses: headings, paragraphs, bullet-free tables, bold, inline code,
horizontal rules and line breaks inside paragraphs. Open the HTML in a browser and print (or save as PDF).

Usage: python scripts/render_conformance.py  ->  CONFORMANCE_print.html
"""

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CSS = """
@page { size: A4 landscape; margin: 14mm 12mm; }
:root { --ink: #111; --muted: #555; --rule: #c9c9c9; --head: #f1f1ef; }
body { font: 9.5pt/1.4 "Segoe UI", Arial, sans-serif; color: var(--ink); background: #fff; margin: 0 auto; max-width: 1180px; padding: 16px; }
h1 { font-size: 17pt; margin: 0 0 4px; }
h2 { font-size: 12.5pt; margin: 18px 0 6px; break-after: avoid; }
p { margin: 6px 0; }
table { width: 100%; border-collapse: collapse; margin: 6px 0 10px; table-layout: fixed; }
th, td { border: 1px solid var(--rule); padding: 4px 6px; vertical-align: top; text-align: left; word-wrap: break-word; }
th { background: var(--head); font-weight: 600; }
tr { break-inside: avoid; }
code { font: 8.5pt Consolas, monospace; word-break: break-all; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 14px 0; }
.status { font-weight: 700; white-space: nowrap; }
.BUILT { color: #0b6b2b; } .PARTIAL { color: #8a5a00; } .PHASE-2 { color: #8a1c1c; }
col.c1 { width: 17%; } col.c2 { width: 8%; } col.c3 { width: 25%; } col.c4 { width: 50%; }
"""


def inline(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`"):
            out.append(f"<code>{html.escape(p[1:-1])}</code>")
        else:
            p = html.escape(p.replace("\\|", "|"))
            p = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", p, flags=re.S)
            out.append(p)
    return "".join(out)


def cells(line: str) -> list[str]:
    line = line.strip().strip("|")
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)]


def render(md: str) -> str:
    lines, body, i = md.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            body.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.strip() == "---":
            body.append("<hr>")
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            head, rows = cells(block[0]), [cells(r) for r in block[2:]]
            status_table = len(head) == 4 and head[1] == "Status"
            t = ["<table>"]
            if status_table:
                t.append('<colgroup><col class="c1"><col class="c2"><col class="c3"><col class="c4"></colgroup>')
            t.append("<thead><tr>" + "".join(f"<th>{inline(h)}</th>" for h in head) + "</tr></thead><tbody>")
            for r in rows:
                tds = []
                for j, c in enumerate(r):
                    if status_table and j == 1:
                        cls = c.split()[0] if c else ""
                        tds.append(f'<td><span class="status {cls}">{inline(c)}</span></td>')
                    else:
                        tds.append(f"<td>{inline(c)}</td>")
                t.append("<tr>" + "".join(tds) + "</tr>")
            t.append("</tbody></table>")
            body.append("".join(t))
            continue
        elif line.strip():
            para = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(("#", "|", "---")):
                i += 1
                para.append(lines[i])
            body.append("<p>" + inline("\n".join(para)).replace("\n", "<br>") + "</p>")
        i += 1
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>Conformance with the architecture diagram</title><style>{CSS}</style></head><body>"
            + "\n".join(body) + "</body></html>")


def main() -> None:
    src = ROOT / "CONFORMANCE.md"
    out = ROOT / "CONFORMANCE_print.html"
    out.write_text(render(src.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
