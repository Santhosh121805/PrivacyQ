"""Plain-English labels for NSCH columns, read from the official Stata codebook (.do file shipped with the data).

``label var x "..."`` gives each variable's official label; ``label define x_lab k "..."`` gives the meaning of each
category code. One-hot columns produced by the preprocessor are named ``<variable>_c<code>`` and are labelled
"<variable label>: <category label>". Nothing is invented: a column whose label is not in the codebook keeps its code.
"""

from __future__ import annotations

import re
from pathlib import Path

_VAR = re.compile(r'^\s*label var\s+(\w+)\s+"(.*)"\s*$')
_DEF = re.compile(r'^\s*label define\s+(\w+)\s+(-?\d+)\s+"(.*?)"')
_VAL = re.compile(r'^\s*(?:cap\s+)?label values\s+(\w+)\s+(\w+)')


def parse_codebook(path: str | Path) -> tuple[dict[str, str], dict[str, dict[int, str]]]:
    var_labels, defines, value_of = {}, {}, {}
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        if m := _VAR.match(line):
            var_labels[m.group(1).lower()] = m.group(2).strip()
        elif m := _DEF.match(line):
            defines.setdefault(m.group(1), {})[int(m.group(2))] = m.group(3).strip()
        elif m := _VAL.match(line):
            value_of[m.group(1).lower()] = m.group(2)
    values = {var: defines.get(lab, {}) for var, lab in value_of.items()}
    return var_labels, values


def plain_label(column: str, var_labels: dict, values: dict) -> str:
    """Plain-English label for a raw or one-hot column (falls back to the code itself)."""
    m = re.fullmatch(r"(.+)_c(-?\d+)", column)
    if m and m.group(1) in var_labels:
        var, code = m.group(1), int(m.group(2))
        cat = values.get(var, {}).get(code)
        return f"{var_labels[var]}: {cat}" if cat else f"{var_labels[var]}: code {code}"
    return var_labels.get(column, column)
