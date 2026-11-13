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


# Display-only abbreviations of codebook meanings, for columns whose header already names the attribute.
# sc_hispanic_r: codebook 1 = "Hispanic or Latino Origin", 2 = "Not Hispanic or Latino Origin"; under the header
# "Hispanic" these read as Yes / No. The full codebook label stays in the tooltip.
SHORT_VALUES = {"sc_hispanic_r": {1: "Yes", 2: "No"}}


def display_value(column: str, value, kind: str, values: dict) -> tuple[str, str, bool]:
    """Human-readable text for one value as loaded by src/m1_data/nsch.py.

    The loader recodes yes/no answers 1/2 -> 1/0 and writes nominal categories as "c<code>"; ordinal codes are
    kept. This maps the loaded value back to its NSCH code and returns (text, tooltip, known). ``known`` is False
    when the column has value labels but this code is not among them: the code itself is shown, never a guess.
    """
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—", "missing answer", True
    loaded = value if isinstance(value, str) else f"{float(value):g}"
    labels = values.get(column, {})
    if kind == "nominal":
        code = int(str(value)[1:])
    elif kind == "binary":
        code = 1 if float(value) == 1 else 2
    else:
        f = float(value)
        code = int(f) if f.is_integer() else None
    tip = f"raw value {loaded}" + (f" (NSCH code {code})" if kind in ("binary", "nominal") else "")
    if column == "fpl_i1":                                   # percent of the federal poverty level, top/bottom coded
        lab = labels.get(code)
        text = (lab.replace(" or ", "% or ", 1) if lab else f"{float(value):g}%") + " of federal poverty level"
        return text, tip, True
    if not labels:                                           # numeric variable with no value labels (e.g. age)
        return loaded, tip, True
    if code in labels:
        full = labels[code]
        short = SHORT_VALUES.get(column, {}).get(code)
        return (short or full), (f"{tip}: {full}" if short else tip), True
    return loaded, f"{tip}: not in the NSCH codebook", False
