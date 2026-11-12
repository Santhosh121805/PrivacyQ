"""Loader for the U.S. National Survey of Children's Health (NSCH) public-use file.

Source: U.S. Census Bureau / HRSA MCHB, topical public-use Stata files from
https://www2.census.gov/programs-surveys/nsch/datasets/<year>/

Target definitions follow the NSCH Data Resource Center convention: a child
"currently has" a condition when the caregiver reports that a doctor or other
health care provider ever said the child has it (K2Q3xA = 1) AND the child
currently has it (K2Q3xB = 1). Children never diagnosed, or diagnosed in the
past but not currently, are coded 0. Rows with an unknown status are dropped.

Important caveat for reporting: this is a *caregiver-reported provider
diagnosis*, not a clinical screening instrument (e.g. PHQ-A). Undiagnosed
adolescents are labelled 0, so the label under-counts true prevalence.

Features are an explicit whitelist of risk/context variables. Diagnoses,
diagnosis-dependent items, mental-health service use and special-health-care
needs screeners are excluded to prevent target leakage (see ``LEAKAGE_PREFIXES``
and ``LEAKAGE_COLUMNS``). Survey weights, state and stratum are carried as
metadata columns only and never used as features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.m1_data.loader import DatasetLoader

TARGETS = {
    "depression_current": ("k2q32a", "k2q32b"),
    "anxiety_current": ("k2q33a", "k2q33b"),
}

# name -> (kind, description). kind: "binary" (1=yes, 2=no -> 1/0),
# "ordinal" (numeric codes kept as is), "nominal" (one-hot encoded).
FEATURES: dict[str, tuple[str, str]] = {
    # demographics / household
    "sc_age_years": ("ordinal", "Age in years"),
    "sc_sex": ("binary", "Sex (1=male recoded to 1, female to 0)"),
    "sc_racer": ("nominal", "Race recode"),
    "sc_hispanic_r": ("binary", "Hispanic origin"),
    "fpl_i1": ("ordinal", "Family poverty ratio, first implicate (50-400)"),
    "higrade": ("ordinal", "Highest adult education"),
    "family_r": ("nominal", "Family structure"),
    "totkids_r": ("ordinal", "Number of children in household"),
    "house_gen": ("nominal", "Parental nativity"),
    "hhlanguage": ("nominal", "Primary household language"),
    "instype": ("nominal", "Insurance type"),
    # caregiver health (family mental-health proxy)
    "a1_menthealth": ("ordinal", "Adult 1 mental/emotional health (1=excellent..5=poor)"),
    "a2_menthealth": ("ordinal", "Adult 2 mental/emotional health"),
    "a1_physhealth": ("ordinal", "Adult 1 physical health"),
    # adverse childhood experiences
    "ace1": ("ordinal", "Hard to cover basics (1=never..4=very often)"),
    "ace3": ("binary", "Parent/guardian divorced"),
    "ace4": ("binary", "Parent/guardian died"),
    "ace5": ("binary", "Parent/guardian served time in jail"),
    "ace6": ("binary", "Witnessed adults hitting each other"),
    "ace7": ("binary", "Victim/witness of neighbourhood violence"),
    "ace8": ("binary", "Lived with someone mentally ill/suicidal/depressed"),
    "ace9": ("binary", "Lived with someone with alcohol/drug problem"),
    "ace10": ("binary", "Treated unfairly because of race"),
    "ace11": ("binary", "Treated unfairly because of health condition"),
    # sleep, screens, activity
    "hoursleep": ("ordinal", "Average sleep hours, binned"),
    "bedtime": ("ordinal", "How often same bedtime"),
    "screentime": ("ordinal", "Daily screen time, binned"),
    "physactiv": ("ordinal", "Days of 60 min physical activity"),
    "k7q30": ("binary", "Sports team/lessons"),
    "k7q31": ("binary", "Clubs/organisations"),
    "k7q32": ("binary", "Organised activities/lessons"),
    "k7q37": ("binary", "Community service"),
    "k7q38": ("binary", "Paid work"),
    # school
    "k7q02r_r": ("ordinal", "Days missed school (illness/injury)"),
    "k7q04r_r": ("ordinal", "Times school contacted household about problems"),
    "repeated": ("binary", "Repeated a grade"),
    "grades": ("nominal", "Grades in school (6 = no grades given)"),
    "k7q82_r": ("ordinal", "Cares about doing well in school"),
    "k7q83_r": ("ordinal", "Does required homework"),
    "k7q84_r": ("ordinal", "Finishes tasks started"),
    "k7q85_r": ("ordinal", "Stays calm when challenged"),
    # social
    "bullied_r": ("ordinal", "Bullied by others (1=never..5=almost every day)"),
    "bully": ("ordinal", "Bullies others"),
    "makefriend": ("ordinal", "Difficulty making/keeping friends"),
    "k9q96": ("binary", "Has another adult mentor"),
    "k10q40_r": ("ordinal", "Safe in neighbourhood"),
    "k10q41_r": ("ordinal", "Safe at school"),
    "k8q21": ("ordinal", "Talks with parent about things that matter"),
    "k8q11": ("ordinal", "Days family eats meal together"),
    # family resilience (Likert)
    "hopeful": ("ordinal", "Family stays hopeful"),
    "strengths": ("ordinal", "Family draws on strengths"),
    "wktosolve": ("ordinal", "Family works together"),
    "talkabout": ("ordinal", "Family talks together"),
    "k8q35": ("binary", "Caregiver has emotional support"),
    # physical health
    "k2q01": ("ordinal", "General health (1=excellent..5=poor)"),
    "bmiclass": ("nominal", "BMI percentile class"),
    "physicalpain": ("binary", "Chronic physical pain"),
    "concussion": ("binary", "Ever concussion/brain injury"),
    "bodyimage": ("ordinal", "Concern over body image"),
    # economic hardship / housing
    "foodsit": ("ordinal", "Household food situation"),
    "k11q62": ("binary", "Free/reduced-cost meals"),
    "everhomeless": ("nominal", "Ever homeless/shelter"),
    "homeevic": ("ordinal", "Worried about eviction"),
    "missmortgage": ("nominal", "Unable to pay mortgage/rent"),
    "k11q43r": ("ordinal", "Times moved to a new address"),
    "vape": ("binary", "Vaping inside home"),
}

METADATA = {"fipsst": "state_fips", "fwc": "survey_weight", "year": "survey_year"}

LEAKAGE_PREFIXES = ("k2q3", "k2q60", "sc_k2q", "k4q2")
LEAKAGE_COLUMNS = {
    "treatneed", "addtreat", "autismtreat", "autismmed", "sc_cshcn", "hcability",
    "hcextent", "memorycond", "k8q31", "k8q32", "k8q34", "hospitaler",
}


def assert_no_leakage(columns: Sequence[str]) -> None:
    leaked = [
        c for c in columns
        if c.startswith(LEAKAGE_PREFIXES) or c in LEAKAGE_COLUMNS
    ]
    if leaked:
        raise ValueError(f"target-leaking columns selected as features: {leaked}")


@dataclass
class NSCHLoader(DatasetLoader):
    paths: Sequence[str] = ()
    target: str = "depression_current"
    min_age: int = 12
    max_age: int = 17
    name: str = "NSCH"
    is_synthetic: bool = False
    label_column: str = "label"
    group_column: str | None = "state_fips"
    metadata_columns: tuple[str, ...] = field(default=tuple(METADATA.values()), init=False)

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("NSCHLoader requires at least one .dta path")
        if self.target not in TARGETS:
            raise ValueError(f"unknown target {self.target!r}; choose from {sorted(TARGETS)}")
        assert_no_leakage(list(FEATURES))
        years = []
        for p in self.paths:
            digits = "".join(ch for ch in Path(p).stem if ch.isdigit())
            years.append(digits[:4] if digits else "?")
        self.name = f"NSCH-{'+'.join(years)}-age{self.min_age}-{self.max_age}-{self.target}"

    def load(self) -> pd.DataFrame:
        frames = []
        for path in self.paths:
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"{path} not found; download it with scripts/download_nsch.py"
                )
            raw = pd.read_stata(path, convert_categoricals=False)
            raw.columns = raw.columns.str.lower()
            frames.append(raw)
        raw = pd.concat(frames, ignore_index=True)

        raw = raw[raw["sc_age_years"].between(self.min_age, self.max_age)]
        ever, current = TARGETS[self.target]
        label = pd.Series(np.nan, index=raw.index)
        label[raw[ever] == 2] = 0
        label[(raw[ever] == 1) & (raw[current] == 2)] = 0
        label[(raw[ever] == 1) & (raw[current] == 1)] = 1
        raw = raw[label.notna()]
        label = label[label.notna()].astype(int)

        out = pd.DataFrame(index=raw.index)
        for column, (kind, _) in FEATURES.items():
            values = pd.to_numeric(raw[column], errors="coerce")
            if kind == "binary":
                out[column] = values.map({1.0: 1.0, 2.0: 0.0})
            elif kind == "nominal":
                out[column] = values.map(lambda v: v if pd.isna(v) else f"c{int(v)}")
            else:
                out[column] = values.astype(float)
        for source, target in METADATA.items():
            out[target] = raw[source].to_numpy()
        out[self.label_column] = label.to_numpy()
        return out.reset_index(drop=True)


def feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [(name, kind, desc) for name, (kind, desc) in FEATURES.items()],
        columns=["variable", "kind", "description"],
    )


def _current(ever: pd.Series, current: pd.Series) -> pd.Series:
    """Same rule as the binary loader: ever=no -> 0 (current is skipped by design);
    ever=yes & current=no -> 0; ever=yes & current=yes -> 1; anything else -> NaN (undetermined)."""
    label = pd.Series(np.nan, index=ever.index)
    label[ever == 2] = 0
    label[(ever == 1) & (current == 2)] = 0
    label[(ever == 1) & (current == 1)] = 1
    return label


def three_class_from_codes(dep_ever, dep_cur, anx_ever, anx_cur) -> pd.Series:
    """0 = none, 1 = anxiety only, 2 = depression (with or without anxiety); NaN when undetermined.

    Both conditions use the "currently has" (B) variant: K2Q32B depression, K2Q33B anxiety.
    A depression-positive row is class 2 whatever its anxiety answer, so it is never undetermined;
    a depression-negative row with undetermined anxiety cannot be placed in 0 or 1 and is NaN.
    Merging classes 0 and 1 gives the binary depression_current target exactly."""
    dep = _current(pd.Series(dep_ever), pd.Series(dep_cur))
    anx = _current(pd.Series(anx_ever), pd.Series(anx_cur))
    out = pd.Series(np.nan, index=dep.index)
    out[dep == 1] = 2
    out[(dep == 0) & (anx == 0)] = 0
    out[(dep == 0) & (anx == 1)] = 1
    return out


def three_class_labels(loader: NSCHLoader) -> np.ndarray:
    """Three-class label aligned row-for-row with ``loader.load()`` (depression_current target),
    NaN where anxiety is undetermined among depression-negative rows."""
    if loader.target != "depression_current":
        raise ValueError("three-class labels are aligned with the depression_current loader")
    frames = []
    for path in loader.paths:
        raw = pd.read_stata(path, convert_categoricals=False,
                            columns=["sc_age_years", "k2q32a", "k2q32b", "k2q33a", "k2q33b"])
        raw.columns = raw.columns.str.lower()
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["sc_age_years"].between(loader.min_age, loader.max_age)]
    raw = raw[_current(raw["k2q32a"], raw["k2q32b"]).notna()]          # same rows as load()
    return three_class_from_codes(raw["k2q32a"], raw["k2q32b"], raw["k2q33a"], raw["k2q33b"]).to_numpy()
