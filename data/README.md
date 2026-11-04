# Data

Raw and processed data are never committed. Everything under `data/` except this
file is git-ignored.

## Primary dataset: NSCH 2024 (real, public-use)

- **What:** National Survey of Children's Health 2024, topical public-use file,
  U.S. Census Bureau for HRSA Maternal and Child Health Bureau.
- **Source:** https://www2.census.gov/programs-surveys/nsch/datasets/2024/nsch_2024_topical_Stata.zip
- **Fetch:** `python scripts/download_nsch.py 2024` → `data/raw/nsch_2024/nsch_2024e_topical.dta`
- **SHA-256 of the zip downloaded 2026-11-04:** `a12d1839cae8b843884e7fb5b520835844ee6304c496ba7873d7f8a4152fe2e0`
- **Population used:** selected children aged 12–17 (21,960 of 51,375 records).
- **Target `depression_current`:** caregiver reports a provider ever said the child
  has depression (K2Q32A = 1) *and* the child currently has it (K2Q32B = 1).
  Never-diagnosed and past-only children are 0; unknown status is dropped.
  This is a caregiver-reported diagnosis, not a screening instrument, so
  undiagnosed depression is counted as negative.
- **Features:** a 66-variable whitelist in `src/m1_data/nsch.py` covering
  demographics, caregiver mental health, ACEs, sleep, screen time, activity, school,
  bullying, family resilience, physical health and economic hardship. Diagnoses,
  diagnosis-dependent items, mental-health treatment and special-health-care-needs
  screeners are excluded (`assert_no_leakage`).
- **Metadata (never used as features):** state FIPS (for federated "site"
  partitions), survey weight `fwc`, survey year. Models are trained unweighted, so
  results describe the sample, not U.S. population estimates.

### Limitation: the label measures diagnosis, not depression

The target is caregiver-reported, provider-diagnosed depression. A large share
of depression is never diagnosed (the WHO estimate cited in this project's
literature survey is ~63.6% of cases), and undiagnosed adolescents sit in the
negative class. The label therefore captures *who gets diagnosed*, not *who is
depressed*. A model trained on it identifies adolescents who resemble those
already diagnosed, which under-represents groups with poorer access to care.

In addition, NSCH is a one-time cross-sectional survey, and some predictors
(e.g. difficulty making friends, staying calm when challenged, missed school)
may be consequences of depression rather than causes. All results are framed as
**identifying who is at risk now, not forecasting who will become depressed.**

NSCH 2023 is downloadable with the same variables but is deliberately not pooled:
the quantum models are subsample-limited, so extra rows buy nothing and would add
a survey-year effect.

## Synthetic data

`SyntheticAdolescentLoader` generates survey-shaped data for tests and smoke runs.
Runs on it write to `results/SYNTHETIC-*` with a `SYNTHETIC_DATA_WARNING.txt`, and
every row is tagged `SYNTHETIC:` in its `data_source` column.
