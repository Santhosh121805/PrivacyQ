# Data

Raw and processed data are never committed. Everything under `data/` except this
file is git-ignored.

## Primary dataset: NSCH 2024 (real, public-use)

- **What:** National Survey of Children's Health 2024, topical public-use file,
  U.S. Census Bureau for HRSA Maternal and Child Health Bureau.
- **Source:** https://www2.census.gov/programs-surveys/nsch/datasets/2024/nsch_2024_topical_Stata.zip
- **Fetch:** `python scripts/download_nsch.py 2024` → `data/raw/nsch_2024/nsch_2024e_topical.dta`
- **SHA-256 of the zip downloaded 2026-11-04:** `a12d1839cae8b843884e7fb5b520835844ee6304c496ba7873d7f8a4152fe2e0`
- **Population used:** 21,789 adolescents aged 12–17 with a determinable depression label. Of the 51,375
  records, 21,960 are aged 12–17; the 171 whose depression answer is missing (K2Q32A missing, or K2Q32A = 1 with
  K2Q32B missing) are dropped. 21,789 is the population of every experiment.
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

## Second dataset: Mental Health Text Classification (4-class, public)

- **What:** Mukherjee, P. (2025). *Mental Health Text Classification Dataset (4-Class)*, Hugging Face Hub,
  `ourafla/Mental-Health_Text-Classification_Dataset`. English social-media posts and short statements labelled
  Suicidal / Depression / Anxiety / Normal. Derived from three public corpora (Komati, *Suicide and Depression
  Detection*, Kaggle; Sarkar, *Sentiment Analysis for Mental Health*, Kaggle; Murarka et al. 2021, *Reddit Mental
  Health Classification*, GitHub). Cite all four.
- **Licence:** CC BY 4.0 — `license: cc-by-4.0` in the dataset card metadata and the Hub tag `license:cc-by-4.0`
  (checked 2026-11-18). The card limits use to research and education: "not a clinical tool and must not be used
  for real-world diagnosis, triage, or crisis intervention."
- **Version used:** Hub commit `e47ef4f68a0d28d9f174cb8af2323400b14f1a73` (last modified 2025-12-18), files fetched
  from `https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset/resolve/<commit>/<file>`
  into `data/raw/mhtext/` (git-ignored). SHA-256:
  - `mental_heath_unbanlanced.csv` `d41cdc3ec3b8dde8f5553accaccd1a01a59c1440d83d7af6856fab77605c9471`
  - `mental_health_combined_test.csv` `2806f9518b92915b11ea35b16955ac3826fb5d981fc5f9152ee2282f7cffa418`
  - `mental_health_feature_engineered.csv` `122f1e702d2971d4c80c90fb3329dfa1485b56e1ccd0f4c5266604299858887e`
- **Files are CSV, not parquet.** The card describes two columns and 48,945 rows for the main file; the file holds
  three columns (`Unique_ID`, `text`, `status`) and 49,612 rows. No source-corpus column exists.
- **Known problems found on inspection (2026-11-18):** 496 of the 992 rows of the shipped "balanced test" file
  also occur in the main file (after lower-casing and whitespace normalisation), so that file is not used as a
  held-out set. The main file has 684 exact duplicates (normalised) and further near-duplicates; see the Phase 0
  report. Free text written by real people is more identifying than survey codes.

## Synthetic data

`SyntheticAdolescentLoader` generates survey-shaped data for tests and smoke runs.
Runs on it write to `results/SYNTHETIC-*` with a `SYNTHETIC_DATA_WARNING.txt`, and
every row is tagged `SYNTHETIC:` in its `data_source` column.
