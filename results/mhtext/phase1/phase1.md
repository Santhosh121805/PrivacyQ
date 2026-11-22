# Mental Health Text — Phase 1 build

Seed 2026. Raw files verified against the Phase 0 SHA-256 record. Keyword flags ['has_suicidal_keyword', 'has_stress_keyword', 'has_help_keyword'] removed (label leakage). Shipped test file not used (496 of its 992 rows are in the main file).

Label mapping, 4-class: Normal = 0, Anxiety = 1, Depression = 2, Suicidal = 3
Label mapping, 3-class: Anxiety = 0, Depression = 1, Suicidal = 2

## Class balance before deduplication

| Rows | n | Normal | Anxiety | Depression | Suicidal |
|---|---|---|---|---|---|
| All rows | 49,612 | 18,391 (37.07%) | 5,503 (11.09%) | 14,506 (29.24%) | 11,212 (22.60%) |
| ID-bearing (main) | 40,012 | 15,991 (39.97%) | 3,103 (7.76%) | 12,106 (30.26%) | 8,812 (22.02%) |
| Quarantined block | 9,600 | 2,400 (25.00%) | 2,400 (25.00%) | 2,400 (25.00%) | 2,400 (25.00%) |

## Deduplication (before any split)

- Exact duplicates (normalised text): 664 rows dropped; 13 groups with conflicting labels → 33 rows dropped entirely.
- Near-duplicates (TF-IDF cosine ≥ 0.90): 669 pairs; 381 rows dropped; 43 conflicting-label groups → 90 rows dropped entirely.
- Rows 49,612 → 48,444. ID-bearing kept 38,859 (dropped 1,153); quarantined block kept 9,585 (dropped 15). Near-duplicate removal repeated with IDF refitted on the survivors until no pair ≥ 0.90 remained (4 passes).
- Confirmation pass on the 15,000-row subsample plus the quarantined block (IDF refitted on exactly those rows, before the hold-out was drawn): 10 main and 3 quarantined rows dropped, so the main population is 14,990 rows.

Conflicting-label groups (the same post filed under two classes, mostly Depression vs Suicidal) show that boundary is soft in the source data. They are dropped, not resolved, and they cap achievable accuracy between those classes.

| After dedup | n | Normal | Anxiety | Depression | Suicidal |
|---|---|---|---|---|---|
| ID-bearing (main) | 38,859 | 15,528 (39.96%) | 2,903 (7.47%) | 11,729 (30.18%) | 8,699 (22.39%) |
| Quarantined block | 9,585 | 2,395 (24.99%) | 2,396 (25.00%) | 2,397 (25.01%) | 2,397 (25.01%) |

## 4-class view (subsample 15,000 → 14,990 after the confirmation pass, seed 2026; hold-out 20%)

| Split | Rows | Normal | Anxiety | Depression | Suicidal | Median words by class | p95 words |
|---|---|---|---|---|---|---|---|
| dev | 11,992 | 4,789 (39.93%) | 897 (7.48%) | 3,621 (30.20%) | 2,685 (22.39%) | Normal 10, Anxiety 77, Depression 82, Suicidal 70 | 194 |
| holdout | 2,998 | 1,198 (39.96%) | 224 (7.47%) | 905 (30.19%) | 671 (22.38%) | Normal 10, Anxiety 72, Depression 93, Suicidal 70 | 201 |
| quarantine | 9,582 | 2,393 (24.97%) | 2,396 (25.01%) | 2,397 (25.02%) | 2,396 (25.01%) | Normal 31, Anxiety 115, Depression 96, Suicidal 119 | 457 |

## 3-class view (subsample 15,000 → 14,990 after the confirmation pass, seed 2026; hold-out 20%)

| Split | Rows | Anxiety | Depression | Suicidal | Median words by class | p95 words |
|---|---|---|---|---|---|---|
| dev | 7,203 | 897 (12.45%) | 3,621 (50.27%) | 2,685 (37.28%) | Anxiety 77, Depression 82, Suicidal 70 | 209 |
| holdout | 1,800 | 224 (12.44%) | 905 (50.28%) | 671 (37.28%) | Anxiety 72, Depression 93, Suicidal 70 | 214 |
| quarantine | 7,189 | 2,396 (33.33%) | 2,397 (33.34%) | 2,396 (33.33%) | Anxiety 115, Depression 96, Suicidal 119 | 508 |

The 3-class view is the 4-class rows with Normal removed and the same split assignment (nested).
Checks passed: no normalised text in two splits; every main-population row has a Unique_ID.
Processed file: data/processed/mhtext/mhtext_phase1.csv (SHA-256 64d49a67fbdd98a8…). Total 611 s.
