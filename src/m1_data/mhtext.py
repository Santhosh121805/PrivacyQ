"""Mental Health Text Classification (4-class) loader: second data source beside NSCH 2024.

Hugging Face ``ourafla/Mental-Health_Text-Classification_Dataset`` (CC BY 4.0), commit e47ef4f, raw files in
``data/raw/mhtext/`` (git-ignored). See data/README.md for provenance and the Phase 0 inspection.

Phase 1 decisions (all recorded in the build manifest):
  * Main population = the 40,012 rows that carry a ``Unique_ID``. The trailing 9,600-row block (no ID, exactly
    2,400 per class, no sentiment features) has unknown provenance and is QUARANTINED: held aside, reported
    separately, never mixed into training or the hold-out.
  * Deduplicate BEFORE splitting: exact duplicates after lower-casing and whitespace normalisation, then
    near-duplicates with TF-IDF cosine >= 0.90 (word 1-2-grams). A duplicate group whose rows carry different
    labels is dropped entirely (those conflicts cap achievable Depression/Suicidal accuracy; they are not resolved).
  * Stratified 15,000-row subsample (seed fixed), then a stratified 20 % hold-out, opened once.
  * Three-class view (Normal dropped) is the same rows minus Normal, with the same split assignment, so the
    four- and three-class runs are nested and directly comparable.
  * The three keyword flags are label leakage and are blocked from every model input.

The shipped ``mental_health_combined_test.csv`` is not used: half of it (496 / 992) is in the main file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.m1_data.loader import DatasetLoader

# Explicit, fixed label mappings (printed by the build script and stored in the manifest).
CLASSES_4 = {"Normal": 0, "Anxiety": 1, "Depression": 2, "Suicidal": 3}
CLASSES_3 = {"Anxiety": 0, "Depression": 1, "Suicidal": 2}
MAPPINGS = {4: CLASSES_4, 3: CLASSES_3}

# Columns that could give the answer away or identify a row: never a model input.
KEYWORD_LEAKAGE = ("has_suicidal_keyword", "has_stress_keyword", "has_help_keyword")
BLOCKED_COLUMNS = frozenset({*KEYWORD_LEAKAGE, "status", "label", "label_name", "Unique_ID", "split", "provenance",
                             "dup_group", "text_norm"})
# Numeric features shipped with the dataset (computed by the dataset author, not here).
SHIPPED_FEATURES = ("text_length", "word_count", "num_urls", "num_emojis", "num_special_chars", "num_excess_punct",
                    "avg_word_length", "stopword_ratio", "type_token_ratio", "polarity", "subjectivity",
                    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio")
RAW_MAIN = "mental_health_feature_engineered.csv"       # = the training file row for row, plus shipped features
SPLITS = ("dev", "holdout", "quarantine")


class MHTextDataError(RuntimeError):
    """The files on disk do not match what Phase 0 recorded; raised instead of guessing."""


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def assert_no_blocked(columns: Iterable[str]) -> None:
    """Fail if any blocked (leaking or identifying) column is about to reach a model."""
    hit = sorted(set(map(str, columns)) & BLOCKED_COLUMNS)
    if hit:
        raise MHTextDataError(f"blocked columns selected as model inputs: {hit}")


def model_inputs(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """The only path from the table to a model: shipped numeric features (or ``columns``), guarded."""
    cols = list(columns) if columns is not None else [c for c in SHIPPED_FEATURES if c in frame.columns]
    assert_no_blocked(cols)
    return frame[cols]


# ------------------------------------------------------------------ deduplication (before any split)
def dedup(frame: pd.DataFrame, threshold: float = 0.90, batch: int = 1500,
          max_passes: int = 6) -> tuple[pd.DataFrame, dict]:
    """Drop exact and near-duplicate texts; drop whole groups whose labels conflict.

    Rows are kept in file order, so the first occurrence survives (ID-bearing rows precede the quarantined
    block). Similarity: TF-IDF cosine on word 1-2-grams (sublinear tf, min_df 2) over normalised text; groups
    are connected components of the >= ``threshold`` graph. Returns the kept rows and a count log.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = frame.copy()
    df["text_norm"] = df.text.map(normalise)
    log = {"rows_in": int(len(df))}

    # exact duplicates
    n_labels = df.groupby("text_norm").status.transform("nunique")
    exact_conflict = n_labels > 1
    log["exact_conflict_groups"] = int(df.loc[exact_conflict, "text_norm"].nunique())
    log["exact_conflict_rows_dropped"] = int(exact_conflict.sum())
    df = df[~exact_conflict]
    dup = df.text_norm.duplicated(keep="first")
    log["exact_duplicate_rows_dropped"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)

    # near-duplicates, repeated with IDF refitted on the survivors until no pair >= threshold remains
    log.update(near_duplicate_pairs=0, near_conflict_groups=0, near_conflict_rows_dropped=0,
               near_duplicate_rows_dropped=0, near_passes=0)
    for _ in range(max_passes):
        X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, dtype=np.float32).fit_transform(df.text_norm)
        rows, cols = [], []
        for s in range(0, X.shape[0], batch):
            S = (X[s:s + batch] @ X.T).tocoo()
            keep = (S.data >= threshold) & (S.col > S.row + s)
            rows.append(S.row[keep] + s)
            cols.append(S.col[keep])
        i, j = np.concatenate(rows), np.concatenate(cols)
        log["near_passes"] += 1
        if not len(i):
            break
        log["near_duplicate_pairs"] += int(len(i))
        G = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(len(df), len(df)))
        _, comp = connected_components(G, directed=False)
        df["dup_group"] = comp
        conflict = df.groupby("dup_group").status.transform("nunique") > 1
        log["near_conflict_groups"] += int(df.loc[conflict, "dup_group"].nunique())
        log["near_conflict_rows_dropped"] += int(conflict.sum())
        df = df[~conflict].copy()
        near_dup = df.dup_group.duplicated(keep="first")        # singletons are never duplicated
        log["near_duplicate_rows_dropped"] += int(near_dup.sum())
        df = df[~near_dup].drop(columns="dup_group").reset_index(drop=True)
    else:
        raise MHTextDataError(f"near-duplicate removal did not settle in {max_passes} passes")
    log["rows_out"] = int(len(df))
    log["threshold"] = threshold
    log["similarity"] = "TF-IDF cosine, word 1-2-grams, sublinear tf, min_df=2, on lower-cased whitespace-normalised text"
    return df, log


def read_raw(root: str | Path) -> pd.DataFrame:
    """The main file with shipped features; provenance = 'id' (has Unique_ID) or 'quarantine' (trailing block)."""
    path = Path(root) / RAW_MAIN
    if not path.exists():
        raise MHTextDataError(f"{path} not found — fetch the files listed in data/README.md")
    df = pd.read_csv(path)
    unknown = set(df.status.unique()) - set(CLASSES_4)
    if unknown:
        raise MHTextDataError(f"unexpected labels {unknown}")
    df["provenance"] = np.where(df.Unique_ID.isna(), "quarantine", "id")
    first_q = int(np.argmax(df.provenance.to_numpy() == "quarantine"))
    if not (df.provenance.iloc[first_q:] == "quarantine").all():
        raise MHTextDataError("the ID-less rows are not one trailing block; the Phase 0 finding no longer holds")
    return df


# ------------------------------------------------------------------ loader over the processed file
@dataclass
class MHTextLoader(DatasetLoader):
    """Processed table from scripts/build_mhtext_splits.py. ``n_classes`` 4 or 3 (3 drops Normal).

    ``splits`` picks rows by the locked assignment: ``dev`` and ``holdout`` form the main population;
    ``quarantine`` is the held-aside 9,600-row block (after its own deduplication).
    """

    processed: str = "data/processed/mhtext/mhtext_phase1.csv"
    n_classes: int = 4
    splits: tuple[str, ...] = ("dev", "holdout")
    name: str = "MHText-4class"
    label_column: str = "label"
    group_column: str | None = None
    metadata_columns: tuple[str, ...] = field(default=("Unique_ID", "split", "provenance"), init=False)

    def __post_init__(self) -> None:
        if self.n_classes not in MAPPINGS:
            raise ValueError("n_classes must be 4 or 3")
        bad = set(self.splits) - set(SPLITS)
        if bad:
            raise ValueError(f"unknown splits {bad}")
        self.name = f"MHText-{self.n_classes}class"
        self.mapping = dict(MAPPINGS[self.n_classes])

    def load(self) -> pd.DataFrame:
        path = Path(self.processed)
        if not path.exists():
            raise MHTextDataError(f"{path} not found — run scripts/build_mhtext_splits.py first")
        df = pd.read_csv(path, keep_default_na=True)
        df = df[df.split.isin(self.splits)]
        df = df[df.status.isin(self.mapping)].reset_index(drop=True)
        df["label_name"] = df.status
        df[self.label_column] = df.status.map(self.mapping).astype(int)
        return df

    def text(self, frame: pd.DataFrame) -> pd.Series:
        return frame.text.astype(str)
