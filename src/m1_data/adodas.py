"""AdoDAS (Adolescent Depression, Anxiety and Stress; ACM MM 2026 Grand Challenge) participant-level loader.

Second data source beside NSCH 2024, same ``DatasetLoader`` interface. Data stay under ``data/raw/adodas/``
(git-ignored). Layout, from the official baseline README (github.com/Multimodel-signal-lab/AdoDAS2026):

    <root>/train.csv, <root>/val.csv                     participant labels (test labels are withheld)
    <root>/<split>/<anon_school>/<anon_class>/<anon_pid>/audio/<group>/[<model_tag>/]<session>/...
    <root>/<split>/<anon_school>/<anon_class>/<anon_pid>/video/<group>/[<model_tag>/]<session>/...

Sessions: A01 (scripted reading), B01-B03 (open responses: yesterday, happiest memory, saddest memory).
Labels (project page, "Annotation and Labels"): DASS-21 items d01..d21 in {0,1,2,3}; subscale = 2 x item sum;
binary target = Mild-or-above, i.e. Depression >= 10, Anxiety >= 8, Stress >= 15 (scaled scores).

Nothing here guesses a value. Column names that the documentation does not fix (the participant id, any
pre-computed binary columns) are detected; if a binary column is present it must agree with the label
derived from the items, otherwise loading fails. Unknown columns are kept, untouched, and listed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.m1_data.loader import DatasetLoader

SESSIONS = ("A01", "B01", "B02", "B03")
OPEN_SESSIONS = ("B01", "B02", "B03")
ITEMS = tuple(f"d{i:02d}" for i in range(1, 22))
SUBSCALE_ITEMS = {
    "depression": ("d03", "d05", "d10", "d13", "d16", "d17", "d21"),
    "anxiety": ("d02", "d04", "d07", "d09", "d15", "d19", "d20"),
    "stress": ("d01", "d06", "d08", "d11", "d12", "d14", "d18"),
}
MILD_OR_ABOVE = {"depression": 10, "anxiety": 8, "stress": 15}       # scaled score cut-offs (project page table)
TARGETS = tuple(SUBSCALE_ITEMS)
ID_CANDIDATES = ("person_id", "anon_pid", "pid", "participant_id", "subject_id", "id")
BINARY_CANDIDATES = {"depression": ("depression", "d", "dep", "label_d", "y_d"),
                     "anxiety": ("anxiety", "a", "anx", "label_a", "y_a"),
                     "stress": ("stress", "s", "str", "label_s", "y_s")}
LABELLED_SPLITS = ("train", "val")
ALL_SPLITS = ("train", "val", "test")
SPLIT_DIRS = {"train": ("train", "Train"), "val": ("val", "Val", "valid", "validation"), "test": ("test", "Test")}


class AdoDASDataError(RuntimeError):
    """The files on disk do not match the documented release; raised instead of guessing."""


# ------------------------------------------------------------------ labels
def subscale_scores(items: pd.DataFrame) -> pd.DataFrame:
    """Scaled DASS-21 subscale scores (2 x sum of the subscale's seven items)."""
    return pd.DataFrame({t: 2 * items[list(cols)].sum(axis=1, min_count=7) for t, cols in SUBSCALE_ITEMS.items()},
                        index=items.index)


def binary_labels(items: pd.DataFrame) -> pd.DataFrame:
    """Track A1 targets: 1 = Mild-or-above, 0 = Normal; NaN if any of the subscale's items is missing."""
    scores = subscale_scores(items)
    out = pd.DataFrame(index=items.index)
    for t, cut in MILD_OR_ABOVE.items():
        out[t] = np.where(scores[t].isna(), np.nan, (scores[t] >= cut).astype(float))
    return out


# ------------------------------------------------------------------ files
def _split_dir(root: Path, split: str) -> Path | None:
    for name in SPLIT_DIRS[split]:
        if (root / name).is_dir():
            return root / name
    return None


def _id_column(columns: Iterable[str]) -> str:
    lower = {c.lower(): c for c in columns}
    for cand in ID_CANDIDATES:
        if cand in lower:
            return lower[cand]
    raise AdoDASDataError(f"no participant id column; expected one of {ID_CANDIDATES}, found {list(columns)}")


def read_label_csv(path: Path, split: str) -> tuple[pd.DataFrame, list[str]]:
    """One row per participant: person_id, split, d01..d21, derived targets; plus untouched extra columns."""
    raw = pd.read_csv(path)
    idc = _id_column(raw.columns)
    lower = {c.lower(): c for c in raw.columns}
    missing = [i for i in ITEMS if i not in lower]
    if missing:
        raise AdoDASDataError(f"{path.name}: DASS-21 item columns missing: {missing}")
    out = pd.DataFrame({"person_id": raw[idc].astype(str), "split": split})
    for i in ITEMS:
        col = pd.to_numeric(raw[lower[i]], errors="coerce")
        bad = col.dropna()[~col.dropna().isin([0, 1, 2, 3])]
        if len(bad):
            raise AdoDASDataError(f"{path.name}: {i} has values outside 0-3: {sorted(bad.unique())[:5]}")
        out[i] = col.to_numpy()
    derived = binary_labels(out[list(ITEMS)])
    for t in TARGETS:
        out[t] = derived[t].to_numpy()
        given = next((lower[c] for c in BINARY_CANDIDATES[t] if c in lower and lower[c] not in (idc,)), None)
        if given is not None:
            g = pd.to_numeric(raw[given], errors="coerce").to_numpy()
            ok = ~np.isnan(g) & ~np.isnan(out[t].to_numpy())
            if (g[ok] != out[t].to_numpy()[ok]).any():
                raise AdoDASDataError(f"{path.name}: column {given!r} disagrees with the {t} label derived "
                                      f"from the DASS-21 items in {(g[ok] != out[t].to_numpy()[ok]).sum()} rows")
    used = {idc, *(lower[i] for i in ITEMS)}
    used |= {lower[c] for t in TARGETS for c in BINARY_CANDIDATES[t] if c in lower}
    extra = [c for c in raw.columns if c not in used]
    for c in extra:
        out[f"aux__{c}"] = raw[c].to_numpy()
    if out.person_id.duplicated().any():
        raise AdoDASDataError(f"{path.name}: duplicated participant ids: "
                              f"{out.person_id[out.person_id.duplicated()].head().tolist()}")
    return out, extra


def scan_feature_tree(root: Path) -> pd.DataFrame:
    """One row per (participant, session): split, school, class, and the feature groups found on disk."""
    rows = []
    for split in ALL_SPLITS:
        sdir = _split_dir(root, split)
        if sdir is None:
            continue
        for pdir in sorted(p for p in sdir.glob("*/*/*") if p.is_dir()):
            school, klass, pid = pdir.parts[-3], pdir.parts[-2], pdir.parts[-1]
            found: dict[str, set[str]] = {}
            for modality in ("audio", "video"):
                for gdir in (pdir / modality).glob("*") if (pdir / modality).is_dir() else []:
                    for sess in SESSIONS:
                        if any(s.is_dir() for s in gdir.glob(f"**/{sess}")):
                            found.setdefault(sess, set()).add(f"{modality}/{gdir.name}")
            for sess in SESSIONS:
                rows.append({"person_id": pid, "split": split, "school_id": school, "class_id": klass,
                             "session": sess, "n_feature_groups": len(found.get(sess, ())),
                             "feature_groups": ";".join(sorted(found.get(sess, ())))})
    return pd.DataFrame(rows, columns=["person_id", "split", "school_id", "class_id", "session",
                                       "n_feature_groups", "feature_groups"])


# ------------------------------------------------------------------ leakage guards
def assert_subject_disjoint(assignment: Mapping[str, Sequence[str]]) -> None:
    """Fail if any participant id appears in more than one split / fold / client."""
    seen: dict[str, str] = {}
    clashes = []
    for name, ids in assignment.items():
        for pid in set(map(str, ids)):
            if pid in seen:
                clashes.append((pid, seen[pid], name))
            else:
                seen[pid] = name
    if clashes:
        raise AdoDASDataError(f"{len(clashes)} participant(s) in two partitions, e.g. {clashes[:3]}")


def assert_sessions_together(sessions: pd.DataFrame, part_of: Mapping[str, str], column: str = "split") -> None:
    """Fail if one participant's sessions fall in different partitions.

    ``sessions`` has one row per (person_id, session); ``part_of`` maps person_id -> partition name, or, when
    empty, the partition is read from ``sessions[column]``.
    """
    parts = sessions.person_id.astype(str).map(part_of) if part_of else sessions[column]
    if parts.isna().any() and part_of:
        missing = sessions.person_id[parts.isna()].unique()[:5]
        raise AdoDASDataError(f"sessions of participants without a partition, e.g. {list(missing)}")
    split_count = pd.DataFrame({"pid": sessions.person_id.astype(str), "part": parts}).groupby("pid").part.nunique()
    broken = split_count[split_count > 1]
    if len(broken):
        raise AdoDASDataError(f"{len(broken)} participant(s) have sessions in different partitions, "
                              f"e.g. {broken.index[:3].tolist()}")


def subject_folds(frame: pd.DataFrame, target: str, n_splits: int, n_repeats: int, seed: int) -> list[dict]:
    """Repeated stratified, participant-grouped K-fold over the rows of ``frame`` (one row per participant).

    Returns positions into ``frame``. Every fold is checked to be subject-disjoint before it is returned.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = frame[target].to_numpy()
    groups = frame.person_id.astype(str).to_numpy()
    if np.isnan(y.astype(float)).any():
        raise AdoDASDataError(f"{target}: missing labels in the rows passed to subject_folds")
    folds = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y.astype(int), groups)):
            assert_subject_disjoint({"train": groups[tr], "val": groups[va]})
            folds.append({"repeat": r, "fold": k, "train": tr, "val": va})
    return folds


# ------------------------------------------------------------------ loader
@dataclass
class AdoDASLoader(DatasetLoader):
    """Participant-level AdoDAS table: ids, split, school, class, DASS-21 items, three binary targets.

    ``target`` picks the label column used by the generic runners (one of depression / anxiety / stress).
    ``splits`` restricts the rows (default: the two labelled splits; test labels are withheld).
    """

    root: str = "data/raw/adodas"
    target: str = "depression"
    splits: tuple[str, ...] = LABELLED_SPLITS
    require_feature_tree: bool = True
    name: str = "AdoDAS-2026"
    label_column: str = "label"
    group_column: str | None = "person_id"
    metadata_columns: tuple[str, ...] = field(default=("person_id", "split", "school_id", "class_id"), init=False)

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        bad = set(self.splits) - set(ALL_SPLITS)
        if bad:
            raise ValueError(f"unknown splits {bad}")
        self.name = f"AdoDAS-2026-{self.target}"

    def _root(self) -> Path:
        root = Path(self.root)
        if not root.is_dir():
            raise AdoDASDataError(f"AdoDAS data not found at {root} — download and unzip the release there")
        return root

    def labels(self) -> pd.DataFrame:
        root, frames, self.extra_columns_ = self._root(), [], {}
        for split in self.splits:
            if split == "test":
                path = root / "test.csv"
                if not path.exists():
                    continue                                       # withheld by the organisers
            else:
                path = root / f"{split}.csv"
                if not path.exists():
                    raise AdoDASDataError(f"missing label file {path}")
            df, extra = read_label_csv(path, split)
            self.extra_columns_[split] = extra
            frames.append(df)
        lab = pd.concat(frames, ignore_index=True)
        assert_subject_disjoint({s: lab.person_id[lab.split == s] for s in lab.split.unique()})
        return lab

    def sessions(self) -> pd.DataFrame:
        sess = scan_feature_tree(self._root())
        if sess.empty and self.require_feature_tree:
            raise AdoDASDataError(f"no participant folders under {self.root}/<split>/<school>/<class>/<pid>/")
        if not sess.empty:
            assert_sessions_together(sess, {}, column="split")
            assert_sessions_together(sess, {}, column="school_id")
            per_pid = sess.drop_duplicates("person_id").groupby("person_id").split.nunique()
            if (per_pid > 1).any():
                raise AdoDASDataError("participant folder found under two splits")
        return sess

    def load(self) -> pd.DataFrame:
        lab = self.labels()
        sess = self.sessions()
        if "test" in self.splits and not (lab.split == "test").any() and not sess.empty:
            test_ids = sess.loc[sess.split == "test", "person_id"].drop_duplicates()
            lab = pd.concat([lab, pd.DataFrame({"person_id": test_ids.to_numpy(), "split": "test"})],
                            ignore_index=True)                    # labels withheld: rows carry NaN targets
            assert_subject_disjoint({s: lab.person_id[lab.split == s] for s in lab.split.unique()})
        if not sess.empty:
            where = sess.drop_duplicates("person_id")[["person_id", "split", "school_id", "class_id"]]
            merged = lab.merge(where, on="person_id", how="left", suffixes=("", "_tree"))
            clash = merged.split_tree.notna() & (merged.split_tree != merged.split)
            if clash.any():
                raise AdoDASDataError(f"{clash.sum()} participants are in a different split in the csv and the "
                                      f"feature tree, e.g. {merged.person_id[clash].head(3).tolist()}")
            if self.require_feature_tree and merged.school_id.isna().any():
                raise AdoDASDataError(f"{merged.school_id.isna().sum()} labelled participants have no feature "
                                      f"folder (no school id), e.g. {merged.person_id[merged.school_id.isna()].head(3).tolist()}")
            lab = merged.drop(columns="split_tree")
        else:
            lab["school_id"], lab["class_id"] = np.nan, np.nan
        lab[self.label_column] = lab[self.target]
        return lab


def class_balance(frame: pd.DataFrame) -> pd.DataFrame:
    """Positive rate of each target per split (and overall), with counts. Read from the data, never assumed."""
    rows = []
    for split, part in [*frame.groupby("split"), ("all", frame)]:
        for t in TARGETS:
            y = part[t].dropna()
            rows.append({"split": split, "target": t, "n": int(len(y)), "positives": int(y.sum()),
                         "positive_rate": float(y.mean()) if len(y) else float("nan")})
    return pd.DataFrame(rows)


def item_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of each response 0-3 per DASS-21 item."""
    return pd.DataFrame({i: frame[i].value_counts(normalize=True).reindex([0, 1, 2, 3]).fillna(0.0) for i in ITEMS}).T


def school_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-school participant counts and label rates (the natural federated clients)."""
    g = frame.groupby("school_id")
    out = g.size().rename("participants").to_frame()
    for t in TARGETS:
        out[f"{t}_rate"] = g[t].mean()
    out["classes"] = g.class_id.nunique()
    for s in ALL_SPLITS:
        out[f"n_{s}"] = frame[frame.split == s].groupby("school_id").size().reindex(out.index).fillna(0).astype(int)
    return out.reset_index()
