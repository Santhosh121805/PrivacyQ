"""Re-identification risk of a released free-text corpus, on the same scale as the NSCH quasi-identifier table.

NSCH (src/m9_audit/reidentification.py): an attacker who knows a person's quasi-identifier values singles them
out with probability 1 / (rows sharing those values) - "prosecutor" risk. The text analogue used here: an attacker
who knows a FRAGMENT of a person's post (k consecutive words - e.g. a sentence they saw quoted, or wrote
themselves elsewhere) searches the corpus for it; the equivalence class is the set of posts that contain the
fragment, so prosecutor risk = 1 / (posts containing it). Same summary fields as the NSCH table: uniqueness,
k-violation rate, average and maximum prosecutor risk.

Also here:
- ``any_unique_ngram_rate``: share of posts with at least one k-word span found in no other post (an attacker who
  knows ANY such span of the post);
- ``scrub``: removes direct identifiers (URLs, e-mails, @/u/ handles, r/ communities, every number);
- ``self_disclosures``: regex counts of posts that state an age, a sex, a U.S. state or a school level - the
  NSCH quasi-identifiers written into the text itself (patterns are a lower bound; matches are counted, not read);
- ``linkage``: nearest-neighbour linkage of a fragment's embedding to the released per-post vectors.

Tokens are lower-cased whitespace tokens of the Phase 1 normalised text. Nothing here stores or prints text.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from src.m1_data.mhtext import normalise


def tokens(texts: Sequence[str]) -> list[list[str]]:
    return [normalise(str(t)).split() for t in texts]


def _grams(tok: list[str], k: int) -> set[int]:
    return {hash(" ".join(tok[i:i + k])) for i in range(len(tok) - k + 1)}


def choose_spans(toks: list[list[str]], k: int, seed: int) -> list[list[str]]:
    """One random k-word span per post (whole post if shorter than k)."""
    rng = np.random.default_rng(seed)
    out = []
    for t in toks:
        if len(t) <= k:
            out.append(list(t))
        else:
            s = int(rng.integers(0, len(t) - k + 1))
            out.append(t[s:s + k])
    return out


def span_class_sizes(toks: list[list[str]], spans: list[list[str]]) -> np.ndarray:
    """Number of posts that contain each span (>= 1: the span's own post)."""
    want: dict[int, dict[int, int]] = {}
    for sp in spans:
        if sp:
            want.setdefault(len(sp), {})[hash(" ".join(sp))] = 0
    for L, counts in want.items():
        for t in toks:
            for h in _grams(t, L) & counts.keys():
                counts[h] += 1
    return np.array([max(1, want[len(sp)][hash(" ".join(sp))]) if sp else len(toks) for sp in spans])


def risk_summary(size: np.ndarray, k: int = 5) -> dict:
    return {"n_rows": int(len(size)), "uniqueness": float(np.mean(size == 1)), "k": k,
            "k_violation_rate": float(np.mean(size < k)), "min_class_size": int(size.min()),
            "median_class_size": float(np.median(size)), "avg_prosecutor_risk": float(np.mean(1.0 / size)),
            "max_prosecutor_risk": float(np.max(1.0 / size))}


def fragment_risk(toks: list[list[str]], span_words: int, seed: int, k: int = 5) -> dict:
    spans = choose_spans(toks, span_words, seed)
    out = risk_summary(span_class_sizes(toks, spans), k)
    out["span_words"] = span_words
    out["posts_shorter_than_span"] = int(sum(len(t) < span_words for t in toks))
    return out


def any_unique_ngram_rate(toks: list[list[str]], n: int) -> float:
    grams = [_grams(t, n) for t in toks]
    count: dict[int, int] = {}
    for g in grams:
        for h in g:
            count[h] = count.get(h, 0) + 1
    return float(np.mean([any(count[h] == 1 for h in g) for g in grams]))


_SCRUB = [
    (re.compile(r"https?://\S+|www\.\S+", re.I), " <url> "),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), " <email> "),
    (re.compile(r"(?<![\w/])@\w+"), " <user> "),
    (re.compile(r"(?<!\w)/?u/\w+", re.I), " <user> "),
    (re.compile(r"(?<!\w)/?r/\w+", re.I), " <community> "),
    (re.compile(r"\d+"), " <num> "),
]


def scrub(text: str) -> str:
    for rx, rep in _SCRUB:
        text = rx.sub(rep, text)
    return text


_AGE = re.compile(
    r"\b(\d{1,2})\s*(?:years?\s*old|yrs?\s*old|y/?o)\b"
    r"|[(\[]\s*(\d{1,2})\s*[mf]\s*[)\]]|[(\[]\s*[mf]\s*(\d{1,2})\s*[)\]]"
    r"|\b(?:i'?m|i am|im)\s+(?:only\s+|just\s+)?(\d{1,2})(?=\s*(?:[,.!?;)]|$|and\b|years?\b|now\b))", re.I)
_SEX = re.compile(r"[(\[]\s*\d{1,2}\s*[mf]\s*[)\]]|[(\[]\s*[mf]\s*\d{1,2}\s*[)\]]"
                  r"|\b(?:i'?m|i am|im|as)\s+an?\s+(?:\d{1,2}\s*(?:years?\s*old|yo)\s+)?"
                  r"(?:girl|boy|guy|woman|man|female|male|dude|lady|teenage girl|teenage boy)\b", re.I)
_STATES = ("Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|"
           "Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|"
           "Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|"
           "North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|"
           "Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming")
_STATE = re.compile(rf"\b(?:{_STATES})\b")                       # case-sensitive: proper nouns only
_SCHOOL = re.compile(r"\b(?:high school|middle school|junior high|\d{1,2}(?:th|st|nd|rd) grade|"
                     r"(?:freshman|sophomore|junior|senior) year)\b", re.I)


def _ages(text: str) -> list[int]:
    return [int(next(g for g in m.groups() if g)) for m in _AGE.finditer(text)]


def self_disclosures(texts: Sequence[str]) -> dict:
    ages = [[a for a in _ages(str(t)) if 10 <= a <= 79] for t in texts]
    has_age = np.array([bool(a) for a in ages])
    teen = np.array([any(12 <= a <= 17 for a in x) for x in ages])
    sex = np.array([bool(_SEX.search(str(t))) for t in texts])
    state = np.array([bool(_STATE.search(str(t))) for t in texts])
    school = np.array([bool(_SCHOOL.search(str(t))) for t in texts])
    return {"n_rows": int(len(texts)), "states_age": float(has_age.mean()), "states_age_12_17": float(teen.mean()),
            "states_sex": float(sex.mean()), "names_us_state": float(state.mean()),
            "mentions_school_level": float(school.mean()), "age_and_sex": float((has_age & sex).mean()),
            "age_sex_and_state": float((has_age & sex & state).mean()),
            "any_of_age_sex_state": float((has_age | sex | state).mean())}


def linkage(queries: np.ndarray, released: np.ndarray, metric: str = "cosine", batch: int = 2048) -> dict:
    """Query i belongs to released row i. Rank of the true row among all released rows (1 = linked)."""
    Q, R = np.asarray(queries, np.float32), np.asarray(released, np.float32)
    if metric == "cosine":
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        R = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    ranks = np.empty(len(Q), int)
    r2 = (R ** 2).sum(1)
    for s in range(0, len(Q), batch):
        q = Q[s:s + batch]
        score = q @ R.T if metric == "cosine" else -(r2[None, :] - 2 * q @ R.T)   # higher = closer
        own = score[np.arange(len(q)), np.arange(s, s + len(q))]
        ranks[s:s + len(q)] = (score > own[:, None]).sum(1) + 1
    return {"n_queries": int(len(Q)), "top1": float(np.mean(ranks == 1)), "top5": float(np.mean(ranks <= 5)),
            "top10": float(np.mean(ranks <= 10)), "median_rank": float(np.median(ranks)),
            "mrr": float(np.mean(1.0 / ranks)), "chance_top1": 1.0 / len(R)}
