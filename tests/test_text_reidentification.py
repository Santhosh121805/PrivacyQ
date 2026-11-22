import numpy as np

from src.m9_audit import text_reidentification as tr


def test_fragment_class_sizes_count_posts_containing_span():
    toks = tr.tokens(["the cat sat on the mat", "a cat sat on a hat", "dogs run fast"])
    sizes = tr.span_class_sizes(toks, [["cat", "sat", "on"], ["the", "mat"], ["dogs", "run", "fast"]])
    assert sizes.tolist() == [2, 1, 1]


def test_repeated_span_in_one_post_counts_once():
    toks = tr.tokens(["go go go go", "stop"])
    assert tr.span_class_sizes(toks, [["go", "go"]]).tolist() == [1]


def test_risk_summary_matches_nsch_definitions():
    s = tr.risk_summary(np.array([1, 1, 2, 2, 10]), k=5)
    assert s["uniqueness"] == 0.4 and s["k_violation_rate"] == 0.8
    assert np.isclose(s["avg_prosecutor_risk"], (1 + 1 + 0.5 + 0.5 + 0.1) / 5) and s["max_prosecutor_risk"] == 1.0


def test_short_posts_use_whole_post():
    spans = tr.choose_spans(tr.tokens(["one two", "a b c d e f g"]), 5, seed=0)
    assert spans[0] == ["one", "two"] and len(spans[1]) == 5


def test_any_unique_ngram():
    toks = tr.tokens(["a b c", "a b c", "a b d"])
    assert np.isclose(tr.any_unique_ngram_rate(toks, 2), 1 / 3)


def test_scrub_removes_direct_identifiers():
    s = tr.scrub("I'm 16, see https://x.com/a or mail me@ex.com, @bob u/alice r/depression")
    for leaked in ("16", "x.com", "me@ex.com", "@bob", "alice", "r/depression"):
        assert leaked not in s


def test_self_disclosures():
    d = tr.self_disclosures(["I (16F) live in Texas", "i am 15 and in high school", "nothing here", "I'm 20 minutes late"])
    assert d["states_age"] == 0.5 and d["states_age_12_17"] == 0.5
    assert d["states_sex"] == 0.25 and d["names_us_state"] == 0.25 and d["mentions_school_level"] == 0.25


def test_linkage_ranks():
    R = np.eye(4)
    res = tr.linkage(R + 0.01, R, "euclidean")
    assert res["top1"] == 1.0
    res = tr.linkage(R[[1, 0, 2, 3]], R, "cosine")
    assert res["top1"] == 0.5
