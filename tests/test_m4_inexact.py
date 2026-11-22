"""M4 Phase 2: inexact top-K (m4_rank/inexact.py). Needs M1's real index and the frozen artefact; skipped when they are missing."""

from __future__ import annotations

import pytest

from common.config import ROOT

INDEX = ROOT / "data" / "processed" / "index" / "index.pkl.gz"
ARTEFACT = ROOT / "data" / "processed" / "inexact.pkl.gz"
pytestmark = pytest.mark.skipif(not (INDEX.exists() and ARTEFACT.exists()),
                                reason="needs the real index and `python -m m4_rank.inexact build`")

QUERIES = ["punishment for murder under section 103 BNS", "presumption of dowry death BNS 80", "offence of sedition",
           "anticipatory bail protection limited to a fixed period section 438", "section 80"]


@pytest.fixture(scope="module")
def ix():
    from m4_rank import inexact

    return inexact


@pytest.mark.parametrize("query", QUERIES)
def test_exact_method_reproduces_m1_search_bit_for_bit(ix, query):
    from m1_index import search

    a, b = search(query, 50), ix.inexact_search(query, 50, "exact")
    assert [h.doc_id for h in a] == [h.doc_id for h in b]
    assert [h.rel for h in a] == [h.rel for h in b]
    assert [h.zone_scores for h in a] == [h.zone_scores for h in b]


@pytest.mark.parametrize("method", ["champion", "tiered", "elimination", "cluster"])
def test_methods_score_fewer_judgments_and_keep_the_hit_contract(ix, method):
    query = QUERIES[0]
    ix.inexact_search(query, 20, "exact")
    exact_work = ix.last_stats().postings_scored
    hits = ix.inexact_search(query, 20, method)
    st = ix.last_stats()
    assert hits and len(hits) <= 20
    for h in hits:
        h.validate()
    assert [h.rel for h in hits] == sorted((h.rel for h in hits), reverse=True)
    assert st.fallback == "" and st.postings_scored < exact_work


def test_scores_on_a_subset_equal_the_exact_scores(ix):
    """Pruning changes WHICH judgments are scored, never their score (unless terms are eliminated)."""
    exact = {h.doc_id: h.rel for h in ix.inexact_search(QUERIES[1], 5000, "exact")}
    for method in ("champion", "tiered", "cluster"):
        for h in ix.inexact_search(QUERIES[1], 50, method):
            assert h.rel == exact[h.doc_id]


def test_operators_and_filters_fall_back_to_exact_search(ix):
    from m1_index import search

    q = '"dying declaration" AND murder'
    assert [h.doc_id for h in ix.inexact_search(q, 10, "champion")] == [h.doc_id for h in search(q, 10)]
    assert ix.last_stats().fallback.startswith("query operators")
    ix.inexact_search("murder", 10, "champion", filters={"bench_size": 3})
    assert ix.last_stats().fallback == "filters"


def test_elimination_drops_low_idf_terms(ix):
    ix.inexact_search(QUERIES[0], 20, "elimination")
    assert "section" not in ix.last_stats().terms


def test_search_fn_has_search_signature_and_rejects_unknown(ix):
    fn = ix.search_fn("champion")
    assert fn("offence of sedition", k=3)
    with pytest.raises(ValueError):
        ix.search_fn("nope")
    assert ix.inexact_search("   ", 10, "tiered") == []
