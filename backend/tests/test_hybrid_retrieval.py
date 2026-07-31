"""
Tests for hybrid retrieval in app/agents/graph.py — specifically the
BM25 supplementation logic in `_do_retrieval`.

Covers:
  - BM25 results that duplicate an existing FAISS chunk_db_id are skipped
  - BM25 results that FAISS missed are added with the fixed supplement
    score and "Keyword Match" confidence tag
  - BM25 failures degrade gracefully (FAISS-only results still returned)
  - BM25 supplement chunks are correctly capped below FINAL_RELEVANCE_THRESHOLD
    so they can never pass the strict gate on their own
"""
from unittest.mock import patch

import pytest

from app.agents import graph as g
import app.retrieval.bm25_store as bm25_module


def _fake_faiss_chunk(chunk_db_id, similarity=0.5, filename="notes.pdf"):
    """Mimics one entry of the chunks_data list built inside _do_retrieval
    from FAISS results, before BM25 supplementation runs."""
    return {
        "content": f"faiss content {chunk_db_id}",
        "similarity": similarity,
        "confidence": "Moderate",
        "filename": filename,
        "page_num": 1,
        "heading": "",
        "chunk_index": 0,
        "document_id": 1,
        "chunk_db_id": chunk_db_id,
    }


def _fake_bm25_result(chunk_db_id, content="bm25 content", filename="notes.pdf"):
    """Mimics one (score, meta) tuple returned by BM25Store.search()."""
    meta = {
        "chunk_db_id": chunk_db_id,
        "content": content,
        "filename": filename,
        "page_num": 2,
        "heading": "",
        "chunk_index": 5,
        "document_id": 1,
    }
    return (12.4, meta)  # BM25 score is unbounded/arbitrary — never used directly


# ── BM25 supplements chunks FAISS missed ─────────────────────────────────────

def test_bm25_adds_chunk_faiss_missed():
    """
    A chunk BM25 finds that FAISS did NOT surface should be added to the
    results, tagged with the fixed supplement score and 'Keyword Match'.
    """
    state = {"db": object(), "user_id": 1, "intent": "qa"}

    with patch.object(g, "get_query_embedding", return_value=[0.0] * 384), \
         patch.object(g, "get_faiss_store") as mock_faiss_store, \
         patch.object(g, "_do_bm25_search", return_value=[_fake_bm25_result(chunk_db_id=999)]):

        mock_faiss_store.return_value.search.return_value = []  # FAISS found nothing

        results = g._do_retrieval("DDL", state)

    assert len(results) == 1
    assert results[0]["chunk_db_id"] == 999
    assert results[0]["confidence"] == "Keyword Match"
    assert results[0]["similarity"] == g.BM25_SUPPLEMENT_SCORE


# ── BM25 does NOT duplicate a chunk FAISS already found ──────────────────────

def test_bm25_does_not_duplicate_existing_faiss_chunk():
    """
    If BM25 returns a chunk_db_id that FAISS already surfaced, it must be
    skipped — not appended as a second, lower-confidence duplicate entry.
    """
    state = {"db": object(), "user_id": 1, "intent": "qa"}

    with patch.object(g, "get_query_embedding", return_value=[0.0] * 384), \
         patch.object(g, "get_faiss_store") as mock_faiss_store, \
         patch.object(g, "_do_bm25_search", return_value=[_fake_bm25_result(chunk_db_id=42)]):

        # Simulate FAISS having already returned this chunk by pre-seeding
        # what _do_retrieval would have built from faiss_store.search results.
        mock_faiss_store.return_value.search.return_value = [
            (0.6, {
                "chunk_db_id": 42,
                "filename": "notes.pdf",
                "page_num": 1,
                "heading": "",
                "chunk_index": 0,
                "document_id": 1,
                "content_preview": "faiss content 42",
            })
        ]

        results = g._do_retrieval("DDL", state)

    # Only ONE entry for chunk_db_id 42 — not two
    matching = [r for r in results if r["chunk_db_id"] == 42]
    assert len(matching) == 1
    # And it should be the FAISS version (real similarity), not the BM25
    # supplement score, proving FAISS stayed authoritative.
    assert matching[0]["similarity"] == 0.6
    assert matching[0]["confidence"] != "Keyword Match"


# ── BM25 failure degrades gracefully ──────────────────────────────────────────

def test_bm25_failure_falls_back_to_faiss_only():
    """
    If BM25 search throws for any reason (empty index, import error, etc.),
    retrieval must still return the FAISS results rather than crashing the
    whole query.
    """
    state = {"db": object(), "user_id": 1, "intent": "qa"}

    with patch.object(g, "get_query_embedding", return_value=[0.0] * 384), \
         patch.object(g, "get_faiss_store") as mock_faiss_store, \
         patch.object(bm25_module, "BM25Store", side_effect=RuntimeError("index corrupt")):

        mock_faiss_store.return_value.search.return_value = [
            (0.6, {
                "chunk_db_id": 1,
                "filename": "notes.pdf",
                "page_num": 1,
                "heading": "",
                "chunk_index": 0,
                "document_id": 1,
                "content_preview": "faiss content 1",
            })
        ]

        # _do_bm25_search itself catches this internally and returns [],
        # so we call the real function (not mocked) to test that path.
        results = g._do_retrieval("DDL", state)

    assert len(results) == 1
    assert results[0]["chunk_db_id"] == 1


# ── BM25 supplement score can never pass the strict final relevance gate ────

def test_bm25_supplement_score_below_final_threshold():
    """
    Regression guard: BM25_SUPPLEMENT_SCORE must stay below
    FINAL_RELEVANCE_THRESHOLD, otherwise a keyword-only match could pass
    response_node's strict gate and get presented as high-confidence,
    which defeats the purpose of tagging it separately as 'Keyword Match'.
    """
    assert g.BM25_SUPPLEMENT_SCORE < g.FINAL_RELEVANCE_THRESHOLD
    # And it should still clear the softer REWRITE_THRESHOLD, so a
    # keyword-only hit doesn't trigger an unnecessary query rewrite retry.
    assert g.BM25_SUPPLEMENT_SCORE >= g.REWRITE_THRESHOLD


# ── Results stay sorted after merging FAISS + BM25 ────────────────────────────

def test_combined_results_sorted_by_similarity_descending():
    state = {"db": object(), "user_id": 1, "intent": "qa"}

    with patch.object(g, "get_query_embedding", return_value=[0.0] * 384), \
         patch.object(g, "get_faiss_store") as mock_faiss_store, \
         patch.object(g, "_do_bm25_search", return_value=[_fake_bm25_result(chunk_db_id=999)]):

        mock_faiss_store.return_value.search.return_value = [
            (0.9, {
                "chunk_db_id": 1,
                "filename": "notes.pdf",
                "page_num": 1,
                "heading": "",
                "chunk_index": 0,
                "document_id": 1,
                "content_preview": "high relevance faiss hit",
            })
        ]

        results = g._do_retrieval("DDL", state)

    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True)