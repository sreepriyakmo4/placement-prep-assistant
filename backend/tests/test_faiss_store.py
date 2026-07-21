"""
Tests for app/retrieval/faiss_store.py — specifically the per-user filtering
and deduplication logic in `search()`, since a bug here is a real data-leak
risk (one user seeing another user's notes) rather than just a quality bug.
"""
import numpy as np
import pytest

from app.retrieval.faiss_store import FAISSStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh, isolated FAISSStore backed by a temp directory per test."""
    from app.core import config
    monkeypatch.setattr(config.settings, "FAISS_INDEX_PATH", str(tmp_path))
    s = FAISSStore()
    return s


def _random_unit_vector(dim=384, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


def test_search_filters_by_user_id(store):
    vec_a = _random_unit_vector(seed=1)
    vec_b = _random_unit_vector(seed=2)

    store.add_embeddings(
        [vec_a, vec_b],
        [
            {"user_id": 1, "filename": "user1_notes.pdf", "content_preview": "alpha"},
            {"user_id": 2, "filename": "user2_notes.pdf", "content_preview": "beta"},
        ],
    )

    # Query with a vector identical to user 2's — user 1 must never see it.
    results = store.search(vec_b, top_k=5, user_id=1, min_score=0.0)
    filenames = [meta["filename"] for _, meta in results]

    assert "user2_notes.pdf" not in filenames


def test_search_returns_own_data(store):
    vec_a = _random_unit_vector(seed=1)
    store.add_embeddings([vec_a], [{"user_id": 1, "filename": "mine.pdf", "content_preview": "x"}])

    results = store.search(vec_a, top_k=5, user_id=1, min_score=0.0)
    assert len(results) == 1
    assert results[0][1]["filename"] == "mine.pdf"


def test_search_respects_min_score(store):
    vec = _random_unit_vector(seed=1)
    store.add_embeddings([vec], [{"user_id": 1, "filename": "f.pdf", "content_preview": "x"}])

    # A near-orthogonal query vector should score far below a high min_score.
    orthogonal_query = _random_unit_vector(seed=42)
    results = store.search(orthogonal_query, top_k=5, user_id=1, min_score=0.99)
    assert results == []


def test_search_deduplicates_near_identical_chunks(store):
    vec = _random_unit_vector(seed=1)
    # Two chunks with an identical content_preview prefix (e.g. produced by
    # overlapping chunking) should be deduplicated to one result.
    store.add_embeddings(
        [vec, vec],
        [
            {"user_id": 1, "filename": "f.pdf", "content_preview": "duplicate paragraph text here"},
            {"user_id": 1, "filename": "f.pdf", "content_preview": "duplicate paragraph text here"},
        ],
    )
    results = store.search(vec, top_k=5, user_id=1, min_score=0.0)
    assert len(results) == 1


def test_search_on_empty_index_returns_empty_list(store):
    vec = _random_unit_vector(seed=1)
    assert store.search(vec, top_k=5, user_id=1) == []


def test_delete_by_document_removes_only_that_documents_vectors(store):
    vec_a = _random_unit_vector(seed=1)
    vec_b = _random_unit_vector(seed=2)

    store.add_embeddings(
        [vec_a, vec_b],
        [
            {"user_id": 1, "document_id": 100, "filename": "doc100.pdf", "content_preview": "a"},
            {"user_id": 1, "document_id": 200, "filename": "doc200.pdf", "content_preview": "b"},
        ],
    )

    store.delete_by_document(100)

    remaining_filenames = [m["filename"] for m in store.metadata]
    assert remaining_filenames == ["doc200.pdf"]
    assert store.index.ntotal == 1