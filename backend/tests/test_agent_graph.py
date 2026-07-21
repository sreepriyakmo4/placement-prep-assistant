"""
Tests for app/agents/graph.py — specifically the conditional-retry and
relevance-gating logic, since that's the part of the system most likely to
silently regress (e.g. someone tweaks a threshold and breaks the fallback
behavior without noticing).

Covers:
  - route_after_retrieval: the conditional edge that decides retry vs. respond
  - query_rewrite_node: LLM rewrite with graceful fallback to original query
  - response_node: the *second*, stricter relevance gate that decides whether
    to actually use retrieved chunks or fall back to general knowledge
  - intent_router: keyword-based intent classification
"""
from unittest.mock import MagicMock, patch

import pytest

from app.agents import graph as g


# ── intent_router ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected_intent", [
    ("What is deadlock?", "qa"),
    ("Explain how the OSI model works", "explain"),
    ("Generate a quiz on sorting algorithms", "quiz"),
    ("Conduct a mock interview for SDE role", "interview"),
    ("Difference between TCP and UDP", "explain"),   # "difference between" is an explain keyword
    ("asdkjfh random gibberish query", "qa"),          # default fallback
])
def test_intent_router_classifies_correctly(query, expected_intent):
    state = {"query": query}
    result = g.intent_router(state)
    assert result["intent"] == expected_intent


# ── route_after_retrieval ─────────────────────────────────────────────────────

def test_route_after_retrieval_no_chunks_triggers_rewrite():
    state = {"retrieved_chunks": []}
    assert g.route_after_retrieval(state) == "query_rewrite"


def test_route_after_retrieval_low_score_triggers_rewrite(make_chunk):
    # top score below REWRITE_THRESHOLD (0.30)
    state = {"retrieved_chunks": [make_chunk(similarity=0.29)]}
    assert g.route_after_retrieval(state) == "query_rewrite"


def test_route_after_retrieval_high_score_goes_to_response(make_chunk):
    # top score at/above REWRITE_THRESHOLD (0.30)
    state = {"retrieved_chunks": [make_chunk(similarity=0.30)]}
    assert g.route_after_retrieval(state) == "response_node"


def test_route_after_retrieval_uses_top_chunk_only(make_chunk):
    # First chunk is low, but if chunks aren't sorted this test would catch
    # a regression where a *later* high-scoring chunk is mistakenly used
    # instead of chunks[0].
    state = {"retrieved_chunks": [
        make_chunk(similarity=0.10),
        make_chunk(similarity=0.95),
    ]}
    assert g.route_after_retrieval(state) == "query_rewrite"


# ── query_rewrite_node ────────────────────────────────────────────────────────

def test_query_rewrite_node_uses_llm_output():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="deadlock definition OS"))]

    with patch.object(g._client.chat.completions, "create", return_value=fake_response):
        state = {"query": "can you please tell me what is a deadlock"}
        result = g.query_rewrite_node(state)

    assert result["rewritten_query"] == "deadlock definition OS"
    assert result["was_rewritten"] is True


def test_query_rewrite_node_falls_back_to_original_on_empty_llm_output():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=""))]

    with patch.object(g._client.chat.completions, "create", return_value=fake_response):
        state = {"query": "what is deadlock"}
        result = g.query_rewrite_node(state)

    # Empty/garbage LLM output must not silently become the search query —
    # it should fall back to the user's original text.
    assert result["rewritten_query"] == "what is deadlock"


def test_query_rewrite_node_falls_back_to_original_on_exception():
    with patch.object(g._client.chat.completions, "create", side_effect=RuntimeError("groq down")):
        state = {"query": "what is deadlock"}
        result = g.query_rewrite_node(state)

    assert result["rewritten_query"] == "what is deadlock"
    assert result["was_rewritten"] is True  # still flagged, even though the rewrite itself failed


def test_query_rewrite_node_rejects_overlong_output():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="x" * 500))]

    with patch.object(g._client.chat.completions, "create", return_value=fake_response):
        state = {"query": "what is deadlock"}
        result = g.query_rewrite_node(state)

    assert result["rewritten_query"] == "what is deadlock"


# ── response_node: the second relevance gate ─────────────────────────────────

def _mock_llm_generate(text="Mocked answer."):
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=text))]
    return fake_response


def test_response_node_discards_chunks_below_final_threshold(make_chunk):
    """
    Even if route_after_retrieval let a chunk through the *first* threshold
    (0.30) without a retry, response_node applies a stricter *second*
    threshold (0.40). Anything below that must be discarded and treated as
    a general-knowledge fallback, not presented as 'from your notes'.
    """
    state = {
        "query": "some query",
        "intent": "qa",
        "retrieved_chunks": [make_chunk(similarity=0.35)],  # between the two thresholds
        "chat_history": [],
        "was_rewritten": False,
    }
    with patch.object(g._client.chat.completions, "create", return_value=_mock_llm_generate()):
        result = g.response_node(state)

    assert result["sources"] == []  # chunks were discarded
    assert "general knowledge" in result["answer"].lower() or "🌐" in result["answer"]


def test_response_node_keeps_chunks_above_final_threshold(make_chunk):
    state = {
        "query": "some query",
        "intent": "qa",
        "retrieved_chunks": [make_chunk(similarity=0.85)],
        "chat_history": [],
        "was_rewritten": False,
    }
    with patch.object(g._client.chat.completions, "create", return_value=_mock_llm_generate()):
        result = g.response_node(state)

    assert len(result["sources"]) == 1
    assert "uploaded study materials" in result["answer"].lower()


def test_response_node_flags_rewritten_query_in_answer(make_chunk):
    state = {
        "query": "some query",
        "intent": "qa",
        "retrieved_chunks": [make_chunk(similarity=0.85)],
        "chat_history": [],
        "was_rewritten": True,
        "rewritten_query": "better search terms",
    }
    with patch.object(g._client.chat.completions, "create", return_value=_mock_llm_generate()):
        result = g.response_node(state)

    assert "rephrased" in result["answer"].lower()


def test_response_node_handles_llm_failure_gracefully(make_chunk):
    state = {
        "query": "some query",
        "intent": "qa",
        "retrieved_chunks": [make_chunk(similarity=0.85)],
        "chat_history": [],
        "was_rewritten": False,
    }
    with patch.object(g._client.chat.completions, "create", side_effect=RuntimeError("groq down")):
        result = g.response_node(state)

    assert "error" in result["answer"].lower()


# ── Regression guard on the thresholds themselves ────────────────────────────

def test_final_threshold_is_stricter_than_rewrite_threshold():
    """
    This is a cheap but high-value guard: if someone 'simplifies' the code
    to a single threshold later, this test fails loudly instead of silently
    changing the fallback behavior described in the README/interview notes.
    """
    assert g.FINAL_RELEVANCE_THRESHOLD > g.REWRITE_THRESHOLD