"""
Shared pytest fixtures.

Run from the `backend/` folder:
    cd backend
    pytest -v

This conftest also stubs out heavy optional dependencies (sentence-transformers
model download, real Groq network calls) so the suite runs fast and offline.
"""
import os
import sys
import types
import pytest

# Make `app.*` importable regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Provide harmless defaults so `app.core.config.Settings()` doesn't blow up
# if no .env / real API key is present in CI.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-do-not-use-in-prod-12345")


@pytest.fixture
def make_chunk():
    """Factory for a fake retrieved-chunk dict matching graph.py's shape."""
    def _make(similarity=0.5, content="Some study note content.", filename="notes.pdf",
              page_num=1, heading="Intro", chunk_index=0, chunk_db_id=1, document_id=1,
              confidence="Moderate"):
        return {
            "content": content,
            "similarity": round(similarity, 4),
            "confidence": confidence,
            "filename": filename,
            "page_num": page_num,
            "heading": heading,
            "chunk_index": chunk_index,
            "document_id": document_id,
            "chunk_db_id": chunk_db_id,
        }
    return _make