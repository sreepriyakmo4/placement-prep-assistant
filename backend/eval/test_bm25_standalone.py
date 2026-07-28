"""
Standalone smoke test for BM25Store -- run this BEFORE wiring BM25 into
anything else, just to confirm the index builds and searches correctly
against your real database.

This does NOT touch ingestion, the agent, or any existing app code.
It's throwaway/manual -- safe to delete once you've confirmed it works.

Run from the backend/ folder (or inside the backend container):
    python -m eval.test_bm25_standalone
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.base import SessionLocal
from app.retrieval.bm25_store import BM25Store

USER_ID = 1  # same user_id your eval script uses

TEST_QUERIES = [
    "What is a primary key?",
    "DDL",
    "subquery",
    "GROUP BY HAVING",
]


def run_test():
    db = SessionLocal()
    try:
        store = BM25Store(db, USER_ID)

        print(f"\n{'='*60}")
        print(f"BM25 STANDALONE TEST")
        print(f"Indexed chunks: {len(store.chunk_records)}")
        print(f"{'='*60}\n")

        for query in TEST_QUERIES:
            print(f"Query: '{query}'")
            results = store.search(query, top_k=3)
            if not results:
                print("      (no results)")
            for rank, (score, meta) in enumerate(results, start=1):
                preview = meta["content_preview"][:120].replace("\n", " ")
                print(f"      rank {rank} | score={round(score, 3)} | {meta['filename']}")
                print(f"                 {preview}...")
            print()
    finally:
        db.close()


if __name__ == "__main__":
    run_test()