"""
RAG Retrieval Evaluation Script
Measures: did FAISS return a chunk from the correct document for each question?

Run from the backend/ folder:
    python -m eval.run_retrieval_eval
"""

import json
import sys
import os

# So we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval.embeddings import get_query_embedding
from app.retrieval.faiss_store import get_faiss_store

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
TOP_K = 5
USER_ID = 1  # your user id in the app


def run_eval():
    # Load golden set
    with open(GOLDEN_SET_PATH, "r") as f:
        golden_set = json.load(f)

    store = get_faiss_store()
    print(f"\n{'='*60}")
    print(f"RAG RETRIEVAL EVALUATION")
    print(f"Total questions: {len(golden_set)}")
    print(f"Checking top-{TOP_K} retrieved chunks per question")
    print(f"{'='*60}\n")

    passed = 0
    failed = 0
    failed_questions = []

    for i, item in enumerate(golden_set):
        question = item["question"]
        expected_filename = item["source_filename"]

        # Get embedding for the question
        embedding = get_query_embedding(question)

        # Search FAISS
        results = store.search(
            embedding,
            top_k=TOP_K,
            user_id=USER_ID,
            min_score=0.0,  # no threshold, we want to see all results
        )

        # Check if any of the top-K results came from the correct file
        retrieved_filenames = [meta.get("filename", "") for _, meta in results]
        hit = any(expected_filename in fname for fname in retrieved_filenames)

        status = "✅ PASS" if hit else "❌ FAIL"
        if hit:
            passed += 1
        else:
            failed += 1
            failed_questions.append(question)

        # Show top result score for context
        top_score = round(results[0][0], 3) if results else 0
        print(f"Q{i+1}: {status} | score={top_score} | {question[:55]}")

    total = passed + failed
    precision = round((passed / total) * 100, 1)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Passed:     {passed}/{total}")
    print(f"Failed:     {failed}/{total}")
    print(f"Precision@{TOP_K}: {precision}%")

    if failed_questions:
        print(f"\nFailed questions:")
        for q in failed_questions:
            print(f"  - {q}")

    print(f"\n{'='*60}")
    
    print(f"RESUME METRIC: Retrieval Precision@{TOP_K} = {precision}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_eval()