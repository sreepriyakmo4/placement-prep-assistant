"""
BM25 keyword-search index -- a classic word-overlap ranking algorithm,
used alongside FAISS (meaning-based search) to catch exact keyword /
acronym matches that pure embedding search can sometimes miss
(e.g. "DDL" vs "Data Definition Language").

This is intentionally simple and standalone right now: it is NOT wired
into ingestion or the agent yet. It rebuilds its index from Postgres
`chunks` on demand, since the dataset size here (hundreds of chunks)
makes that fast and avoids needing a separate persisted index file
like FAISS uses.
"""
import logging
import re
from typing import List, Tuple, Dict, Any

from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer: lowercase, split on non-alphanumeric characters.
    BM25 works on word overlap, so this doesn't need to be fancy -- just
    consistent between indexing and querying.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Store:
    """
    Per-user BM25 keyword index, rebuilt from the `chunks` table.

    Mirrors the shape of FAISSStore (search returns (score, metadata) pairs)
    so it's easy to combine results from both in the agent later.
    """

    def __init__(self, db: Session, user_id: int):
        self.user_id = user_id
        self.chunk_records: List[Dict[str, Any]] = []
        self._build_index(db, user_id)

    def _build_index(self, db: Session, user_id: int):
        rows = (
            db.query(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .filter(Document.user_id == user_id)
            .all()
        )

        self.chunk_records = []
        tokenized_corpus = []

        for chunk, doc in rows:
            self.chunk_records.append({
                "chunk_db_id": chunk.id,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "filename": doc.filename,
                "page_num": chunk.page_num,
                "heading": chunk.heading or "",
                "content": chunk.content,
                "content_preview": chunk.content[:300],
            })
            tokenized_corpus.append(_tokenize(chunk.content))

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

        logger.info(
            f"BM25Store built for user={user_id}: {len(self.chunk_records)} chunks indexed"
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Returns (score, metadata) pairs, sorted best-first -- same shape as
        FAISSStore.search() so results from both can be merged later.

        BM25 scores are NOT bounded 0-1 like FAISS cosine similarity --
        they're relative rank scores, only meaningfully comparable to each
        other within the same query, not across different queries.
        """
        if not self.bm25 or not self.chunk_records:
            logger.warning("BM25 search called but index is empty!")
            return []

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        scored_chunks = list(zip(scores, self.chunk_records))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        top_results = scored_chunks[:top_k]

        results = [(float(score), meta) for score, meta in top_results if score > 0]

        logger.info(
            f"BM25 search: user={self.user_id}, returned {len(results)} chunks, "
            f"scores={[round(r[0], 3) for r in results]}"
        )
        return results