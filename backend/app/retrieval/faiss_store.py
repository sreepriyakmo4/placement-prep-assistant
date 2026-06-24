import faiss
import numpy as np
import os
import pickle
import logging
from typing import List, Tuple, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    FAISS vector store with per-user filtering and cosine similarity search.

    Uses IndexFlatIP (inner product) on L2-normalised vectors,
    which is mathematically equivalent to cosine similarity.
    Scores range from 0.0 (no similarity) to 1.0 (identical).
    """

    def __init__(self):
        self.index_path = settings.FAISS_INDEX_PATH
        self.index: faiss.IndexFlatIP = None
        self.metadata: List[Dict[str, Any]] = []
        self._load_or_create()

    def _load_or_create(self):
        os.makedirs(self.index_path, exist_ok=True)
        idx_file = os.path.join(self.index_path, "index.bin")
        meta_file = os.path.join(self.index_path, "metadata.pkl")

        if os.path.exists(idx_file) and os.path.exists(meta_file):
            try:
                self.index = faiss.read_index(idx_file)
                with open(meta_file, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(
                    f"FAISS loaded: {self.index.ntotal} vectors "
                    f"from {self.index_path}"
                )
            except Exception as e:
                logger.error(f"Failed to load FAISS index, creating fresh: {e}")
                self._create_fresh()
        else:
            self._create_fresh()

    def _create_fresh(self):
        # 384-dim for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(384)
        self.metadata = []
        logger.info("FAISS: created fresh IndexFlatIP(384)")

    def save(self):
        try:
            faiss.write_index(
                self.index,
                os.path.join(self.index_path, "index.bin")
            )
            with open(os.path.join(self.index_path, "metadata.pkl"), "wb") as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"FAISS save failed: {e}")
            raise

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]],
    ) -> List[int]:
        """
        Add embeddings + metadata. Normalises vectors to unit length
        so IndexFlatIP gives cosine similarity scores.
        Returns list of FAISS positions assigned.
        """
        if not embeddings:
            return []

        vectors = np.array(embeddings, dtype=np.float32)

        # Normalise to unit vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vectors = vectors / norms

        start_pos = self.index.ntotal
        self.index.add(vectors)

        faiss_ids = list(range(start_pos, start_pos + len(embeddings)))
        self.metadata.extend(metadata_list)
        self.save()

        logger.info(
            f"FAISS: added {len(embeddings)} vectors. "
            f"Total now: {self.index.ntotal}"
        )
        return faiss_ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        user_id: int = None,
        min_score: float = 0.15,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search FAISS index.
        Returns (similarity_score, metadata) tuples sorted best-first.

        min_score=0.15 is intentionally low — short technical questions
        ("what is deadlock?") often score 0.15–0.28 against relevant chunks.
        The LLM prompt handles quality filtering by using only what's relevant.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS search called but index is empty!")
            return []

        vector = np.array([query_embedding], dtype=np.float32)

        # Normalise query vector too
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # Fetch extra candidates when filtering by user to ensure top_k results
        fetch_k = min(
            top_k * 10 if user_id else top_k * 3,
            self.index.ntotal
        )
        scores, indices = self.index.search(vector, fetch_k)

        logger.debug(
            f"FAISS raw search: fetched {fetch_k} candidates, "
            f"top score={scores[0][0]:.4f} for user={user_id}"
        )

        results = []
        seen_previews = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            similarity = float(score)
            if similarity < min_score:
                continue

            meta = self.metadata[idx]

            # Per-user filtering
            if user_id is not None and meta.get("user_id") != user_id:
                continue

            # Deduplicate near-identical chunks
            preview_key = meta.get("content_preview", "")[:80]
            if preview_key in seen_previews:
                continue
            seen_previews.add(preview_key)

            results.append((similarity, meta))

            if len(results) >= top_k:
                break

        results.sort(key=lambda x: x[0], reverse=True)

        logger.info(
            f"FAISS search: user={user_id}, returned {len(results)} chunks, "
            f"scores={[round(r[0],3) for r in results]}"
        )
        return results

    def delete_by_document(self, document_id: int):
        """Rebuild index without vectors from the given document."""
        if self.index.ntotal == 0:
            return

        keep_indices = [
            i for i, m in enumerate(self.metadata)
            if m.get("document_id") != document_id
        ]

        if len(keep_indices) == self.index.ntotal:
            return  # nothing to delete

        if not keep_indices:
            self._create_fresh()
            self.save()
            return

        all_vectors = np.zeros((self.index.ntotal, 384), dtype=np.float32)
        self.index.reconstruct_n(0, self.index.ntotal, all_vectors)

        kept_vectors = all_vectors[keep_indices]
        kept_metadata = [self.metadata[i] for i in keep_indices]

        self._create_fresh()
        if len(kept_vectors) > 0:
            self.index.add(kept_vectors)
            self.metadata = kept_metadata

        self.save()
        logger.info(
            f"FAISS: deleted doc {document_id}, "
            f"{self.index.ntotal} vectors remain"
        )


# ── Singleton ──────────────────────────────────────────────────────────────────

_faiss_store: FAISSStore = None


def get_faiss_store() -> FAISSStore:
    global _faiss_store
    if _faiss_store is None:
        _faiss_store = FAISSStore()
    return _faiss_store