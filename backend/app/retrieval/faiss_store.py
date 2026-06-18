import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Any
from app.core.config import settings


class FAISSStore:
    """
    FAISS store that maps vector index positions to chunk metadata.
    
    Key improvement over the old version:
    - Stores full metadata (user_id, document_id, chunk_index, filename, page_num, heading)
      alongside each vector so retrieval_node never needs a separate DB lookup
      just to know what document a chunk belongs to.
    - Supports per-user filtering so User A's notes never bleed into User B's answers.
    - Returns (distance, metadata) tuples instead of (chunk_db_id, distance).
    """

    def __init__(self):
        self.index_path = settings.FAISS_INDEX_PATH
        self.index: faiss.IndexFlatIP = None   # Inner product on L2-normalised vectors = cosine similarity
        self.metadata: List[Dict[str, Any]] = []  # parallel list to index positions
        self._load_or_create()

    def _load_or_create(self):
        os.makedirs(self.index_path, exist_ok=True)
        idx_file = os.path.join(self.index_path, "index.bin")
        meta_file = os.path.join(self.index_path, "metadata.pkl")

        if os.path.exists(idx_file) and os.path.exists(meta_file):
            self.index = faiss.read_index(idx_file)
            with open(meta_file, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            # 384-dim for all-MiniLM-L6-v2
            # IndexFlatIP with L2-normalised vectors gives cosine similarity
            self.index = faiss.IndexFlatIP(384)
            self.metadata = []

    def save(self):
        faiss.write_index(
            self.index,
            os.path.join(self.index_path, "index.bin")
        )
        with open(os.path.join(self.index_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]],
    ) -> List[int]:
        """
        Add embeddings with their metadata.
        metadata_list items should contain:
          user_id, document_id, chunk_db_id, chunk_index,
          filename, page_num, heading, content_preview
        Returns list of faiss positions assigned.
        """
        vectors = np.array(embeddings, dtype=np.float32)

        # L2-normalise so IndexFlatIP gives cosine similarity (score 0-1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = vectors / norms

        start_pos = self.index.ntotal
        self.index.add(vectors)

        faiss_ids = list(range(start_pos, start_pos + len(embeddings)))
        self.metadata.extend(metadata_list)
        self.save()
        return faiss_ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        user_id: int = None,
        min_score: float = 0.35,   # cosine similarity threshold (0-1)
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search FAISS index.
        Returns list of (similarity_score, metadata) tuples,
        filtered by user_id and min_score, sorted by score descending.
        
        similarity_score is cosine similarity (higher = more relevant).
        0.35 threshold keeps only chunks with at least moderate relevance.
        """
        if self.index.ntotal == 0:
            return []

        vector = np.array([query_embedding], dtype=np.float32)

        # L2-normalise query too
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # Fetch more candidates if filtering by user_id
        fetch_k = min(top_k * 6 if user_id else top_k * 2, self.index.ntotal)
        scores, indices = self.index.search(vector, fetch_k)

        results = []
        seen_content = set()  # deduplicate near-identical chunks

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            similarity = float(score)
            if similarity < min_score:
                continue

            meta = self.metadata[idx]

            # Per-user filtering
            if user_id and meta.get("user_id") != user_id:
                continue

            # Deduplicate: skip if very similar content already added
            preview = meta.get("content_preview", "")[:100]
            if preview in seen_content:
                continue
            seen_content.add(preview)

            results.append((similarity, meta))

            if len(results) >= top_k:
                break

        # Sort by similarity descending (best first)
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def delete_by_document(self, document_id: int):
        """
        Remove all vectors belonging to a document.
        FAISS FlatIP doesn't support in-place deletion,
        so we rebuild the index without those vectors.
        """
        if self.index.ntotal == 0:
            return

        keep_indices = [
            i for i, m in enumerate(self.metadata)
            if m.get("document_id") != document_id
        ]

        if len(keep_indices) == self.index.ntotal:
            return  # nothing to delete

        if not keep_indices:
            self.index = faiss.IndexFlatIP(384)
            self.metadata = []
            self.save()
            return

        # Reconstruct index with only kept vectors
        all_vectors = np.zeros((self.index.ntotal, 384), dtype=np.float32)
        self.index.reconstruct_n(0, self.index.ntotal, all_vectors)

        kept_vectors = all_vectors[keep_indices]
        kept_metadata = [self.metadata[i] for i in keep_indices]

        self.index = faiss.IndexFlatIP(384)
        self.metadata = []

        if len(kept_vectors) > 0:
            self.index.add(kept_vectors)
            self.metadata = kept_metadata

        self.save()


# ── Singleton ──────────────────────────────────────────────────────────────────

_faiss_store: FAISSStore = None


def get_faiss_store() -> FAISSStore:
    global _faiss_store
    if _faiss_store is None:
        _faiss_store = FAISSStore()
    return _faiss_store