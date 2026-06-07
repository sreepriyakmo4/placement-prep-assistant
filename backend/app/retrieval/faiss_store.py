import faiss
import numpy as np
import os
import pickle
from typing import List, Tuple
from app.core.config import settings


class FAISSStore:
    def __init__(self):
        self.index_path = settings.FAISS_INDEX_PATH
        self.index: faiss.IndexFlatL2 = None
        self.id_map: List[int] = []
        self._load_or_create()

    def _load_or_create(self):
        os.makedirs(self.index_path, exist_ok=True)
        idx_file = os.path.join(self.index_path, "index.bin")
        map_file = os.path.join(self.index_path, "id_map.pkl")
        if os.path.exists(idx_file) and os.path.exists(map_file):
            self.index = faiss.read_index(idx_file)
            with open(map_file, "rb") as f:
                self.id_map = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(384)  # MiniLM dim
            self.id_map = []

    def save(self):
        faiss.write_index(self.index, os.path.join(self.index_path, "index.bin"))
        with open(os.path.join(self.index_path, "id_map.pkl"), "wb") as f:
            pickle.dump(self.id_map, f)

    def add_embeddings(self, embeddings: List[List[float]], chunk_db_ids: List[int]) -> List[int]:
        vectors = np.array(embeddings, dtype=np.float32)
        start_pos = self.index.ntotal
        self.index.add(vectors)
        faiss_ids = list(range(start_pos, start_pos + len(embeddings)))
        self.id_map.extend(chunk_db_ids)
        self.save()
        return faiss_ids

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(vector, min(top_k, self.index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.id_map):
                results.append((self.id_map[idx], float(dist)))
        return results


_faiss_store: FAISSStore = None

def get_faiss_store() -> FAISSStore:
    global _faiss_store
    if _faiss_store is None:
        _faiss_store = FAISSStore()
    return _faiss_store