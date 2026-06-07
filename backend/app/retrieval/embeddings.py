from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model

def get_embedding(text: str) -> List[float]:
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

def get_query_embedding(text: str) -> List[float]:
    return get_embedding(text)

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()