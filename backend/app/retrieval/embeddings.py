import logging
from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return _model


def get_query_embedding(text: str) -> List[float]:
    """
    Embed a single query string for FAISS search.
    normalize_embeddings=True produces unit vectors so that
    IndexFlatIP (inner product) equals cosine similarity.
    """
    model = get_model()
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.tolist()


# Keep get_embedding as alias so any existing imports still work
get_embedding = get_query_embedding


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of document chunks for indexing.
    batch_size=32 is efficient for CPU inference.
    normalize_embeddings=True so FAISS inner product = cosine similarity.
    """
    if not texts:
        return []

    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    logger.info(f"Embedded {len(texts)} chunks, shape={embeddings.shape}")
    return embeddings.tolist()