from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # all-MiniLM-L6-v2: fast, 384-dim, great for semantic similarity
        # If you want higher accuracy at the cost of speed, swap to:
        # "BAAI/bge-small-en-v1.5"  (still 384-dim, better retrieval quality)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def get_embedding(text: str) -> List[float]:
    """Embed a single text (used for query embedding at search time)."""
    model = get_model()
    # prompt_name="query" tells the model this is a search query,
    # not a document — improves retrieval for asymmetric search tasks.
    # Falls back gracefully if model doesn't support it.
    try:
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            prompt_name="query",
            normalize_embeddings=True,   # pre-normalise for cosine sim
        )
    except TypeError:
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return embedding.tolist()


def get_query_embedding(text: str) -> List[float]:
    """Alias for clarity — used when embedding a user query."""
    return get_embedding(text)


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of document chunks.
    Uses batch encoding for efficiency and normalises embeddings.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,   # pre-normalise for cosine sim
    )
    return embeddings.tolist()