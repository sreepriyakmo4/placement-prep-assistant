from typing import List
import google.generativeai as genai
from app.core.config import settings

genai.configure(api_key=settings.GOOGLE_API_KEY)


def get_embedding(text: str) -> List[float]:
    result = genai.embed_content(
        model=settings.EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]


def get_query_embedding(text: str) -> List[float]:
    result = genai.embed_content(
        model=settings.EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result["embedding"]


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for text in texts:
        embeddings.append(get_embedding(text))
    return embeddings
