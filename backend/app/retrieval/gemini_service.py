from typing import List, Optional
from groq import Groq
from app.retrieval.embeddings import get_embedding, get_embeddings_batch
from app.core.config import settings

_client = Groq(api_key=settings.GROQ_API_KEY)


class GroqService:
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return get_embeddings_batch(texts)

    async def embed_query(self, text: str) -> List[float]:
        return get_embedding(text)

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = _client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
        )
        return response.choices[0].message.content


gemini_service = GroqService()