from google import genai
from google.genai import types
from app.core.config import settings

client = genai.Client(api_key=settings.GOOGLE_API_KEY)

def get_embedding(text: str):
    result = client.models.embed_content(
        model="models/text-embedding-004",
        contents=text
    )
    return result.embeddings[0].values

def get_query_embedding(text: str):
    return get_embedding(text)

def get_embeddings_batch(texts):
    return [get_embedding(t) for t in texts]