import logging
import sys

from pydantic import field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_INSECURE_DEFAULT_SECRET = "change-me-in-production"


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/placement_prep"
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24
    GROQ_API_KEY: str
    HF_API_KEY: str = ""
    FAISS_INDEX_PATH: str = "./faiss_index"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K_CHUNKS: int = 5
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @field_validator("SECRET_KEY")
    @classmethod
    def secret_key_must_be_set(cls, v: str) -> str:
        if not v or v == _INSECURE_DEFAULT_SECRET:
            raise ValueError(
                "SECRET_KEY is not set (or still uses the insecure default). "
                "Set a real value in your .env file, e.g.:\n"
                "    SECRET_KEY=$(python -c \"import secrets; print(secrets.token_hex(32))\")\n"
                "Refusing to start with a public/guessable JWT signing key."
            )
        if len(v) < 32:
            raise ValueError(
                f"SECRET_KEY is only {len(v)} characters — use at least 32 "
                "random hex/base64 characters (`python -c \"import secrets; "
                "print(secrets.token_hex(32))\"`)."
            )
        return v

    @field_validator("GROQ_API_KEY")
    @classmethod
    def groq_api_key_must_be_set(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(GROQ_API_KEY=gsk_...). Without it every chat/quiz request "
                "would fail at call time instead of at startup."
            )
        return v

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    try:
        return Settings()
    except Exception as e:
        logger.critical("Configuration error — refusing to start.\n%s", e)
        print(
            f"\n{'=' * 70}\nCONFIGURATION ERROR — the app will not start:\n\n{e}\n{'=' * 70}\n",
            file=sys.stderr,
        )
        sys.exit(1)


settings = get_settings()