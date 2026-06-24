import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.base import Base, engine
from app.api import auth, documents, chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# Create all DB tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Placement Preparation Assistant API",
    description="AI-powered placement preparation with RAG and LangGraph",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.on_event("startup")
def on_startup():
    """Log FAISS state on startup so we can confirm index is loaded."""
    try:
        from app.retrieval.faiss_store import get_faiss_store
        store = get_faiss_store()
        logger.info(
            f"FAISS index ready: {store.index.ntotal} vectors loaded "
            f"from {store.index_path}"
        )
    except Exception as e:
        logger.error(f"FAISS startup check failed: {e}")


@app.get("/health")
def health():
    try:
        from app.retrieval.faiss_store import get_faiss_store
        store = get_faiss_store()
        return {
            "status": "ok",
            "faiss_vectors": store.index.ntotal,
        }
    except Exception as e:
        return {"status": "ok", "faiss_error": str(e)}