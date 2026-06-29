import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.base import Base, engine, SessionLocal
from app.api import auth, documents, chat, quiz

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
app.include_router(quiz.router)


def _rebuild_faiss_if_needed():
    """
    On startup, check if the FAISS index is out of sync with the DB.

    This happens when:
    - docker compose down -v was run (wipes named volumes including faiss_data)
    - The FAISS index file was deleted manually
    - The container was rebuilt without preserving the faiss volume

    In these cases, chunks exist in PostgreSQL but FAISS is empty.
    We detect this and rebuild FAISS from all chunks in the DB.
    """
    from app.retrieval.faiss_store import get_faiss_store
    from app.retrieval.embeddings import get_embeddings_batch
    from app.db.models import Chunk, Document, DocumentStatus

    store = get_faiss_store()
    faiss_count = store.index.ntotal

    db = SessionLocal()
    try:
        db_chunk_count = db.query(Chunk).count()
        done_doc_count = db.query(Document).filter(
            Document.status == DocumentStatus.DONE
        ).count()

        logger.info(
            f"Startup check: FAISS={faiss_count} vectors, "
            f"DB={db_chunk_count} chunks, "
            f"done_docs={done_doc_count}"
        )

        # If FAISS is empty but DB has chunks, rebuild
        if faiss_count == 0 and db_chunk_count > 0:
            logger.warning(
                f"FAISS index is empty but DB has {db_chunk_count} chunks. "
                f"Rebuilding FAISS index from database..."
            )

            # Load all chunks with their document info
            chunks = (
                db.query(Chunk)
                .join(Document)
                .filter(Document.status == DocumentStatus.DONE)
                .order_by(Chunk.id)
                .all()
            )

            if not chunks:
                logger.info("No 'done' chunks found — nothing to rebuild")
                return

            logger.info(f"Rebuilding FAISS from {len(chunks)} chunks...")

            texts = [c.content for c in chunks]
            embeddings = get_embeddings_batch(texts)

            metadata_list = []
            for chunk, embedding in zip(chunks, embeddings):
                doc = chunk.document
                metadata_list.append({
                    "user_id": doc.user_id,
                    "document_id": chunk.document_id,
                    "chunk_db_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "filename": doc.filename,
                    "page_num": chunk.page_num or 1,
                    "heading": chunk.heading or "",
                    "content_preview": chunk.content[:300],
                })

            faiss_ids = store.add_embeddings(embeddings, metadata_list)

            # Update faiss_id back in DB
            for chunk, fid in zip(chunks, faiss_ids):
                chunk.faiss_id = fid
            db.commit()

            logger.info(
                f"FAISS rebuild complete: {len(faiss_ids)} vectors indexed. "
                f"Index now has {store.index.ntotal} total vectors."
            )
        else:
            logger.info(
                f"FAISS index OK: {faiss_count} vectors "
                f"({db_chunk_count} DB chunks)"
            )

    except Exception as e:
        logger.exception(f"FAISS rebuild check failed: {e}")
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    """On startup: check FAISS health and rebuild from DB if needed."""
    try:
        _rebuild_faiss_if_needed()
    except Exception as e:
        logger.error(f"Startup FAISS check failed: {e}")


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