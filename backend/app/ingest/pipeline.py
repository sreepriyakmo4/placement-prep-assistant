import logging
from app.db.base import SessionLocal
from app.db.models import Document, Chunk, DocumentStatus
from app.retrieval.embeddings import get_embeddings_batch
from app.retrieval.faiss_store import get_faiss_store
from .pdf_processor import extract_text_from_pdf, split_into_chunks

logger = logging.getLogger(__name__)


def ingest_document(document_id: int, file_bytes: bytes):
    """
    Full ingestion pipeline — creates its own DB session.

    WHY: FastAPI's background tasks run after the response is sent.
    The db session from get_db() is already CLOSED by then (the generator's
    finally block runs). Passing a closed session to the background task
    causes silent failures — chunks never saved, FAISS never populated,
    document stays stuck on 'pending'.

    Fix: create a fresh SessionLocal() here and own its lifecycle.
    """
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            logger.error(f"Document {document_id} not found in DB")
            return

        doc.status = DocumentStatus.PROCESSING
        db.commit()

        # Step 1: Extract text page-by-page
        pages = extract_text_from_pdf(file_bytes)
        if not pages:
            logger.error(f"No text extracted from document {document_id}")
            doc.status = DocumentStatus.FAILED
            db.commit()
            return

        # Step 2: Semantic chunking
        chunk_dicts = split_into_chunks(pages)
        if not chunk_dicts:
            logger.error(f"No chunks produced for document {document_id}")
            doc.status = DocumentStatus.FAILED
            db.commit()
            return

        logger.info(f"Document {document_id}: {len(pages)} pages → {len(chunk_dicts)} chunks")

        # Step 3: Save chunks to PostgreSQL
        chunk_objects = []
        for chunk_dict in chunk_dicts:
            chunk = Chunk(
                document_id=document_id,
                chunk_index=chunk_dict["chunk_index"],
                content=chunk_dict["content"],
                page_num=chunk_dict.get("page_num", 1),
                heading=chunk_dict.get("heading", ""),
                faiss_id=None,
            )
            db.add(chunk)
            chunk_objects.append((chunk, chunk_dict))

        db.commit()
        for chunk_obj, _ in chunk_objects:
            db.refresh(chunk_obj)

        # Step 4: Embed all chunks in batch
        texts = [cd["content"] for _, cd in chunk_objects]
        embeddings = get_embeddings_batch(texts)

        # Step 5: Build metadata and store in FAISS
        metadata_list = []
        for (chunk_obj, chunk_dict), _ in zip(chunk_objects, embeddings):
            content = chunk_dict["content"]
            metadata_list.append({
                "user_id": doc.user_id,
                "document_id": document_id,
                "chunk_db_id": chunk_obj.id,
                "chunk_index": chunk_obj.chunk_index,
                "filename": doc.filename,
                "page_num": chunk_dict.get("page_num", 1),
                "heading": chunk_dict.get("heading", ""),
                "content_preview": content[:300],
            })

        faiss_store = get_faiss_store()
        faiss_ids = faiss_store.add_embeddings(embeddings, metadata_list)

        # Update faiss_id back in DB
        for (chunk_obj, _), fid in zip(chunk_objects, faiss_ids):
            chunk_obj.faiss_id = fid

        doc.status = DocumentStatus.DONE
        db.commit()
        logger.info(
            f"Document {document_id} ingestion complete: "
            f"{len(chunk_objects)} chunks indexed in FAISS"
        )

    except Exception as e:
        logger.exception(f"Ingestion failed for document {document_id}: {e}")
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.status = DocumentStatus.FAILED
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
