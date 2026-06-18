import logging
from sqlalchemy.orm import Session
from app.db.models import Document, Chunk, DocumentStatus
from app.retrieval.embeddings import get_embeddings_batch
from app.retrieval.faiss_store import get_faiss_store
from .pdf_processor import extract_text_from_pdf, split_into_chunks

logger = logging.getLogger(__name__)


def ingest_document(db: Session, document_id: int, file_bytes: bytes):
    """
    Full ingestion pipeline:
    1. Extract text page-by-page (preserving page numbers)
    2. Semantic chunking (paragraph-aware, heading-aware, with overlap)
    3. Save chunks to PostgreSQL
    4. Embed chunks in batch
    5. Store embeddings + rich metadata in FAISS
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        return

    try:
        doc.status = DocumentStatus.PROCESSING
        db.commit()

        # Step 1: Extract
        pages = extract_text_from_pdf(file_bytes)
        if not pages:
            logger.error(f"No text extracted from document {document_id}")
            doc.status = DocumentStatus.FAILED
            db.commit()
            return

        # Step 2: Chunk semantically
        chunk_dicts = split_into_chunks(pages)
        if not chunk_dicts:
            logger.error(f"No chunks produced for document {document_id}")
            doc.status = DocumentStatus.FAILED
            db.commit()
            return

        logger.info(f"Document {document_id}: {len(pages)} pages → {len(chunk_dicts)} chunks")

        # Step 3: Save to DB
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

        # Step 4: Embed
        texts = [cd["content"] for _, cd in chunk_objects]
        embeddings = get_embeddings_batch(texts)

        # Step 5: Build metadata list for FAISS
        metadata_list = []
        for (chunk_obj, chunk_dict), embedding in zip(chunk_objects, embeddings):
            content = chunk_dict["content"]
            metadata_list.append({
                "user_id": doc.user_id,
                "document_id": document_id,
                "chunk_db_id": chunk_obj.id,
                "chunk_index": chunk_obj.chunk_index,
                "filename": doc.filename,
                "page_num": chunk_dict.get("page_num", 1),
                "heading": chunk_dict.get("heading", ""),
                # Store first 300 chars as preview so the agent can
                # show a snippet without a DB round-trip
                "content_preview": content[:300],
            })

        # Store in FAISS
        faiss_store = get_faiss_store()
        faiss_ids = faiss_store.add_embeddings(embeddings, metadata_list)

        # Update faiss_id in DB
        for (chunk_obj, _), fid in zip(chunk_objects, faiss_ids):
            chunk_obj.faiss_id = fid

        doc.status = DocumentStatus.DONE
        db.commit()
        logger.info(f"Document {document_id} ingestion complete: {len(chunk_objects)} chunks indexed")

    except Exception as e:
        logger.exception(f"Ingestion failed for document {document_id}: {e}")
        doc.status = DocumentStatus.FAILED
        db.commit()
        raise e