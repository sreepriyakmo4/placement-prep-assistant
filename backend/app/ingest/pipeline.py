from sqlalchemy.orm import Session
from app.db.models import Document, Chunk, DocumentStatus
from app.retrieval.embeddings import get_embeddings_batch
from app.retrieval.faiss_store import get_faiss_store
from .pdf_processor import extract_text_from_pdf, split_text


def ingest_document(db: Session, document_id: int, file_bytes: bytes):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        return

    try:
        doc.status = DocumentStatus.PROCESSING
        db.commit()

        raw_text = extract_text_from_pdf(file_bytes)
        chunks = split_text(raw_text)

        if not chunks:
            doc.status = DocumentStatus.FAILED
            db.commit()
            return

        # Save chunks to DB first to get IDs
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(
                document_id=document_id,
                chunk_index=i,
                content=chunk_text,
                faiss_id=None,
            )
            db.add(chunk)
            chunk_objects.append(chunk)
        db.commit()
        for c in chunk_objects:
            db.refresh(c)

        # Generate embeddings
        embeddings = get_embeddings_batch(chunks)

        # Store in FAISS
        faiss_store = get_faiss_store()
        chunk_db_ids = [c.id for c in chunk_objects]
        faiss_ids = faiss_store.add_embeddings(embeddings, chunk_db_ids)

        # Update faiss_id in DB
        for chunk_obj, fid in zip(chunk_objects, faiss_ids):
            chunk_obj.faiss_id = fid
        doc.status = DocumentStatus.DONE
        db.commit()

    except Exception as e:
        doc.status = DocumentStatus.FAILED
        db.commit()
        raise e
