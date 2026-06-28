from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from datetime import datetime
from app.db.base import get_db
from app.db.models import Document, User
from app.api.deps import get_current_user
from app.ingest.pipeline import ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentOut(BaseModel):
    id: int
    filename: str
    status: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()

    doc = Document(user_id=current_user.id, filename=file.filename, status="pending")
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # CRITICAL FIX: pass only document_id + file_bytes, NOT the db session.
    # The request's db session is closed before the background task runs.
    # ingest_document now creates its own fresh session internally.
    background_tasks.add_task(ingest_document, doc.id, file_bytes)

    return doc


@router.get("", response_model=List[DocumentOut])
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    docs = (
        db.query(Document)
        .filter(Document.user_id == current_user.id)
        .order_by(Document.uploaded_at.desc())
        .all()
    )
    return docs


@router.delete("/{doc_id}")
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == current_user.id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove vectors from FAISS before deleting from DB
    try:
        from app.retrieval.faiss_store import get_faiss_store
        get_faiss_store().delete_by_document(doc_id)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"FAISS delete failed for doc {doc_id}: {e}")

    db.delete(doc)
    db.commit()
    return {"message": "Deleted"}

