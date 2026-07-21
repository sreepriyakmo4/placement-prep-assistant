from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, EmailStr


# ─── Auth ────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    full_name: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# ─── Documents ────────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: int
    filename: str
    original_filename: str
    status: str
    chunk_count: int
    file_size: Optional[int]
    error_message: Optional[str]
    uploaded_at: datetime

    class Config:
        from_attributes = True


# ─── Sessions & Messages ──────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    title: Optional[str] = "New Chat"


class SessionOut(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Source(BaseModel):
    filename: str
    chunk_index: int
    document_id: int
    chunk_id: int
    content_preview: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    intent: Optional[str]
    sources: Optional[List[Source]]
    created_at: datetime

    class Config:
        from_attributes = True


class SessionDetail(BaseModel):
    id: int
    title: str
    created_at: datetime
    messages: List[MessageOut]

    class Config:
        from_attributes = True


# ─── Chat ────────────────────────────────────────────────────────────────────

class ChatQuery(BaseModel):
    query: str
    session_id: Optional[int] = None
    mode: Optional[str] = None  # override auto-detection


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: List[Source]
    session_id: int
    message_id: int
