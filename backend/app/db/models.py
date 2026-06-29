from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Enum, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from .base import Base


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sessions = relationship("Session", back_populates="user", cascade="all, delete")
    documents = relationship("Document", back_populates="user", cascade="all, delete")
    quiz_attempts = relationship("QuizAttempt", back_populates="user", cascade="all, delete")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String, nullable=False)
    status = Column(String, default=DocumentStatus.PENDING)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete")
    quiz_questions = relationship("QuizQuestion", back_populates="document", cascade="all, delete")
    quiz_attempts = relationship("QuizAttempt", back_populates="document", cascade="all, delete")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_num = Column(Integer, nullable=True)      # which PDF page this chunk came from
    heading = Column(String, nullable=True)         # nearest heading above this chunk
    faiss_id = Column(Integer, nullable=True)

    document = relationship("Document", back_populates="chunks")


class QuizQuestion(Base):
    """Stores generated MCQ questions for a document"""
    __tablename__ = "quiz_questions"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    question = Column(Text, nullable=False)
    options = Column(Text, nullable=False)  # JSON: ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"]
    correct_answer = Column(String, nullable=False)  # "A", "B", "C", or "D"
    topic = Column(String, nullable=False)  # e.g. "Constraints", "Joins", "Indexing"
    explanation = Column(Text, nullable=True)  # Why is this answer correct
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="quiz_questions")


class QuizAttempt(Base):
    """Tracks each time a user takes a quiz on a document"""
    __tablename__ = "quiz_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    score = Column(Integer, nullable=False)  # e.g. 13 (out of 15)
    total_questions = Column(Integer, default=15)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="quiz_attempts")
    document = relationship("Document", back_populates="quiz_attempts")
    answers = relationship("QuizAnswer", back_populates="attempt", cascade="all, delete")


class QuizAnswer(Base):
    """Stores user's answer to each question in a quiz attempt"""
    __tablename__ = "quiz_answers"

    id = Column(Integer, primary_key=True, index=True)
    attempt_id = Column(Integer, ForeignKey("quiz_attempts.id", ondelete="CASCADE"), nullable=False)
    question_id = Column(Integer, ForeignKey("quiz_questions.id"), nullable=False)
    user_answer = Column(String, nullable=False)  # "A", "B", "C", or "D"
    is_correct = Column(Boolean, nullable=False)
    topic = Column(String, nullable=False)  # denormalized for easy filtering

    attempt = relationship("QuizAttempt", back_populates="answers")