import json
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.agents.graph import run_agent, run_agent_stream
from app.core.limiter import limiter
from app.db.base import get_db
from app.db.models import Message, Session as ChatSession, User

router = APIRouter(prefix="/chat", tags=["chat"])


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[int] = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class SessionOut(BaseModel):
    id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    intent: str
    session_id: int


@router.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
def query(
    request: Request,
    body: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Get or create session
    if body.session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == body.session_id,
            ChatSession.user_id == current_user.id,
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

    else:
        title = body.query[:50] + ("..." if len(body.query) > 50 else "")
        session = ChatSession(user_id=current_user.id, title=title)
        db.add(session)
        db.commit()
        db.refresh(session)

    # Build chat history
    existing_messages = (
        db.query(Message)
        .filter(Message.session_id == session.id)
        .order_by(Message.created_at)
        .all()
    )

    chat_history = [
        {"role": m.role, "content": m.content}
        for m in existing_messages
    ]

    # Save user message
    user_msg = Message(
        session_id=session.id,
        role="user",
        content=body.query,
    )
    db.add(user_msg)
    db.commit()

    # Run agent
    result = run_agent(
        query=body.query,
        chat_history=chat_history,
        db=db,
        user_id=current_user.id,
    )

    # Save assistant message
    assistant_msg = Message(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        sources=json.dumps(result["sources"]),
    )
    db.add(assistant_msg)
    db.commit()

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        intent=result["intent"],
        session_id=session.id,
    )


@router.post("/query/stream")
@limiter.limit("10/minute")
def query_stream(
    request: Request,
    body: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Server-Sent Events version of /chat/query.
    """

    if body.session_id:
        session = db.query(ChatSession).filter(
            ChatSession.id == body.session_id,
            ChatSession.user_id == current_user.id,
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

    else:
        title = body.query[:50] + ("..." if len(body.query) > 50 else "")
        session = ChatSession(
            user_id=current_user.id,
            title=title,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    existing_messages = (
        db.query(Message)
        .filter(Message.session_id == session.id)
        .order_by(Message.created_at)
        .all()
    )

    chat_history = [
        {"role": m.role, "content": m.content}
        for m in existing_messages
    ]

    user_msg = Message(
        session_id=session.id,
        role="user",
        content=body.query,
    )
    db.add(user_msg)
    db.commit()

    session_id = session.id
    query_text = body.query
    user_id = current_user.id

    def event_generator():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        final_answer = ""
        final_sources = []
        final_intent = "qa"

        for event in run_agent_stream(
            query=query_text,
            chat_history=chat_history,
            db=db,
            user_id=user_id,
        ):
            if event.get("type") == "done":
                final_answer = event.get("answer", "")
                final_sources = event.get("sources", [])
                final_intent = event.get("intent", "qa")

            yield f"data: {json.dumps(event)}\n\n"

        assistant_msg = Message(
            session_id=session_id,
            role="assistant",
            content=final_answer,
            sources=json.dumps(final_sources),
        )
        db.add(assistant_msg)
        db.commit()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions", response_model=List[SessionOut])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )

    return sessions


@router.get("/sessions/{session_id}", response_model=List[MessageOut])
def get_session_messages(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at)
        .all()
    )

    return messages


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id,
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    db.delete(session)
    db.commit()

    return {"message": "Deleted"}