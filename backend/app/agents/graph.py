from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from groq import Groq
from sqlalchemy.orm import Session
from app.core.config import settings
from app.retrieval.embeddings import get_query_embedding
from app.retrieval.faiss_store import get_faiss_store
from app.db.models import Chunk, Document

_client = Groq(api_key=settings.GROQ_API_KEY)


class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_chunks: List[dict]
    answer: str
    sources: List[dict]
    chat_history: List[dict]


def intent_router(state: AgentState) -> AgentState:
    query = state["query"].lower()
    intent = "qa"
    explain_keywords = ["explain", "how does", "what is", "describe", "elaborate", "detail"]
    quiz_keywords = ["quiz", "mcq", "multiple choice", "test me", "questions on", "generate questions"]
    interview_keywords = ["interview", "ask me", "interviewer", "prepare me", "mock interview"]

    if any(k in query for k in quiz_keywords):
        intent = "quiz"
    elif any(k in query for k in interview_keywords):
        intent = "interview"
    elif any(k in query for k in explain_keywords):
        intent = "explain"

    return {**state, "intent": intent}


def retrieval_node(state: AgentState, db: Session) -> AgentState:
    query_embedding = get_query_embedding(state["query"])
    faiss_store = get_faiss_store()
    results = faiss_store.search(query_embedding, top_k=settings.TOP_K_CHUNKS)

    chunks_data = []
    for chunk_db_id, distance in results:
        # Filter out irrelevant chunks using distance threshold
        if distance > 1.5:
            continue
        chunk = db.query(Chunk).filter(Chunk.id == chunk_db_id).first()
        if chunk:
            doc = db.query(Document).filter(Document.id == chunk.document_id).first()
            chunks_data.append({
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "filename": doc.filename if doc else "Unknown",
                "distance": distance,
            })

    return {**state, "retrieved_chunks": chunks_data}


def response_node(state: AgentState) -> AgentState:
    intent = state["intent"]
    query = state["query"]
    chunks = state["retrieved_chunks"]

    history_text = ""
    if state.get("chat_history"):
        recent = state["chat_history"][-6:]
        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])

    if chunks:
        context = "\n\n".join([
            f"[Source: {c['filename']}, Chunk #{c['chunk_index']}]\n{c['content']}"
            for c in chunks
        ])
        context_instruction = (
            "You have relevant content from the student's uploaded study materials. "
            "Base your answer primarily on this content. "
            "At the very beginning of your response, add this line exactly: "
            "'📚 *Answer based on your uploaded documents.*'\n"
        )
    else:
        context = ""
        context_instruction = (
            "No relevant content was found in the student's uploaded study materials for this query. "
            "Answer from your general knowledge. "
            "At the very beginning of your response, add this line exactly: "
            "'🌐 *Answer from general knowledge (not found in your uploaded documents).*'\n"
        )

    system_prompts = {
        "qa": "You are a placement preparation assistant. Give a concise, accurate answer. Be direct and to the point.",
        "explain": "You are a placement preparation tutor. Provide a thorough explanation with: 1) Clear definition 2) How it works 3) Real-world examples 4) Interview tips.",
        "quiz": "You are a placement preparation quiz generator. Create exactly 5 MCQs. Format: Q1. [question]\nA) ...\nB) ...\nC) ...\nD) ...\n\nAt the end: ANSWERS: Q1-X, Q2-X, Q3-X, Q4-X, Q5-X",
        "interview": "You are an interviewer conducting a technical placement interview. Ask 4-5 questions starting easy and increasing difficulty.",
    }

    user_message = f"""{context_instruction}

CONTEXT FROM STUDY MATERIALS:
{context if context else "No relevant content found in uploaded documents."}

{"CHAT HISTORY:" + chr(10) + history_text if history_text else ""}

USER QUERY: {query}

Provide your response now:"""

    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompts.get(intent, system_prompts["qa"])},
            {"role": "user", "content": user_message},
        ],
        max_tokens=2048,
        temperature=0.7,
    )
    answer = response.choices[0].message.content

    sources = [
        {"filename": c["filename"], "chunk_number": c["chunk_index"]}
        for c in chunks
    ]

    return {**state, "answer": answer, "sources": sources}


def build_graph(db: Session):
    workflow = StateGraph(AgentState)
    workflow.add_node("intent_router", intent_router)
    workflow.add_node("retrieval_node", lambda s: retrieval_node(s, db))
    workflow.add_node("response_node", response_node)
    workflow.set_entry_point("intent_router")
    workflow.add_edge("intent_router", "retrieval_node")
    workflow.add_edge("retrieval_node", "response_node")
    workflow.add_edge("response_node", END)
    return workflow.compile()


def run_agent(query: str, chat_history: List[dict], db: Session) -> dict:
    graph = build_graph(db)
    initial_state: AgentState = {
        "query": query,
        "intent": None,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "chat_history": chat_history,
    }
    result = graph.invoke(initial_state)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "intent": result["intent"],
    }