"""
LangGraph placement preparation agent.

Key improvements:
- Per-user FAISS filtering (your notes only answer your questions)
- Confidence scoring based on cosine similarity
- Prompts that enforce exact wording from teacher's notes
- Graceful fallback to general knowledge when no relevant chunks found
"""
import logging
from typing import TypedDict, List, Optional, Dict, Any

from langgraph.graph import StateGraph, END
from groq import Groq

from app.core.config import settings
from app.retrieval.embeddings import get_query_embedding
from app.retrieval.faiss_store import get_faiss_store
from app.db.models import Chunk

logger = logging.getLogger(__name__)

_client = Groq(api_key=settings.GROQ_API_KEY)


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    user_id: int
    intent: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]  # {content, similarity, metadata}
    answer: str
    sources: List[Dict]
    chat_history: List[Dict]
    db: Any   # SQLAlchemy session (passed through state)


# ── Node 1: Intent Router ──────────────────────────────────────────────────────

def intent_router(state: AgentState) -> AgentState:
    query = state["query"].lower()

    quiz_kw      = ["quiz", "mcq", "multiple choice", "test me", "questions on", "generate questions", "practice questions"]
    interview_kw = ["interview", "ask me", "interviewer", "prepare me", "mock interview", "conduct interview"]
    explain_kw   = ["explain", "how does", "how do", "what is", "describe", "elaborate", "detail", "tell me about", "walk me through"]

    if any(k in query for k in quiz_kw):
        intent = "quiz"
    elif any(k in query for k in interview_kw):
        intent = "interview"
    elif any(k in query for k in explain_kw):
        intent = "explain"
    else:
        intent = "qa"

    return {**state, "intent": intent}


# ── Node 2: Retrieval ──────────────────────────────────────────────────────────

def retrieval_node(state: AgentState) -> AgentState:
    db = state.get("db")
    user_id = state.get("user_id")

    try:
        query_embedding = get_query_embedding(state["query"])
        faiss_store = get_faiss_store()

        # Search with per-user filtering and cosine similarity threshold
        results = faiss_store.search(
            query_embedding,
            top_k=settings.TOP_K_CHUNKS,
            user_id=user_id,
            min_score=0.30,   # only chunks with ≥30% cosine similarity
        )

        chunks_data = []
        for similarity, meta in results:
            chunk_db_id = meta.get("chunk_db_id")
            content = None

            # Fetch full content from DB (preview is only 300 chars)
            if db and chunk_db_id:
                chunk = db.query(Chunk).filter(Chunk.id == chunk_db_id).first()
                if chunk:
                    content = chunk.content

            if not content:
                content = meta.get("content_preview", "")

            if not content:
                continue

            # Confidence label based on cosine similarity
            if similarity >= 0.75:
                confidence = "Very High"
            elif similarity >= 0.60:
                confidence = "High"
            elif similarity >= 0.45:
                confidence = "Moderate"
            else:
                confidence = "Low"

            chunks_data.append({
                "content": content,
                "similarity": round(similarity, 4),
                "confidence": confidence,
                "filename": meta.get("filename", "Unknown"),
                "page_num": meta.get("page_num", "?"),
                "heading": meta.get("heading", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "document_id": meta.get("document_id"),
                "chunk_db_id": chunk_db_id,
            })

        logger.info(
            f"Retrieval for user {user_id}: query='{state['query'][:60]}' "
            f"→ {len(chunks_data)} chunks (best similarity: "
            f"{chunks_data[0]['similarity'] if chunks_data else 'N/A'})"
        )

    except Exception as e:
        logger.exception(f"Retrieval failed: {e}")
        chunks_data = []

    return {**state, "retrieved_chunks": chunks_data}


# ── Node 3: Response Generator ─────────────────────────────────────────────────

# Core instruction repeated in every prompt so the LLM never forgets it
FIDELITY_INSTRUCTION = """
CRITICAL INSTRUCTIONS — follow these without exception:
1. Your answer MUST be based on the CONTEXT FROM STUDY MATERIALS provided below.
2. Use the EXACT phrases, terminology, and definitions from the context — do not paraphrase or substitute synonyms.
3. If the context uses a specific term (e.g. "mutual exclusion", "semaphore"), use that exact term.
4. Quote sentences from the context directly when they are the clearest way to answer.
5. Preserve the structure from the notes — if the material has numbered points or bullet lists, reflect that structure.
6. If the context does NOT contain enough information to answer confidently, say so explicitly, then supplement from general knowledge clearly labelled as "[General Knowledge]".
7. Never invent definitions or examples that are not in the context.
""".strip()

SYSTEM_BASE = (
    "You are a placement preparation assistant helping a student study from their own teacher's notes and study materials. "
    "Your job is to retrieve and present exactly what is in those notes — not to rephrase or improve them. "
    "Students need the exact wording their teacher used so they can match exam answers precisely."
)

PROMPTS = {
    "qa": """{fidelity}

CONTEXT FROM STUDY MATERIALS:
{context}

PREVIOUS CONVERSATION:
{history}

STUDENT QUESTION: {query}

Answer using the exact wording and definitions from the study materials above.
If the answer is directly stated in the materials, quote or closely paraphrase it.
Start your answer with the most relevant sentence from the context.""",

    "explain": """{fidelity}

CONTEXT FROM STUDY MATERIALS:
{context}

PREVIOUS CONVERSATION:
{history}

TOPIC TO EXPLAIN: {query}

Provide a structured explanation using the study materials. Follow this format:

**Definition** (exact wording from notes):
[quote or closely follow the definition in the context]

**How It Works** (from your notes):
[explain using the structure and terms in the context]

**Key Points** (from your notes):
[list the main points exactly as they appear in the material]

**Example** (from notes or clearly labelled as [General Knowledge]):
[example]

**What to Say in an Interview**:
[based on the above notes, what phrasing should the student use?]""",

    "quiz": """{fidelity}

CONTEXT FROM STUDY MATERIALS:
{context}

PREVIOUS CONVERSATION:
{history}

TOPIC: {query}

Generate exactly 5 MCQs. Each question MUST be answerable from the context above.
The correct answer should use the exact wording from the study material.

Format:

**Quiz: [Topic Name]**

**Q1.** [Question based on the context]
A) [Option]
B) [Option]  
C) [Option]
D) [Option]

[Repeat for Q2-Q5]

---
**ANSWERS:**
Q1: [Letter] — [Exact phrase from notes that confirms this answer]
[Repeat for Q2-Q5]""",

    "interview": """{fidelity}

CONTEXT FROM STUDY MATERIALS:
{context}

PREVIOUS CONVERSATION:
{history}

INTERVIEW REQUEST: {query}

Conduct a mock interview. Base the expected answers on the study materials provided.

**Technical Interview Simulation**

*[Brief intro as interviewer]*

**Q1 (Warm-up):** [Question]
✅ *Strong answer (from your notes):* [What a good answer should say, using exact terms from context]

**Q2 (Core concept):** [Question]
✅ *Strong answer (from your notes):* [...]

**Q3 (Application):** [Question]
✅ *Strong answer (from your notes):* [...]

**Q4 (Deep dive):** [Question]
✅ *Strong answer (from your notes):* [...]

**Q5 (Scenario):** [Question]
✅ *Strong answer (from your notes):* [...]

---
**Study Tip:** The key terms your interviewer will listen for: [list exact terms from the context]""",
}


def response_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "qa")
    chunks = state.get("retrieved_chunks", [])
    query = state["query"]

    # Build history string (last 3 exchanges = 6 messages)
    history_parts = []
    for msg in (state.get("chat_history") or [])[-6:]:
        role = msg.get("role", "user").capitalize()
        history_parts.append(f"{role}: {msg['content']}")
    history = "\n".join(history_parts) if history_parts else "No previous conversation."

    # Build context and source list
    if chunks:
        context_parts = []
        for c in chunks:
            heading = f"[{c['heading']}] " if c.get("heading") else ""
            context_parts.append(
                f"📄 Source: {c['filename']} | Page {c['page_num']} | "
                f"Relevance: {c['confidence']} ({c['similarity']:.0%})\n"
                f"{heading}{c['content']}"
            )
        context = "\n\n---\n\n".join(context_parts)
        prefix = "📚 *Answer based on your uploaded study materials.*\n\n"
    else:
        context = "No relevant content found in the uploaded study materials for this query."
        prefix = (
            "🌐 *No matching content found in your uploaded documents. "
            "Answering from general knowledge — please verify against your notes.*\n\n"
        )

    prompt_template = PROMPTS.get(intent, PROMPTS["qa"])
    prompt = prompt_template.format(
        fidelity=FIDELITY_INSTRUCTION,
        context=context,
        history=history,
        query=query,
    )

    try:
        response = _client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_BASE},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.2,   # low temperature = more faithful to source material
        )
        raw_answer = response.choices[0].message.content
        answer = prefix + raw_answer
    except Exception as e:
        logger.exception(f"Groq generation failed: {e}")
        answer = "⚠️ Error generating response. Please check your GROQ_API_KEY and try again."

    sources = [
        {
            "filename": c["filename"],
            "page_num": c["page_num"],
            "chunk_index": c["chunk_index"],
            "heading": c["heading"],
            "similarity": c["similarity"],
            "confidence": c["confidence"],
            "content_preview": c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
        }
        for c in chunks
    ]

    return {**state, "answer": answer, "sources": sources}


# ── Build Graph ────────────────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("intent_router", intent_router)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("response_node", response_node)
    workflow.set_entry_point("intent_router")
    workflow.add_edge("intent_router", "retrieval_node")
    workflow.add_edge("retrieval_node", "response_node")
    workflow.add_edge("response_node", END)
    return workflow.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(
    query: str,
    chat_history: List[Dict],
    db,
    user_id: int,
) -> Dict[str, Any]:
    graph = get_graph()
    initial_state: AgentState = {
        "query": query,
        "user_id": user_id,
        "intent": None,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "chat_history": chat_history,
        "db": db,
    }
    result = graph.invoke(initial_state)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "intent": result["intent"],
    }