"""
LangGraph placement preparation agent.

- Per-user FAISS filtering (your notes only answer your questions)
- Intent-aware retrieval: explain/quiz/interview fetch MORE chunks for full coverage
- Comprehensive explain prompt: covers ALL types/items from the document
- Confidence scoring based on cosine similarity
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
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    sources: List[Dict]
    chat_history: List[Dict]
    db: Any


# ── Node 1: Intent Router ──────────────────────────────────────────────────────

def intent_router(state: AgentState) -> AgentState:
    query = state["query"].lower()

    quiz_kw      = ["quiz", "mcq", "multiple choice", "test me", "questions on",
                    "generate questions", "practice questions"]
    interview_kw = ["interview", "ask me", "interviewer", "prepare me",
                    "mock interview", "conduct interview"]
    explain_kw   = ["explain", "how does", "how do", "what is", "describe",
                    "elaborate", "detail", "tell me about", "walk me through",
                    "types of", "list", "difference between", "compare",
                    "advantages", "disadvantages", "features of"]

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

# How many chunks to retrieve per intent.
# explain/quiz/interview need MORE chunks for comprehensive coverage.
# e.g. "types of OS" may span 6-8 chunks — we need all of them.
INTENT_TOP_K = {
    "qa":        5,
    "explain":   12,   # fetch up to 12 chunks so no type/item is missed
    "quiz":      10,   # need broad coverage to generate good MCQs
    "interview": 10,   # need multiple topics for varied interview questions
}


def retrieval_node(state: AgentState) -> AgentState:
    db = state.get("db")
    user_id = state.get("user_id")
    intent = state.get("intent", "qa")

    # Use intent-specific top_k for broader or narrower retrieval
    top_k = INTENT_TOP_K.get(intent, settings.TOP_K_CHUNKS)

    try:
        query_embedding = get_query_embedding(state["query"])
        faiss_store = get_faiss_store()

        results = faiss_store.search(
            query_embedding,
            top_k=top_k,
            user_id=user_id,
            min_score=0.15,
        )

        chunks_data = []
        for similarity, meta in results:
            chunk_db_id = meta.get("chunk_db_id")
            content = None

            # Fetch full content from DB
            if db and chunk_db_id:
                try:
                    chunk = db.query(Chunk).filter(Chunk.id == chunk_db_id).first()
                    if chunk:
                        content = chunk.content
                except Exception as e:
                    logger.warning(f"DB lookup failed for chunk {chunk_db_id}: {e}")

            if not content:
                content = meta.get("content_preview", "")

            if not content:
                continue

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
            f"Retrieval intent={intent} user={user_id} "
            f"query='{state['query'][:60]}' "
            f"→ {len(chunks_data)}/{top_k} chunks "
            f"(best={chunks_data[0]['similarity'] if chunks_data else 'none'})"
        )

    except Exception as e:
        logger.exception(f"Retrieval failed: {e}")
        chunks_data = []

    return {**state, "retrieved_chunks": chunks_data}


# ── Node 3: Response Generator ─────────────────────────────────────────────────

FIDELITY_INSTRUCTION = """
CRITICAL INSTRUCTIONS — follow these without exception:
1. Your answer MUST be based ENTIRELY on the CONTEXT FROM STUDY MATERIALS below.
2. Use the EXACT phrases, terminology, and definitions from the context.
3. COVER EVERYTHING in the context that is relevant to the question — do NOT skip any type, category, or item mentioned.
4. If the context lists N types/categories/methods, your answer must address ALL N of them.
5. Quote or closely paraphrase sentences from the context — do not substitute synonyms.
6. Preserve structure: if notes have numbered points or bullet lists, reflect that structure.
7. If the context does NOT contain enough information, say so explicitly, then add [General Knowledge] clearly labelled.
8. Never invent definitions or examples not in the context.
""".strip()

SYSTEM_BASE = (
    "You are a placement preparation assistant helping a student study from their own "
    "teacher's notes and study materials. Your job is to retrieve and present EVERYTHING "
    "that is in those notes relevant to the question — not just a summary. "
    "Students need complete coverage of all types, categories, and details their teacher wrote "
    "so they can give full answers in exams and interviews."
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

Provide a COMPLETE and COMPREHENSIVE explanation using ALL content from the study materials above.

IMPORTANT: If the context mentions multiple types, categories, methods, or items — you MUST cover EVERY SINGLE ONE with at least a sentence each. Do not summarise or skip any.

Structure your response as follows:

**Overview / Definition** (exact wording from notes):
[Give the definition or overview exactly as stated in the context]

**[List every type / category / method mentioned — use the exact heading from notes]**

For EACH type/category found in the context, write:
- **[Name]**: [Full explanation from the notes — at least 2-3 sentences per item]

**Key Points to Remember** (from your notes):
[Bullet list of the most important facts, using exact phrasing from context]

**What to Say in an Interview**:
[Exact terminology and phrasing the student should use, drawn from the context]""",

    "quiz": """{fidelity}

CONTEXT FROM STUDY MATERIALS:
{context}

PREVIOUS CONVERSATION:
{history}

TOPIC: {query}

Generate exactly 5 MCQs. Each question MUST be directly answerable from the context above.
Cover DIFFERENT aspects from the context — do not repeat the same concept twice.
The correct answer must use the exact wording from the study material.

Format:

**Quiz: [Topic Name]**

**Q1.** [Question based on the context]
A) [Option]
B) [Option]
C) [Option]
D) [Option]

[Repeat for Q2–Q5]

---
**ANSWERS:**
Q1: [Letter] — [Exact phrase from notes that confirms this answer]
[Repeat for Q2–Q5]""",

    "interview": """{fidelity}

CONTEXT FROM STUDY MATERIALS:
{context}

PREVIOUS CONVERSATION:
{history}

INTERVIEW REQUEST: {query}

Conduct a mock interview. Base ALL expected answers on the study materials provided.
Cover different aspects of the topic across the 5 questions.

**Technical Interview Simulation**

*[Brief intro as interviewer]*

**Q1 (Warm-up):** [Question]
✅ *Strong answer (from your notes):* [What a good answer should say — use exact terms from context]

**Q2 (Core concept):** [Question]
✅ *Strong answer (from your notes):* [...]

**Q3 (Types/Categories):** [Question about specific types or categories from the notes]
✅ *Strong answer (from your notes):* [Cover ALL relevant types from the context]

**Q4 (Deep dive):** [Question]
✅ *Strong answer (from your notes):* [...]

**Q5 (Application/Scenario):** [Question]
✅ *Strong answer (from your notes):* [...]

---
**Study Tip:** Key terms your interviewer will listen for: [list exact terms from context]""",
}


def response_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "qa")
    chunks = state.get("retrieved_chunks", [])
    query = state["query"]

    history_parts = []
    for msg in (state.get("chat_history") or [])[-6:]:
        role = msg.get("role", "user").capitalize()
        history_parts.append(f"{role}: {msg['content']}")
    history = "\n".join(history_parts) if history_parts else "No previous conversation."

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
            max_tokens=4096,   # increased from 2048 — comprehensive answers need space
            temperature=0.2,
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
            # Increased from 200 to 400 chars so source badge shows meaningful context
            "content_preview": (
                c["content"][:400] + "..."
                if len(c["content"]) > 400
                else c["content"]
            ),
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

