"""
LangGraph placement preparation agent.

Graph structure:
    intent_router
        ↓
    retrieval_node
        ↓
    route_after_retrieval  ← CONDITIONAL EDGE (new)
        ↓ (good)               ↓ (poor confidence)
    response_node        query_rewrite_node
        ↓                      ↓
       END               retrieval_node_retry
                               ↓
                         response_node
                               ↓
                              END

- If top retrieval score >= 0.3  → go straight to response_node
- If top retrieval score <  0.3  → rewrite the query, retry retrieval once,
                                    then go to response_node regardless
- Per-user FAISS filtering (your notes only answer your questions)
- Intent-aware retrieval: explain/quiz/interview fetch MORE chunks
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
    rewritten_query: Optional[str]   # set by query_rewrite_node if triggered
    was_rewritten: bool               # flag so response_node can note this
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

INTENT_TOP_K = {
    "qa":        5,
    "explain":   12,
    "quiz":      10,
    "interview": 10,
}


def _do_retrieval(query: str, state: AgentState) -> List[Dict[str, Any]]:
    """
    Shared retrieval logic used by both retrieval_node and retrieval_node_retry.
    Accepts the query string explicitly so the retry node can pass the
    rewritten query without touching state["query"].
    """
    db = state.get("db")
    user_id = state.get("user_id")
    intent = state.get("intent", "qa")
    top_k = INTENT_TOP_K.get(intent, settings.TOP_K_CHUNKS)

    try:
        query_embedding = get_query_embedding(query)
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

    except Exception as e:
        logger.exception(f"Retrieval failed: {e}")
        chunks_data = []

    return chunks_data


def retrieval_node(state: AgentState) -> AgentState:
    chunks_data = _do_retrieval(state["query"], state)

    top_score = chunks_data[0]["similarity"] if chunks_data else 0
    logger.info(
        f"Retrieval intent={state.get('intent')} user={state.get('user_id')} "
        f"query='{state['query'][:60]}' "
        f"→ {len(chunks_data)} chunks (best={top_score})"
    )
    return {**state, "retrieved_chunks": chunks_data}


# ── Conditional Edge: route after retrieval ────────────────────────────────────

# If the best chunk score is below this threshold, retrieval is considered poor
# and we branch to the query rewrite node instead of going straight to response.
REWRITE_THRESHOLD = 0.30


def route_after_retrieval(state: AgentState) -> str:
    """
    Decides the next node after retrieval_node.

    Returns:
        "query_rewrite"  — top score is low, try rephrasing the query
        "response_node"  — retrieval looks good, generate the answer
    """
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        # No results at all — rewrite and retry
        logger.info("route_after_retrieval: no chunks found → query_rewrite")
        return "query_rewrite"

    top_score = chunks[0]["similarity"]

    if top_score < REWRITE_THRESHOLD:
        logger.info(
            f"route_after_retrieval: low confidence ({top_score:.3f} < "
            f"{REWRITE_THRESHOLD}) → query_rewrite"
        )
        return "query_rewrite"

    logger.info(
        f"route_after_retrieval: good confidence ({top_score:.3f}) → response_node"
    )
    return "response_node"


# ── Node 3: Query Rewrite ──────────────────────────────────────────────────────

REWRITE_SYSTEM = (
    "You are a search query optimizer for a student placement preparation assistant. "
    "Your only job is to rewrite a student's question into a better search query "
    "that will find relevant content in technical study notes about topics like "
    "operating systems, databases, data structures, microcontrollers, and programming. "
    "Output ONLY the rewritten query — no explanation, no preamble, no punctuation at the end."
)

REWRITE_PROMPT = """The following student question did not retrieve good results from the study material database.
Rewrite it as a clearer, more specific search query that is more likely to match technical notes.

Rules:
- Keep it short (5-10 words)
- Use technical keywords the notes would contain
- Remove conversational filler ("can you", "please", "I want to know")
- If the question is already technical, try a synonym or broader/narrower phrasing

Original question: {query}

Rewritten query:"""


def query_rewrite_node(state: AgentState) -> AgentState:
    """
    Uses Groq to rephrase the original query into a better search query,
    then stores the result in state["rewritten_query"].
    The actual retry retrieval happens in retrieval_node_retry.
    """
    original_query = state["query"]

    try:
        response = _client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": REWRITE_PROMPT.format(query=original_query)},
            ],
            max_tokens=60,
            temperature=0.3,
        )
        rewritten = response.choices[0].message.content.strip()

        # Safety: if Groq returns something too long or empty, fall back
        if not rewritten or len(rewritten) > 200:
            rewritten = original_query

        logger.info(
            f"query_rewrite_node: '{original_query[:60]}' "
            f"→ '{rewritten[:60]}'"
        )

    except Exception as e:
        logger.warning(f"Query rewrite failed: {e} — using original query")
        rewritten = original_query

    return {**state, "rewritten_query": rewritten, "was_rewritten": True}


# ── Node 4: Retrieval Retry ────────────────────────────────────────────────────

def retrieval_node_retry(state: AgentState) -> AgentState:
    """
    Runs retrieval again using the rewritten query.
    If the retry also returns poor results, we keep whatever we got
    (response_node handles empty chunks gracefully with a general-knowledge fallback).
    """
    rewritten_query = state.get("rewritten_query") or state["query"]
    chunks_data = _do_retrieval(rewritten_query, state)

    top_score = chunks_data[0]["similarity"] if chunks_data else 0
    logger.info(
        f"retrieval_node_retry: rewritten='{rewritten_query[:60]}' "
        f"→ {len(chunks_data)} chunks (best={top_score})"
    )

    return {**state, "retrieved_chunks": chunks_data}


# ── Node 5: Response Generator ─────────────────────────────────────────────────

FIDELITY_INSTRUCTION = """
CRITICAL INSTRUCTIONS — follow these without exception:
1. Your answer MUST be based ENTIRELY on the CONTEXT FROM STUDY MATERIALS below.
2. Use the EXACT phrases, terminology, and definitions from the context. Do NOT paraphrase
   unnecessarily — reproduce the wording from the notes faithfully wherever it already answers
   the question well.
3. COVER EVERYTHING in the context that is relevant to the question — do NOT skip any type,
   category, item, or bullet point mentioned. Do NOT compress the context into a short summary.
   Expand your answer to use ALL retrieved chunks, not just the most relevant one or two.
4. If the context lists N types/categories/methods/points, your answer must address ALL N of
   them, each as its own clearly labelled item — never collapse a list into a single sentence
   or paragraph.
5. PRESERVE THE ORIGINAL DOCUMENT STRUCTURE:
   - Keep the same headings and subheadings used in the notes whenever a [heading] tag is
     present in the context (use them as Markdown headings, e.g. "## Heading", "### Subheading").
   - If a chunk has no explicit heading, infer a short heading from its first line/topic rather
     than dropping structure.
   - Preserve bullet lists as bullet lists, numbered lists as numbered lists, and tables as
     Markdown tables — never flatten them into prose.
   - Preserve important keywords, terms, and labels exactly as written in the notes.
6. If retrieved chunks belong to the SAME heading/topic, merge them smoothly under that one
   heading. If chunks belong to DIFFERENT headings/topics, present them under SEPARATE headings
   in the order that best matches the notes — do not blend unrelated topics into one paragraph.
7. The final answer should read like well-organized study notes extracted directly from the
   documents — NOT like an LLM-generated summary. Favor completeness and structure over brevity.
8. If the context does NOT contain enough information, say so explicitly, then add
   [General Knowledge] clearly labelled.
9. Never invent definitions, examples, headings, or facts not present in the context.

Priority order when these instructions interact: Accuracy > Completeness > Formatting > Brevity.
""".strip()

SYSTEM_BASE = (
    "You are a placement preparation assistant helping a student study from their own "
    "teacher's notes and study materials. Your job is to retrieve and present EVERYTHING "
    "that is in those notes relevant to the question — not just a summary. "
    "Students need complete coverage of all types, categories, and details their teacher wrote "
    "so they can give full answers in exams and interviews.\n\n"
    "You output well-organized study notes extracted directly from the source material — "
    "with the same headings, subheadings, bullet points, numbered lists, and tables the notes "
    "already use — NOT a compressed LLM-style summary. Reproduce the structure and wording of "
    "the notes faithfully; only add light connective phrasing where needed for readability. "
    "Never merge a multi-point list into a single paragraph, and never drop a heading, bullet, "
    "or item that appears in the retrieved context."
)

PROMPTS = {
    "qa": """{fidelity}

CONTEXT FROM STUDY MATERIALS:
{context}

PREVIOUS CONVERSATION:
{history}

STUDENT QUESTION: {query}

Answer using the exact wording, headings, and structure from the study materials above.
- If the context answers this in a single sentence or definition, give that directly using the
  original wording — no need to add headings for a one-line factual answer.
- If the context contains a list, multiple points, steps, or sub-parts relevant to the question,
  reproduce ALL of them as a Markdown bullet/numbered list under the original heading (or a short
  inferred heading) — do NOT compress them into a single paragraph.
- If the relevant content spans more than one heading/topic in the notes, present each under its
  own "## Heading" in Markdown rather than blending them together.
Start your answer with the most relevant sentence or heading from the context.""",

    "explain": """{fidelity}

CONTEXT FROM STUDY MATERIALS:
{context}

PREVIOUS CONVERSATION:
{history}

TOPIC TO EXPLAIN: {query}

Produce COMPLETE, WELL-ORGANIZED STUDY NOTES using ALL content from the study materials above —
not a summary. The output should read like the original notes, reorganized only enough to answer
the question, never compressed into a few sentences.

How to structure the answer:

1. Look at the "[heading]" tag attached to each source chunk above (when present). Use those
   EXACT headings/subheadings as Markdown headings in your answer (## for a main heading, ###
   for a sub-heading), in the order they make sense for the topic.
2. If two or more chunks share the same heading/topic, MERGE them smoothly under that one
   heading — do not repeat the heading twice.
3. If chunks belong to DIFFERENT headings/topics, keep them under SEPARATE headings — do not
   blend unrelated topics into one paragraph.
4. Within each heading, reproduce the notes faithfully:
   - Keep bullet points as "-" lists, numbered steps as "1. 2. 3." lists, and any tabular data
     as a Markdown table. Never flatten a list into prose.
   - Use the exact terminology and definitions from the notes; avoid unnecessary paraphrasing.
   - If the context lists multiple types/categories/methods under a heading, give EVERY one of
     them its own bullet or sub-heading with its full explanation — do not skip or shorten any.
5. If a chunk has no heading, give it a short heading inferred from its content rather than
   dropping it or merging it silently into another topic.
6. End with a short "## Key Points to Remember" section as a bullet list of the most important
   facts, using exact phrasing from the context — this is a recap, not a replacement for the
   detailed sections above.

Do not omit any heading, bullet, list item, or table that appears in the retrieved context.""",

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
    was_rewritten = state.get("was_rewritten", False)
    rewritten_query = state.get("rewritten_query", "")

    # Use the original query for display; the rewritten one was only for retrieval
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

        # Tell the user if the answer used a rewritten query
        if was_rewritten:
            prefix = (
                f"📚 *Answer based on your uploaded study materials.*\n"
                f"> 🔄 *Your question was automatically rephrased to improve search results.*\n\n"
            )
        else:
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
            max_tokens=4096,
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

    # Register all nodes
    workflow.add_node("intent_router", intent_router)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("query_rewrite_node", query_rewrite_node)
    workflow.add_node("retrieval_node_retry", retrieval_node_retry)
    workflow.add_node("response_node", response_node)

    # Entry point
    workflow.set_entry_point("intent_router")

    # Linear edges
    workflow.add_edge("intent_router", "retrieval_node")

    # ── CONDITIONAL EDGE ──────────────────────────────────────────────────────
    # After first retrieval, check confidence.
    # Good confidence  → go straight to response_node
    # Poor confidence  → rewrite query, retry retrieval, then response_node
    workflow.add_conditional_edges(
        "retrieval_node",
        route_after_retrieval,
        {
            "response_node":  "response_node",
            "query_rewrite":  "query_rewrite_node",
        }
    )

    # Rewrite path: rewrite → retry retrieval → response
    workflow.add_edge("query_rewrite_node", "retrieval_node_retry")
    workflow.add_edge("retrieval_node_retry", "response_node")

    # End
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
        "rewritten_query": None,
        "was_rewritten": False,
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


# ── Streaming variant ──────────────────────────────────────────────────────────

def run_agent_stream(
    query: str,
    chat_history: List[Dict],
    db,
    user_id: int,
):
    """
    Generator yielding dicts:
      {"type": "status",  "message": "..."}
      {"type": "chunk",   "content": "..."}
      {"type": "done",    "answer": "...", "sources": [...], "intent": "..."}
      {"type": "error",   "message": "..."}
    """
    state: AgentState = {
        "query": query,
        "rewritten_query": None,
        "was_rewritten": False,
        "user_id": user_id,
        "intent": None,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "chat_history": chat_history,
        "db": db,
    }

    # Node 1: intent
    state = intent_router(state)

    # Node 2: first retrieval
    yield {"type": "status", "message": "🔍 Searching relevant documents..."}
    state = retrieval_node(state)

    # Conditional edge
    route = route_after_retrieval(state)

    if route == "query_rewrite":
        yield {"type": "status", "message": "🔄 Refining your question for better results..."}
        state = query_rewrite_node(state)

        yield {"type": "status", "message": "🔍 Searching again with refined query..."}
        state = retrieval_node_retry(state)

    # Build context + prompt (identical to response_node logic)
    intent = state.get("intent", "qa")
    chunks = state.get("retrieved_chunks", [])
    was_rewritten = state.get("was_rewritten", False)

    history_parts = []
    for msg in (chat_history or [])[-6:]:
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

        if was_rewritten:
            prefix = (
                f"📚 *Answer based on your uploaded study materials.*\n"
                f"> 🔄 *Your question was automatically rephrased to improve search results.*\n\n"
            )
        else:
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

    sources = [
        {
            "filename": c["filename"],
            "page_num": c["page_num"],
            "chunk_index": c["chunk_index"],
            "heading": c["heading"],
            "similarity": c["similarity"],
            "confidence": c["confidence"],
            "content_preview": (
                c["content"][:400] + "..." if len(c["content"]) > 400 else c["content"]
            ),
        }
        for c in chunks
    ]

    yield {"type": "status", "message": "🧠 Generating response..."}

    full_answer = prefix
    if prefix:
        yield {"type": "chunk", "content": prefix}

    try:
        stream = _client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_BASE},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.2,
            stream=True,
        )
        for event in stream:
            delta = None
            if event.choices:
                delta = event.choices[0].delta.content
            if delta:
                full_answer += delta
                yield {"type": "chunk", "content": delta}
    except Exception as e:
        logger.exception(f"Groq streaming generation failed: {e}")
        err_text = "⚠️ Error generating response. Please check your GROQ_API_KEY and try again."
        full_answer += err_text
        yield {"type": "chunk", "content": err_text}

    yield {
        "type": "done",
        "answer": full_answer,
        "sources": sources,
        "intent": intent,
    }