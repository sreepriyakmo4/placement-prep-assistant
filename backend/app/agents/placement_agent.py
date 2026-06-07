"""
LangGraph placement preparation agent.
Nodes: intent_router -> retrieval_node -> response_node
"""
import json
import logging
from typing import TypedDict, List, Optional, Dict, Any

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from app.retrieval.gemini_service import gemini_service
from app.retrieval.faiss_store import faiss_store
from app.db.schemas import Source

logger = logging.getLogger(__name__)


# ─── State ───────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    sources: List[Dict]
    mode_override: Optional[str]
    history: List[Dict[str, str]]


# ─── Node 1: Intent Router ────────────────────────────────────────────────────

INTENT_PROMPT = """You are an intent classifier for a placement preparation assistant.
Classify the following student query into EXACTLY one of these intents:
- qa: factual question needing a concise answer
- explain: request for detailed explanation with examples
- quiz: request to generate quiz or practice questions
- interview: request for interview questions or mock interview

Respond with ONLY the intent word, nothing else.

Examples:
"What is deadlock?" -> qa
"Explain the OSI model layers" -> explain
"Give me quiz questions on data structures" -> quiz
"Ask me Amazon SDE interview questions" -> interview
"What is the difference between TCP and UDP?" -> qa
"Generate 5 MCQs on sorting algorithms" -> quiz
"How does garbage collection work? Explain in detail" -> explain
"Conduct a mock interview for Google SDE role" -> interview

Query: {query}
Intent:"""


async def intent_router(state: AgentState) -> AgentState:
    # Allow caller to override intent detection
    if state.get("mode_override"):
        state["intent"] = state["mode_override"]
        return state

    try:
        intent_raw = await gemini_service.generate(
            INTENT_PROMPT.format(query=state["query"])
        )
        intent = intent_raw.strip().lower()
        if intent not in ("qa", "explain", "quiz", "interview"):
            intent = "qa"
        state["intent"] = intent
    except Exception as e:
        logger.warning(f"Intent detection failed: {e}, defaulting to qa")
        state["intent"] = "qa"
    return state


# ─── Node 2: Retrieval ────────────────────────────────────────────────────────

async def retrieval_node(state: AgentState) -> AgentState:
    try:
        query_embedding = await gemini_service.embed_query(state["query"])
        results = faiss_store.search(query_embedding, top_k=5)
        state["retrieved_chunks"] = [
            {"distance": dist, "metadata": meta} for dist, meta in results
        ]
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")
        state["retrieved_chunks"] = []
    return state


# ─── Node 3: Response Generator ──────────────────────────────────────────────

SYSTEM_BASE = """You are an expert placement preparation coach helping students ace technical interviews.
You have access to relevant study materials. Always base your answers on the provided context.
Be encouraging, precise, and practical."""

PROMPTS = {
    "qa": """Based on the context below, answer the question concisely and accurately.

Context:
{context}

Question: {query}

Chat History:
{history}

Provide a clear, direct answer. If the context doesn't cover the question, say so honestly.""",

    "explain": """Based on the context below, provide a comprehensive explanation.

Context:
{context}

Topic to explain: {query}

Chat History:
{history}

Structure your response as:
1. **Core Concept**: Clear definition
2. **How it Works**: Step-by-step breakdown
3. **Example**: Concrete, relatable example
4. **Interview Tips**: What interviewers look for
5. **Common Mistakes**: What to avoid""",

    "quiz": """Based on the context below, generate a placement-focused quiz.

Context:
{context}

Topic: {query}

Chat History:
{history}

Generate exactly 5 multiple-choice questions in this format:

**Quiz: [Topic Name]**

**Q1.** [Question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]

[Repeat for Q2-Q5]

---
**ANSWERS:**
Q1: [Letter] - [Brief explanation]
Q2: [Letter] - [Brief explanation]
...

Make questions progressively harder. Focus on concepts likely asked in placements.""",

    "interview": """Based on the context below, conduct a technical interview simulation.

Context:
{context}

Interview request: {query}

Chat History:
{history}

Act as a senior interviewer. Ask 5 interview questions that:
1. Start with a warm-up question
2. Progress to core technical concepts
3. Include a problem-solving question
4. Ask a design/architecture question
5. End with a behavioral question

Format:
**Technical Interview Simulation**

*[Brief intro as interviewer]*

**Question 1 (Warm-up):** [Question]
*Expected answer: [What a good answer covers]*

[Continue for Q2-Q5]

---
**Evaluation Guide:** [What to look for in responses]"""
}


async def response_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "qa")
    chunks = state.get("retrieved_chunks", [])

    # Build context string from retrieved chunks, fetching content from DB
    context_parts = []
    sources = []

    for chunk_info in chunks:
        meta = chunk_info.get("metadata", {})
        # We'll look up chunk content from metadata (populated during search)
        content = meta.get("content", "")
        if not content:
            content = "[Chunk content unavailable]"
        filename = meta.get("filename", "Unknown")
        chunk_idx = meta.get("chunk_index", 0)
        doc_id = meta.get("document_id", 0)
        chunk_id = meta.get("chunk_id", 0)

        context_parts.append(
            f"[Source: {filename}, Chunk {chunk_idx}]\n{content}"
        )
        sources.append({
            "filename": filename,
            "chunk_index": chunk_idx,
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "content_preview": content[:150] + "..." if len(content) > 150 else content,
        })

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found in uploaded materials."

    # Build history string
    history_parts = []
    for msg in (state.get("history") or [])[-6:]:  # last 3 exchanges
        role = msg.get("role", "user").capitalize()
        history_parts.append(f"{role}: {msg['content']}")
    history = "\n".join(history_parts) if history_parts else "No previous conversation."

    prompt_template = PROMPTS.get(intent, PROMPTS["qa"])
    prompt = prompt_template.format(
        context=context,
        query=state["query"],
        history=history,
    )

    try:
        answer = await gemini_service.generate(prompt, system=SYSTEM_BASE)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        answer = "I encountered an error generating a response. Please try again."

    state["answer"] = answer
    state["sources"] = sources
    return state


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("intent_router", intent_router)
    graph.add_node("retrieval_node", retrieval_node)
    graph.add_node("response_node", response_node)

    graph.set_entry_point("intent_router")
    graph.add_edge("intent_router", "retrieval_node")
    graph.add_edge("retrieval_node", "response_node")
    graph.add_edge("response_node", END)

    return graph.compile()


# Singleton
placement_agent = build_agent()


async def run_agent(
    query: str,
    history: List[Dict[str, str]] = None,
    mode_override: Optional[str] = None,
    db=None,
) -> Dict[str, Any]:
    """Run the placement preparation agent and enrich sources with DB content."""
    initial_state: AgentState = {
        "query": query,
        "intent": "",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "mode_override": mode_override,
        "history": history or [],
    }

    # Run the graph
    result = await placement_agent.ainvoke(initial_state)

    # Enrich chunks with DB content if db session provided
    if db and result["retrieved_chunks"]:
        from sqlalchemy import select
        from app.db.models import Chunk

        # Get all faiss_ids from retrieved chunks to batch query
        enriched_sources = []
        for chunk_info in result["retrieved_chunks"]:
            meta = chunk_info.get("metadata", {})
            doc_id = meta.get("document_id")
            chunk_idx = meta.get("chunk_index")

            if doc_id and chunk_idx is not None:
                stmt = select(Chunk).where(
                    Chunk.document_id == doc_id,
                    Chunk.chunk_index == chunk_idx,
                )
                chunk_result = await db.execute(stmt)
                chunk = chunk_result.scalar_one_or_none()
                if chunk:
                    meta["content"] = chunk.content
                    meta["chunk_id"] = chunk.id
                    preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                    enriched_sources.append({
                        "filename": meta.get("filename", "Unknown"),
                        "chunk_index": chunk_idx,
                        "document_id": doc_id,
                        "chunk_id": chunk.id,
                        "content_preview": preview,
                    })

        result["sources"] = enriched_sources

        # Re-run response node with enriched content
        result = await placement_agent.ainvoke({
            **initial_state,
            "intent": result["intent"],
            "retrieved_chunks": result["retrieved_chunks"],
        })
        result["sources"] = enriched_sources

    return result
