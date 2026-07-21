# 🎓 PlacementAI — AI-Powered Placement Preparation Assistant

A full-stack AI assistant to help students prepare for placements by uploading study materials and interacting with an intelligent agent powered by **Groq (Llama 3.3) + LangGraph + FAISS**.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│   Login | Chat | Documents | Sessions | Source Citations    │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────┐
│                      FastAPI Backend                         │
│   /auth  /documents  /chat  /quiz                             │
└───────┬──────────────────┬──────────────────────────────────┘
        │                  │
┌───────▼──────┐  ┌────────▼──────────────────────────────────┐
│  PostgreSQL  │  │           LangGraph Agent                  │
│  users       │  │  intent_router → retrieval_node →          │
│  sessions    │  │  (conditional: query_rewrite → retry) →    │
│  messages    │  │  response_node                             │
│  documents   │  └────────┬──────────────────────────────────┘
│  chunks      │           │
└──────────────┘  ┌────────▼────────────────┐
                  │   FAISS Vector Store    │
                  │   Local MiniLM Embeddings│
                  │   Top-K cosine search   │
                  └─────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- PostgreSQL 15
- Groq API key

---

### 1. Clone & Setup

```bash
git clone <repo>
cd placement-prep
```

---

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY, SECRET_KEY, and DATABASE_URL
```

**`.env` file:**
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/placement_prep
SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
GROQ_API_KEY=gsk_your-groq-api-key
FAISS_INDEX_PATH=./faiss_index
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

> `SECRET_KEY` and `GROQ_API_KEY` are required — the app will refuse to start without them (see `app/core/config.py`).

---

### 3. Database Setup

```bash
# Create the database
psql -U postgres -c "CREATE DATABASE placement_prep;"
```

Tables are auto-created on startup via SQLAlchemy's `Base.metadata.create_all()` — no separate migration step is needed for local dev.

---

### 4. Start Backend

```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

---

### 5. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Configure API URL (optional, defaults to localhost:8000)
echo "VITE_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

Frontend available at: http://localhost:5173

---

### 6. Docker (Full Stack)

```bash
# From root directory
docker-compose up --build
```

- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 📁 Project Structure

```
placement-prep/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── auth.py          # Register & Login endpoints
│   │   │   ├── chat.py          # Chat query & session endpoints
│   │   │   ├── documents.py     # PDF upload & management
│   │   │   ├── quiz.py          # Quiz generation, submission, history
│   │   │   └── deps.py          # JWT auth dependency
│   │   ├── agents/
│   │   │   └── graph.py         # LangGraph StateGraph agent
│   │   ├── retrieval/
│   │   │   ├── faiss_store.py   # FAISS vector store wrapper
│   │   │   └── embeddings.py    # Local sentence-transformers embedding utils
│   │   ├── ingest/
│   │   │   ├── pdf_processor.py # PyMuPDF text extraction + chunking
│   │   │   └── pipeline.py      # End-to-end ingestion pipeline
│   │   ├── db/
│   │   │   ├── base.py          # SQLAlchemy engine & session
│   │   │   └── models.py        # ORM models
│   │   ├── core/
│   │   │   ├── config.py        # Pydantic settings (fail-fast on missing secrets)
│   │   │   └── security.py      # JWT & password hashing
│   │   └── main.py               # FastAPI app entry point
│   ├── eval/
│   │   └── run_retrieval_eval.py # Retrieval quality evaluation harness
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── features/
│   │   │   ├── auth/
│   │   │   │   ├── AuthContext.tsx   # React auth context + state
│   │   │   │   └── LoginPage.tsx     # Login/Register page
│   │   │   ├── chat/
│   │   │   │   └── ChatPage.tsx      # Main chat interface (SSE streaming)
│   │   │   ├── documents/
│   │   │   │   └── DocumentsPanel.tsx # PDF management panel
│   │   │   └── quiz/
│   │   │       └── QuizPage.tsx      # Quiz taking + results
│   │   ├── lib/
│   │   │   ├── api.ts           # Axios API client
│   │   │   └── utils.ts         # Helpers
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   ├── tailwind.config.js
│   └── Dockerfile
└── docker-compose.yml
```

---

## 🤖 LangGraph Agent Modes

The agent detects intent from your query using keyword matching, then routes to a mode-specific prompt:

| Mode | Trigger Keywords | Behavior |
|------|-----------------|----------|
| **Q&A** | Default | Concise, direct answers |
| **Explain** | "explain", "how does", "what is" | Detailed explanation + examples + tips |
| **Quiz** | "quiz", "mcq", "test me" | 5 MCQs with answers at the end |
| **Interview** | "interview", "ask me", "mock" | Interviewer-style progressive questions |

If the top retrieval score falls below a confidence threshold, the agent automatically rewrites the query (via Groq) and retries retrieval once before falling back to general knowledge — implemented as a conditional edge in the LangGraph state machine (`app/agents/graph.py`).

---

## 📋 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login, returns JWT |
| POST | `/documents/upload` | Upload PDF (multipart) |
| GET | `/documents` | List user's documents |
| DELETE | `/documents/{id}` | Delete document |
| POST | `/chat/query` | Send message, get AI response |
| POST | `/chat/query/stream` | Send message, stream response via SSE |
| GET | `/chat/sessions` | List chat sessions |
| GET | `/chat/sessions/{id}` | Get session messages |
| DELETE | `/chat/sessions/{id}` | Delete session |
| POST | `/quiz/generate/{doc_id}` | Generate a 15-question MCQ quiz for a document |
| POST | `/quiz/submit/{doc_id}` | Submit answers, get score + weak-topic breakdown |
| GET | `/quiz/history/{doc_id}` | Get quiz attempt history for a document |

---

## 🧪 Retrieval Evaluation

A retrieval quality harness lives at `backend/eval/run_retrieval_eval.py`. It runs a golden set of labeled questions against the FAISS index and reports Precision@5 (whether the correct source document appears in the top-5 retrieved chunks).

```bash
cd backend
python -m eval.run_retrieval_eval
```

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | postgres://... | PostgreSQL connection string |
| `SECRET_KEY` | *(required, no default)* | JWT signing secret — must be 32+ random characters |
| `GROQ_API_KEY` | *(required, no default)* | Groq API key for LLM generation and query rewriting |
| `FAISS_INDEX_PATH` | `./faiss_index` | FAISS persistence directory |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated allowed origins |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers embedding model |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model used for generation |
| `CHUNK_SIZE` | `800` | Target text chunk size in chars |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K_CHUNKS` | `5` | Default number of chunks retrieved |

---

## 🔑 Getting a Groq API Key

1. Go to [Groq Console](https://console.groq.com/keys)
2. Create a new API key
3. Add it to your `.env` file as `GROQ_API_KEY`

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, SQLAlchemy, LangChain, LangGraph
- **AI**: Groq (Llama 3.3) for generation, local `sentence-transformers` (MiniLM) for embeddings
- **Vector DB**: FAISS (CPU, `IndexFlatIP` for cosine similarity)
- **Database**: PostgreSQL
- **Frontend**: React 18, TypeScript, TailwindCSS, React Query, Axios
- **Auth**: JWT (python-jose), bcrypt

---

## 📝 Notes

- FAISS index is persisted to disk at `FAISS_INDEX_PATH` and reloaded on startup; if the DB has chunks but the index is empty (e.g. after a volume wipe), it's automatically rebuilt from Postgres on startup.
- PDF ingestion runs in the background; status updates to `done` when complete.
- Uploaded files are validated by both extension and magic bytes (`%PDF-`) before being queued for ingestion.
- Chat history (last 6 messages) is sent to Groq for context continuity.
- Source citations (filename, page, heading, similarity score) are stored as JSON alongside each assistant message.
- Chunking is paragraph- and heading-aware (not naive fixed-length splitting): text is split on paragraph boundaries, headings always start a new chunk, and each chunk carries forward ~300 characters of the previous chunk's last paragraph for context continuity.
- Handwritten or scanned PDFs currently fail ingestion (PyMuPDF returns empty text for image-based pages) — OCR support is a planned improvement.