# 🎓 PlacementAI — AI-Powered Placement Preparation Assistant

A full-stack AI assistant to help students prepare for placements by uploading study materials and interacting with an intelligent agent powered by **Gemini + LangGraph + FAISS**.

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
│   /auth  /documents  /chat                                   │
└───────┬──────────────────┬──────────────────────────────────┘
        │                  │
┌───────▼──────┐  ┌────────▼──────────────────────────────────┐
│  PostgreSQL  │  │           LangGraph Agent                  │
│  users       │  │  intent_router → retrieval_node →          │
│  sessions    │  │  response_node                             │
│  messages    │  └────────┬──────────────────────────────────┘
│  documents   │           │
│  chunks      │  ┌────────▼────────────────┐
└──────────────┘  │   FAISS Vector Store    │
                  │   Gemini Embeddings     │
                  │   Top-5 semantic search │
                  └─────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- PostgreSQL 15
- Google Gemini API key

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
# Edit .env and add your GOOGLE_API_KEY and DATABASE_URL
```

**`.env` file:**
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/placement_prep
SECRET_KEY=your-super-secret-key-minimum-32-chars
GOOGLE_API_KEY=your-google-gemini-api-key
FAISS_INDEX_PATH=./faiss_index
```

---

### 3. Database Setup

```bash
# Create the database
psql -U postgres -c "CREATE DATABASE placement_prep;"

# Run migrations
alembic upgrade head

# OR let the app auto-create tables on startup (Base.metadata.create_all)
```

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
│   │   │   └── deps.py          # JWT auth dependency
│   │   ├── agents/
│   │   │   └── graph.py         # LangGraph StateGraph agent
│   │   ├── retrieval/
│   │   │   ├── faiss_store.py   # FAISS vector store wrapper
│   │   │   └── embeddings.py    # Gemini embedding utils
│   │   ├── ingest/
│   │   │   ├── pdf_processor.py # PyMuPDF text extraction + chunking
│   │   │   └── pipeline.py      # End-to-end ingestion pipeline
│   │   ├── db/
│   │   │   ├── base.py          # SQLAlchemy engine & session
│   │   │   └── models.py        # ORM models
│   │   ├── core/
│   │   │   ├── config.py        # Pydantic settings
│   │   │   └── security.py      # JWT & password hashing
│   │   └── main.py               # FastAPI app entry point
│   ├── alembic/                 # Database migrations
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── features/
│   │   │   ├── auth/
│   │   │   │   ├── AuthContext.tsx   # React auth context + state
│   │   │   │   └── LoginPage.tsx     # Login/Register page
│   │   │   ├── chat/
│   │   │   │   └── ChatPage.tsx      # Main chat interface
│   │   │   └── documents/
│   │   │       └── DocumentsPanel.tsx # PDF management panel
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

The agent automatically detects intent from your query:

| Mode | Trigger Keywords | Behavior |
|------|-----------------|----------|
| **Q&A** | Default | Concise, direct answers |
| **Explain** | "explain", "how does", "what is" | Detailed explanation + examples + tips |
| **Quiz** | "quiz", "mcq", "test me" | 5 MCQs with answers at the end |
| **Interview** | "interview", "ask me", "mock" | Interviewer-style progressive questions |

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
| GET | `/chat/sessions` | List chat sessions |
| GET | `/chat/sessions/{id}` | Get session messages |
| DELETE | `/chat/sessions/{id}` | Delete session |

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | postgres://... | PostgreSQL connection string |
| `SECRET_KEY` | - | JWT signing secret (required) |
| `GOOGLE_API_KEY` | - | Gemini API key (required) |
| `FAISS_INDEX_PATH` | `./faiss_index` | FAISS persistence directory |
| `CHUNK_SIZE` | `500` | Text chunk size in chars |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_CHUNKS` | `5` | Number of chunks retrieved |

---

## 🔑 Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file as `GOOGLE_API_KEY`

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, SQLAlchemy, Alembic, LangChain, LangGraph
- **AI**: Google Gemini 1.5 Flash, Gemini Embeddings
- **Vector DB**: FAISS (CPU)
- **Database**: PostgreSQL
- **Frontend**: React 18, TypeScript, TailwindCSS, React Query, Axios
- **Auth**: JWT (python-jose), bcrypt

---

## 📝 Notes

- FAISS index is persisted to disk at `FAISS_INDEX_PATH` and reloaded on startup
- PDF ingestion runs in the background; status updates to `done` when complete
- Chat history (last 6 messages) is sent to Gemini for context continuity
- Source citations are stored as JSON alongside each assistant message
