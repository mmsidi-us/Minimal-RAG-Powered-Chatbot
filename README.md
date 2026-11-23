# 🔍 Containerized RAG Assistant

A minimal, full-stack Retrieval-Augmented Generation (RAG) chatbot packaged with Docker Compose. It ingests local documents into **ChromaDB** using cosine similarity, retrieves relevant context chunks via a **FastAPI** backend, generates grounded answers locally using **Ollama**, and provides an interactive **Streamlit** chat interface with source citations and confidence scoring.

---

## 🏗️ Architecture

```text
┌─────────────────────────┐         HTTP (REST)         ┌─────────────────────────┐
│   Streamlit Frontend    │ ──────────────────────────► │     FastAPI Backend     │
│     (Port 8501)         │ ◄────────────────────────── │       (Port 8000)       │
└─────────────────────────┘   JSON {answer, sources,    └───────────┬─────────────┘
                                    confidence}                     │
                                                        ┌───────────┴─────────────┐
                                                        │                         │
                                                        ▼                         ▼
                                            ┌───────────────────────┐ ┌───────────────────────┐
                                            │       ChromaDB        │ │        Ollama         │
                                            │  (Cosine Vector DB)   │ │  (Local LLM Server)   │
                                            │  Volume: chroma_data  │ │  Port: 11434          │
                                            └───────────────────────┘ │  Volume: ollama_data  │
                                                                      └───────────────────────┘

Services Overview
frontend (Streamlit): Chat UI with session-persisted history, sidebar health diagnostics, document counter, one-click re-indexing, color-coded confidence badges, and expandable source citations.

backend (FastAPI): Exposes /ask, /ingest, /stats, /health, and / endpoints; handles paragraph-based chunking, ChromaDB cosine retrieval, distance-threshold filtering, prompt construction, and Ollama chat generation.

ollama (Ollama): Local LLM inference server running llama3.2:1b with persistent model storage via a named Docker volume.

📁 Project Structure

containerized-rag/
├── docker-compose.yml          # Multi-service orchestration & named volumes
├── .env.example                # Environment variable template
├── .gitignore                  # Excludes .env, local vector DB, and cache files
├── backend/
│   ├── Dockerfile              # FastAPI container definition
│   ├── main.py                 # API routes & Pydantic schemas
│   ├── rag.py                  # Ingestion, retrieval, prompt builder, generation, confidence
│   ├── config.py               # Environment-driven Settings class
│   ├── requirements.txt        # Backend Python dependencies
│   ├── docs/                   # Corpus files (.txt / .md)
│   └── tests/
│       └── test_api.py         # Pytest API smoke & validation tests
├── frontend/
│   ├── Dockerfile              # Streamlit container definition
│   ├── app.py                  # Streamlit UI & backend API client
│   └── requirements.txt        # Frontend Python dependencies
└── README.md

⚙️ Configuration
Copy .env.example to .env before starting the stack:

Windows (Command Prompt):
copy .env.example .env

Linux / macOS / PowerShell:
cp .env.example .env

VariableDescriptionDefault (Docker)OLLAMA_URLInternal URL for the Ollama servicehttp://ollama:11434MODEL_NAMELocal Ollama model used for generationllama3.2:1bCHROMA_PATHPersistent directory inside the backend container/app/rag_dbMAX_RESULTSNumber of top chunks retrieved per query3CONFIDENCE_THRESHOLDMaximum cosine distance allowed (0.0 – 2.0)1.0DEBUGEnable verbose debug loggingfalse
🚀 Quick Start
(Docker Compose)
1. Build and Start All Services
docker-compose up --build -d

2. Pull the Required Ollama Model
Download llama3.2:1b into the running Ollama container (persisted in ollama_data):
docker exec -it <ollama-container-name> ollama pull llama3.2:1b

3. Ingest Documents & Start Chatting
Open the Streamlit UI at http://localhost:8501.

Verify in the sidebar that Backend, ChromaDB, and Ollama show as connected/ok.

Click Re-index Documents in the sidebar to ingest all .txt and .md files from backend/docs/.

Ask questions in the chat input and inspect the Confidence badge and expandable Sources section.

🔌 API Reference
Interactive Swagger documentation is available at http://localhost:8000/docs.

Method	Endpoint	Description
GET	/	Health/welcome check
GET	/health	Connectivity status for ChromaDB and Ollama + chunk count
GET	/stats	Current chunk count, active LLM model, and ChromaDB path
POST	/ingest	Loads .txt/.md files from ./docs, chunks by paragraph, and upserts to ChromaDB
POST	/ask	Retrieves matching chunks, applies distance guardrails, and generates a cited answer

🛡️ Guardrails & Reliability
Cosine Distance Filtering: Chunks exceeding CONFIDENCE_THRESHOLD (max_distance) are discarded before prompt construction so irrelevant context does not pollute the LLM response.

No-Context Fallback: If ChromaDB is empty or no chunks pass the distance threshold, /ask returns a safe fallback message with confidence: "low" instead of hallucinating.

Grounded System Prompt: Instructs the model to answer strictly from the provided context, cite source filenames, and keep answers concise.

Input Validation: Pydantic validators reject empty or whitespace-only questions (HTTP 422).

🧪 Running Tests
Run the backend test suite inside the container (or locally from backend/):
docker exec -it <backend-container-name> pytest tests/ -v

🔮 Future Improvements
Overlapping Chunking: Add configurable token/character overlap to preserve context across paragraph boundaries.

Expanded File Support: Support .pdf and .docx ingestion alongside .txt and .md.

Hybrid Search & Reranking: Combine dense vector search with keyword (BM25) scoring or a cross-encoder reranker for higher retrieval precision.

Streaming Responses: Stream tokens from Ollama /api/chat directly into the Streamlit chat UI.

### How to Apply It
1. Open `README.md` in your project root and replace its contents with the Markdown above.
2. Replace `<ollama-container-name>` and `<backend-container-name>` with your actual container names (`minimalrag-poweredchatbot-ollama-1` and `minimalrag-poweredchatbot-backend-1`) if you want copy-paste commands tailored to your folder name.
3. Stage, commit, and push the updated `README.md` to GitHub:
   ```cmd
   git add README.md
   git commit -m "Add comprehensive README with architecture diagram and setup guide"
   git push
