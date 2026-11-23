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
```

### Services Overview
* **`frontend` (Streamlit):** Chat UI with session-persisted history, sidebar health diagnostics, document counter, one-click re-indexing, color-coded confidence badges, and expandable source citations.
* **`backend` (FastAPI):** Exposes `/ask`, `/ingest`, `/stats`, `/health`, and `/` endpoints; handles paragraph-based chunking, ChromaDB cosine retrieval, distance-threshold filtering, prompt construction, and Ollama chat generation.
* **`ollama` (Ollama):** Local LLM inference server running `llama3.2:1b` with persistent model storage via a named Docker volume.

---

## 📁 Project Structure

```text
Minimal-RAG-Powered-Chatbot/
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
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` before starting the stack:

* **Windows (Command Prompt):**
  ```cmd
  copy .env.example .env
  ```
* **Linux / macOS / PowerShell:**
  ```bash
  cp .env.example .env
  ```

| Variable | Description | Default (Docker) |
| :--- | :--- | :--- |
| `OLLAMA_URL` | Internal URL for the Ollama service | `http://ollama:11434` |
| `MODEL_NAME` | Local Ollama model used for generation | `llama3.2:1b` |
| `CHROMA_PATH` | Persistent directory inside the backend container | `/app/rag_db` |
| `MAX_RESULTS` | Number of top chunks retrieved per query | `3` |
| `CONFIDENCE_THRESHOLD` | Maximum cosine distance allowed (`0.0` – `2.0`) | `1.0` |
| `DEBUG` | Enable verbose debug logging | `false` |

---

## 🚀 Quick Start (Docker Compose)

### 1. Build and Start All Services
```bash
docker-compose up --build -d
```

### 2. Pull the Required Ollama Model
Download `llama3.2:1b` into the running Ollama container (persisted in `ollama_data`):
```bash
docker exec -it minimalrag-poweredchatbot-ollama-1 ollama pull llama3.2:1b
```
*(Tip: Run `docker-compose ps` if your folder name produces a different container prefix.)*

### 3. Ingest Documents & Start Chatting
1. Open the Streamlit UI at **`http://localhost:8501`**.
2. Verify in the sidebar that **Backend**, **ChromaDB**, and **Ollama** show as connected/ok.
3. Click **Re-index Documents** in the sidebar to ingest all `.txt` and `.md` files from `backend/docs/`.
4. Ask questions in the chat input and inspect the **Confidence** badge and expandable **Sources** section.

---

## 🔌 API Reference

Interactive Swagger documentation is available at **`http://localhost:8000/docs`**.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health/welcome check |
| `GET` | `/health` | Connectivity status for ChromaDB and Ollama + chunk count |
| `GET` | `/stats` | Current chunk count, active LLM model, and ChromaDB path |
| `POST` | `/ingest` | Loads `.txt`/`.md` files from `./docs`, chunks by paragraph, and upserts to ChromaDB |
| `POST` | `/ask` | Retrieves matching chunks, applies distance guardrails, and generates a cited answer |

---

## 🛡️ Guardrails & Reliability

* **Cosine Distance Filtering:** Chunks exceeding `CONFIDENCE_THRESHOLD` (`max_distance`) are discarded before prompt construction so irrelevant context does not pollute the LLM response.
* **No-Context Fallback:** If ChromaDB is empty or no chunks pass the distance threshold, `/ask` returns a safe fallback message with `confidence: "low"` instead of hallucinating.
* **Grounded System Prompt:** Instructs the model to answer strictly from the provided context, cite source filenames, and keep answers concise.
* **Input Validation:** Pydantic validators reject empty or whitespace-only questions (`HTTP 422`).

---

## 🧪 Running Tests

Run the backend test suite inside the container:

```bash
docker exec -it minimalrag-poweredchatbot-backend-1 pytest tests/ -v
```

---

## 🔮 Future Improvements

* **Overlapping Chunking:** Add configurable token/character overlap to preserve context across paragraph boundaries.
* **Expanded File Support:** Support `.pdf` and `.docx` ingestion alongside `.txt` and `.md`.
* **Hybrid Search & Reranking:** Combine dense vector search with keyword (BM25) scoring or a cross-encoder reranker for higher retrieval precision.
* **Streaming Responses:** Stream tokens from Ollama `/api/chat` directly into the Streamlit chat UI.
