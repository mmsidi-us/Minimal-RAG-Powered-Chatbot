# Containerized RAG Assistant

A containerized Retrieval-Augmented Generation (RAG) chatbot using **FastAPI**, **ChromaDB**, **Ollama**, and **Streamlit**, orchestrated via **Docker Compose**.

## Architecture

```text
[Streamlit UI :8501] --HTTP--> [FastAPI Backend :8000] --Cosine Query--> [ChromaDB (/app/rag_db)]
                                         |
                                         +-------POST /api/chat--------> [Ollama :11434 (/root/.ollama)]