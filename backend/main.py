"""
Module 8 Project — Containerized RAG Assistant
Backend: FastAPI Application
=============================
Run locally:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import requests
from config import settings
import rag

app = FastAPI(title="RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCS_DIR = "./docs"

# ── Request / Response schemas ─────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str
    n_results: int = settings.max_results
    max_distance: float = settings.confidence_threshold

    @field_validator("question")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v


class SourceChunk(BaseModel):
    text: str
    source: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: str


class IngestResponse(BaseModel):
    chunks_ingested: int
    message: str


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"message": "RAG API is running", "docs": "/docs"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    RAG query pipeline:
      1. rag.retrieve(req.question, req.n_results, req.max_distance)
      2. If no chunks, return a helpful "no documents" response
      3. rag.build_prompt(req.question, chunks)
      4. rag.generate(messages)
      5. rag.compute_confidence(chunks)
      6. Return AskResponse
    """
    chunks = rag.retrieve(req.question, req.n_results, req.max_distance)

    if not chunks:
        return AskResponse(
            answer=(
                "I couldn't find relevant information in the ingested documents to answer your question. "
                "Try re-indexing documents or asking a question related to the corpus."
            ),
            sources=[],
            confidence="low",
        )

    messages = rag.build_prompt(req.question, chunks)

    try:
        answer = rag.generate(messages)
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    confidence = rag.compute_confidence(chunks)
    formatted_sources = [
        SourceChunk(
            text=c["text"],
            source=c.get("metadata", {}).get("source", "unknown"),
            distance=round(float(c["distance"]), 4),
        )
        for c in chunks
    ]

    return AskResponse(
        answer=answer,
        sources=formatted_sources,
        confidence=confidence,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Load documents from DOCS_DIR and upsert into ChromaDB.
    """
    chunks = rag.load_documents(DOCS_DIR)
    if not chunks:
        return IngestResponse(
            chunks_ingested=0,
            message=f"No .txt or .md documents found in {DOCS_DIR}.",
        )

    rag.collection.upsert(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["id"] for c in chunks],
    )

    return IngestResponse(
        chunks_ingested=len(chunks),
        message=f"Successfully ingested {len(chunks)} chunks from {DOCS_DIR}.",
    )


@app.get("/stats")
def stats():
    """
    Return document count, model name, and db path.
    """
    return {
        "document_count": rag.collection.count(),
        "model": settings.model_name,
        "db_path": settings.chroma_path,
    }


@app.get("/health")
def health():
    """
    Return status of ChromaDB and Ollama.
    """
    doc_count = 0
    chromadb_status = "ok"
    try:
        doc_count = rag.collection.count()
    except Exception:
        chromadb_status = "error"

    ollama_status = "disconnected"
    try:
        resp = requests.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=3)
        if resp.status_code == 200:
            ollama_status = "connected"
    except requests.RequestException:
        ollama_status = "disconnected"

    overall_status = "ok" if (chromadb_status == "ok" and ollama_status == "connected") else "degraded"

    return {
        "status": overall_status,
        "chromadb": chromadb_status,
        "ollama": ollama_status,
        "document_count": doc_count,
    }