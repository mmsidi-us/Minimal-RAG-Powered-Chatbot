"""
Module 8 Project — Containerized RAG Assistant
Backend: RAG Pipeline
======================
Pipeline overview:
    1. load_documents(directory)  — read .txt / .md files and split into chunks
    2. retrieve(query, n, max_distance) — query ChromaDB for the most relevant chunks
    3. build_prompt(question, chunks) — format system prompt + context
    4. generate(messages)         — call Ollama's /api/chat endpoint
    5. compute_confidence(chunks) — rate result quality from distances
"""

import os
import requests
import chromadb
from config import settings

# ── ChromaDB client ────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
collection = chroma_client.get_or_create_collection(
    "documents",
    configuration={"hnsw": {"space": "cosine"}},
)


# ── 1. Document loading ────────────────────────────────────────────────────

def load_documents(directory: str) -> list[dict]:
    """
    Read all .txt and .md files in `directory`.
    Split each file into paragraphs (split on blank lines).
    Return a list of dicts with keys: "text", "id", "metadata".
    """
    if not os.path.isdir(directory):
        return []

    chunks: list[dict] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith((".txt", ".md")):
            continue

        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for idx, paragraph in enumerate(paragraphs):
            chunks.append(
                {
                    "text": paragraph,
                    "id": f"{filename}_{idx}",
                    "metadata": {
                        "source": filename,
                        "chunk_index": str(idx),
                    },
                }
            )

    return chunks


# ── 2. Retrieval ───────────────────────────────────────────────────────────

def retrieve(query: str, n_results: int = 3, max_distance: float = 1.2) -> list[dict]:
    """
    Query ChromaDB for the `n_results` most relevant chunks.
    Filter out any chunks with distance > max_distance.
    Return a list of dicts with keys: "text", "metadata", "distance".
    """
    if collection.count() == 0:
        return []

    limit = min(max(1, n_results), collection.count())
    results = collection.query(
        query_texts=[query],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    filtered: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        if dist is not None and float(dist) <= max_distance:
            filtered.append(
                {
                    "text": doc,
                    "metadata": meta or {},
                    "distance": float(dist),
                }
            )

    return filtered


# ── 3. Prompt building ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer ONLY from the provided context. "
    "If the context doesn't contain the answer, say you don't have enough "
    "information. Cite source documents by name. Keep responses under 200 words."
)


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """
    Build the messages list for Ollama's /api/chat endpoint.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Source {i}: {source}]\n{text}")

    context_str = "\n\n".join(context_parts)
    system_content = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_str}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]


# ── 4. Generation ──────────────────────────────────────────────────────────

def generate(messages: list[dict]) -> str:
    """
    Send messages to Ollama's /api/chat endpoint and return the response text.
    """
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.model_name,
        "messages": messages,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip() or "No response generated."
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {settings.ollama_url}. Make sure Ollama is running."
        )
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Failed to communicate with Ollama: {exc}") from exc


# ── 5. Confidence scoring ──────────────────────────────────────────────────

def compute_confidence(chunks: list[dict]) -> str:
    """
    Return "high", "medium", or "low" based on the best (lowest) distance score.
    """
    if not chunks:
        return "low"

    best_distance = min(float(c.get("distance", 999.0)) for c in chunks)
    if best_distance < 0.5:
        return "high"
    if best_distance < 1.0:
        return "medium"
    return "low"


# ── Quick test (run as a script) ───────────────────────────────────────────

if __name__ == "__main__":
    docs = load_documents("./docs")
    print(f"Loaded {len(docs)} chunks from ./docs")

    if docs:
        collection.upsert(
            documents=[d["text"] for d in docs],
            metadatas=[d["metadata"] for d in docs],
            ids=[d["id"] for d in docs],
        )
        print(f"Upserted {len(docs)} chunks. Total in collection: {collection.count()}")
        sample = retrieve("What is RAG?")
        print(f"Retrieved {len(sample)} chunks for 'What is RAG?'")