"""
Module 8 Project — Containerized RAG Assistant
Backend: API Tests
===================
Run with:
    pytest tests/ -v
"""

import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402

client = TestClient(app)


def test_root():
    """GET / should return HTTP 200 with a 'message' field."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_health():
    """GET /health should return HTTP 200 with status, chromadb, and ollama keys."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "chromadb" in data
    assert "ollama" in data
    assert "document_count" in data


def test_stats():
    """GET /stats should return HTTP 200 with document_count and model keys."""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "document_count" in data
    assert "model" in data
    assert "db_path" in data


def test_ask_empty_question_rejected():
    """POST /ask with whitespace-only question should return 422."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422