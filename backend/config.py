"""
Module 8 Project — Containerized RAG Assistant
Backend: Configuration
======================
Reads all configuration from environment variables.
"""

import os


class Settings:
    def __init__(self) -> None:
        self.ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.model_name: str = os.environ.get("MODEL_NAME", "llama3.2:1b")
        self.chroma_path: str = os.environ.get("CHROMA_PATH", "./rag_db")
        self.max_results: int = int(os.environ.get("MAX_RESULTS", "3"))
        self.confidence_threshold: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "1.0"))
        self.debug: bool = os.environ.get("DEBUG", "false").lower() == "true"


settings = Settings()