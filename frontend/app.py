"""
Module 8 Project — Containerized RAG Assistant
Frontend: Streamlit Chat UI
============================
Run locally:
    BACKEND_URL=http://localhost:8000 streamlit run app.py
"""

import os
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

CONFIDENCE_COLOURS = {
    "high": "green",
    "medium": "orange",
    "low": "red",
}

# ── Step 1: Page configuration ─────────────────────────────────────────────
st.set_page_config(page_title="RAG Assistant", page_icon="🔍", layout="centered")

# ── Step 2: Session state initialisation ───────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def render_confidence(confidence: str) -> None:
    colour = CONFIDENCE_COLOURS.get(confidence, "gray")
    st.markdown(f":{colour}[Confidence: **{confidence}**]")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for idx, src in enumerate(sources, start=1):
            source_name = src.get("source", "unknown")
            distance = src.get("distance", "N/A")
            text = src.get("text", "")
            st.markdown(f"**{idx}. `{source_name}`** (distance: `{distance}`)")
            st.caption(text)
            if idx < len(sources):
                st.divider()


# ════════════════════════════════════════════════════════════════════════════
# SECTION A — SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("RAG Assistant")
    st.write("Ask questions grounded in your indexed documents via FastAPI, ChromaDB, and Ollama.")

    health_data = None
    try:
        h_resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if h_resp.status_code == 200:
            health_data = h_resp.json()
            st.success("Backend: connected")
            st.caption(
                f"ChromaDB: `{health_data.get('chromadb', 'unknown')}` | "
                f"Ollama: `{health_data.get('ollama', 'unknown')}` | "
                f"Docs/Chunks: `{health_data.get('document_count', 0)}`"
            )
        else:
            st.error(f"Backend: error ({h_resp.status_code})")
    except requests.exceptions.RequestException:
        st.error("Backend: unreachable")

    st.divider()

    if st.button("Re-index Documents", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            try:
                ing_resp = requests.post(f"{BACKEND_URL}/ingest", timeout=120)
                if ing_resp.status_code == 200:
                    ing_data = ing_resp.json()
                    count = ing_data.get("chunks_ingested", 0)
                    msg = ing_data.get("message", "Ingestion complete.")
                    st.success(f"{msg} ({count} chunks)")
                    st.rerun()
                else:
                    st.error(f"Ingestion failed ({ing_resp.status_code}): {ing_resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach the backend at {BACKEND_URL}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION B — MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════
st.title("RAG Assistant")
st.caption("Ask questions grounded in your documents.")

# ── Chat history display ───────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    role = msg.get("role", "assistant")
    with st.chat_message(role):
        st.markdown(msg.get("content", ""))
        if role == "assistant":
            if "confidence" in msg:
                render_confidence(msg["confidence"])
            if "sources" in msg:
                render_sources(msg["sources"])

# ── Chat input ─────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": prompt},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    confidence = data.get("confidence", "low")

                    st.markdown(answer)
                    render_confidence(confidence)
                    render_sources(sources)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "confidence": confidence,
                        }
                    )
                else:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except ValueError:
                        detail = resp.text
                    st.error(f"Error ({resp.status_code}): {detail}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach the backend at {BACKEND_URL}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request error: {exc}")