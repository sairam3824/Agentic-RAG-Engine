from __future__ import annotations

import os
import sys

import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.graph.workflow import AgenticRAGEngine
from src.ingest import extract_text

st.set_page_config(page_title="Agentic RAG Engine", layout="wide")
st.title("Agentic RAG Engine")
st.caption("Traditional RAG retrieves blindly. Agentic RAG retrieves intelligently.")

if "engine" not in st.session_state:
    st.session_state.engine = AgenticRAGEngine()
if "messages" not in st.session_state:
    st.session_state.messages = []

engine: AgenticRAGEngine = st.session_state.engine

with st.sidebar:
    st.subheader("Retrieval Sources")
    selected_sources = st.multiselect(
        "Select sources",
        options=["vector", "web", "sql"],
        default=["vector", "web", "sql"],
    )

    st.subheader("Document Upload")
    uploads = st.file_uploader("Upload PDF/TXT/MD", accept_multiple_files=True, type=["pdf", "txt", "md"])
    if st.button("Index uploaded documents"):
        docs = []
        for uploaded_file in uploads or []:
            text = extract_text(uploaded_file.name, uploaded_file.getvalue())
            if text:
                docs.append(
                    {
                        "content": text,
                        "source": uploaded_file.name,
                        "metadata": {"source": uploaded_file.name},
                    }
                )
        indexed = engine.nodes.vector_retriever.add_documents(docs)
        st.success(f"Indexed {indexed} document chunks.")

    st.subheader("Run Metrics")
    st.write("- Max retrieval loops: 3")
    st.write("- Evaluation includes relevance/completeness/freshness")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    state = engine.run(question=question, sources=selected_sources, max_iterations=3)

    answer = state.get("answer", "I could not produce an answer.")
    confidence = float(state.get("confidence", 0.0) or 0.0)
    iteration_count = int(state.get("iteration_count", 0))
    consulted_sources = state.get("sources", [])

    with st.chat_message("assistant"):
        st.markdown(answer)

        c1, c2, c3 = st.columns(3)
        c1.metric("Iterations", iteration_count)
        c2.metric("Sources consulted", len(consulted_sources))
        c3.metric("Confidence", f"{confidence:.2f}")

        with st.expander("Retrieval Trace", expanded=False):
            st.json(state.get("retrieval_trace", []))

        with st.expander("Evaluated Scores", expanded=False):
            st.json(state.get("evaluation", {}))

        with st.expander("Top Retrieved Docs", expanded=False):
            for idx, doc in enumerate(state.get("retrieved_docs", [])[:5], start=1):
                st.markdown(f"**[{idx}] {doc.get('source', 'unknown')}**")
                st.write(doc.get("content", "")[:400])

        with st.expander("Fact Check", expanded=False):
            st.json(state.get("fact_check", []))

    st.session_state.messages.append({"role": "assistant", "content": answer})
