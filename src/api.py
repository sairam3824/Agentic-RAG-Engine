from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

from src.graph.workflow import AgenticRAGEngine
from src.ingest import extract_text

app = FastAPI(title="agentic-rag-engine", version="0.1.0")
engine = AgenticRAGEngine()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=3, ge=1, le=3)


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    evaluation: dict[str, Any] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    iteration_count: int
    retrieval_trace: list[dict[str, Any]] = Field(default_factory=list)
    fact_check: list[dict[str, Any]] = Field(default_factory=list)
    unanswerable_points: list[str] = Field(default_factory=list)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "agentic-rag-engine",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    state = engine.run(
        question=payload.question,
        sources=payload.sources,
        max_iterations=payload.max_iterations,
    )
    return QueryResponse(
        answer=state.get("answer", ""),
        confidence=float(state.get("confidence", 0.0) or 0.0),
        evaluation=state.get("evaluation", {}),
        sources=state.get("sources", []),
        iteration_count=int(state.get("iteration_count", 0)),
        retrieval_trace=state.get("retrieval_trace", []),
        fact_check=state.get("fact_check", []),
        unanswerable_points=state.get("unanswerable_points", []),
    )


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    docs: list[dict[str, Any]] = []
    for file in files:
        content = await file.read()
        text = extract_text(file.filename, content)
        if not text:
            continue
        docs.append(
            {
                "content": text,
                "source": file.filename,
                "metadata": {"source": file.filename, "content_type": file.content_type or "unknown"},
            }
        )

    indexed = engine.nodes.vector_retriever.add_documents(docs)
    return {"indexed_documents": indexed, "received_files": len(files)}
