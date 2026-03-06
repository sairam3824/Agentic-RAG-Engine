from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

QueryType = Literal["factual", "analytical", "comparison", "how-to"]


class RetrievedDoc(TypedDict, total=False):
    id: str
    content: str
    source: str
    score: float
    metadata: dict[str, Any]


class Evaluation(TypedDict, total=False):
    relevance: float
    completeness: float
    freshness: float
    overall: float
    good_enough: bool
    reason: str


class TraceEvent(TypedDict, total=False):
    node: str
    message: str
    data: dict[str, Any]


class RAGState(TypedDict, total=False):
    question: str
    query: str
    optimized_query: str
    refined_query: str
    query_type: QueryType
    needs_retrieval: bool
    retrieved_docs: list[RetrievedDoc]
    evaluation: Evaluation
    answer: str
    sources: list[str]
    requested_sources: list[str]
    iteration_count: int
    max_iterations: int
    retrieval_trace: list[TraceEvent]
    confidence: float
    unanswerable_points: list[str]
    fact_check: list[dict[str, Any]]
    error: NotRequired[str]
